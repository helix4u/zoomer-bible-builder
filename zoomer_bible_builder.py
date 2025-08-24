#!/usr/bin/env python3
# zoomer_bible_builder.py
# Build a "Zoomer" translation by walking KJV verse-by-verse using a local OpenAI-compatible API,
# preserving a sliding chat history so the model varies phrasing across nearby verses.

import argparse
import json
import os
import re
import sys
import time
import signal
import unicodedata
from typing import Dict, List, Iterator, Optional, Tuple, Set

import requests

BOOKS_URL = "https://raw.githubusercontent.com/aruljohn/Bible-kjv/master/Books.json"
RAW_BOOK_URL = "https://raw.githubusercontent.com/aruljohn/Bible-kjv/master/{fname}"

DEFAULT_API_BASE = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL = "gemma-3-4b-it"

PROGRESS_PATH = "progress_zoomer_bible.json"

# Anti-echo stops... DO NOT include "\n" or you'll clip content.
DEFAULT_STOPS = ["Reference:", "AI:", "---"]

def graceful_exit(signum, frame):
    print("\nStopping cleanly...")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------- util ----------
def http_get_json(url: str, timeout: int = 60) -> any:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def read_json(path: str) -> any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, data: any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def save_progress(i: int) -> None:
    write_json(PROGRESS_PATH, {"index": int(i)})

def load_progress() -> int:
    if not os.path.exists(PROGRESS_PATH):
        return 0
    try:
        data = read_json(PROGRESS_PATH)
        return int(data.get("index", 0))
    except Exception:
        return 0

# normalize curly quotes and dashes to ASCII
_ASCII_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201C": '"', "\u201D": '"',
    "\u2013": "...", "\u2014": "...",
    "\u00A0": " ",
})
def normalize_ascii(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return s.translate(_ASCII_MAP)

def sanitize_one_line(s: str) -> str:
    if not s:
        return ""
    s = normalize_ascii(s)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r'\s*---.*$', '', s)                           # cut anything after '---'
    s = re.sub(r'^\s*(AI|Answer|Output)\s*:\s*', '', s, flags=re.I)
    s = re.sub(r'\s*Reference\s*:.*$', '', s, flags=re.I)
    s = "".join(ch for ch in s if ord(ch) < 0x2500)           # conservative glyph filter
    return s.strip()

def ref_string(book: str, chapter: int, verse: int) -> str:
    return f"{book} {chapter}:{verse}"

def make_user_prompt(book: str, chapter: int, verse: int, text: str) -> str:
    text = normalize_ascii(str(text))
    return f'Reference: {book} {chapter}:{verse}\nText: "{text}"'

def parse_ref_from_output_line(line: str) -> Optional[str]:
    # Expect prefix like: [Genesis 1:1]
    m = re.match(r'^\[([^\]]+)\]', line.strip())
    return m.group(1) if m else None

def normalize_book_filename(book: str) -> str:
    fname = re.sub(r"[\\s'’:-]+", "", book)
    return f"{fname}.json"

def try_fetch_book_json(book: str, repo_dir: Optional[str]) -> dict:
    # local first
    if repo_dir:
        candidates = [
            os.path.join(repo_dir, f"{book}.json"),
            os.path.join(repo_dir, normalize_book_filename(book)),
            os.path.join(repo_dir, book.replace(" ", "") + ".json"),
            os.path.join(repo_dir, book.replace(" ", "").lower() + ".json"),
            os.path.join(repo_dir, book.lower() + ".json"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return read_json(p)
    # remote fallback
    for fname in [normalize_book_filename(book), normalize_book_filename(book).lower()]:
        url = RAW_BOOK_URL.format(fname=fname)
        try:
            return http_get_json(url)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                continue
            raise
    raise FileNotFoundError(f"Could not fetch JSON for book '{book}'.")

def fetch_books_list(repo_dir: Optional[str], cache_path: str = "Books.json") -> List[str]:
    if repo_dir:
        local = os.path.join(repo_dir, "Books.json")
        if os.path.exists(local):
            books = read_json(local)
            if isinstance(books, list) and books:
                return books
    if os.path.exists(cache_path):
        try:
            books = read_json(cache_path)
            if isinstance(books, list) and books:
                return books
        except Exception:
            pass
    books = http_get_json(BOOKS_URL)
    if not isinstance(books, list) or not books:
        raise RuntimeError("Unexpected Books.json structure.")
    write_json(cache_path, books)
    return books

# ---------- robust verse iterator ----------
def _parse_verse_entry(vobj: dict, book_name: str, chap_i: int) -> Tuple[int, str]:
    # Accept both {"1":"text"} and {"verse": 1, "text":"text"} with case variants
    if not isinstance(vobj, dict) or not vobj:
        raise ValueError(f"Unrecognized verse entry in {book_name} {chap_i}: {vobj!r}")
    if len(vobj) == 1:
        k, val = next(iter(vobj.items()))
        k_str = str(k).strip()
        if k_str.isdigit():
            return int(k_str), str(val)
    lower_map = {str(k).lower(): k for k in vobj.keys()}
    if "verse" in lower_map and "text" in lower_map:
        vkey = lower_map["verse"]
        tkey = lower_map["text"]
        return int(str(vobj[vkey]).strip()), str(vobj[tkey])
    raise ValueError(f"Unrecognized verse object keys in {book_name} {chap_i}: {list(vobj.keys())}")

def iter_book_verses(book_obj: dict) -> Iterator[Tuple[str, int, int, str]]:
    book_name = book_obj.get("book")
    chapters = book_obj.get("chapters", [])
    for ch in chapters:
        chap_raw = ch.get("chapter")
        chap_i = int(str(chap_raw).strip())
        verses = ch.get("verses", [])
        for vobj in verses:
            verse_i, vtext = _parse_verse_entry(vobj, book_name, chap_i)
            yield (book_name, chap_i, verse_i, vtext)

def flatten_all_verses(books_list: List[str], repo_dir: Optional[str]) -> Tuple[List[Dict], Dict[str, str]]:
    out = []
    ref_to_text: Dict[str, str] = {}
    for book in books_list:
        bjson = try_fetch_book_json(book, repo_dir)
        for book_name, chap_i, verse_i, vtext in iter_book_verses(bjson):
            ref = f"{book_name} {chap_i}:{verse_i}"
            out.append({"book": book_name, "chapter": chap_i, "verse": verse_i, "text": vtext})
            ref_to_text[ref] = vtext
    return out, ref_to_text

# ---------- history helpers ----------
def tail_lines(path: str, max_lines: int) -> List[str]:
    if not os.path.exists(path) or max_lines <= 0:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [ln.rstrip("\r\n") for ln in lines[-max_lines:]]
    except Exception:
        return []

def build_history_messages(out_path: str, ref_to_text: Dict[str, str], pairs: int) -> List[Dict[str, str]]:
    """
    Returns a list of messages like:
      user:   Reference + Text (from original KJV)
      assistant: [Book C:V] zoomer line
    for the last `pairs` completed verses in the output file.
    """
    msgs: List[Dict[str, str]] = []
    recent = tail_lines(out_path, pairs)
    for line in recent:
        ref = parse_ref_from_output_line(line)
        if not ref:
            continue
        # Reconstruct the original "user" verse prompt from our corpus
        kjv = ref_to_text.get(ref)
        if not kjv:
            continue
        user_msg = f'Reference: {ref}\nText: "{normalize_ascii(kjv)}"'
        # Assistant message is exactly the produced line
        assist_msg = sanitize_one_line(line)
        if not assist_msg:
            continue
        msgs.append({"role": "user", "content": user_msg})
        msgs.append({"role": "assistant", "content": assist_msg})
    return msgs

# ---------- chat ----------
def chat_once(
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.9,
    stream: bool = True,
    max_tokens: int = -1,
    stops: Optional[List[str]] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    extra_headers: Optional[Dict[str,str]] = None,
) -> str:
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    messages = [{"role": "system", "content": system_prompt}]
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }
    if stops:
        payload["stop"] = stops

    if stream:
        with requests.post(api_base, headers=headers, json=payload, stream=True, timeout=600) as r:
            r.raise_for_status()
            out = []
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("data: "):
                    data = raw[len("data: "):].strip()
                else:
                    data = raw.strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    ch0 = obj.get("choices", [{}])[0]
                    delta = ch0.get("delta", {})
                    piece = delta.get("content")
                    if piece is None:
                        msg = ch0.get("message", {})
                        piece = msg.get("content")
                    if piece:
                        out.append(piece)
                except Exception:
                    pass
            return "".join(out).strip()
    else:
        r = requests.post(api_base, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        obj = r.json()
        return obj["choices"][0]["message"]["content"].strip()

def postprocess_line(result: str, expected_prefix: str) -> str:
    if not result:
        return expected_prefix
    m = re.search(r'\[' + re.escape(expected_prefix[1:-1]) + r'\][^\n\r]*', result)
    s = m.group(0) if m else result
    s = sanitize_one_line(s)
    if not s.startswith(expected_prefix):
        body = re.sub(r'^[\[\("]+', "", s).strip()
        s = f"{expected_prefix} {body}"
    return s

def body_ok(line: str, expected_prefix: str, min_chars: int = 6) -> bool:
    body = line[len(expected_prefix):].strip()
    return len(body) >= min_chars

def opening_ngram(s: str, n: int = 2) -> str:
    """Return a lowercase n-gram of the opening tokens, stripping punctuation."""
    toks: List[str] = []
    for t in re.split(r"\s+", s.strip().lower()):
        t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
        if t:
            toks.append(t)
        if len(toks) >= n:
            break
    return " ".join(toks)

# ---------- driver ----------
def run(
    system_prompt_path: str,
    out_txt: str,
    api_base: str,
    model: str,
    api_key: Optional[str],
    start_index: Optional[int],
    end_index: Optional[int],
    temperature: float,
    rate_limit_s: float,
    retries: int,
    stream: bool,
    repo_dir: Optional[str],
    ctx_pairs: int,
):
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        sys_prompt = f.read().strip()

    extra_headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

    print("Loading book list...")
    books_list = fetch_books_list(repo_dir)
    print(f"Books: {len(books_list)}")

    print("Flattening verses...")
    verses, ref_to_text = flatten_all_verses(books_list, repo_dir)
    total = len(verses)
    print(f"Total verses: {total}")

    idx = start_index if start_index is not None else load_progress()
    last = end_index if end_index is not None else total

    print(f"Starting at index: {idx}")
    print(f"Ending before index: {last}")
    print(f"Output file: {out_txt}")
    os.makedirs(os.path.dirname(os.path.abspath(out_txt)), exist_ok=True)
    seen_starts: Set[str] = set()
    if os.path.exists(out_txt):
        with open(out_txt, "r", encoding="utf-8") as fprev:
            for ln in fprev:
                opener = opening_ngram(ln.split("]",1)[-1])
                if opener:
                    seen_starts.add(opener)

    with open(out_txt, "a", encoding="utf-8") as outf:
        while idx < last and idx < total:
            v = verses[idx]
            book, chapter, verse, text = v["book"], int(v["chapter"]), int(v["verse"]), v["text"]

            # Build sliding history from file + ref_to_text, limited to `ctx_pairs`
            history = build_history_messages(out_txt, ref_to_text, ctx_pairs)

            user_prompt = make_user_prompt(book, chapter, verse, text)
            expected_prefix = f"[{ref_string(book, chapter, verse)}]"

            attempt = 0
            while True:
                attempt += 1
                try:
                    raw = chat_once(
                        api_base=api_base,
                        model=model,
                        system_prompt=sys_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        stream=stream,
                        max_tokens=-1,
                        stops=DEFAULT_STOPS,
                        history_messages=history,
                        extra_headers=extra_headers,
                    )
                    raw_for_log = raw
                    line = postprocess_line(raw, expected_prefix)

                    # If body is empty or repeats the same opening as recent outputs, try one fallback
                    if not body_ok(line, expected_prefix):
                        # fallback call: no stops, non-streaming
                        raw2 = chat_once(
                            api_base=api_base,
                            model=model,
                            system_prompt=sys_prompt,
                            user_prompt=user_prompt,
                            temperature=max(0.9, temperature),
                            stream=False,
                            max_tokens=-1,
                            stops=None,
                            history_messages=history,
                            extra_headers=extra_headers,
                        )
                        line2 = postprocess_line(raw2, expected_prefix)
                        if body_ok(line2, expected_prefix):
                            line = line2
                            raw_for_log = raw2

                    # Lightweight anti-dup opener check vs history and all prior lines
                    recent_assistant_lines = [m["content"] for m in history if m["role"] == "assistant"] if history else []
                    recent_starts = {opening_ngram(ln.split("]",1)[-1]) for ln in recent_assistant_lines}
                    this_start = opening_ngram(line.split("]",1)[-1])
                    if this_start in recent_starts or this_start in seen_starts:
                        # Ask once more for variation with a tiny nudge: add a short "user" reminder to history
                        nudge = {"role":"user","content":"Reminder: Vary your opening phrasing... avoid repeating recent openers."}
                        raw3 = chat_once(
                            api_base=api_base,
                            model=model,
                            system_prompt=sys_prompt,
                            user_prompt=user_prompt,
                            temperature=max(1.0, temperature),
                            stream=False,
                            max_tokens=-1,
                            stops=None,
                            history_messages=history + [nudge],
                            extra_headers=extra_headers,
                        )
                        line3 = postprocess_line(raw3, expected_prefix)
                        if body_ok(line3, expected_prefix):
                            line = line3
                            raw_for_log = raw3
                            this_start = opening_ngram(line.split("]",1)[-1])

                    outf.write(line + "\n")
                    outf.flush()
                    save_progress(idx + 1)
                    seen_starts.add(this_start)
                    preview_src = normalize_ascii(raw_for_log or line)
                    preview_src = preview_src.replace("\r", " ").replace("\n", " ").strip()
                    preview = preview_src[:120] + ("..." if len(preview_src) > 120 else "")
                    print(f"{idx+1}/{total}: {preview}")
                    time.sleep(rate_limit_s)
                    break
                except requests.HTTPError as e:
                    code = e.response.status_code if e.response is not None else "?"
                    print(f"HTTP error {code} on verse {idx}... attempt {attempt}/{retries}")
                except requests.RequestException as e:
                    print(f"Network error on verse {idx}: {e}... attempt {attempt}/{retries}")
                except Exception as e:
                    print(f"Unexpected error on verse {idx}: {e}... attempt {attempt}/{retries}")

                if attempt >= retries:
                    print("Giving up on this verse for now. You can resume later.")
                    save_progress(idx)
                    return

                time.sleep(min(5 * attempt, 30))  # backoff

            idx += 1

    print("All done.")

def parse_index_arg(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return int(s)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build a Zoomer translation of the KJV using a local LLM.")
    p.add_argument("--system-prompt", default="system_zoomer_prompt.txt", help="Path to the system prompt file.")
    p.add_argument("--out", default="zoomer_bible.txt", help="Output file to append verses to.")
    p.add_argument("--api-base", default=DEFAULT_API_BASE, help="OpenAI-compatible chat completions endpoint.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    p.add_argument("--api-key", default=os.environ.get("API_KEY"), help="Optional API key for the endpoint.")
    p.add_argument("--start-index", type=parse_index_arg, default=None, help="Zero-based verse index to start from.")
    p.add_argument("--end-index", type=parse_index_arg, default=None, help="Stop before this zero-based index.")
    p.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    p.add_argument("--rate-limit-s", type=float, default=0.2, help="Seconds to sleep between requests.")
    p.add_argument("--retries", type=int, default=5, help="Max retries per verse.")
    p.add_argument("--no-stream", action="store_true", help="Disable streaming.")
    p.add_argument("--repo-dir", default=None, help="Optional path to a local clone of aruljohn/Bible-kjv to read JSON from.")
    p.add_argument("--ctx-pairs", type=int, default=10, help="How many prior user/assistant pairs to include as chat history.")
    args = p.parse_args()

    run(
        system_prompt_path=args.system_prompt,
        out_txt=args.out,
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
        start_index=args.start_index,
        end_index=args.end_index,
        temperature=args.temperature,
        rate_limit_s=args.rate_limit_s,
        retries=args.retries,
        stream=(not args.no_stream),
        repo_dir=args.repo_dir,
        ctx_pairs=args.ctx_pairs,
    )
