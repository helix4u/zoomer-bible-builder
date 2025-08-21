#!/usr/bin/env python3
# zoomer_bible_builder.py
# Build a "Zoomer" translation by walking KJV verse-by-verse using a local OpenAI-compatible API.

import argparse
import json
import os
import re
import sys
import time
import signal
from typing import Dict, List, Iterator, Optional, Tuple

import requests

BOOKS_URL = "https://raw.githubusercontent.com/aruljohn/Bible-kjv/master/Books.json"
RAW_BOOK_URL = "https://raw.githubusercontent.com/aruljohn/Bible-kjv/master/{fname}"

DEFAULT_API_BASE = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL = "gemma-3-4b-it"

PROGRESS_PATH = "progress_zoomer_bible.json"

# Anti-echo stops… DO NOT include "\n" here or you’ll clip the content to just the reference.
DEFAULT_STOPS = ["Reference:", "AI:", "---"]

def graceful_exit(signum, frame):
    print("\nStopping cleanly...")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)

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

# normalize curly quotes and em/en dashes to ASCII
_ASCII_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201C": '"', "\u201D": '"',
    "\u2013": "...", "\u2014": "...",
    "\u00A0": " ",
})
def normalize_ascii(s: str) -> str:
    return s.translate(_ASCII_MAP)

def sanitize_one_line(s: str) -> str:
    if not s:
        return ""
    s = normalize_ascii(s)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    # cut anything after '---'
    s = re.sub(r'\s*---.*$', '', s)
    # remove common scaffolding echoes
    s = re.sub(r'^\s*(AI|Answer|Output)\s*:\s*', '', s, flags=re.I)
    s = re.sub(r'\s*Reference\s*:.*$', '', s, flags=re.I)
    # restrict to basic glyphs
    s = "".join(ch for ch in s if ord(ch) < 0x2500)
    return s.strip()

def ref_string(book: str, chapter: int, verse: int) -> str:
    return f"{book} {chapter}:{verse}"

def make_user_prompt(book: str, chapter: int, verse: int, text: str) -> str:
    text = normalize_ascii(str(text))
    return f'Reference: {book} {chapter}:{verse}\nText: "{text}"'

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

def flatten_all_verses(books_list: List[str], repo_dir: Optional[str]) -> List[Dict]:
    out = []
    for book in books_list:
        bjson = try_fetch_book_json(book, repo_dir)
        for book_name, chap_i, verse_i, vtext in iter_book_verses(bjson):
            out.append({"book": book_name, "chapter": chap_i, "verse": verse_i, "text": vtext})
    return out

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
    extra_headers: Optional[Dict[str,str]] = None,
) -> str:
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
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
    # Prefer the first bracketed match
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

# ---------- driver ----------
def run(
    system_prompt_path: str,
    out_txt: str,
    api_base: str,
    model: str,
    start_index: Optional[int],
    end_index: Optional[int],
    temperature: float,
    rate_limit_s: float,
    retries: int,
    stream: bool,
    repo_dir: Optional[str],
):
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        sys_prompt = f.read().strip()

    print("Loading book list...")
    books_list = fetch_books_list(repo_dir)
    print(f"Books: {len(books_list)}")

    print("Flattening verses...")
    verses = flatten_all_verses(books_list, repo_dir)
    total = len(verses)
    print(f"Total verses: {total}")

    idx = start_index if start_index is not None else load_progress()
    last = end_index if end_index is not None else total

    print(f"Starting at index: {idx}")
    print(f"Ending before index: {last}")
    print(f"Output file: {out_txt}")
    os.makedirs(os.path.dirname(os.path.abspath(out_txt)), exist_ok=True)

    with open(out_txt, "a", encoding="utf-8") as outf:
        while idx < last and idx < total:
            v = verses[idx]
            book, chapter, verse, text = v["book"], int(v["chapter"]), int(v["verse"]), v["text"]

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
                        stops=DEFAULT_STOPS
                    )
                    line = postprocess_line(raw, expected_prefix)

                    # Fallback retry if body came back empty… try once with no stops and non-streaming
                    if not body_ok(line, expected_prefix):
                        raw2 = chat_once(
                            api_base=api_base,
                            model=model,
                            system_prompt=sys_prompt,
                            user_prompt=user_prompt,
                            temperature=max(0.8, temperature),
                            stream=False,
                            max_tokens=-1,
                            stops=None
                        )
                        line2 = postprocess_line(raw2, expected_prefix)
                        if body_ok(line2, expected_prefix):
                            line = line2

                    outf.write(line + "\n")
                    outf.flush()
                    save_progress(idx + 1)
                    preview = line[:120] + ("..." if len(line) > 120 else "")
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
    p.add_argument("--start-index", type=parse_index_arg, default=None, help="Zero-based verse index to start from.")
    p.add_argument("--end-index", type=parse_index_arg, default=None, help="Stop before this zero-based index.")
    p.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    p.add_argument("--rate-limit-s", type=float, default=0.2, help="Seconds to sleep between requests.")
    p.add_argument("--retries", type=int, default=5, help="Max retries per verse.")
    p.add_argument("--no-stream", action="store_true", help="Disable streaming.")
    p.add_argument("--repo-dir", default=None, help="Optional path to a local clone of aruljohn/Bible-kjv to read JSON from.")
    args = p.parse_args()

    run(
        system_prompt_path=args.system_prompt,
        out_txt=args.out,
        api_base=args.api_base,
        model=args.model,
        start_index=args.start_index,
        end_index=args.end_index,
        temperature=args.temperature,
        rate_limit_s=args.rate_limit_s,
        retries=args.retries,
        stream=(not args.no_stream),
        repo_dir=args.repo_dir,
    )
