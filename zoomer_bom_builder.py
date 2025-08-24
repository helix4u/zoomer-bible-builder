#!/usr/bin/env python3
# zoomer_bom_builder.py
# Make a "Zoomer" Book of Mormon by calling a local OpenAI-compatible API,
# using the JSON from https://github.com/johngthecreator/Book_of_Mormon_Scriptures (flat array of verses).
#
# Defaults: pulls book-of-mormon.json if --bom-json not provided.
# Keeps a sliding chat history of the last N verse pairs to reduce repetitive openings.
#
# Notes:
# - Sends Authorization header if --api-key or API_KEY is provided (fixes 401).
# - Omits max_tokens unless > 0 (some providers reject negative values).
# - On HTTP errors, prints response body for easier debugging.

import argparse
import json
import os
import re
import sys
import time
import signal
from typing import Dict, List, Optional, Tuple

import requests

# ---- source (flat array) ----
BOM_JSON_URL = "https://raw.githubusercontent.com/johngthecreator/Book_of_Mormon_Scriptures/main/book-of-mormon.json"
STD_JSON_URL = "https://raw.githubusercontent.com/johngthecreator/Book_of_Mormon_Scriptures/main/standard-works.json"

DEFAULT_API_BASE = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL = "gemma-3-4b-it"

OUT_PATH_DEFAULT = "zoomer_bom.txt"
PROGRESS_PATH = "progress_zoomer_bom.json"

# Anti-echo stops (do NOT include "\n" or you will clip the content to the ref only)
DEFAULT_STOPS = ["Reference:", "AI:", "---"]

def graceful_exit(signum, frame):
    print("\nStopping cleanly...")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)

# ---------- utils ----------
def http_get_json(url: str, timeout: int = 60):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, data):
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

# normalize curly quotes/dashes to ASCII; squash NBSP
_ASCII_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201C": '"', "\u201D": '"',
    "\u2013": "...", "\u2014": "...",
    "\u00A0": " ",
})
def normalize_ascii(s: str) -> str:
    return str(s).translate(_ASCII_MAP)

def sanitize_one_line(s: str) -> str:
    if not s:
        return ""
    s = normalize_ascii(s)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r'\s*---.*$', '', s)                       # cut anything after '---'
    s = re.sub(r'^\s*(AI|Answer|Output)\s*:\s*', '', s, flags=re.I)
    s = re.sub(r'\s*Reference\s*:.*$', '', s, flags=re.I)
    s = "".join(ch for ch in s if ord(ch) < 0x2500)       # conservative glyph filter
    return s.strip()

def ref_string(book: str, chapter: int, verse: int) -> str:
    return f"{book} {chapter}:{verse}"

def ref_from_item(item: dict) -> str:
    # Prefer verse_title if present (e.g., "1 Nephi 1:1"); else compose from fields.
    vt = item.get("verse_title")
    if vt:
        return str(vt).strip()
    return ref_string(item["book_title"], int(item["chapter_number"]), int(item["verse_number"]))

def make_user_prompt(book: str, chapter: int, verse: int, text: str) -> str:
    return f'Reference: {book} {chapter}:{verse}\nText: "{normalize_ascii(text)}"'

def parse_ref_from_output_line(line: str) -> Optional[str]:
    m = re.match(r'^\[([^\]]+)\]', line.strip())
    return m.group(1) if m else None

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
    msgs: List[Dict[str, str]] = []
    recent = tail_lines(out_path, pairs)
    for line in recent:
        ref = parse_ref_from_output_line(line)
        if not ref:
            continue
        kjv = ref_to_text.get(ref)
        if not kjv:
            continue
        user_msg = f'Reference: {ref}\nText: "{normalize_ascii(kjv)}"'
        assist_msg = sanitize_one_line(line)
        if not assist_msg:
            continue
        msgs.append({"role": "user", "content": user_msg})
        msgs.append({"role": "assistant", "content": assist_msg})
    return msgs

# ---------- loading ----------
def load_bom_json(path_or_flag: Optional[str], use_standard_works: bool) -> List[dict]:
    if path_or_flag and os.path.exists(path_or_flag):
        data = read_json(path_or_flag)
    else:
        url = STD_JSON_URL if use_standard_works else BOM_JSON_URL
        data = http_get_json(url)
    if not isinstance(data, list):
        raise RuntimeError("Expected a JSON array of verse objects.")
    # Minimal schema check per repo README (volume/book/chapter/verse fields).
    probe = data[0]
    required = {"book_title", "chapter_number", "verse_number", "scripture_text"}
    if not required.issubset(probe.keys()):
        raise RuntimeError(f"Unexpected verse shape: keys={list(probe.keys())[:8]}")
    return data

def flatten_bom(data: List[dict]) -> Tuple[List[dict], Dict[str, str]]:
    verses: List[dict] = []
    ref_to_text: Dict[str, str] = {}
    for it in data:
        book = str(it["book_title"])
        chap = int(it["chapter_number"])
        vs   = int(it["verse_number"])
        txt  = str(it["scripture_text"])
        ref  = ref_from_item(it)
        verses.append({"book": book, "chapter": chap, "verse": vs, "text": txt, "ref": ref})
        # Map both ref_title and constructed ref to the same text (be robust)
        ref_to_text[ref] = txt
        ref_to_text[ref_string(book, chap, vs)] = txt
    return verses, ref_to_text

# ---------- chat ----------
def chat_once(
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
    temperature: float = 0.9,
    stream: bool = True,
    max_tokens: int = -1,
    stops: Optional[List[str]] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    extra_headers: Optional[Dict[str,str]] = None,
) -> str:
    headers = {"Content-Type": "application/json"}
    # actually use the api_key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        # some local/proxy servers accept either header name
        headers.setdefault("X-API-Key", api_key)
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
        "stream": stream
    }
    # only send max_tokens if it is a positive integer
    if isinstance(max_tokens, int) and max_tokens > 0:
        payload["max_tokens"] = max_tokens
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

def opening_ngram(s: str, n: int = 4) -> str:
    toks = [t for t in re.split(r'\s+', s.strip()) if t]
    return " ".join(toks[:n]).lower()

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
    ctx_pairs: int,
    bom_json_path: Optional[str],
    use_standard_works: bool,
):
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        sys_prompt = f.read().strip()

    print("Loading JSON...")
    data = load_bom_json(bom_json_path, use_standard_works)
    verses, ref_to_text = flatten_bom(data)
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
            book, chapter, verse, text, ref = v["book"], int(v["chapter"]), int(v["verse"]), v["text"], v["ref"]

            # Sliding history from disk (last ctx_pairs lines -> paired user/assistant turns)
            history = build_history_messages(out_txt, ref_to_text, ctx_pairs)

            user_prompt = make_user_prompt(book, chapter, verse, text)
            expected_prefix = f"[{ref}]"

            attempt = 0
            while True:
                attempt += 1
                try:
                    raw = chat_once(
                        api_base=api_base,
                        model=model,
                        api_key=api_key,
                        system_prompt=sys_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        stream=stream,
                        max_tokens=-1,
                        stops=DEFAULT_STOPS,
                        history_messages=history
                    )
                    line = postprocess_line(raw, expected_prefix)

                    # Fallback: non-streaming, no stops, slightly higher temp
                    if not body_ok(line, expected_prefix):
                        raw2 = chat_once(
                            api_base=api_base,
                            model=model,
                            api_key=api_key,
                            system_prompt=sys_prompt,
                            user_prompt=user_prompt,
                            temperature=max(0.9, temperature),
                            stream=False,
                            max_tokens=-1,
                            stops=None,
                            history_messages=history
                        )
                        line2 = postprocess_line(raw2, expected_prefix)
                        if body_ok(line2, expected_prefix):
                            line = line2

                    # Anti-dup opener vs history
                    if history:
                        recent_assistant_lines = [m["content"] for m in history if m["role"] == "assistant"]
                        recent_starts = {opening_ngram(ln.split("]",1)[-1]) for ln in recent_assistant_lines}
                        this_start = opening_ngram(line.split("]",1)[-1])
                        if this_start in recent_starts:
                            nudge = {"role":"user","content":"Reminder: vary the opening phrasing; avoid repeating recent openers."}
                            raw3 = chat_once(
                                api_base=api_base,
                                model=model,
                                api_key=api_key,
                                system_prompt=sys_prompt,
                                user_prompt=user_prompt,
                                temperature=max(1.0, temperature),
                                stream=False,
                                max_tokens=-1,
                                stops=None,
                                history_messages=history + [nudge]
                            )
                            line3 = postprocess_line(raw3, expected_prefix)
                            if body_ok(line3, expected_prefix):
                                line = line3

                    outf.write(line + "\n")
                    outf.flush()
                    save_progress(idx + 1)
                    preview = line[:120] + ("..." if len(line) > 120 else "")
                    print(f"{idx+1}/{total}: {preview}")
                    time.sleep(rate_limit_s)
                    break
                except requests.HTTPError as e:
                    resp = e.response
                    code = resp.status_code if resp is not None else "?"
                    body = resp.text if resp is not None else ""
                    print(f"HTTP error {code} on verse {idx}... attempt {attempt}/{retries}")
                    if body:
                        print(body[:2000])
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
    p = argparse.ArgumentParser(description="Build a Zoomer Book of Mormon using a local LLM.")
    p.add_argument("--system-prompt", default="system_zoomer_prompt.txt", help="Path to the system prompt file.")
    p.add_argument("--out", default=OUT_PATH_DEFAULT, help="Output file to append verses to.")
    p.add_argument("--api-base", default=DEFAULT_API_BASE, help="OpenAI-compatible chat completions endpoint.")
    p.add_argument("--api-key", default=os.environ.get("API_KEY"), help="API key for the endpoint, defaults to API_KEY env var.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    p.add_argument("--start-index", type=parse_index_arg, default=None, help="Zero-based verse index to start from.")
    p.add_argument("--end-index", type=parse_index_arg, default=None, help="Stop before this zero-based index.")
    p.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    p.add_argument("--rate-limit-s", type=float, default=0.2, help="Seconds to sleep between requests.")
    p.add_argument("--retries", type=int, default=5, help="Max retries per verse.")
    p.add_argument("--no-stream", action="store_true", help="Disable streaming.")
    p.add_argument("--ctx-pairs", type=int, default=10, help="How many prior user/assistant pairs to include as chat history.")
    p.add_argument("--bom-json", default=None, help="Path to local JSON; if omitted, auto-downloads book-of-mormon.json.")
    p.add_argument("--standard-works", action="store_true", help="Use the combined standard-works.json instead of just the Book of Mormon.")
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
        ctx_pairs=args.ctx_pairs,
        bom_json_path=args.bom_json,
        use_standard_works=args.standard_works,
    )
