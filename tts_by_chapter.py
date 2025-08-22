#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
from pathlib import Path
from collections import OrderedDict
import requests

# Config defaults (can be overridden by env or CLI)
TTS_API_URL = os.environ.get("TTS_API_URL", "http://localhost:8880/v1/audio/speech")
TTS_MODEL = os.environ.get("TTS_MODEL", "kokoro")
TTS_VOICE = os.environ.get("TTS_VOICE", "af_sky+af+af_nicole")
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.75"))
TTS_FORMAT = os.environ.get("TTS_FORMAT", "mp3")  # mp3, wav, opus, flac, m4a, pcm
INTER_RUN_DELAY = float(os.environ.get("INTER_RUN_DELAY", "2.0"))  # seconds between TTS attempts

# Robust verse header:
# [Genesis 11:7] Text...
# Accepts: leading spaces or BOM, book names with digits or spaces or dots or apostrophes or hyphens,
# arbitrary spaces around colon, and header anywhere in the line.
LINE_RE = re.compile(
    r'\[\s*(?P<book>[A-Za-z0-9][A-Za-z0-9 .\'-]*?)\s+'
    r'(?P<chapter>\d+)\s*:\s*(?P<verse>\d+)\s*\]\s*(?P<text>.*)'
)

def sanitize_name(name: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9 _\-]', '', name).strip()
    cleaned = re.sub(r'\s+', '_', cleaned)
    return cleaned or "Unknown"

def normalize_line(s: str) -> str:
    # Strip UTF-8 BOM and trailing CRLF noise
    return s.lstrip('\ufeff').rstrip('\r\n')

def parse_chapters(text: str, debug: bool = False):
    """
    Returns OrderedDict keyed by (book, chapter) -> list[str] of verse or continuation text.
    Matches [Book Chapter:Verse] anywhere in line. Non matches after the first match
    are appended to the current chapter as continuations.
    """
    chapters = OrderedDict()
    current_key = None
    matched = 0
    debug_nomatch_examples = []

    for raw in text.splitlines():
        line = normalize_line(raw)
        m = LINE_RE.search(line)
        if m:
            matched += 1
            book = m.group("book").strip()
            chapter = int(m.group("chapter"))
            verse_text = m.group("text").strip()
            key = (book, chapter)
            chapters.setdefault(key, []).append(verse_text)
            current_key = key
        else:
            if current_key is not None and line.strip():
                chapters[current_key].append(line.strip())
            else:
                if debug and "[" in line and len(debug_nomatch_examples) < 15:
                    debug_nomatch_examples.append(repr(line))

    if debug:
        sys.stderr.write(f"[debug] verse headers matched: {matched}\n")
        if matched == 0 and debug_nomatch_examples:
            sys.stderr.write("[debug] lines containing '[' that did NOT match pattern:\n")
            for ex in debug_nomatch_examples:
                sys.stderr.write(f"  {ex}\n")

    return chapters

def combine_chapter_text(book: str, chapter: int, parts: list[str]) -> str:
    header = f"{book} chapter {chapter}."
    body = " ".join(p.strip() for p in parts if p.strip())
    return f"{header} {body}".strip()

def tts_request(text: str, voice: str, speed: float, fmt: str, timeout_s: int = 600) -> bytes:
    """
    Single shot TTS request. No retries by design.
    """
    text = text.replace('*', '')
    payload = {
        "model": TTS_MODEL,
        "input": text,
        "voice": voice,
        "response_format": fmt,
        "speed": speed,
    }
    try:
        r = requests.post(TTS_API_URL, json=payload, timeout=timeout_s)
        if r.status_code == 200 and r.content:
            return r.content
        # Log server response text if available, then fail fast
        sys.stderr.write(f"[error] TTS returned {r.status_code}\n")
        try:
            sys.stderr.write((r.text or "")[:800] + "\n")
        except Exception:
            pass
        raise RuntimeError("TTS request failed without retry")
    except requests.RequestException as e:
        raise RuntimeError(f"TTS request error: {e}")

def write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def main():
    ap = argparse.ArgumentParser(description="Generate per-chapter audio from bracketed [Book Chapter:Verse] text.")
    ap.add_argument("input_file", help="Path to input text file")
    ap.add_argument("--outdir", default="output", help="Base output directory")
    ap.add_argument("--voice", default=TTS_VOICE, help="Voice or voice combo string")
    ap.add_argument("--speed", default=TTS_SPEED, type=float, help="TTS speed")
    ap.add_argument("--format", default=TTS_FORMAT, help="Audio format: mp3, wav, opus, flac, m4a, pcm")
    ap.add_argument("--skip-existing", action="store_true", help="Skip chapters that already have an output file")
    ap.add_argument("--debug", action="store_true", help="Print match diagnostics")
    ap.add_argument(
        "--inter-run-delay",
        type=float,
        default=INTER_RUN_DELAY,
        help="Seconds to sleep after each TTS attempt to avoid container lockups"
    )
    args = ap.parse_args()

    in_path = Path(args.input_file)
    if not in_path.exists():
        sys.stderr.write(f"[error] Input file not found: {in_path}\n")
        sys.exit(1)

    try:
        try:
            text = in_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = in_path.read_text(encoding="latin-1")
    except Exception as e:
        sys.stderr.write(f"[error] Reading input failed: {e}\n")
        sys.exit(1)

    chapters = parse_chapters(text, debug=args.debug)
    if not chapters:
        sys.stderr.write("[error] No chapters found. Expected lines like: [Genesis 11:7] Text...\n")
        if not args.debug:
            sys.stderr.write("Tip: re-run with --debug to see why lines are not matching.\n")
        sys.exit(2)

    base_out = Path(args.outdir)
    total = len(chapters)

    for idx, ((book, chapter), parts) in enumerate(chapters.items(), start=1):
        book_dir = base_out / sanitize_name(book)
        out_file = book_dir / f"{sanitize_name(book)}_{chapter:03d}.{args.format}"

        # decide skip first... no delay on skips
        if args.skip_existing and out_file.exists():
            print(f"[{idx}/{total}] SKIP {book} {chapter} ... already exists")
            continue

        chapter_text = combine_chapter_text(book, chapter, parts)
        print(f"[{idx}/{total}] Generating {book} {chapter} ... {len(chapter_text)} chars")

        made_tts_call = False
        try:
            made_tts_call = True
            audio = tts_request(chapter_text, args.voice, args.speed, args.format)
            write_bytes(out_file, audio)
            print(f"[{idx}/{total}] Wrote {out_file}")
        except Exception as e:
            sys.stderr.write(f"[error] Failed on {book} {chapter}: {e}\n")
        finally:
            # throttle only after an actual TTS attempt, never on skips
            if made_tts_call and args.inter_run_delay > 0:
                time.sleep(args.inter_run_delay)

    print("Done.")

if __name__ == "__main__":
    main()
