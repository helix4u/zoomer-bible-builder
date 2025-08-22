#!/usr/bin/env python3
"""graphrag_bible_companion.py

A dynamic Bible companion that displays the original scripture, a rephrased
version in a user-selected style, and brief commentary. It also demonstrates
how a GraphRAG index could be used to surface related concepts or passages.
The GraphRAG pieces are intentionally lightweight so the script can run even
if the optional dependency is not installed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Optional

import requests

try:  # Optional GraphRAG dependency
    from graphrag import GraphRAG  # type: ignore
except Exception:  # pragma: no cover - library is optional
    GraphRAG = None  # type: ignore

BOOKS_URL = "https://raw.githubusercontent.com/aruljohn/Bible-kjv/master/Books.json"
RAW_BOOK_URL = "https://raw.githubusercontent.com/aruljohn/Bible-kjv/master/{fname}"

DEFAULT_API_BASE = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL = "gemma-3-4b-it"


def http_get_json(url: str, timeout: int = 60) -> any:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def read_json(path: str) -> any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_book_filename(book: str) -> str:
    fname = re.sub(r"[\s'’:-]+", "", book)
    return f"{fname}.json"


def try_fetch_book_json(book: str, repo_dir: Optional[str]) -> dict:
    if repo_dir:
        candidates = [
            os.path.join(repo_dir, f"{book}.json"),
            os.path.join(repo_dir, normalize_book_filename(book)),
        ]
        for p in candidates:
            if os.path.exists(p):
                return read_json(p)
    url = RAW_BOOK_URL.format(fname=normalize_book_filename(book))
    return http_get_json(url)


def find_passage(book: str, chapter: int, verse: int, repo_dir: Optional[str]) -> Optional[str]:
    try:
        book_json = try_fetch_book_json(book, repo_dir)
    except Exception:
        return None
    for ch in book_json.get("chapters", []):
        if int(ch.get("chapter")) != chapter:
            continue
        for v in ch.get("verses", []):
            if int(v.get("verse")) == verse:
                return v.get("text")
    return None


def get_rephrased_passage(passage: str, style: str, api_base: str, model: str) -> str:
    system_prompt = (
        f"You are a Bible translator. Rephrase the following passage in a {style} style. "
        "Be creative but stay true to the original meaning."
    )
    user_prompt = f"Passage: \"{passage}\""
    headers = {"Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "stream": False,
    }
    try:
        r = requests.post(api_base, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        obj = r.json()
        return obj["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error contacting API: {e}"


def setup_graphrag_index() -> Optional[GraphRAG]:
    """Sets up a GraphRAG index if the library is available."""
    if GraphRAG is None:
        return None
    # Real usage would build an index from the Bible text.
    # Here we just create an empty index for demonstration.
    try:
        return GraphRAG()
    except Exception:
        return None


def query_graphrag(index: Optional[GraphRAG], query: str) -> str:
    if index is None:
        return "GraphRAG not available."
    try:
        # Real usage might be: index.query(query)
        return str(index.query(query))
    except Exception:
        return "GraphRAG query failed."


def get_commentary(book: str, chapter: int, verse: int, index: Optional[GraphRAG]) -> str:
    prompt = f"Commentary on {book} {chapter}:{verse}"
    return query_graphrag(index, prompt)


def run(book: str, chapter: int, verse: int, style: str, api_base: str, model: str, repo_dir: Optional[str]) -> None:
    index = setup_graphrag_index()
    original = find_passage(book, chapter, verse, repo_dir)
    if not original:
        print("Passage not found.")
        return
    print("--- Original Scripture ---")
    print(original)
    print(f"\n--- Rephrased in {style} ---")
    print(get_rephrased_passage(original, style, api_base, model))
    print("\n--- Commentary ---")
    print(get_commentary(book, chapter, verse, index))
    print("\n--- GraphRAG Related Concepts ---")
    query = f"Themes related to {book} {chapter}:{verse}"
    print(query_graphrag(index, query))


def main() -> None:
    p = argparse.ArgumentParser(description="A GraphRAG-powered Bible companion")
    p.add_argument("book", help="Book of the Bible, e.g. 'Genesis'")
    p.add_argument("chapter", type=int, help="Chapter number")
    p.add_argument("verse", type=int, help="Verse number")
    p.add_argument("--style", default="casual modern English", help="Rephrasing style")
    p.add_argument("--api-base", default=DEFAULT_API_BASE, help="Chat completions endpoint")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    p.add_argument("--repo-dir", default="Bible-kjv", help="Path to KJV JSON repo")
    args = p.parse_args()
    run(
        book=args.book,
        chapter=args.chapter,
        verse=args.verse,
        style=args.style,
        api_base=args.api_base,
        model=args.model,
        repo_dir=args.repo_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
