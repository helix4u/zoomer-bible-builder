#!/usr/bin/env python3
# dynamic_bible_companion.py
# A dynamic Bible companion that provides original scripture, a rephrased version in a chosen style,
# and commentary, with a placeholder for GraphRAG integration.

import argparse
import json
import os
import re
import sys
import time
import signal
from typing import Dict, List, Optional, Tuple

import requests

# --- Constants and Configuration ---
BOOKS_URL = "https://raw.githubusercontent.com/aruljohn/Bible-kjv/master/Books.json"
RAW_BOOK_URL = "https://raw.githubusercontent.com/aruljohn/Bible-kjv/master/{fname}"

DEFAULT_API_BASE = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL = "gemma-3-4b-it"

# --- Utility Functions (similar to zoomer_bible_builder.py) ---

def graceful_exit(signum, frame):
    print("\nStopping cleanly...")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)

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
    # Remote fallback
    url = RAW_BOOK_URL.format(fname=normalize_book_filename(book))
    try:
        return http_get_json(url)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            raise FileNotFoundError(f"Could not fetch JSON for book '{book}'.") from e
        raise

def find_passage(book: str, chapter: int, verse: int, repo_dir: Optional[str]) -> Optional[str]:
    """Finds the text of a specific Bible passage."""
    try:
        book_json = try_fetch_book_json(book, repo_dir)
        for ch in book_json.get("chapters", []):
            if int(ch.get("chapter")) == chapter:
                for v in ch.get("verses", []):
                    if int(v.get("verse")) == verse:
                        return v.get("text")
    except FileNotFoundError:
        return None
    return None

# --- Core Components ---

def get_rephrased_passage(passage: str, style: str, api_base: str, model: str) -> str:
    """
    Rephrases a Bible passage using an LLM.
    """
    system_prompt = f"You are a Bible translator. Rephrase the following passage in a {style} style. Be creative but stay true to the original meaning."
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
        "stream": False
    }

    try:
        r = requests.post(api_base, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        obj = r.json()
        return obj["choices"][0]["message"]["content"].strip()
    except requests.RequestException as e:
        return f"Error: Could not connect to the API. {e}"
    except (KeyError, IndexError) as e:
        return f"Error: Unexpected API response format. {e}"


def get_commentary(book: str, chapter: int, verse: int) -> str:
    """
    Fetches commentary for a specific passage.
    This is a placeholder for now. In a real implementation, this could query a database,
    an external API (like from a seminary or Bible commentary website), or a local knowledge base.
    """
    # Placeholder commentary
    return "This is where a short commentary or explanation would appear. For example, it might explain the historical context, theological significance, or different interpretations of the passage."

def setup_graphrag_index():
    """
    Placeholder for setting up the GraphRAG index.
    This would involve loading the Bible data into a graph structure
    and preparing it for querying.
    """
    print("Setting up GraphRAG index... (placeholder)")
    # In a real implementation, you would use the graphrag library here.
    # from graphrag.query.context_builder.entity_extraction import run
    # run(...)
    pass

def query_graphrag(query: str) -> str:
    """
    Placeholder for querying the GraphRAG index.
    This would be used to find related passages, themes, or concepts.
    """
    print(f"Querying GraphRAG with: '{query}'... (placeholder)")
    # In a real implementation, you would use the graphrag library here.
    # from graphrag.query.llm.oai.chat_openai import ChatOpenAI
    # from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
    # from graphrag.query.structured_search.global_search.search import GlobalSearch
    # ...
    return "GraphRAG would return related passages and concepts here."


# --- Main Driver ---

def run(book: str, chapter: int, verse: int, style: str, api_base: str, model: str, repo_dir: Optional[str]):
    """
    Main function to run the dynamic Bible companion.
    """
    print(f"Looking for {book} {chapter}:{verse}...")

    original_passage = find_passage(book, chapter, verse, repo_dir)

    if not original_passage:
        print("Passage not found.")
        return

    print("\n--- Original Scripture ---")
    print(original_passage)

    print(f"\n--- Rephrased in {style} ---")
    rephrased_passage = get_rephrased_passage(original_passage, style, api_base, model)
    print(rephrased_passage)

    print("\n--- Commentary ---")
    commentary = get_commentary(book, chapter, verse)
    print(commentary)

    # --- GraphRAG Integration Example ---
    print("\n--- GraphRAG Related Concepts (Placeholder) ---")
    # Example query to find related concepts
    graphrag_query = f"What are the main themes in {book} {chapter}:{verse}?"
    related_concepts = query_graphrag(graphrag_query)
    print(related_concepts)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="A dynamic Bible companion.")
    p.add_argument("book", help="The book of the Bible (e.g., 'Genesis').")
    p.add_argument("chapter", type=int, help="The chapter number.")
    p.add_argument("verse", type=int, help="The verse number.")
    p.add_argument("--style", default="casual modern English", help="The style to rephrase the passage in (e.g., 'Gen Z slang', 'poetic', 'scholarly').")
    p.add_argument("--api-base", default=DEFAULT_API_BASE, help="OpenAI-compatible chat completions endpoint.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    p.add_argument("--repo-dir", default="Bible-kjv", help="Optional path to a local clone of aruljohn/Bible-kjv to read JSON from.")

    args = p.parse_args()

    # Setup GraphRAG (placeholder)
    setup_graphrag_index()

    run(
        book=args.book,
        chapter=args.chapter,
        verse=args.verse,
        style=args.style,
        api_base=args.api_base,
        model=args.model,
        repo_dir=args.repo_dir,
    )
