# bible_companion.py
# An interactive Bible companion that provides scripture, paraphrasing, and commentary.

import json
import os
import re
import requests
from typing import Dict, List, Optional, Tuple

# --- Path setup ---
# Get the absolute path of the directory containing this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Constants ---
BOOKS_URL = "https://raw.githubusercontent.com/aruljohn/Bible-kjv/master/Books.json"
RAW_BOOK_URL = "https://raw.githubusercontent.com/aruljohn/Bible-kjv/master/{fname}"
DEFAULT_API_BASE = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL = "gemma-3-4b-it"

# ---------- Copied utility functions from zoomer_bible_builder.py ----------

def http_get_json(url: str, timeout: int = 60) -> any:
    """Fetches JSON data from a URL."""
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def read_json(path: str) -> any:
    """Reads a JSON file from a local path."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, data: any) -> None:
    """Writes data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def normalize_book_filename(book: str) -> str:
    """Converts a book name to its likely filename."""
    fname = re.sub(r"[\\s'’:-]+", "", book)
    return f"{fname}.json"

def try_fetch_book_json(book: str) -> dict:
    """
    Tries to fetch a book's JSON data, first from a local directory,
    then falling back to a remote URL.
    """
    repo_dir = os.path.join(SCRIPT_DIR, "..", "Bible-kjv")
    # local first
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

def fetch_books_list() -> List[str]:
    """Fetches the list of Bible books."""
    repo_dir = os.path.join(SCRIPT_DIR, "..", "Bible-kjv")
    cache_path = os.path.join(SCRIPT_DIR, "..", "Books.json")

    local_books_path = os.path.join(repo_dir, "Books.json")
    if os.path.exists(local_books_path):
        books = read_json(local_books_path)
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

# ---------- New functions for the Bible Companion ----------

def find_verse_in_book(book_json: dict, chapter: int, verse: int) -> Optional[str]:
    """Finds the text of a specific verse in a book's JSON data."""
    for ch in book_json.get("chapters", []):
        if int(ch.get("chapter", 0)) == chapter:
            for v_obj in ch.get("verses", []):
                # Handles both {"1": "text"} and {"verse": 1, "text": "text"} formats
                if len(v_obj) == 1:
                    v_num, v_text = next(iter(v_obj.items()))
                    if int(v_num) == verse:
                        return v_text
                elif int(v_obj.get("verse", 0)) == verse:
                    return v_obj.get("text")
    return None


def parse_reference(ref_string: str) -> Optional[Tuple[str, int, int]]:
    """
    Parses a reference string like "John 3:16" into (book, chapter, verse).
    Handles book names with leading numbers and spaces, like "1 John".
    """
    ref_string = ref_string.strip()
    # Regex to capture book, chapter, and verse.
    # It allows for a leading number and space in the book name.
    match = re.match(r'(\d?\s*[a-zA-Z]+)\s*(\d+):(\d+)', ref_string, re.IGNORECASE)
    if not match:
        return None

    book, chapter, verse = match.groups()
    return book.strip(), int(chapter), int(verse)

# Anti-echo stops... DO NOT include "\n" or you'll clip content.
DEFAULT_STOPS = ["Reference:", "AI:", "---"]

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
    s = re.sub(r'\s*---.*$', '', s)                           # cut anything after '---'
    s = re.sub(r'^\s*(AI|Answer|Output)\s*:\s*', '', s, flags=re.I)
    s = re.sub(r'\s*Reference\s*:.*$', '', s, flags=re.I)
    s = "".join(ch for ch in s if ord(ch) < 0x2500)           # conservative glyph filter
    return s.strip()

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

    # For this interactive tool, we will prefer non-streaming responses
    # as it's simpler to handle.
    payload["stream"] = False

    r = requests.post(api_base, headers=headers, json=payload, timeout=600)
    r.raise_for_status()
    obj = r.json()
    return obj["choices"][0]["message"]["content"].strip()


def get_paraphrase(verse_text: str, style: str) -> str:
    """Gets a paraphrased version of a verse using the LLM."""
    if style == "zoomer":
        # Path relative to the parent directory of the script
        prompt_path = os.path.join(SCRIPT_DIR, "..", "system_zoomer_prompt.txt")
    elif style == "casual":
        # Path relative to the script's own directory
        prompt_path = os.path.join(SCRIPT_DIR, "experiment", "system_casual_prompt.txt")
    else:
        raise ValueError(f"Unknown paraphrase style: {style}")

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        return f"Error: System prompt file not found at '{prompt_path}'"

    user_prompt = f'Text: "{verse_text}"'

    try:
        paraphrase = chat_once(
            api_base=DEFAULT_API_BASE,
            model=DEFAULT_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7, # Lower temperature for more predictable output
            stops=DEFAULT_STOPS
        )
        return sanitize_one_line(paraphrase)
    except requests.RequestException as e:
        # Check for connection error, which is common if local server isn't running
        if "Connection refused" in str(e):
             return "Error: Cannot connect to the language model API. Is the local server running?"
        return f"Error: A network problem occurred. {e}"
    except Exception as e:
        return f"Error: An unexpected error occurred while generating the paraphrase. {e}"


def load_commentaries() -> List[Dict]:
    """Loads commentaries from the JSON file."""
    path = os.path.join(SCRIPT_DIR, "commentaries.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file is missing or invalid, return an empty list
        print("Warning: commentaries.json not found or is invalid.")
        return []

def find_commentary(commentaries: List[Dict], book: str, chapter: int, verse: int) -> Optional[str]:
    """Finds a commentary for a specific verse."""
    for comm in commentaries:
        if (comm["book"] == book and
            comm["chapter"] == chapter and
            comm["verse"] == verse):
            return comm.get("commentary")
    return None

def main():
    print("Welcome to the Bible Companion!")
    print("Enter a Bible reference (e.g., 'John 3:16') to get started, or 'quit' to exit.")

    try:
        # Pre-load the list of books to validate user input
        books_list = fetch_books_list()
        # Create a lowercase-to-proper-case mapping for flexible matching
        books_map = {b.lower(): b for b in books_list}
        print(f"Loaded {len(books_list)} books.")
    except (requests.RequestException, RuntimeError) as e:
        print(f"Fatal Error: Could not load Bible data. {e}")
        return

    # Load commentaries at startup
    commentaries = load_commentaries()
    if commentaries:
        print(f"Loaded {len(commentaries)} commentary entries.")

    while True:
        user_input = input("\nEnter reference > ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            break

        parsed_ref = parse_reference(user_input)
        if not parsed_ref:
            print("Invalid format. Please use 'Book Chapter:Verse' (e.g., 'Genesis 1:1').")
            continue

        book_name_raw, chapter_num, verse_num = parsed_ref

        # Validate the book name
        book_name_lower = book_name_raw.lower()
        if book_name_lower not in books_map:
            print(f"Book '{book_name_raw}' not found. Please check the spelling.")
            continue

        book_name_proper = books_map[book_name_lower]

        try:
            book_json = try_fetch_book_json(book_name_proper)
            verse_text = find_verse_in_book(book_json, chapter_num, verse_num)

            if verse_text:
                print(f"\n{book_name_proper} {chapter_num}:{verse_num} (KJV):")
                print(verse_text)

                # Display commentary if found
                commentary = find_commentary(commentaries, book_name_proper, chapter_num, verse_num)
                if commentary:
                    print("\nCommentary:")
                    print(commentary)

                # Get paraphrase style from user
                style_input = input("\nChoose paraphrase style ('zoomer' or 'casual'), or press Enter to skip: ").lower()
                if style_input in ["zoomer", "casual"]:
                    print(f"\nGenerating {style_input} paraphrase...")
                    paraphrase = get_paraphrase(verse_text, style_input)
                    print(f"\nParaphrase ({style_input}):")
                    print(paraphrase)
                elif style_input:
                    print("Invalid style. Skipping paraphrase.")

            else:
                print(f"Could not find verse {chapter_num}:{verse_num} in {book_name_proper}.")

        except (requests.RequestException, FileNotFoundError) as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
