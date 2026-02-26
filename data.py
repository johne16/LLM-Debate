"""
Data module: Fetch StrategyQA questions from the HuggingFace HTTP API.
"""

import json
import os
import random
import requests


# Module-level cache: fetched once, reused on subsequent calls.
_cache = None
_CACHE_SIZE = 500
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_CACHE_FILE = os.path.join(_DATA_DIR, "strategyqa_cache.json")


def _fetch_and_cache() -> list[dict]:
    """Load questions from local cache file, or fetch from HuggingFace and save."""
    global _cache

    # Try loading from local file first
    if os.path.exists(_CACHE_FILE) and os.path.getsize(_CACHE_FILE) > 0:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            _cache = json.load(f)
        return _cache

    # Fetch from HuggingFace API in pages (endpoint caps at 100 per request)
    questions = []
    page_size = 100
    for offset in range(0, _CACHE_SIZE, page_size):
        length = min(page_size, _CACHE_SIZE - offset)
        url = (
            f"https://datasets-server.huggingface.co/rows"
            f"?dataset=wics/strategy-qa"
            f"&config=strategyQA"
            f"&split=test"
            f"&offset={offset}"
            f"&length={length}"
        )

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        for row_item in data["rows"]:
            row = row_item["row"]
            questions.append({
                "question": row["question"],
                "answer": bool(row["answer"])  # True = Yes, False = No
            })

    # Save to local file for future sessions
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)

    _cache = questions
    return _cache


def fetch_strategy_qa(sample_size: int) -> list[dict]:
    """
    Fetch StrategyQA questions via the HuggingFace datasets server API.
    Uses a module-level cache; only the first call hits the network.

    Args:
        sample_size: Number of questions to retrieve.

    Returns:
        List of dicts, each with keys: 'question' (str), 'answer' (bool).
        answer=True means "Yes", answer=False means "No".
    """
    global _cache
    if _cache is None:
        _fetch_and_cache()

    return _cache[:sample_size]


def fetch_random_question() -> dict:
    """
    Pick a random question from the cached dataset.

    Returns:
        Dict with 'question' (str) and 'answer' (bool).
    """
    global _cache
    if _cache is None:
        _fetch_and_cache()

    return random.choice(_cache)


if __name__ == "__main__":
    # Quick test
    from dotenv import load_dotenv
    load_dotenv()

    qs = fetch_strategy_qa(3)
    for q in qs:
        label = "Yes" if q["answer"] else "No"
        print(f"Q: {q['question']}  | GT: {label}")
