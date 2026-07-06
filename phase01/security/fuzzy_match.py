import re
from difflib import SequenceMatcher


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
SIMILARITY_THRESHOLD = 0.80
SCRAMBLED_KEYWORDS = {
    "ignore",
    "system",
    "developer",
    "assistant",
    "instruction",
    "instructions",
    "prompt",
    "reveal",
    "bypass",
    "override",
    "forget",
    "hidden",
    "secret",
}


def contains_scrambled_keyword(text: str) -> list[str]:
    """
    Return security keywords that appear in scrambled form.
    """
    matches: list[str] = []

    for token in _tokenize(text):
        for keyword in SCRAMBLED_KEYWORDS:
            if _looks_scrambled(token, keyword):
                matches.append(keyword)

    return sorted(set(matches))


def _similarity(word1: str, word2: str) -> float:
    """
    Return similarity score in [0.0, 1.0].
    """
    return SequenceMatcher(None, word1, word2).ratio()


def _looks_scrambled(word: str, keyword: str) -> bool:
    """
    Detect likely misspellings or scrambled versions of a keyword.
    """
    if word == keyword:
        return False

    if abs(len(word) - len(keyword)) > 2:
        return False

    if word[0] != keyword[0]:
        return False

    if word[-1] != keyword[-1]:
        return False

    return _similarity(word, keyword) >= SIMILARITY_THRESHOLD


def _tokenize(text: str) -> list[str]:
    """
    Split text into lowercase alphanumeric tokens.
    """
    return _TOKEN_PATTERN.findall(text.lower())
