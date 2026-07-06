import unicodedata
import re

_WHITESPACE_RE = re.compile(r"\s+")
_ALLOWED_WHITESPACE = {" ", "\t", "\n", "\r"}
_DISALLOWED_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn"}
_SCRIPT_PREFIXES = (
    "LATIN",
    "CYRILLIC",
    "GREEK",
    "ARABIC",
    "HEBREW",
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",
    "CJK",
    "BOPOMOFO",
    "DEVANAGARI",
    "THAI",
    "ARMENIAN",
    "GEORGIAN",
)


def _normalize_unicode(text: str) -> str:
    """
    Normalize text for LLM preprocessing.

    Operations:
    - Unicode normalize (NFKC)
    - Collapse consecutive whitespace
    - Strip leading/trailing whitespace
    """
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _remove_control_characters(text: str) -> str:
    """
    Remove invisible and non-printable Unicode characters.

    Preserves normal whitespace while removing control, format,
    surrogate, private-use, and unassigned code points.
    """
    return "".join(
        ch
        for ch in text
        if (
           ch in _ALLOWED_WHITESPACE
           or unicodedata.category(ch) not in _DISALLOWED_CATEGORIES
        )
    )


def _detect_mixed_scripts(text: str) -> set[str]:
    """
    Detect the Unicode scripts used in a string.
    Returns:
        A set of detected scripts.

    Example:
        >>> _detect_mixed_scripts("hello")
        {"LATIN"}

        >>> _detect_mixed_scripts("paypal")
        {"LATIN", "CYRILLIC"}
    """
    scripts: set[str] = set()

    for ch in text:
        if not ch.isalpha():
            continue

        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue

        for script in _SCRIPT_PREFIXES:
            if name.startswith(script):
                scripts.add(script)
                break
        else:
            scripts.add("UNKNOWN")

    return scripts


def has_mixed_scripts(text: str) -> bool:
    """
    Return True if alphabetic characters come from more than one script.
    """
    return len(_detect_mixed_scripts(text)) > 1


def clean_text(text: str) -> str:
    text = _normalize_unicode(text)
    text = _remove_control_characters(text)

    if has_mixed_scripts(text):
        raise ValueError("Text contains characters from multiple scripts")

    return text
