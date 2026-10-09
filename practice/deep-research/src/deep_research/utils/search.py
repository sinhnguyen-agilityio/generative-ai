"""Normalize web pages and format source passages for research."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

MIN_FULL_TEXT = 500


MIN_TEXT = 200  # results with less text than this are dropped


_LINK_ONLY = re.compile(r"^(?:[-*|]\s*)?!?\[[^\]]*\]\([^)]*\)\s*\|?$")


_NOISE = re.compile(
    r"cookie|subscribe|sign (?:in|up)|log in|all rights reserved|privacy policy|terms of (?:use|service)", re.I
)


def normalize_url(url: str) -> str:
    """Normalize page identity by removing fragments, tracking parameters, and trailing slashes.

    Args:
        url: Page URL, possibly with whitespace or tracking parameters.

    Returns:
        URL with lowercase scheme/host and non-utm query parameters preserved.
    """
    parts = urlsplit(url.strip())
    params = [(k, v) for k, v in parse_qsl(
        parts.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(params), ""))


def clean(text: str) -> str:
    """Remove repeated lines and common navigation, link, and banner noise.

    Args:
        text: Raw page text to clean.

    Returns:
        Remaining content joined with newlines in its original order.
    """
    seen, kept = set(), []
    for line in text.splitlines():
        s = line.strip()
        if not s or s in seen or _LINK_ONLY.match(s):
            continue
        if len(s) < 25 and not s.startswith("#") and not any(c.isdigit() for c in s):
            continue
        if len(s) < 120 and _NOISE.search(s):
            continue
        seen.add(s)
        kept.append(s)
    return "\n".join(kept)


def prepare_pages(results: list[dict]) -> list[dict]:
    """Prepare unique usable pages, falling back to snippets when cleaned text is too short.

    Args:
        results: Normalized search results with raw_content, content, URL, title, and score.

    Returns:
        Pages with url, title, text, snippet, full_text, and score; short results are dropped.
    """
    pages = {}
    for result in results:
        url = normalize_url(result["url"])
        text, snippet = clean(result["raw_content"]), result["content"].strip()
        full_text = len(text) >= MIN_FULL_TEXT
        if not full_text:
            text = snippet
        if url not in pages and len(text) >= MIN_TEXT:
            pages[url] = {
                # One line: a title is rendered into source lines, so it must not break them.
                "url": url, "title": " ".join(result["title"].split()) or url, "text": text,
                "snippet": snippet, "full_text": full_text, "score": result["score"],
            }
    return list(pages.values())


def format_passages(passages: list[dict], sources: dict[str, dict]) -> str:
    """Group passages by URL and label each page with its source number.

    Separate passages of one page with ellipses and pages with a Markdown rule.
    The full-text label describes the source metadata, not the length of the displayed excerpt.

    Args:
        passages: URL/text passages in the order they should be displayed.
        sources: Registry keyed by URL with id, title, and full_text metadata for each passage.

    Returns:
        Citation-numbered evidence text, or "No results." when there are no passages.
    """
    grouped: dict[str, list[str]] = {}
    for passage in passages:
        grouped.setdefault(passage["url"], []).append(passage["text"])
    blocks = []
    for url, texts in grouped.items():
        source = sources[url]
        kind = "full text" if source["full_text"] else "snippet only"
        blocks.append(
            f"[{source['id']}] {source['title']} — {url} ({kind})\n" + "\n...\n".join(texts))
    return "\n\n---\n\n".join(blocks) or "No results."
