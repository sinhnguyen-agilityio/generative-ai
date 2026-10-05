"""Raw page content: cleaning, size limits, chunking and stable source ids.

Tavily returns each page's full text (``raw_content``). It is stored in the VFS and chunked for
the vector index, and never enters an LLM prompt whole: the researcher sees only the chunks most
relevant to its question (``agents/passage_selector.py``). Every function here is pure, so the
same page always gives the same cleaned text and the same chunks — chunks can be rebuilt from the
VFS at any time (for example after switching the embedding model).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

from intelligence_collector.tools.url_tracking import normalize_url

# Hard ceiling on the stored page text (about 15k tokens); longer pages are cut and flagged.
MAX_PAGE_CHARS = 60_000
# Below this, the "page" is usually a cookie wall, paywall or script shell: use the snippet.
MIN_PAGE_CHARS = 500
CHUNK_CHARS = 1_000  # about 250 tokens: one passage the LLMs may read
CHUNK_OVERLAP = 200

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f​-‍⁠﻿]")
_BLANK_RUNS = re.compile(r"\n{3,}")
# Lines repeated this often are navigation, footers or share buttons rather than content.
_BOILERPLATE_REPEATS = 3

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_CHARS,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
)


def source_id_for(url: str) -> str:
    """Stable id of one page: the same URL always maps to the same id."""
    return "src_" + hashlib.sha256(normalize_url(url).encode()).hexdigest()[:12]


def clean_page(text: str) -> tuple[str, bool]:
    """Strip images, control characters, repeated boilerplate lines and blank runs, then cap the
    length. Returns ``(clean_text, truncated)``."""
    text = _CONTROL.sub("", _IMAGE.sub("", text.replace("\r\n", "\n")))
    lines = [line.rstrip() for line in text.split("\n")]
    repeats = Counter(line.strip() for line in lines if line.strip())
    kept = [line for line in lines if repeats[line.strip()] < _BOILERPLATE_REPEATS]
    text = _BLANK_RUNS.sub("\n\n", "\n".join(kept)).strip()
    if len(text) <= MAX_PAGE_CHARS:
        return text, False
    return text[:MAX_PAGE_CHARS], True


def usable_page(text: str) -> bool:
    """Whether a cleaned page has enough text to be worth reading and indexing."""
    return len(text) >= MIN_PAGE_CHARS


def chunk_page(text: str) -> list[str]:
    """Split a page into overlapping chunks along Markdown headings, paragraphs, then sentences."""
    return [chunk for chunk in _SPLITTER.split_text(text) if chunk.strip()]
