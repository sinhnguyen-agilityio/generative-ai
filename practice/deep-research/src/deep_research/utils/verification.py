"""Parse Markdown claims, remove unsupported sentences, and format their evidence."""

from __future__ import annotations

import re

from deep_research.utils.text import CITATION, SENTENCE_END

_TABLE_RULE = re.compile(r"^\|?[\s:|-]+\|?$")


_LIST_MARK = re.compile(r"^\s*(?:(?:[-*+]|\d+[.)])\s+|>\s*)?")


def parse(body: str) -> list[tuple[str, list[str] | None]]:
    """Split a Markdown body into verifiable sentences while preserving structural lines.

    Headings, blank lines, table rules, and fenced code have no sentences. Other lines
    keep their list marker separately; a table row is treated as one sentence.

    Args:
        body: Report body without its generated Sources section.

    Returns:
        Pairs of preserved text/list marker and a sentence list, or None for structural lines.
    """
    lines, fenced = [], False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
        if fenced or not stripped or stripped.startswith(("```", "#")) or _TABLE_RULE.match(stripped):
            lines.append((line, None))
        elif stripped.startswith("|"):
            lines.append(("", [line]))
        else:
            mark = _LIST_MARK.match(line).group(0)
            lines.append((mark, SENTENCE_END.split(line[len(mark):].strip())))
    return lines


def remove(lines: list[tuple[str, list[str] | None]], drop: set[tuple[int, int]]) -> str:
    """Remove selected sentences, empty content lines, and sections left without content.

    Args:
        lines: Parsed report lines from parse.
        drop: Zero-based (line index, sentence index) pairs to remove.

    Returns:
        Rebuilt Markdown body with redundant blank lines collapsed.
    """
    kept = []
    for i, (text, sentences) in enumerate(lines):
        if sentences is None:
            kept.append(text)
        elif rest := [s for j, s in enumerate(sentences) if (i, j) not in drop]:
            kept.append(text + " ".join(rest))
    body, after = [], None  # heading level of the next non-blank line kept; 0 for content
    for line in reversed(kept):
        level = len(line) - len(line.lstrip("#"))
        if level > 1 and (after is None or 0 < after <= level):
            continue
        body.append(line)
        if line.strip():
            after = level
    return re.sub(r"\n{3,}", "\n\n", "\n".join(reversed(body))).strip()


def evidence(units, sources: dict[int, tuple[str, str]], passages: dict[str, list[str]]) -> str:
    """Format evidence for the known sources cited by the report sentences.

    Args:
        units: Indexed sentence tuples (line index, sentence index, text).
        sources: Global source numbers mapped to (title, URL).
        passages: Read source passages grouped by URL.

    Returns:
        Source-numbered passages with placeholders for unread sources, or "None.".
    """
    cited = sorted({int(n) for _, _, s in units for n in CITATION.findall(s)} & sources.keys())
    blocks = []
    for n in cited:
        title, url = sources[n]
        texts = passages.get(url) or ["(no passage of this source was read)"]
        blocks.append(f"[{n}] {title}\n" + "\n---\n".join(texts))
    return "\n\n".join(blocks) or "None."
