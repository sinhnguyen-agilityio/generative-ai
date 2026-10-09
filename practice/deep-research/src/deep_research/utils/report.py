"""Build report source lists from trusted citation metadata."""

from __future__ import annotations

from deep_research.utils.text import CITATION

SOURCES_HEADING = "\n\n## Sources\n"


def with_sources(body: str, sources: dict[int, tuple[str, str]]) -> str:
    """Append a source list containing only citation numbers used in the report body.

    Args:
        body: Report body with valid numbered citations.
        sources: Source-number mapping to (title, URL); every cited number must be present.

    Returns:
        Body with a Sources section, or the unchanged body when it has no citations.
    """
    cited = sorted({int(n) for n in CITATION.findall(body)})
    if not cited:
        return body
    return body + SOURCES_HEADING + "\n".join(f"[{n}] {sources[n][0]}: {sources[n][1]}" for n in cited)
