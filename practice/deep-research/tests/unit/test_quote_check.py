"""Verbatim quote checks used for searcher reports, gap coverage and claims (FR-013a)."""

from intelligence_collector.source_verification import (
    normalize_for_quote,
    quote_in_text,
    verified_prefix,
)
from intelligence_collector.storage.evidence_index import stored_text_for_source
from intelligence_collector.storage.vfs import VFS

PAGE = "WAL mode: readers don’t block writers.\n\nHowever,   there can only be ONE writer at a time — always."


def test_normalization_ignores_case_spacing_quotes_and_dashes():
    assert normalize_for_quote("  Don’t — X ") == "don't - x"
    assert quote_in_text("there can only be one writer at a time", PAGE)
    assert quote_in_text("readers don't block writers", PAGE)
    assert quote_in_text("one writer at a time - always", PAGE)


def test_paraphrase_blank_and_wrong_quotes_fail():
    assert not quote_in_text("only a single writer is allowed", PAGE)
    assert not quote_in_text("   ", PAGE)
    assert not quote_in_text('"."', PAGE)


def test_verified_prefix_keeps_the_longest_verifying_start():
    quote = "there can only be one writer at a time and many readers in parallel"
    assert verified_prefix(quote, PAGE, 30) == "there can only be one writer at a time"
    assert verified_prefix(quote, PAGE, 3) == "there can only"
    assert verified_prefix("nothing here", PAGE, 30) == ""


def test_stored_text_includes_the_page_or_the_snippet(tmp_path):
    vfs = VFS(tmp_path)
    vfs.write("run", "research/t1/raw/0.md", PAGE, "raw_page")
    with_page = stored_text_for_source(
        vfs, "run", {"raw_path": "research/t1/raw/0.md", "excerpt": "selected passage"}
    )
    assert "ONE writer" in with_page and "selected passage" in with_page
    snippet_only = stored_text_for_source(vfs, "run", {"excerpt": "snippet text"})
    assert snippet_only == "snippet text"
