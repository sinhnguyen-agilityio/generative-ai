"""Raw page handling: stable ids, cleaning, size limits and chunking are pure functions."""

from intelligence_collector.tools.page_content import (
    CHUNK_CHARS,
    MAX_PAGE_CHARS,
    MIN_PAGE_CHARS,
    chunk_page,
    clean_page,
    source_id_for,
    usable_page,
)


def test_source_id_is_stable_per_page_and_distinct_across_pages():
    same = [
        "https://docs.test/guide",
        "https://DOCS.test/guide/",
        "https://docs.test/guide#section",
    ]
    assert len({source_id_for(url) for url in same}) == 1
    assert source_id_for("https://docs.test/guide") != source_id_for("https://docs.test/other")
    assert source_id_for("https://docs.test/guide").startswith("src_")


def test_clean_page_strips_images_controls_boilerplate_and_blank_runs():
    nav = "Home | Docs | Blog"
    page = "\n".join(
        [
            nav,
            "# Checkpointing",
            "![diagram](https://docs.test/img.png)",
            "A checkpoint​ saves graph state\x07 after every step.",
            "",
            "",
            "",
            nav,
            "Threads resume from the latest checkpoint.",
            nav,
        ]
    )

    text, truncated = clean_page(page)

    assert truncated is False
    assert nav not in text  # repeated three times: navigation, not content
    assert "![diagram]" not in text and "​" not in text and "\x07" not in text
    assert "\n\n\n" not in text
    assert text == (
        "# Checkpointing\n\nA checkpoint saves graph state after every step.\n\n"
        "Threads resume from the latest checkpoint."
    )


def test_clean_page_caps_long_pages_and_flags_the_cut():
    text, truncated = clean_page("word " * (MAX_PAGE_CHARS // 2))
    assert truncated is True
    assert len(text) == MAX_PAGE_CHARS


def test_short_pages_are_not_usable():
    assert not usable_page("x" * (MIN_PAGE_CHARS - 1))
    assert usable_page("x" * MIN_PAGE_CHARS)


def test_chunks_overlap_follow_headings_and_are_deterministic():
    sections = [
        f"## Section {n}\n" + " ".join(f"Sentence {n}.{i} about topic {n}." for i in range(80))
        for n in range(3)
    ]
    page = "\n\n".join(sections)

    chunks = chunk_page(page)

    assert chunks == chunk_page(page)  # same page, same chunks: they can be rebuilt any time
    assert len(chunks) > 3
    assert all(len(chunk) <= CHUNK_CHARS for chunk in chunks)
    # Every section heading starts a chunk, and no text is lost.
    assert sum(chunk.startswith("## Section") for chunk in chunks) == 3
    joined = " ".join(chunks)
    assert all(f"Sentence {n}.79 about topic {n}." in joined for n in range(3))
    # Neighbouring chunks inside a section share some text (overlap).
    assert any(chunks[i][-40:].split()[-1] in chunks[i + 1] for i in range(len(chunks) - 1))
