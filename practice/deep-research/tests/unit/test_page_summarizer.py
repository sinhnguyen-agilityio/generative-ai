"""Page summarizer: the LLM reads a capped page and returns a summary with real quotes only."""

import json
import threading

from langchain_core.messages import AIMessage

from intelligence_collector.agents.page_summarizer import (
    SUMMARIZER_INPUT_CHARS,
    PageSummarizer,
    format_page_evidence,
    verbatim_excerpts,
)
from intelligence_collector.contracts import PageSummary
from intelligence_collector.models.router import ModelRouter

PAGE = (
    "# Checkpointing\n\nA checkpoint saves the graph state after every super-step. "
    "Threads resume from the latest checkpoint.\n\n" + "Background details. " * 60
)


class SummaryModel:
    """Records every summarizer prompt; answers with one real and one invented quote."""

    model_name = "fake-summarizer"

    def __init__(self, fail_urls=()):
        self.prompts, self.fail_urls, self.lock = [], set(fail_urls), threading.Lock()

    def invoke(self, messages, config=None):
        prompt = messages[-1].content
        with self.lock:
            self.prompts.append(prompt)
        if any(url in prompt for url in self.fail_urls):
            raise RuntimeError("summarizer unavailable")
        return AIMessage(
            content=json.dumps(
                {
                    "summary": "Checkpoints save state after each super-step.",
                    "key_excerpts": [
                        "A checkpoint saves the graph state  after every super-step.",
                        "Checkpoints are stored forever.",
                    ],
                }
            )
        )


def test_verbatim_excerpts_keep_only_quotes_found_in_the_page():
    quotes = ["threads RESUME from the\nlatest checkpoint.", "An invented sentence.", "  "]
    assert verbatim_excerpts(quotes, PAGE) == ["threads RESUME from the\nlatest checkpoint."]


def test_page_evidence_lists_the_quotes_under_the_summary():
    summary = PageSummary(summary="Short summary.", key_excerpts=["Quote one."])
    assert format_page_evidence(summary) == 'Short summary.\n\nKey excerpts:\n- "Quote one."'
    assert format_page_evidence(PageSummary(summary="Only text.")) == "Only text."


def test_summarize_drops_invented_quotes_and_caps_the_page_it_sends():
    model = SummaryModel()
    long_page = PAGE + "x" * (SUMMARIZER_INPUT_CHARS * 2)

    summary = PageSummarizer(ModelRouter(model)).summarize(
        "How does LangGraph checkpointing work?",
        url="https://docs.test/persistence",
        title="Persistence",
        page=long_page,
    )

    assert summary.key_excerpts == ["A checkpoint saves the graph state  after every super-step."]
    page_in_prompt = model.prompts[0].split("<untrusted_page>\n", 1)[1]
    assert len(page_in_prompt) < SUMMARIZER_INPUT_CHARS + 100
    assert "How does LangGraph checkpointing work?" in model.prompts[0]


def test_read_pages_summarizes_usable_pages_and_keeps_snippets_otherwise():
    model = SummaryModel(fail_urls={"https://docs.test/broken"})
    rows = [
        {"url": "https://docs.test/a", "title": "A", "content": "snippet a", "raw_content": PAGE},
        {"url": "https://docs.test/short", "content": "snippet short", "raw_content": "Too short."},
        {"url": "https://docs.test/none", "content": "snippet none"},
        {"url": "https://docs.test/broken", "content": "snippet broken", "raw_content": PAGE},
    ]

    read = PageSummarizer(ModelRouter(model)).read_pages("checkpointing", rows)

    assert [row["url"] for row in read] == [row["url"] for row in rows]  # order kept
    summarized, short, missing, broken = read
    assert summarized["content"].startswith("Checkpoints save state after each super-step.")
    assert "Key excerpts:" in summarized["content"]
    assert summarized["snippet"] == "snippet a"
    assert summarized["raw_content"].startswith("# Checkpointing")
    assert summarized["raw_truncated"] is False
    for row, snippet in ((short, "snippet short"), (missing, "snippet none")):
        assert row["content"] == snippet and row["raw_content"] == ""
    assert broken["content"] == "snippet broken"  # the snippet is still evidence
    assert "summarizer unavailable" in broken["summary_error"]
    assert broken["raw_content"]  # the page itself is still stored and indexed
    # Only usable pages reach the model (the router may retry the failing one).
    sent = {prompt.split("Page URL: ", 1)[1].split("\n", 1)[0] for prompt in model.prompts}
    assert sent == {"https://docs.test/a", "https://docs.test/broken"}
