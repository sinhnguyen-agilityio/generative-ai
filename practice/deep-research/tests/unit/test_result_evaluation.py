"""Result evaluation (spec 002, FR-005; contracts/supervisor-tools.md)."""

import json

from langchain_core.messages import AIMessage

from intelligence_collector.agents.result_evaluation import evaluate_results
from intelligence_collector.contracts import EvidencePoint, ResearchTodo
from intelligence_collector.models.router import ModelRouter

TODO = ResearchTodo(
    todo_id="t1",
    research_question="How many concurrent writers does SQLite support?",
    evidence_points=[
        EvidencePoint(point_id="E1", text="Writers at the same time"),
        EvidencePoint(point_id="E2", text="Effect of WAL mode"),
    ],
)


class Evaluator:
    """Scripted result-evaluator model; records the rows it was shown."""

    model_name = "fake-evaluator"

    def __init__(self, answer=None, fail=False):
        self.answer = answer
        self.fail = fail
        self.prompts = []

    def invoke(self, messages, config=None):
        self.prompts.append(str(messages[-1].content))
        if self.fail:
            raise RuntimeError("provider down; api_key=sk-very-secret")
        return AIMessage(content=json.dumps(self.answer))


def row(url, content="passage", raw="full page", title="T"):
    return {"url": url, "content": content, "raw_content": raw, "title": title}


def evaluate(rows, model, seen=()):
    return evaluate_results(
        rows,
        todo=TODO,
        question=TODO.research_question,
        query="sqlite writers",
        search_index=2,
        seen_urls=set(seen),
        router=ModelRouter(model),
    )


def test_duplicates_and_empty_rows_are_rejected_without_a_model_call():
    model = Evaluator({"rows": []})
    outcome = evaluate(
        [
            row("https://a.test/x"),
            row("https://a.test/x#frag"),  # same page within the batch
            row("https://seen.test/page"),
            row("https://empty.test", content="", raw=""),
        ],
        model,
        seen={"https://seen.test/page"},
    )
    reasons = [(e.decision, e.reason.value if e.reason else None) for e in outcome.evaluations]
    assert reasons[1:] == [
        ("rejected", "duplicate"),
        ("rejected", "duplicate"),
        ("rejected", "no_usable_content"),
    ]
    # Only the first, new and usable row was shown to the model, as row [1].
    assert "[1]" in model.prompts[0] and "[2]" not in model.prompts[0]


def test_model_decisions_reasons_covers_and_snippet_only():
    model = Evaluator(
        {
            "rows": [
                {"index": 1, "decision": "accepted", "covers": ["E1", "E9", "E1"]},
                {"index": 2, "decision": "rejected", "reason": "low_credibility"},
                {"index": 3, "decision": "rejected", "reason": "outside_scope"},
                {"index": 4, "decision": "accepted", "covers": ["E2"]},
            ]
        }
    )
    outcome = evaluate(
        [
            row("https://a.test"),
            row("https://forum.test"),
            row("https://other.test"),
            row("https://snippet.test", raw=""),
        ],
        model,
    )
    first, forum, other, snippet = outcome.evaluations
    assert first.decision == "accepted" and first.covers == ["E1"]  # unknown id dropped
    assert not first.snippet_only
    assert (forum.reason.value, other.reason.value) == ("low_credibility", "outside_scope")
    assert snippet.decision == "accepted" and snippet.snippet_only
    assert all(e.query == "sqlite writers" and e.search_index == 2 for e in outcome.evaluations)
    assert not outcome.failed and outcome.corrections == []


def test_omitted_row_is_rejected_off_topic_with_a_correction():
    model = Evaluator({"rows": [{"index": 1, "decision": "accepted", "covers": []}]})
    outcome = evaluate([row("https://a.test"), row("https://b.test")], model)
    assert outcome.evaluations[1].decision == "rejected"
    assert outcome.evaluations[1].reason.value == "off_topic"
    assert outcome.corrections and "https://b.test" in outcome.corrections[0]


def test_failed_evaluation_accepts_nothing_and_redacts_the_error():
    outcome = evaluate([row("https://a.test")], Evaluator(fail=True))
    assert outcome.failed
    assert all(e.decision == "rejected" for e in outcome.evaluations)
    assert "sk-very-secret" not in outcome.error
