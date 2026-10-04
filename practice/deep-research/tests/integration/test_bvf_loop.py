"""Build -> Verify (three gates) -> diagnose/repair loop, with deterministic fake models and tools.

These tests check workflow behavior, not answer quality: the fakes return scripted drafts and
verdicts (see evals/ for the quality evaluation with real evidence).
"""

import json

from intelligence_collector.agents.report_judge import ReportJudge
from intelligence_collector.agents.report_writer import ReportWriter
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.contracts import ResearchTodo
from intelligence_collector.graph.build import build_graph
from intelligence_collector.graph.nodes import ResearchNodes
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.source_verification import load_catalog
from intelligence_collector.tools.mcp import ControlledMCP
from tests.fixtures.fake_models import (
    FakeAcceptanceModel,
    FakeReportModel,
    FakeResearchModel,
    cited_report,
)

URL = "https://controlled.test/cache"
QUESTION = "How does a local software cache store and reuse data?"
BRIEF = {
    "research_question": "How does a local software cache store and reuse data?",
    "required_answers": ["how a local software cache stores and reuses data"],
}
GOOD = cited_report(["A cache stores reusable data for later access [S1]."], {"S1": URL})
IMPROVED = cited_report(
    [
        "A cache stores reusable data for later access [S1].",
        "The cache reuses that stored data across later requests [S1].",
    ],
    {"S1": URL},
)


def _pass(criterion="Answers the research question"):
    return {"criterion": criterion, "met": True, "justification": "Satisfied."}


def _fail(
    criterion="Answers the research question",
    justification="Not yet addressed.",
    missing=(),
    issue_type="draft_defect",
):
    return {
        "criterion": criterion,
        "met": False,
        "justification": justification,
        "missing_points": list(missing),
        "issue_type": issue_type,
    }


def _judge(verdicts) -> FakeAcceptanceModel:
    """Scripted coverage judge: one ReportJudgeVerdict-shaped dict per Verify attempt."""
    return FakeAcceptanceModel(verdicts)


def _faithful(score=0.95, unsupported=()):
    calls = []

    def evaluator(question, report, contexts):
        calls.append((question, report, contexts))
        return {"score": score, "unsupported": list(unsupported)}

    evaluator.calls = calls
    return evaluator


class CountingMCP(ControlledMCP):
    """ControlledMCP shared across sub-agents that counts every search (stand-in for Tavily)."""

    searches = 0

    def search(self, *args, **kwargs):
        CountingMCP.searches += 1
        return super().search(*args, **kwargs)


def _nodes(
    stores,
    report_model,
    *,
    research_model=None,
    budget=3,
    report_judge=None,
    report_guidelines=(),
    faithfulness_evaluator=None,
    require_faithfulness=False,
    context_precision_evaluator=None,
    response_relevancy_evaluator=None,
    mcp_rows=None,
):
    vfs, index = stores
    CountingMCP.searches = 0
    rows = mcp_rows or [{"url": URL, "content": "A cache stores reusable data for later access."}]
    researcher = ResearchAgent(
        ModelRouter(research_model or FakeResearchModel()), lambda: CountingMCP(rows)
    )
    return ResearchNodes(
        vfs,
        index,
        researcher,
        report_writer=ReportWriter(ModelRouter(report_model)),
        report_judge=ReportJudge(ModelRouter(report_judge or _judge([{"results": [_pass()]}]))),
        report_guidelines=report_guidelines,
        recovery_budget=budget,
        faithfulness_evaluator=faithfulness_evaluator
        if faithfulness_evaluator is not None
        else _faithful(),
        require_faithfulness=require_faithfulness,
        context_precision_evaluator=context_precision_evaluator,
        response_relevancy_evaluator=response_relevancy_evaluator,
    )


def _run(stores, report_model, *, run_id="bvf", **kwargs):
    nodes = _nodes(stores, report_model, **kwargs)
    todo = ResearchTodo(todo_id="todo-1", research_question=QUESTION)
    return build_graph(research_nodes=nodes).invoke(
        {
            "run_id": run_id,
            "thread_id": run_id,
            "scoped_request": BRIEF,
            "todos": [todo.model_dump(mode="json")],
        }
    )


def test_passing_first_draft_stops_without_an_improvement_cycle(stores):
    vfs, _ = stores
    model = FakeReportModel([GOOD])
    faithfulness = _faithful(0.95)

    result = _run(stores, model, faithfulness_evaluator=faithfulness)

    assert result["stage"] == "Awaiting Approval"
    assert result["verification_outcome"] == "PASS"
    assert result["draft_version"] == 1 and len(model.prompts("write")) == 1
    assert result["recovery_attempt"] == 0 if "recovery_attempt" in result else True
    assert result.get("bvf_history", []) == []
    assert [c["gate"] for c in result["verification"]["check_results"]] == [
        "coverage",
        "faithfulness",
    ]
    assert result["faithfulness_score"] == 0.95 and len(faithfulness.calls) == 1
    assert CountingMCP.searches == 1  # one task, one search, no extra research
    record = json.loads(
        vfs.read("bvf", f"verification/result.{result['verification']['verification_id']}.json")
    )
    assert record["outcome"] == "PASS" and record["faithfulness_score"] == 0.95


def test_writer_receives_question_checklist_constraints_and_answer_first_rules(stores):
    model = FakeReportModel([GOOD])

    _run(stores, model)

    prompt = model.prompts("write")[0]
    assert "Required answers (answer each one in '## Answer')" in prompt
    assert "- how a local software cache stores and reuses data" in prompt
    assert 'never write "the provided evidence"' in prompt
    assert "never label the report INCOMPLETE" in prompt
    assert "[S1]" in prompt and URL in prompt


def test_coverage_failure_is_repaired_from_stored_evidence_without_new_research(stores):
    vfs, _ = stores
    model = FakeReportModel([GOOD, IMPROVED])
    judge = _judge(
        [
            {
                "results": [
                    _fail(
                        justification="Mentions caching but never says how data is reused.",
                        missing=["states how the cache reuses stored data"],
                    )
                ]
            },
            {"results": [_pass()]},
        ]
    )

    result = _run(stores, model, report_judge=judge)

    assert result["stage"] == "Awaiting Approval"
    assert [v["passed"] for v in result["verification_history"]] == [False, True]
    assert result["recovery_attempt"] == 1
    assert CountingMCP.searches == 1  # the evidence was already stored: no Tavily call
    history = json.loads(vfs.read("bvf", "verification/bvf_history.json"))
    assert history[0]["route"] == "vfs"
    assert history[0]["issues"][0]["missing_points"] == ["states how the cache reuses stored data"]
    repair_prompt = model.prompts("write")[1]
    assert "REPAIR of draft v1" in repair_prompt
    assert "states how the cache reuses stored data" in repair_prompt
    assert GOOD in repair_prompt  # the previous draft is repaired, not discarded
    failed = result["verification_history"][0]["check_results"][0]
    assert failed["item"] == "how a local software cache stores and reuses data"
    assert failed["missing_points"] == ["states how the cache reuses stored data"]


def test_repair_keeps_answers_that_already_passed(stores):
    brief = {**BRIEF, "required_answers": ["how data is stored", "how data is reused"]}
    model = FakeReportModel([GOOD, IMPROVED])
    judge = _judge(
        [
            {"results": [_pass("[C1]"), _fail("[C2]", missing=["states the reuse mechanism"])]},
            {"results": [_pass("[C1]"), _pass("[C2]")]},
        ]
    )
    nodes = _nodes(stores, model, report_judge=judge)
    todo = ResearchTodo(todo_id="todo-1", research_question=QUESTION)
    result = build_graph(research_nodes=nodes).invoke(
        {
            "run_id": "keep",
            "thread_id": "keep",
            "scoped_request": brief,
            "todos": [todo.model_dump(mode="json")],
        }
    )

    assert result["verification_outcome"] == "PASS"
    repair_prompt = model.prompts("write")[1]
    passing = repair_prompt.split("passed:\n", 1)[1].split("\n\n", 1)[0]
    assert "- how data is stored" in passing
    assert "how data is reused" not in passing


def test_evidence_gap_triggers_targeted_research_for_the_affected_task_only(stores):
    vfs, _ = stores
    model = FakeReportModel([GOOD, IMPROVED])
    judge = _judge(
        [
            {"results": [_fail(issue_type="evidence_gap", missing=["eviction policy"])]},
            {"results": [_pass()]},
        ]
    )

    result = _run(stores, model, report_judge=judge)

    assert result["stage"] == "Awaiting Approval"
    assert CountingMCP.searches == 2  # exactly one targeted search for the reopened task
    todo = result["todos"][0]
    assert todo["research_attempt"] == 2
    assert todo["research_instruction"] == "Find evidence for: eviction policy"
    assert json.loads(vfs.read("bvf", "verification/bvf_history.json"))[0]["route"] == "research"
    assert "New targeted evidence" in model.prompts("write")[1]


def test_repeated_gap_after_targeted_research_does_not_research_again_without_a_planner(stores):
    model = FakeReportModel([GOOD])
    judge = _judge([{"results": [_fail(issue_type="evidence_gap", missing=["eviction policy"])]}])

    result = _run(stores, model, report_judge=judge, budget=3)

    routes = [entry["route"] for entry in result["bvf_history"]]
    assert routes[0] == "research" and "research" not in routes[1:]
    assert CountingMCP.searches == 2
    assert result["stage"] == "NEEDS_REVIEW"


def test_repeating_the_same_failure_stops_at_no_progress(stores):
    vfs, _ = stores
    judge = _judge([{"results": [_fail(justification="Still never states the reuse mechanism.")]}])

    result = _run(stores, FakeReportModel([GOOD]), report_judge=judge, budget=5)

    assert result["stage"] == "NEEDS_REVIEW"
    assert result["recovery_attempt"] == 2  # stopped by no-progress, before the budget of 5
    review = json.loads(vfs.read("bvf", "verification/needs_review.json"))
    assert review["reason"] == "no progress after repair"
    assert "never states the reuse mechanism" in review["unresolved"]


def test_cycle_budget_is_a_maximum(stores):
    """A different criterion fails each attempt, so only the budget stops the loop."""
    vfs, _ = stores
    guidelines = [
        "The report MUST be written in Markdown format.",
        "The report MUST be written in English.",
    ]
    judge = _judge(
        [
            {"results": [_fail()]},
            {
                "results": [
                    _fail(
                        criterion="Fairly represents the gathered evidence",
                        issue_type="presentation",
                    )
                ]
            },
            {
                "results": [
                    _fail(
                        criterion="Explains uncertainty and evidence gaps",
                        issue_type="presentation",
                    )
                ]
            },
        ]
    )

    result = _run(
        stores, FakeReportModel([GOOD]), report_judge=judge, budget=2, report_guidelines=guidelines
    )

    assert result["stage"] == "NEEDS_REVIEW"
    assert result["recovery_attempt"] == 2
    assert len(result["verification_history"]) == 3
    review = json.loads(vfs.read("bvf", "verification/needs_review.json"))
    assert review["reason"] == "improvement cycle budget 2 exhausted"


def test_presentation_failure_edits_presentation_only(stores):
    guidelines = ["The report MUST be written in English."]
    model = FakeReportModel([GOOD, IMPROVED])
    judge = _judge(
        [
            {
                "results": [
                    _pass(),
                    _fail(
                        criterion=guidelines[0],
                        justification="Written in French.",
                        issue_type="presentation",
                    ),
                ]
            },
            {"results": [_pass(), _pass(criterion=guidelines[0])]},
        ]
    )

    result = _run(stores, model, report_judge=judge, report_guidelines=guidelines)

    assert result["stage"] == "Awaiting Approval"
    assert result["bvf_history"][0]["route"] == "presentation"
    assert "Change only presentation" in model.prompts("write")[1]
    assert CountingMCP.searches == 1


def test_faithfulness_must_be_strictly_above_090(stores):
    vfs, _ = stores
    scores = iter(
        [
            {"score": 0.9, "unsupported": ["The cache is 40% faster."]},
            {"score": 0.97, "unsupported": []},
        ]
    )
    model = FakeReportModel([GOOD, IMPROVED])

    result = _run(stores, model, faithfulness_evaluator=lambda q, r, c: next(scores))

    first = result["verification_history"][0]
    assert first["passed"] is False
    assert [c["gate"] for c in first["check_results"] if not c["passed"]] == ["faithfulness"]
    assert "The cache is 40% faster." in model.prompts("write")[1]  # targeted at the claim
    assert result["verification_outcome"] == "PASS" and result["faithfulness_score"] == 0.97


def test_evaluator_error_never_rewrites_the_report(stores):
    calls = []

    def failing(question, report, contexts):
        calls.append(report)
        raise RuntimeError("judge LLM rate limited")

    model = FakeReportModel([GOOD])
    result = _run(stores, model, faithfulness_evaluator=failing)

    assert result["stage"] == "NEEDS_REVIEW"
    assert result["verification_outcome"] == "ERROR"
    assert result["verification"] is None
    assert len(model.prompts("write")) == 1 and result["draft_version"] == 1
    assert len(calls) == 2  # first attempt plus one bounded retry, same draft each time
    assert result.get("bvf_history", []) == []
    assert "Faithfulness evaluation failed" in result["verifier_error"]


def test_retry_after_error_reuses_saved_coverage_checks(stores):
    outcomes = iter([RuntimeError("timeout"), {"score": 0.95, "unsupported": []}])

    def flaky(question, report, contexts):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    judge = _judge([{"results": [_pass()]}])
    result = _run(stores, FakeReportModel([GOOD]), report_judge=judge, faithfulness_evaluator=flaky)

    assert result["verification_outcome"] == "PASS"
    assert len(judge.calls) == 1  # coverage was not re-judged for the same draft


def test_nan_or_missing_faithfulness_score_is_error(stores):
    result = _run(
        stores, FakeReportModel([GOOD]), faithfulness_evaluator=lambda q, r, c: float("nan")
    )
    assert result["verification_outcome"] == "ERROR" and result["stage"] == "NEEDS_REVIEW"


def test_required_faithfulness_without_evaluator_is_error(stores):
    vfs, index = stores
    nodes = _nodes(stores, FakeReportModel([GOOD]), require_faithfulness=True)
    nodes.bvf.faithfulness_evaluator = None
    todo = ResearchTodo(todo_id="todo-1", research_question=QUESTION)
    result = build_graph(research_nodes=nodes).invoke(
        {
            "run_id": "cfg",
            "thread_id": "cfg",
            "scoped_request": BRIEF,
            "todos": [todo.model_dump(mode="json")],
        }
    )

    assert result["verification_outcome"] == "ERROR"
    assert "not configured" in result["verifier_error"]


def test_small_saved_evidence_is_read_directly_without_chroma(stores):
    vfs, _ = stores
    sources = [
        {"url": f"https://controlled.test/page{n}", "content": f"Cache detail {n}"}
        for n in range(8)
    ]

    class RecordingIndex:
        retrievals = 0

        def __init__(self, inner):
            self.inner = inner

        def index(self, chunks):
            return self.inner.index(chunks)

        def retrieve(self, *args, **kwargs):
            RecordingIndex.retrievals += 1
            return self.inner.retrieve(*args, **kwargs)

    index = RecordingIndex(stores[1])
    researcher = ResearchAgent(
        ModelRouter(FakeResearchModel()), lambda: ControlledMCP([sources]), max_sources=8
    )
    model = FakeReportModel([GOOD])
    nodes = ResearchNodes(
        vfs,
        index,
        researcher,
        report_writer=ReportWriter(ModelRouter(model)),
        report_judge=ReportJudge(ModelRouter(_judge([{"results": [_pass()]}]))),
        evidence_top_k=3,
        faithfulness_evaluator=_faithful(),
    )
    todo = ResearchTodo(todo_id="todo-1", research_question=QUESTION)
    build_graph(research_nodes=nodes).invoke(
        {
            "run_id": "small",
            "thread_id": "small",
            "scoped_request": BRIEF,
            "todos": [todo.model_dump(mode="json")],
        }
    )

    assert model.prompts("write")[0].count("Excerpt:") == 8
    assert RecordingIndex.retrievals == 0


def test_large_corpus_is_narrowed_for_the_writer_but_not_for_verification(stores):
    vfs, index = stores
    sources = [
        {"url": f"https://controlled.test/page{n}", "content": f"Cache related detail {n} " * 120}
        for n in range(8)
    ]
    researcher = ResearchAgent(
        ModelRouter(FakeResearchModel()), lambda: ControlledMCP([sources]), max_sources=8
    )
    report_model = FakeReportModel([GOOD])
    nodes = ResearchNodes(
        vfs,
        index,
        researcher,
        report_writer=ReportWriter(ModelRouter(report_model)),
        report_judge=ReportJudge(ModelRouter(_judge([{"results": [_pass()]}]))),
        evidence_top_k=3,
        faithfulness_evaluator=_faithful(),
    )
    nodes.bvf.direct_read_chars = 1000  # this corpus is too large to read directly
    todo = ResearchTodo(todo_id="todo-1", research_question=QUESTION)
    result = build_graph(research_nodes=nodes).invoke(
        {
            "run_id": "narrow",
            "thread_id": "narrow",
            "scoped_request": BRIEF,
            "todos": [todo.model_dump(mode="json")],
        }
    )

    assert report_model.prompts("write")[0].count("Excerpt:") == 3
    assert len(load_catalog(vfs, "narrow", result["evidence"]).bundles[0].source_ids) == 8


def test_new_draft_invalidates_verification_and_approval(stores):
    nodes = _nodes(stores, FakeReportModel([IMPROVED]))
    vfs, _ = stores
    todo = ResearchTodo(todo_id="todo-1", research_question=QUESTION)
    state = nodes.researching(
        {
            "run_id": "inv",
            "thread_id": "inv",
            "scoped_request": BRIEF,
            "todos": [todo.model_dump(mode="json")],
        }
    )
    state.update(
        draft=GOOD,
        draft_version=1,
        verification={"passed": True, "artifact_hash": "x"},
        verification_outcome="PASS",
        approval={"approved": True},
        publication={"published": False},
        faithfulness_score=0.95,
    )

    result = nodes.bvf.synthesizing(state)

    assert result["draft"] == IMPROVED and result["draft_version"] == 2
    assert result["verification"] is None and result["verification_outcome"] == ""
    assert result["approval"] is None and result["publication"] is None
    assert result["faithfulness_score"] is None


def test_context_precision_scores_the_ranked_evidence_the_final_draft_was_written_from(stores):
    vfs, _ = stores
    calls = []

    def evaluator(questions, report):
        calls.append((questions, report))
        return {"score": 0.8, "per_question": [{"question": QUESTION, "contexts": 1, "score": 0.8}]}

    result = _run(stores, FakeReportModel([GOOD]), context_precision_evaluator=evaluator)

    assert result["stage"] == "Awaiting Approval"
    questions, report = calls[0]
    assert report == GOOD
    assert questions == [
        {"question": QUESTION, "contexts": ["A cache stores reusable data for later access."]}
    ]
    assert result["writer_evidence"] == [
        {"todo_id": "todo-1", "question": QUESTION, "source_ids": ["S1"]}
    ]
    assert json.loads(vfs.read("bvf", "synthesis/retrieval.v1.json")) == result["writer_evidence"]
    assert result["context_precision_score"] == 0.8


def test_optional_metric_failure_never_blocks_approval(stores):
    def failing(*args):
        raise RuntimeError("judge LLM rate limited")

    result = _run(
        stores,
        FakeReportModel([GOOD]),
        context_precision_evaluator=failing,
        response_relevancy_evaluator=lambda q, r: 0.7,
    )

    assert result["stage"] == "Awaiting Approval"
    assert result["faithfulness_score"] == 0.95  # the gate score is not overwritten
    assert result["response_relevancy_score"] == 0.7
    assert "rate limited" in result["context_precision_error"]


def test_recovery_revision_replies_in_chat_and_shows_the_new_report(stores):
    """Observed in the app: a revise recovery produced a new draft, but the chat stayed silent."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    scores = iter([0.5, 0.95])
    nodes = _nodes(
        stores,
        FakeReportModel([GOOD, IMPROVED]),
        budget=0,
        faithfulness_evaluator=lambda q, r, c: {"score": next(scores), "unsupported": ["x"]},
    )
    graph = build_graph(research_nodes=nodes, checkpointer=MemorySaver(), with_approval=True)
    config = {"configurable": {"thread_id": "rec"}}
    todo = ResearchTodo(todo_id="todo-1", research_question=QUESTION)

    paused = graph.invoke(
        {
            "run_id": "rec",
            "thread_id": "rec",
            "scoped_request": BRIEF,
            "todos": [todo.model_dump(mode="json")],
        },
        config,
    )
    assert paused["stage"] == "NEEDS_REVIEW"
    assert paused["messages"][-1].content.startswith("Report draft v1 could not be verified")
    assert "Faithfulness" in paused["messages"][-1].content

    result = graph.invoke(
        Command(resume={"action": "revise", "feedback": "Keep it to one sentence."}), config
    )

    assert result["stage"] == "Awaiting Approval" and result["draft"] == IMPROVED
    chat = [(m.type, m.content) for m in result["messages"]]
    assert ("human", "Recovery action: revise — Keep it to one sentence.") in chat
    assert chat[-1] == (
        "ai",
        "Report draft v2 passed verification (coverage, citations, "
        "faithfulness) and is ready for your review below.",
    )


def test_needs_review_ui_shows_the_unverified_draft_and_hides_options_after_a_new_report():
    from intelligence_collector import app

    class UI:
        def __init__(self):
            self.markdown_calls, self.captions, self.expanded = [], [], []

        def chat_message(self, role):
            from contextlib import nullcontext

            return nullcontext()

        def markdown(self, text):
            self.markdown_calls.append(text)

        def caption(self, text):
            self.captions.append(text)

        def expander(self, label, expanded=False):
            from contextlib import nullcontext

            self.expanded.append(expanded)
            return nullcontext()

        def selectbox(self, *args, **kwargs):
            return "revise"

        def text_area(self, *args, **kwargs):
            return ""

        def button(self, *args, **kwargs):
            return False

    snapshot = {"stage": "NEEDS_REVIEW", "draft": GOOD, "draft_version": 3, "messages": []}
    ui = UI()
    app._render_transcript(ui, snapshot)
    assert GOOD in ui.markdown_calls
    assert any("Unverified draft v3" in caption for caption in ui.captions)

    app._render_recovery_controls(ui, snapshot)
    app._render_recovery_controls(ui, {**snapshot, "bvf_history": [{"kind": "RECOVERY_DECISION"}]})
    assert ui.expanded == [True, False]

    hidden = UI()
    app._render_recovery_controls(hidden, {**snapshot, "stage": "Awaiting Approval"})
    assert hidden.expanded == []
