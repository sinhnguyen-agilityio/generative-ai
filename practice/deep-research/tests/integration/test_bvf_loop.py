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
BRIEF = {"research_question": "I want a technical report on local software cache behavior from primary sources."}
GOOD = cited_report(["A cache stores reusable data for later access [S1]."], {"S1": URL})
IMPROVED = cited_report(
    ["A cache stores reusable data for later access [S1].",
     "The cache reuses that stored data across later requests [S1]."], {"S1": URL},
)


def _pass(criterion="Answers the research question"):
    return {"criterion": criterion, "met": True, "justification": "Satisfied."}


def _fail(criterion="Answers the research question", justification="Not yet addressed."):
    return {"criterion": criterion, "met": False, "justification": justification}


def _judge(verdicts) -> FakeAcceptanceModel:
    """Scripted report-judge fake: each item is a full ReportJudgeVerdict-shaped dict (one
    ``results`` list). The BVF loop calls the judge once per Verify attempt."""
    return FakeAcceptanceModel(verdicts)


def _passing_judge() -> FakeAcceptanceModel:
    """A judge fake that always reports every criterion met, for tests that only care about the
    BVF loop mechanics around the writer/researcher, not judge behavior itself."""
    return _judge([{"results": [_pass()]}])


def _run(stores, report_model, *, research_model=None, budget=3, run_id="bvf",
         report_judge=None, report_guidelines=(),
         faithfulness_evaluator=None, context_precision_evaluator=None,
         response_relevancy_evaluator=None):
    vfs, index = stores
    researcher = ResearchAgent(
        ModelRouter(research_model or FakeResearchModel()),
        lambda: ControlledMCP([{"url": URL, "content": "A cache stores reusable data for later access."}]),
    )
    nodes = ResearchNodes(
        vfs,
        index,
        researcher,
        report_writer=ReportWriter(ModelRouter(report_model)),
        report_judge=ReportJudge(ModelRouter(report_judge or _passing_judge())),
        report_guidelines=report_guidelines,
        recovery_budget=budget,
        faithfulness_evaluator=faithfulness_evaluator,
        context_precision_evaluator=context_precision_evaluator,
        response_relevancy_evaluator=response_relevancy_evaluator,
    )
    todo = ResearchTodo(todo_id="todo-1", research_question=QUESTION)
    return build_graph(research_nodes=nodes).invoke({
        "run_id": run_id,
        "thread_id": run_id,
        "scoped_request": BRIEF,
        "todos": [todo.model_dump(mode="json")],
    })


def test_failed_draft_is_reflected_on_and_fixed_with_lessons(stores):
    vfs, _ = stores
    model = FakeReportModel([GOOD, IMPROVED], reflections=[{
        "diagnosis": "The draft never actually describes how the cache reuses stored data.",
        "lessons": ["State explicitly how the cache reuses stored data, cited."],
        "strategy": "Add one sentence stating the reuse mechanism, cited.",
    }])
    judge = _judge([
        {"results": [_fail(justification="The report never states how the cache reuses data.")]},
        {"results": [_pass()]},
    ])

    result = _run(stores, model, report_judge=judge)

    assert result["stage"] == "Awaiting Approval"
    assert result["verification"]["passed"] is True
    assert [v["passed"] for v in result["verification_history"]] == [False, True]
    assert result["recovery_attempt"] == 1
    assert result["bvf_lessons"] == ["State explicitly how the cache reuses stored data, cited."]
    reflect_prompt = model.prompts("reflect")[0]
    assert "[criterion_1]" in reflect_prompt
    assert "never states how the cache reuses data" in reflect_prompt
    assert "No progress since the previous attempt: no" in reflect_prompt
    fix_prompt = model.prompts("write")[1]
    assert "BVF fix attempt 1" in fix_prompt
    assert "criterion_1" in fix_prompt
    assert "reuses stored data" in fix_prompt
    assert "Add one sentence stating the reuse mechanism" in fix_prompt
    assert "BVF fix attempt" not in model.prompts("write")[0]
    history = json.loads(vfs.read("bvf", "verification/bvf_history.json"))
    assert history[0]["failed_checks"] == ["criterion_1"]
    assert history[0]["strategy"].startswith("Add one sentence")
    assert vfs.read("bvf", "reports/draft.v1.md") and vfs.read("bvf", "reports/draft.v2.md")
    assert vfs.read("bvf", "reports/previous.md").decode() == GOOD


def test_repeating_the_same_failure_escalates_strategy_then_needs_review(stores):
    vfs, _ = stores
    model = FakeReportModel([GOOD])
    judge = _judge([{"results": [_fail(justification="Still never states the reuse mechanism.")]}])

    result = _run(stores, model, report_judge=judge, budget=5)

    assert result["stage"] == "NEEDS_REVIEW"
    assert result["status"] == "needs_review"
    assert result["verification"]["passed"] is False
    assert result["recovery_attempt"] == 2  # stopped by no-progress, well before the budget of 5
    assert "No progress since the previous attempt: yes" in model.prompts("reflect")[1]
    review = json.loads(vfs.read("bvf", "verification/needs_review.json"))
    assert review["reason"] == "no progress after reflection"
    assert "never states the reuse mechanism" in review["unresolved"]
    assert review["lessons"]


def test_budget_exhaustion_ends_in_needs_review_without_verification(stores):
    """A different criterion fails each attempt (fingerprint keeps changing), so the loop is
    stopped by the recovery budget itself rather than by no-progress stall detection."""
    vfs, _ = stores
    model = FakeReportModel([GOOD])
    judge = _judge([
        {"results": [_fail(criterion="Answers the research question")]},
        {"results": [_fail(criterion="Fairly represents the gathered evidence")]},
        {"results": [_fail(criterion="Explains uncertainty and evidence gaps")]},
    ])

    result = _run(stores, model, report_judge=judge, budget=2)

    assert result["stage"] == "NEEDS_REVIEW"
    assert result["recovery_attempt"] == 2
    assert len(result["verification_history"]) == 3
    assert not any(v["passed"] for v in result["verification_history"])
    review = json.loads(vfs.read("bvf", "verification/needs_review.json"))
    assert review["reason"] == "recovery budget 2 exhausted"


def test_missing_evidence_is_fixed_by_rewriting_never_by_new_research(stores):
    """The fix is a rewrite only: an unsupported claim is removed or disclosed under Limitations
    from the same evidence, and no new research task is ever started inside the loop."""
    research = FakeResearchModel(findings={QUESTION: {
        "summary": f"A cache stores reusable data [{URL}]",
        "sufficient": False,
        "unresolved_gaps": ["cache eviction policy details"],
    }})
    disclosed = cited_report(
        ["A cache stores reusable data for later access [S1]."],
        {"S1": URL},
        limitations=["Cache eviction policy details were not found in the sources."],
    )
    model = FakeReportModel([GOOD, disclosed], reflections=[{
        "diagnosis": "The draft omits that eviction policy details are missing from the evidence.",
        "lessons": ["Disclose every unresolved gap under '## Limitations'."],
        "strategy": "Add a Limitations section stating the missing eviction policy details.",
    }])
    judge = _judge([
        {"results": [_fail(justification="The evidence says nothing about eviction policy.")]},
        {"results": [_pass()]},
    ])

    result = _run(stores, model, research_model=research, report_judge=judge)

    assert result["stage"] == "Awaiting Approval", result["verification"]["repair_details"]
    assert len(result["todos"]) == 1  # no gap-research task was added
    assert model.prompts("reflect")  # every failure is reflected on before the rewrite
    fix_prompt = model.prompts("write")[1]
    assert "BVF fix attempt 1" in fix_prompt
    assert "cache eviction policy details" in fix_prompt  # the gap reaches the writer
    assert "No new evidence is available" in fix_prompt


def test_evidence_selection_narrows_the_build_prompt_but_not_verification(stores):
    """The writer's prompt sees only evidence_top_k sources per question; Verify still reads
    every source that was actually captured, independent of that narrowing."""
    vfs, index = stores
    sources = [
        {"url": f"https://controlled.test/page{n}", "content": f"Cache related detail number {n}"}
        for n in range(8)
    ]
    researcher = ResearchAgent(
        ModelRouter(FakeResearchModel()),
        lambda: ControlledMCP([sources]),
        max_sources=8,  # this test needs more captured pages than the default per-task budget
    )
    report_model = FakeReportModel([GOOD])
    nodes = ResearchNodes(
        vfs, index, researcher,
        report_writer=ReportWriter(ModelRouter(report_model)),
        report_judge=ReportJudge(ModelRouter(_passing_judge())),
        recovery_budget=3,
        evidence_top_k=3,
    )
    todo = ResearchTodo(todo_id="todo-1", research_question=QUESTION)

    result = build_graph(research_nodes=nodes).invoke({
        "run_id": "narrow", "thread_id": "narrow",
        "scoped_request": BRIEF, "todos": [todo.model_dump(mode="json")],
    })

    build_prompt = report_model.prompts("write")[0]
    assert build_prompt.count("Excerpt:") == 3  # narrowed to evidence_top_k, not all 8

    full_catalog = load_catalog(vfs, "narrow", result["evidence"])
    assert len(full_catalog.bundles[0].source_ids) == 8  # Verify still sees every captured source


def test_judge_prompt_carries_the_fact_check_criterion_and_the_evidence(stores):
    """The judge's checklist is just the fact check (plus configured guidelines): it re-reads the
    report against the gathered evidence, with no planner acceptance criteria involved."""
    judge = _judge([{"results": [_pass()]}])

    result = _run(stores, FakeReportModel([GOOD]), report_judge=judge)

    assert result["stage"] == "Awaiting Approval"
    prompt = judge.prompts()[0].casefold()
    assert "hallucinations" in prompt
    assert "gathered evidence" in prompt
    assert "fairly represents" not in prompt  # fixed representation/language criteria are gone


def test_configured_guidelines_flow_into_the_judge_prompt_and_a_violation_is_fixed(stores):
    """Operator-configured guidelines (e.g. Settings.report_guidelines) reach the judge exactly
    like planner criteria do, and an unmet guideline drives a rewrite like any other failure."""
    guidelines = ["The report MUST be written in Markdown format.",
                  "The report MUST be written in English."]
    model = FakeReportModel([GOOD, IMPROVED], reflections=[{
        "diagnosis": "The draft was not written in English.",
        "lessons": ["Always write the report in English."],
        "strategy": "Rewrite the draft in English.",
    }])
    judge = _judge([
        {"results": [_fail(criterion=guidelines[1], justification="The draft is in French.")]},
        {"results": [_pass(criterion=guidelines[1])]},
    ])

    result = _run(stores, model, report_judge=judge, report_guidelines=guidelines)

    assert result["stage"] == "Awaiting Approval"
    prompt = judge.prompts()[0]
    assert guidelines[0] in prompt
    assert guidelines[1] in prompt
    assert result["verification_history"][0]["passed"] is False
    assert "The draft is in French" in result["verification_history"][0]["repair_details"]


def test_fact_check_criterion_reaches_the_judge_and_a_hallucination_is_fixed(stores):
    """The built-in factual-accuracy criterion is always included (not configurable, unlike
    guidelines), and a caught hallucination routes to a rewrite like any other failure."""
    model = FakeReportModel([GOOD, IMPROVED], reflections=[{
        "diagnosis": "The draft invented a statistic the evidence never states.",
        "lessons": ["Never state a number that isn't in the cited source excerpt."],
        "strategy": "Remove the invented statistic and restate only what the evidence supports.",
    }])
    judge = _judge([
        {"results": [_fail(
            criterion="Contains no factual inaccuracies, hallucinations, or internal "
                      "inconsistencies",
            justification="The draft claims a 40% speedup that appears nowhere in [S1].",
        )]},
        {"results": [_pass(criterion="Contains no factual inaccuracies, hallucinations, or "
                                      "internal inconsistencies")]},
    ])

    result = _run(stores, model, report_judge=judge)

    assert result["stage"] == "Awaiting Approval"
    prompt = judge.prompts()[0]
    assert "hallucinations" in prompt
    assert "read the whole report as one continuous piece" in prompt.casefold()
    first = result["verification_history"][0]
    assert first["passed"] is False
    assert "40% speedup" in first["repair_details"]
    assert "Remove the invented statistic" in model.prompts("write")[1]


def test_faithfulness_evaluator_is_disabled_by_default(stores):
    """No evaluator injected: behaves exactly as before this feature — no score, no error."""
    result = _run(stores, FakeReportModel([GOOD]))

    assert result["stage"] == "Awaiting Approval"
    assert result["faithfulness_score"] is None
    assert result["faithfulness_error"] == ""


def test_faithfulness_score_is_recorded_only_once_verification_passes(stores):
    vfs, _ = stores
    calls = []

    def evaluator(question, report, contexts):
        calls.append((question, report, contexts))
        return 0.95

    model = FakeReportModel([GOOD, GOOD])
    judge = _judge([
        {"results": [_fail(justification="Not yet good enough.")]},
        {"results": [_pass()]},
    ])

    result = _run(stores, model, report_judge=judge, faithfulness_evaluator=evaluator)

    assert result["stage"] == "Awaiting Approval"
    # Called once, only for the passing draft — never for the failed first attempt.
    assert len(calls) == 1
    question, report, contexts = calls[0]
    assert question == BRIEF["research_question"]
    assert report == GOOD
    assert contexts == ["A cache stores reusable data for later access."]
    assert result["faithfulness_score"] == 0.95
    assert result["faithfulness_error"] == ""
    verification_id = result["verification"]["verification_id"]
    persisted = json.loads(vfs.read("bvf", f"verification/quality.{verification_id}.json"))
    assert persisted["faithfulness_score"] == 0.95
    assert persisted["question"] == BRIEF["research_question"]
    assert persisted["faithfulness_context_count"] == 1


def test_faithfulness_evaluator_failure_is_recorded_but_never_blocks_approval(stores):
    def failing_evaluator(question, report, contexts):
        raise RuntimeError("judge LLM rate limited")

    result = _run(stores, FakeReportModel([GOOD]), faithfulness_evaluator=failing_evaluator)

    assert result["stage"] == "Awaiting Approval"
    assert result["verification"]["passed"] is True
    assert result["faithfulness_score"] is None
    assert "rate limited" in result["faithfulness_error"]


def test_context_precision_scores_the_ranked_evidence_the_final_draft_was_written_from(stores):
    vfs, _ = stores
    calls = []

    def evaluator(questions, report):
        calls.append((questions, report))
        return {"score": 0.8, "per_question": [{"question": QUESTION, "contexts": 1, "score": 0.8}]}

    result = _run(stores, FakeReportModel([GOOD]), context_precision_evaluator=evaluator)

    assert result["stage"] == "Awaiting Approval"
    assert len(calls) == 1
    questions, report = calls[0]
    assert report == GOOD
    # The per-question ranking the writer saw, resolved to excerpt text in rank order.
    assert questions == [{
        "question": QUESTION, "contexts": ["A cache stores reusable data for later access."],
    }]
    assert result["writer_evidence"] == [
        {"todo_id": "todo-1", "question": QUESTION, "source_ids": ["S1"]},
    ]
    assert json.loads(vfs.read("bvf", "synthesis/retrieval.v1.json")) == result["writer_evidence"]
    assert result["context_precision_score"] == 0.8
    assert result["context_precision_details"][0]["score"] == 0.8
    # Only the configured metrics run; the others stay empty.
    assert result["faithfulness_score"] is None and result["response_relevancy_score"] is None


def test_response_relevancy_scores_the_report_against_the_research_question(stores):
    vfs, _ = stores
    calls = []

    def evaluator(question, report):
        calls.append((question, report))
        return 0.72

    result = _run(stores, FakeReportModel([GOOD]), response_relevancy_evaluator=evaluator)

    assert calls == [(BRIEF["research_question"], GOOD)]
    assert result["response_relevancy_score"] == 0.72
    verification_id = result["verification"]["verification_id"]
    persisted = json.loads(vfs.read("bvf", f"verification/quality.{verification_id}.json"))
    assert persisted["response_relevancy_score"] == 0.72


def test_one_failing_metric_never_drops_the_others_or_blocks_approval(stores):
    def failing(*args):
        raise RuntimeError("judge LLM rate limited")

    result = _run(
        stores, FakeReportModel([GOOD]),
        faithfulness_evaluator=lambda q, r, c: 0.9,
        context_precision_evaluator=failing,
        response_relevancy_evaluator=lambda q, r: 0.7,
    )

    assert result["stage"] == "Awaiting Approval"
    assert result["faithfulness_score"] == 0.9
    assert result["response_relevancy_score"] == 0.7
    assert result["context_precision_score"] is None
    assert "rate limited" in result["context_precision_error"]
