import json

from intelligence_collector.agents.acceptance_verifier import AcceptanceVerifier
from intelligence_collector.agents.report_writer import ReportWriter
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.contracts import ResearchTodo
from intelligence_collector.graph.build import build_graph
from intelligence_collector.graph.nodes import ResearchNodes
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.source_verification import gap_id, load_catalog, main as verify_script
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


def _uncited(sentence: str) -> str:
    return cited_report(["A cache stores reusable data for later access [S1].", sentence], {"S1": URL})


def _run(stores, report_model, *, research_model=None, budget=3, run_id="bvf",
          acceptance_verifier=None, acceptance_criteria=None, faithfulness_evaluator=None):
    vfs, index = stores
    researcher = ResearchAgent(
        ModelRouter(research_model or FakeResearchModel(), mode="fake"),
        lambda: ControlledMCP([{"url": URL, "content": "A cache stores reusable data for later access."}]),
    )
    nodes = ResearchNodes(
        vfs,
        index,
        researcher,
        report_writer=ReportWriter(ModelRouter(report_model, mode="fake")),
        acceptance_verifier=acceptance_verifier,
        recovery_budget=budget,
        faithfulness_evaluator=faithfulness_evaluator,
    )
    todo = ResearchTodo(todo_id="todo-1", research_question=QUESTION)
    return build_graph(research_nodes=nodes).invoke({
        "run_id": run_id,
        "thread_id": run_id,
        "scoped_request": BRIEF,
        "todos": [todo.model_dump(mode="json")],
        "acceptance_criteria": acceptance_criteria or [],
    })


def test_failed_draft_is_reflected_on_and_fixed_with_lessons(stores):
    vfs, _ = stores
    bad = _uncited("Caches also make every workload faster overall.")
    model = FakeReportModel([bad, GOOD], reflections=[{
        "diagnosis": "The draft added a generalisation that no source states.",
        "lessons": ["Only state what a cited excerpt says; drop unsourced generalisations."],
        "strategy": "Remove the unsupported sentence instead of guessing a citation.",
    }])

    result = _run(stores, model)

    assert result["stage"] == "Awaiting Approval"
    assert result["verification"]["passed"] is True
    assert [v["passed"] for v in result["verification_history"]] == [False, True]
    assert result["recovery_attempt"] == 1
    assert result["bvf_lessons"] == ["Only state what a cited excerpt says; drop unsourced generalisations."]
    reflect_prompt = model.prompts("reflect")[0]
    assert "[claims_cited] Uncited claim" in reflect_prompt
    assert "No progress since the previous attempt: no" in reflect_prompt
    fix_prompt = model.prompts("write")[1]
    assert "BVF fix attempt 1" in fix_prompt
    assert "[claims_cited]" in fix_prompt
    assert "drop unsourced generalisations" in fix_prompt
    assert "Remove the unsupported sentence" in fix_prompt
    assert "Caches also make every workload faster overall." in fix_prompt  # previous draft
    assert "BVF fix attempt" not in model.prompts("write")[0]
    history = json.loads(vfs.read("bvf", "verification/bvf_history.json"))
    assert history[0]["failed_checks"] == ["claims_cited"]
    assert history[0]["strategy"].startswith("Remove")
    assert vfs.read("bvf", "reports/draft.v1.md") and vfs.read("bvf", "reports/draft.v2.md")
    assert vfs.read("bvf", "reports/previous.md").decode() == bad


def test_repeating_the_same_failure_escalates_strategy_then_needs_review(stores):
    vfs, _ = stores
    model = FakeReportModel([_uncited("Caches also make every workload faster overall.")])

    result = _run(stores, model, budget=5)

    assert result["stage"] == "NEEDS_REVIEW"
    assert result["status"] == "needs_review"
    assert result["verification"]["passed"] is False
    assert result["recovery_attempt"] == 2  # stopped by no-progress, well before the budget of 5
    assert "No progress since the previous attempt: yes" in model.prompts("reflect")[1]
    review = json.loads(vfs.read("bvf", "verification/needs_review.json"))
    assert review["reason"] == "no progress after reflection"
    assert "Uncited claim" in review["unresolved"]
    assert review["lessons"]


def test_budget_exhaustion_ends_in_needs_review_without_verification(stores):
    vfs, _ = stores
    drafts = [_uncited(f"Unsupported caching statement number {word} appears here.")
              for word in ("one", "two", "three", "four")]
    model = FakeReportModel(drafts)

    result = _run(stores, model, budget=2)

    assert result["stage"] == "NEEDS_REVIEW"
    assert result["recovery_attempt"] == 2
    assert len(result["verification_history"]) == 3
    assert not any(v["passed"] for v in result["verification_history"])
    review = json.loads(vfs.read("bvf", "verification/needs_review.json"))
    assert review["reason"] == "recovery budget 2 exhausted"


def test_evidence_gap_triggers_targeted_research_then_disclosure(stores):
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
    model = FakeReportModel([GOOD, disclosed])

    result = _run(stores, model, research_model=research)

    assert result["stage"] == "Awaiting Approval", result["verification"]["repair_details"]
    first = result["verification_history"][0]
    assert first["issue_type"] == "EVIDENCE_GAP"
    assert model.prompts("reflect") == []  # evidence gaps are repaired by research, not rewording
    assert len(result["todos"]) == 2
    assert result["todos"][1]["research_question"].startswith("cache eviction policy details")
    assert gap_id("todo-1", "cache eviction policy details") in result["researched_gaps"]
    assert "BVF fix attempt 1" in model.prompts("write")[1]


def test_evidence_selection_narrows_the_build_prompt_but_not_verification(stores):
    """The writer's prompt sees only evidence_top_k sources per question; Verify still reads
    every source that was actually captured, independent of that narrowing."""
    vfs, index = stores
    sources = [
        {"url": f"https://controlled.test/page{n}", "content": f"Cache related detail number {n}"}
        for n in range(8)
    ]
    researcher = ResearchAgent(
        ModelRouter(FakeResearchModel(), mode="fake"),
        lambda: ControlledMCP([sources]),
    )
    report_model = FakeReportModel([GOOD])
    nodes = ResearchNodes(
        vfs, index, researcher,
        report_writer=ReportWriter(ModelRouter(report_model, mode="fake")),
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


def test_standalone_verification_script_reproduces_the_graph_verdict(stores, capsys):
    vfs, _ = stores
    _run(stores, FakeReportModel([GOOD]), run_id="script")

    assert verify_script(["--run-id", "script", "--vfs-root", str(vfs.root)]) == 0
    assert '"passed": true' in capsys.readouterr().out

    vfs.write("script", "reports/latest.md", _uncited("Unsupported caching claim appears right here."), "latest_report")
    assert verify_script(["--run-id", "script", "--vfs-root", str(vfs.root)]) == 1


CRITERIA = ["States how a cache stores data for later reuse."]


def test_acceptance_check_only_runs_after_the_deterministic_script_passes(stores):
    """A structurally broken draft never reaches the (paid) acceptance-criteria LLM call."""
    acceptance = FakeAcceptanceModel([{
        "results": [{"criterion": CRITERIA[0], "met": True, "justification": "n/a"}],
    }])
    model = FakeReportModel([_uncited("Caches also make every workload faster overall."), GOOD])

    result = _run(
        stores, model,
        acceptance_verifier=AcceptanceVerifier(ModelRouter(acceptance, mode="fake")),
        acceptance_criteria=CRITERIA,
    )

    assert result["stage"] == "Awaiting Approval"
    assert len(acceptance.calls) == 1  # only called once the draft finally passes structurally


def test_unmet_acceptance_criterion_fails_verification_and_is_fixed(stores):
    bad_then_good = FakeAcceptanceModel([
        {"results": [{"criterion": CRITERIA[0], "met": False, "justification": "Never states this."}]},
        {"results": [{"criterion": CRITERIA[0], "met": True, "justification": "Now states it."}]},
    ])
    model = FakeReportModel([GOOD, GOOD], reflections=[{
        "diagnosis": "The draft never actually describes cache storage/reuse.",
        "lessons": ["Explicitly state how the cache stores and reuses data."],
        "strategy": "Add one sentence stating the storage/reuse mechanism, cited.",
    }])

    result = _run(
        stores, model,
        acceptance_verifier=AcceptanceVerifier(ModelRouter(bad_then_good, mode="fake")),
        acceptance_criteria=CRITERIA,
    )

    assert result["stage"] == "Awaiting Approval"
    assert [v["passed"] for v in result["verification_history"]] == [False, True]
    first_failure = result["verification_history"][0]
    assert first_failure["issue_type"] == "DRAFT_DEFECT"
    assert "acceptance_criteria" in first_failure["checks"]
    assert "Never states this" in first_failure["repair_details"]
    assert len(bad_then_good.calls) == 2


def test_acceptance_check_is_skipped_without_a_configured_verifier(stores):
    """No acceptance_verifier injected: verification behaves exactly as before this feature."""
    result = _run(stores, FakeReportModel([GOOD]), acceptance_criteria=CRITERIA)

    assert result["stage"] == "Awaiting Approval"
    assert "acceptance_criteria" not in result["verification"]["checks"]


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

    model = FakeReportModel([_uncited("Caches also make every workload faster overall."), GOOD])

    result = _run(stores, model, faithfulness_evaluator=evaluator)

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
    persisted = json.loads(vfs.read("bvf", f"verification/faithfulness.{verification_id}.json"))
    assert persisted["score"] == 0.95
    assert persisted["question"] == BRIEF["research_question"]
    assert persisted["context_count"] == 1


def test_faithfulness_evaluator_failure_is_recorded_but_never_blocks_approval(stores):
    def failing_evaluator(question, report, contexts):
        raise RuntimeError("judge LLM rate limited")

    result = _run(stores, FakeReportModel([GOOD]), faithfulness_evaluator=failing_evaluator)

    assert result["stage"] == "Awaiting Approval"
    assert result["verification"]["passed"] is True
    assert result["faithfulness_score"] is None
    assert "rate limited" in result["faithfulness_error"]
