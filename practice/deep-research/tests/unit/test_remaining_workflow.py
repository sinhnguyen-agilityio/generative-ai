import pytest

from intelligence_collector.agents.reviewer_checks import effective_requests, reviewer_format_checks
from intelligence_collector.graph.instrumentation import node_wrapper
from intelligence_collector.graph.migration import normalize_checkpoint
from intelligence_collector.graph.operation_journal import OperationJournal
from intelligence_collector.storage.vfs import VFS


def test_word_limit_is_measured_without_modifying_report():
    report = "One two three [S1]\n\n## Sources\n- [S1] Source - https://test"
    assert reviewer_format_checks(report, ["exactly 3 words"])[0].met
    assert not reviewer_format_checks(report, ["at most 2 words"])[0].met
    assert report.startswith("One two three")


def test_latest_simple_word_constraint_supersedes_older_constraint():
    assert effective_requests(["100 words only", "write the report with 50 words only"]) == [
        "write the report with 50 words only"
    ]
    assert effective_requests(["100 words and emphasize risks", "50 words only"]) == [
        "100 words and emphasize risks",
        "50 words only",
    ]


def test_old_checkpoint_defaults_do_not_create_research_or_approval():
    state = normalize_checkpoint({"run_id": "run"})
    assert "todos" not in state and "approval" not in state
    assert state["checkpoint_schema_version"] == 1
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_checkpoint({"checkpoint_schema_version": 2})


def test_completed_synthesis_reused_after_lifecycle_failure(tmp_path):
    journal = OperationJournal(VFS(tmp_path))
    calls = []

    def synthesize(state):
        calls.append(1)
        return {**state, "draft": "Saved report"}

    def lifecycle(name, status):
        if status == "success":
            raise RuntimeError("simulated crash before checkpoint")

    state = {"run_id": "run", "thread_id": "thread"}
    first = node_wrapper("synthesizing", synthesize, lifecycle=lifecycle, operation_journal=journal)
    with pytest.raises(RuntimeError):
        first(state)
    resumed = node_wrapper("synthesizing", synthesize, operation_journal=journal)
    assert resumed(state)["draft"] == "Saved report"
    assert calls == [1]


def test_research_restart_restores_references_without_research(tmp_path):
    from intelligence_collector.agents.worker import run_worker
    from intelligence_collector.contracts import ResearchTodo

    vfs = VFS(tmp_path)
    vfs.write_json(
        "run",
        "research/task/evidence/result.json",
        {"task_id": "task", "sources": [], "indexed_chunks": []},
        "evidence_bundle",
    )
    todo, refs, chunks = run_worker(
        vfs, None, "run", ResearchTodo(todo_id="task", research_question="Question?"), None
    )
    assert todo.status.value == "completed"
    assert refs[0].kind == "evidence_bundle" and refs[0].run_id == "run"
    assert not chunks


def test_report_judge_rejects_incomplete_or_duplicate_checklists():
    from intelligence_collector.agents.report_judge import ReportJudge
    from intelligence_collector.contracts import ReportJudgeVerdict, ResearchBrief

    class Router:
        def invoke_messages(self, *args, **kwargs):
            return ReportJudgeVerdict(
                results=[{"criterion": "Only one", "met": True, "justification": "Okay"}]
            )

    from intelligence_collector.agents.report_judge import CoverageCriterion

    criteria = [
        CoverageCriterion("C1", "Only one", "output"),
        CoverageCriterion("C2", "Omitted", "output"),
    ]
    with pytest.raises(ValueError, match="no verdict for C2"):
        ReportJudge(Router()).assess(
            ResearchBrief(research_question="Research the local cache behavior"),
            criteria,
            "Report",
            "Evidence",
        )


def test_recovery_ui_submits_explicit_action_and_retains_identity(tmp_path, monkeypatch):
    from intelligence_collector.ui import resources, state
    from intelligence_collector.ui.components import recovery

    class Session(dict):
        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    class UI:
        session_state = Session()
        reran = False

        def caption(self, text):
            pass

        def expander(self, label, expanded=False):
            from contextlib import nullcontext

            return nullcontext()

        spinner = expander

        def selectbox(self, label, choices, **kwargs):
            return "revise"

        def text_area(self, *args, **kwargs):
            return "Use a simple paragraph"

        def button(self, label, **kwargs):
            return label == "Apply recovery action"

        def rerun(self):
            self.reran = True

        def error(self, text):
            pytest.fail(text)

    decisions = []
    snapshot = {"run_id": "run", "thread_id": "thread", "stage": "NEEDS_REVIEW"}

    class Service:
        def resume_interrupt(self, thread_id, decision):
            decisions.append((thread_id, decision))
            return {**snapshot, "stage": "Awaiting Approval"}

    ui = UI()
    monkeypatch.setattr(recovery, "st", ui)
    monkeypatch.setattr(state, "st", ui)
    monkeypatch.setattr(resources, "recovery_service", lambda: Service())
    recovery.render_recovery_controls(snapshot)
    assert decisions == [
        (
            "thread",
            {"action": "revise", "feedback": "Use a simple paragraph", "additional_attempts": 0},
        )
    ]
    assert ui.session_state.scope_state["run_id"] == "run"
    assert ui.session_state.scope_thread_id == "thread"
    assert ui.reran
