"""Langfuse trace topology of real graph runs, read back from the exported OpenTelemetry spans.

The SDK, the LangChain callback, the masking and the context propagation are the production
ones; only the exporter keeps spans in memory (``tests/fixtures/langfuse_capture.py``).
"""

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from intelligence_collector.agents.feedback_router import FeedbackRouter
from intelligence_collector.agents.report_judge import ReportJudge
from intelligence_collector.agents.report_writer import ReportWriter
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.config import LangfuseSettings
from intelligence_collector.contracts import ResearchTodo
from intelligence_collector.graph.build import build_graph
from intelligence_collector.graph.nodes import ResearchNodes
from intelligence_collector.graph.recovery import WorkflowRecoveryService
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.observability.tracing import RunTracer, traced
from intelligence_collector.storage.run_registry import RunRegistry
from intelligence_collector.tools.mcp import ControlledMCP
from tests.fixtures.chat_model import FakeChatModel
from tests.fixtures.fake_models import (
    FakeAcceptanceModel,
    FakeReportModel,
    FakeResearchModel,
    cited_report,
)
from tests.fixtures.langfuse_capture import LangfuseCapture
from tests.integration.test_approval_gate import BRIEF, GOOD, PASS, URL

RUN, THREAD = "run-topology", "thread-topology"
QUESTIONS = [
    "How do caches store data?",
    "How do caches evict entries?",
    "How do caches stay consistent?",
]
BULLETED = GOOD.replace("A cache stores", "- A cache stores")


class Stack:
    """The fake models and MCP clients of one graph, with call counters."""

    def __init__(
        self,
        *,
        drafts=(GOOD,),
        feedback=({"route": "approve", "reason": "ok"},),
        failing_query: str = "",
    ):
        self.researcher = FakeChatModel(model_name="fake-researcher", delegate=FakeResearchModel())
        self.writer_fake = FakeReportModel(list(drafts))
        self.writer = FakeChatModel(model_name="fake-writer", delegate=self.writer_fake)
        self.judge = FakeChatModel(model_name="fake-judge", delegate=FakeAcceptanceModel([PASS]))
        self.router = FakeChatModel(model_name="fake-router", scripted=list(feedback))
        self.mcps: list[ControlledMCP] = []
        self.failing_query = failing_query

    def mcp(self):
        stack = self

        class MCP(ControlledMCP):
            def search(self, query, **kwargs):
                if stack.failing_query and stack.failing_query in query:
                    raise ConnectionError("Tavily MCP unavailable")
                return super().search(query, **kwargs)

        client = MCP([{"url": URL, "content": "A cache stores reusable data."}])
        self.mcps.append(client)
        return client

    @property
    def model_calls(self) -> int:
        return sum(model.calls for model in (self.researcher, self.writer, self.judge, self.router))

    @property
    def mcp_calls(self) -> int:
        return sum(client.calls for client in self.mcps)

    def graph(self, tmp_path: Path, tracer: RunTracer, *, run_registry=None):
        from intelligence_collector.storage.vector_index import LocalVectorIndex
        from intelligence_collector.storage.vfs import VFS

        vfs = VFS(tmp_path / "vfs")
        nodes = ResearchNodes(
            vfs,
            LocalVectorIndex(tmp_path / "index.json"),
            ResearchAgent(ModelRouter(self.researcher), self.mcp),
            report_writer=ReportWriter(ModelRouter(self.writer)),
            report_judge=ReportJudge(ModelRouter(self.judge)),
            recovery_budget=3,
            max_concurrent_researchers=3,
        )
        return build_graph(
            research_nodes=nodes,
            vfs=vfs,
            checkpointer=MemorySaver(),
            with_approval=True,
            output_dir=str(tmp_path / "exports"),
            feedback_router=FeedbackRouter(ModelRouter(self.router)),
            tracer=tracer,
            run_registry=run_registry,
        )


def _state():
    todos = [
        ResearchTodo(todo_id=f"todo-{n}", research_question=question).model_dump(mode="json")
        for n, question in enumerate(QUESTIONS)
    ]
    return {"run_id": RUN, "thread_id": THREAD, "scoped_request": BRIEF, "todos": todos}


def _start(graph, tracer):
    with tracer.execution(
        run_id=RUN, thread_id=THREAD, kind="start", input={"request": "cache behaviour"}
    ) as execution:
        result = graph.invoke(
            _state(), execution.graph_config({"configurable": {"thread_id": THREAD}})
        )
        execution.finish(result)
    return result


def test_one_invocation_is_one_trace_with_nodes_workers_tools_and_generations(tmp_path):
    capture = LangfuseCapture()
    stack = Stack()
    result = _start(stack.graph(tmp_path, capture.tracer), capture.tracer)
    spans = capture.spans()
    tree = capture.render()

    assert result["stage"] == "Awaiting Approval"
    # One root: the execution. Everything shares the run's deterministic trace ID and session.
    ids = {span.span_id for span in spans}
    roots = [span for span in spans if span.parent_id not in ids]
    assert [root.name for root in roots] == ["research.execution"], tree
    assert {span.trace_id for span in spans} == {capture.tracer.trace_id_for(RUN)}
    assert {span.session_id for span in spans} == {THREAD}
    execution = roots[0]
    assert execution.metadata["execution_kind"] == "start"
    assert execution.output == {
        "stage": "Awaiting Approval",
        "draft_version": 1,
        "verification_outcome": "PASS",
        "paused_at": "publication_approval",
    }
    # research.execution -> research_graph -> node
    graph_run = capture.named("research_graph")[0]
    assert graph_run.parent_id == execution.span_id
    for node in ("researching", "research_complete", "synthesizing", "verifying", "approval"):
        assert capture.named(node)[0].parent_id == graph_run.span_id, tree
    assert capture.named("verifying")[0].metadata["draft_version"] == 1
    # node -> supervisor decisions, and worker tasks (one per todo, run concurrently) -> tool
    # call and generations
    researching = capture.named("researching")[0]
    decisions = capture.named("model.supervisor")
    # think -> ConductResearch x3 (one parallel batch) -> think -> ResearchComplete
    assert len(decisions) == 4
    assert {d.parent_id for d in decisions} == {researching.span_id}, tree
    workers = capture.named("research_agent")
    assert sorted(w.metadata["todo_id"] for w in workers) == ["todo-0", "todo-1", "todo-2"]
    assert {w.parent_id for w in workers} == {researching.span_id}
    worker_of = {w.span_id: w.metadata["todo_id"] for w in workers}
    for search in capture.named("mcp.search"):
        assert search.type == "tool"
        owner = next(
            s
            for s in spans
            if s.name == "research_agent" and s.span_id in _ancestor_ids(capture, search)
        )
        assert worker_of[owner.span_id] == search.metadata["todo_id"]
        assert search.output == {"results": 1, "urls": [URL]}
    persists = capture.named("research.persist")
    assert {p.parent_id for p in persists} == {researching.span_id}
    # LLM calls are generations with model name and usage, under the node/worker that made them.
    generations = [span for span in spans if span.type == "generation"]
    assert len(generations) == stack.model_calls
    synthesis = next(g for g in generations if "synthesizing" in capture.ancestors(g))
    assert capture.ancestors(synthesis)[:2] == ["model.synthesizer", "synthesizing"]
    assert all(
        "research_agent" in capture.ancestors(g)
        for g in generations
        if "model.researcher" in capture.ancestors(g)
    )


def _ancestor_ids(capture, span):
    ids, current = set(), capture.parent(span)
    while current is not None:
        ids.add(current.span_id)
        current = capture.parent(current)
    return ids


def test_tool_error_is_recorded_on_its_own_observation_and_siblings_survive(tmp_path):
    capture = LangfuseCapture()
    stack = Stack(failing_query="evict")
    _start(stack.graph(tmp_path, capture.tracer), capture.tracer)

    searches = capture.named("mcp.search")
    failed = [s for s in searches if s.level == "ERROR"]
    assert [s.metadata["todo_id"] for s in failed] == ["todo-1"]
    assert "ConnectionError" in failed[0].status_message
    assert sorted(s.metadata["todo_id"] for s in searches if s.level != "ERROR") == [
        "todo-0",
        "todo-2",
    ]
    # The failed tool call is a task outcome: the workers, persistence and graph completed.
    assert len(capture.named("research.persist")) == 3
    assert capture.named("research.execution")[0].level != "ERROR"


def test_node_failure_marks_the_node_and_execution_and_keeps_completed_nodes(tmp_path):
    capture = LangfuseCapture()
    stack = Stack()

    class BrokenWriter:
        def invoke(self, messages):
            raise RuntimeError("writer exploded")

    stack.writer.delegate = BrokenWriter()
    graph = stack.graph(tmp_path, capture.tracer)
    with pytest.raises(ValueError, match="writer exploded"):
        _start(graph, capture.tracer)

    # Each failed attempt is an ERROR on its own model run, under the failing node.
    attempts = capture.named("model.synthesizer")
    assert [a.level for a in attempts] == ["ERROR", "ERROR"]
    assert [a.metadata["attempt"] for a in attempts] == [1, 2]
    assert capture.named("synthesizing")[0].level == "ERROR"
    assert capture.named("research.execution")[0].level == "ERROR"
    assert capture.named("researching")[0].level != "ERROR"
    assert len(capture.named("research_agent")) == 3


def test_resumes_after_pauses_join_the_run_trace_even_in_a_new_process(tmp_path):
    first = LangfuseCapture()
    stack = Stack(
        drafts=(GOOD, BULLETED),
        feedback=({"route": "revise", "reason": "layout"}, {"route": "approve", "reason": "ok"}),
    )
    registry = RunRegistry(tmp_path / "registry.sqlite")
    graph = stack.graph(tmp_path, first.tracer, run_registry=registry)
    _start(graph, first.tracer)

    # The report feedback is answered after a restart: a new tracer (and Langfuse client) with
    # nothing in memory from the first execution.
    second = LangfuseCapture()
    service = WorkflowRecoveryService(graph, registry, second.tracer)
    revised = service.resume_interrupt(
        THREAD, {"reply": "Please use bullet points.", "actor": "reviewer"}
    )
    assert revised["stage"] == "Awaiting Approval" and revised["draft"] == BULLETED
    published = service.resume_interrupt(THREAD, {"approved": True, "actor": "reviewer"})

    assert published["stage"] == "Published"
    executions = first.named("research.execution") + second.named("research.execution")
    assert [
        (e.metadata["execution_kind"], e.metadata.get("interrupt_kind")) for e in executions
    ] == [("start", None), ("resume", "publication_approval"), ("resume", "publication_approval")]
    assert {e.trace_id for e in executions} == {first.tracer.trace_id_for(RUN)}
    assert {e.session_id for e in executions} == {THREAD}
    # Each execution is closed when its invocation returns; none stays open over a pause.
    assert executions[0].output["paused_at"] == "publication_approval"
    assert executions[1].output["paused_at"] == "publication_approval"
    assert "paused_at" not in executions[2].output
    # The resumed work is under its own execution, in the same trace.
    revision = second.named("synthesizing")[0]
    assert "research.execution" in second.ancestors(revision)
    assert second.named("classify_feedback")[0].metadata["run_id"] == RUN


def test_plan_approval_resume_runs_research_in_a_new_execution_of_the_same_trace(stores, tmp_path):
    from intelligence_collector.agents.scope import ScopeAgent
    from tests.integration.test_full_graph_workflow import (
        REPORT,
        REQUEST,
        SequencedModel,
        _scope_responses,
    )

    vfs, index = stores
    capture = LangfuseCapture()
    registry = RunRegistry(tmp_path / "registry.sqlite")
    scope_model = FakeChatModel(
        model_name="fake-scope", delegate=SequencedModel(_scope_responses())
    )
    nodes = ResearchNodes(
        vfs,
        index,
        ResearchAgent(
            ModelRouter(FakeChatModel(delegate=FakeResearchModel())),
            lambda: ControlledMCP([{"url": URL, "content": "A cache stores data."}]),
        ),
        report_writer=ReportWriter(ModelRouter(FakeChatModel(delegate=FakeReportModel([REPORT])))),
        report_judge=ReportJudge(
            ModelRouter(
                FakeChatModel(
                    delegate=FakeAcceptanceModel(
                        [
                            {
                                "results": [
                                    {"criterion": "all", "met": True, "justification": "Answered."}
                                ]
                            }
                        ]
                    )
                )
            )
        ),
        faithfulness_evaluator=lambda q, r, c: {"score": 0.96, "unsupported": []},
        require_faithfulness=True,
    )
    graph = build_graph(
        scope_agent=ScopeAgent(ModelRouter(scope_model)),
        vfs=vfs,
        research_nodes=nodes,
        checkpointer=MemorySaver(),
        with_approval=True,
        tracer=capture.tracer,
        run_registry=registry,
    )
    with capture.tracer.execution(run_id=RUN, thread_id=THREAD, kind="start") as execution:
        paused = graph.invoke(
            {"run_id": RUN, "thread_id": THREAD, "request": REQUEST},
            execution.graph_config({"configurable": {"thread_id": THREAD}}),
        )
        execution.finish(paused)
    WorkflowRecoveryService(graph, registry, capture.tracer).resume_interrupt(
        THREAD, {"reply": "approve", "actor": "reviewer"}
    )

    first, second = capture.named("research.execution")
    assert first.output["paused_at"] == "plan_approval"
    assert (second.metadata["execution_kind"], second.metadata["interrupt_kind"]) == (
        "resume",
        "plan_approval",
    )
    assert first.trace_id == second.trace_id
    # Nothing was retrieved before the approval; research belongs to the resumed execution.
    research = capture.named("researching")[0]
    assert capture.ancestors(research)[-1] == "research.execution"
    assert capture.parent(capture.parent(research)).span_id == second.span_id
    for name in ("clarify_with_user", "write_research_brief", "planning"):
        node = capture.named(name)[0]
        assert capture.parent(capture.parent(node)).span_id == first.span_id
    evaluator = capture.named("ragas.faithfulness")[0]
    assert evaluator.type == "evaluator" and evaluator.output == {"score": 0.96, "unsupported": 0}
    assert capture.ancestors(evaluator)[0] == "verifying"
    assert [(score["name"], score["value"], score["trace_id"]) for score in capture.scores] == [
        ("faithfulness", 0.96, first.trace_id)
    ]


def test_instrumentation_never_repeats_model_or_tool_calls(tmp_path):
    untraced = Stack()
    _start(untraced.graph(tmp_path / "off", RunTracer()), RunTracer())
    capture = LangfuseCapture()
    traced_stack = Stack()
    _start(traced_stack.graph(tmp_path / "on", capture.tracer), capture.tracer)

    assert traced_stack.model_calls == untraced.model_calls
    assert traced_stack.mcp_calls == untraced.mcp_calls == 3
    generations = [span for span in capture.spans() if span.type == "generation"]
    # One generation per LLM call: no second, hand-made model observation.
    assert len(generations) == traced_stack.model_calls
    assert not [span for span in capture.spans() if span.name.startswith("model.validation")]


def test_disabled_or_misconfigured_langfuse_does_not_change_the_workflow(tmp_path, monkeypatch):
    monkeypatch.setattr("langfuse.Langfuse.auth_check", lambda self: False)
    misconfigured = RunTracer.from_settings(
        LangfuseSettings(
            enabled=True, public_key="pk-wrong-project", secret_key="sk", host="http://127.0.0.1:9"
        )
    )
    assert not misconfigured.enabled
    assert "LANGFUSE_BASE_URL" in misconfigured.disabled_reason
    monkeypatch.setenv("INTELLIGENCE_LANGFUSE_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    missing_keys = RunTracer.from_env()
    assert not missing_keys.enabled and "credentials" in missing_keys.disabled_reason

    for name, tracer in (
        ("misconfigured", misconfigured),
        ("missing", missing_keys),
        ("off", RunTracer()),
    ):
        result = _start(Stack().graph(tmp_path / name, tracer), tracer)
        assert result["stage"] == "Awaiting Approval"
        assert result["verification"]["passed"] is True


def test_exported_payloads_are_redacted_and_bounded():
    capture = LangfuseCapture()
    with capture.tracer.execution(run_id=RUN, thread_id=THREAD, kind="start"):
        with traced(
            "operation.fetch",
            input={
                "api_key": "sk-live-secret",
                "url": "https://source.test/page?tavilyApiKey=tvly-secret&q=1",
                "excerpt": "x" * 10_000,
            },
        ):
            pass
    span = capture.named("operation.fetch")[0]
    exported = repr(span.input)
    assert "sk-live-secret" not in exported and "tvly-secret" not in exported
    assert "https://source.test/page?tavilyApiKey=[REDACTED]&q=1" in exported
    assert len(span.input["excerpt"]) < 4100 and "truncated" in span.input["excerpt"]


def test_traced_is_inert_outside_an_execution():
    with traced("mcp.search", as_type="tool") as observation:
        observation.output = {"results": 0}
    assert observation.output == {"results": 0}


def test_long_report_rewrite_keeps_one_execution_per_invocation(tmp_path):
    """A revision cycle (verify FAIL -> fix -> rewrite) stays inside one execution."""
    capture = LangfuseCapture()
    stack = Stack(drafts=(GOOD, GOOD))
    stack.judge.delegate = FakeAcceptanceModel(
        [
            {
                "results": [
                    {
                        "criterion": "all criteria",
                        "met": False,
                        "justification": "A claim lost its citation.",
                    }
                ]
            },
            PASS,
        ]
    )
    _start(stack.graph(tmp_path, capture.tracer), capture.tracer)

    assert len(capture.named("research.execution")) == 1
    verifications = capture.named("verifying")
    assert [v.metadata.get("draft_version") for v in verifications] == [1, 2]
    assert capture.named("fixing")[0].metadata["improvement_cycle"] == 1


def test_cited_report_fixture_matches_the_single_source():
    assert cited_report(["x [S1]."], {"S1": URL}).endswith(URL)
