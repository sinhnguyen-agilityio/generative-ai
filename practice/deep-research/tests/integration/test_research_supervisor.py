"""The research supervisor is an LLM: it thinks, delegates approved tasks (in parallel), follows up
on gaps, completes or proposes a replan — while code keeps it inside the approved plan."""

import json

from langchain_core.messages import AIMessage

from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.supervisor import ReplanProposal, ResearchSupervisor
from intelligence_collector.contracts import ResearchTodo, TodoStatus
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.tools.mcp import ControlledMCP
from tests.fixtures.fake_models import ScriptedSupervisorModel

COMPLETE = ("ResearchComplete", {})


def think(text="Plan the next step."):
    return ("think_tool", {"reflection": text})


def conduct(todo_id, topic="Find primary documentation."):
    return ("ConductResearch", {"todo_id": todo_id, "research_topic": topic})


class RoundAwareModel(ScriptedSupervisorModel):
    """Compression answers differently for a follow-up round, so merged findings are visible."""

    def invoke(self, messages, config=None):
        if "compress raw web search results" in str(messages[0].content):
            follow_up = "Research only this missing evidence" in str(messages[-1].content)
            summary = (
                "Round 2: release date [https://t.test/2]"
                if follow_up
                else ("Round 1: overview [https://t.test/1]")
            )
            return AIMessage(
                content=json.dumps({"summary": summary, "sufficient": True, "unresolved_gaps": []})
            )
        return super().invoke(messages, config)


def _todos():
    return [
        ResearchTodo(todo_id="t1", research_question="What is LangGraph?"),
        ResearchTodo(todo_id="t2", research_question="How does LangGraph persist state?"),
    ]


def _supervisor(stores, model, *, empty_after=None, **kwargs):
    """Supervisor over sub-agents whose MCP clients each return one new page; with
    `empty_after`, clients created after that many return nothing."""
    vfs, index = stores
    clients = []

    def factory():
        n = len(clients)
        rows = (
            []
            if empty_after is not None and n >= empty_after
            else [{"url": f"https://t.test/{n}", "content": f"Evidence {n}."}]
        )
        clients.append(ControlledMCP([rows]))
        return clients[-1]

    researcher = ResearchAgent(ModelRouter(model), factory)
    return ResearchSupervisor(researcher, vfs, index, **kwargs), clients


def _system_prompt(model, turn=0) -> str:
    return str(model.supervisor_prompts[turn][0].content)


def test_supervisor_thinks_delegates_one_parallel_batch_then_completes(stores):
    model = ScriptedSupervisorModel(
        [
            lambda tasks: [think("Two independent tasks: research both at once.")],
            lambda tasks: [conduct(todo_id, question) for todo_id, question in tasks],
            lambda tasks: [think("Both tasks are sufficient.")],
            lambda tasks: [COMPLETE],
        ]
    )
    supervisor, clients = _supervisor(stores, model)

    results = supervisor.delegate("run", _todos(), brief="Explain LangGraph.")

    assert [todo.status for todo, _, _ in results] == [TodoStatus.COMPLETED] * 2
    assert results.replan is None
    assert len(model.supervisor_prompts) == 4 and len(clients) == 2
    first_input = str(model.supervisor_prompts[0][-1].content)
    assert "Research brief: Explain LangGraph." in first_input
    assert "- todo_id: t1\n  question: What is LangGraph?" in first_input
    replies = model.tool_replies()
    assert replies[0] == "Reflection recorded: Two independent tasks: research both at once."
    assert replies[1].startswith("Task t1 (round 1): completed\nSufficient: yes\nSources: 1")
    assert replies[2].startswith("Task t2 (round 1): completed")
    assert "Findings:\nFindings for What is LangGraph?" in replies[1]
    # Without a planner the supervisor is never offered a replan.
    assert "ProposeReplan" not in _system_prompt(model)


def test_follow_up_researches_only_the_gap_and_appends_its_findings(stores):
    vfs, _ = stores
    model = RoundAwareModel(
        [
            lambda tasks: [conduct("t1"), conduct("t2")],
            lambda tasks: [conduct("t1", "Find the first stable release date.")],
            lambda tasks: [COMPLETE],
        ]
    )
    supervisor, clients = _supervisor(stores, model)

    results = supervisor.delegate("run", _todos())

    (t1, t1_refs, _), (t2, _, _) = results
    assert (t1.research_attempt, t1.status) == (2, TodoStatus.COMPLETED)
    assert t1.research_instruction == "Find the first stable release date."
    assert t2.research_attempt == 1 and len(clients) == 3
    bundle = json.loads(vfs.read("run", "research/t1/attempts/2/evidence/result.json"))
    assert bundle["summary"] == (
        "Round 1: overview [https://t.test/1]\n\nRound 2: release date [https://t.test/2]"
    )
    evidence = [ref for ref in t1_refs if ref.kind == "evidence_bundle"]
    assert len(evidence) == 2  # both rounds' bundles stay in the run's evidence
    assert model.tool_replies()[-1].startswith("Task t1 (round 2): completed")


def test_code_keeps_the_supervisor_inside_the_approved_plan(stores):
    model = ScriptedSupervisorModel(
        [
            lambda tasks: [conduct("t9"), conduct("t1", "a"), conduct("t1", "b"), COMPLETE],
            lambda tasks: [conduct("t1", "again")],
            lambda tasks: [conduct("t1", "a third time")],
            lambda tasks: [conduct("t2")],
            lambda tasks: [COMPLETE],
        ]
    )
    supervisor, _ = _supervisor(stores, model)

    results = supervisor.delegate("run", _todos())

    replies = model.tool_replies()
    assert replies[0] == "Refused: unknown todo_id 't9'. Use one of: t1, t2."
    assert replies[1].startswith("Task t1 (round 1): completed")
    assert replies[2] == "Refused: task t1 is already in this batch."
    assert replies[3] == (
        "Not complete: every task must be researched at least once. Still unresearched: t2"
    )
    assert replies[4].startswith("Task t1 (round 2): completed")
    assert replies[5] == "Refused: task t1 already had 2 research rounds."
    assert replies[6].startswith("Task t2 (round 1): completed")
    assert [todo.status for todo, _, _ in results] == [TodoStatus.COMPLETED] * 2


def test_turn_limit_never_skips_an_approved_task(stores):
    # The model never delegates: first it answers without tools, then it only reflects.
    model = ScriptedSupervisorModel([lambda tasks: [], lambda tasks: [think()]])
    supervisor, clients = _supervisor(stores, model, max_turns=3)

    results = supervisor.delegate("run", _todos())

    assert len(model.supervisor_prompts) == 3
    reminder = str(model.supervisor_prompts[1][-1].content)
    assert reminder.startswith("Not complete: every task must be researched at least once.")
    assert [todo.status for todo, _, _ in results] == [TodoStatus.COMPLETED] * 2
    assert len(clients) == 2  # code delegated both tasks after the last turn


def test_failed_follow_up_keeps_the_earlier_evidence(stores):
    vfs, _ = stores
    model = ScriptedSupervisorModel(
        [
            lambda tasks: [conduct("t1"), conduct("t2")],
            lambda tasks: [conduct("t1", "Find benchmarks.")],
            lambda tasks: [COMPLETE],
        ]
    )
    supervisor, _ = _supervisor(stores, model, empty_after=2)

    (t1, _, _), _ = supervisor.delegate("run", _todos())

    assert model.tool_replies()[-1].startswith("Task t1 (round 2): failed")
    assert (t1.research_attempt, t1.status) == (1, TodoStatus.COMPLETED)
    assert json.loads(vfs.read("run", "research/t1/evidence/result.json"))["sources"]


def test_replan_is_offered_only_when_enabled_and_needs_known_tasks(stores):
    script = [
        lambda tasks: [conduct("t1"), conduct("t2")],
        lambda tasks: [("ProposeReplan", {"reason": "Bad premise", "todo_ids": ["t9"]})],
        lambda tasks: [
            ("ProposeReplan", {"reason": "t2 asks about a removed API", "todo_ids": ["t2"]})
        ],
    ]
    model = ScriptedSupervisorModel(script)
    supervisor, _ = _supervisor(stores, model)
    supervisor.replan_enabled = True

    results = supervisor.delegate("run", _todos(), allow_replan=True)

    assert "ProposeReplan" in _system_prompt(model)
    assert model.tool_replies()[-1] == "Refused: a replan needs a reason and known todo_ids."
    assert results.replan == ReplanProposal(reason="t2 asks about a removed API", todo_ids=["t2"])

    vfs, index = stores
    disabled = ScriptedSupervisorModel([script[2], lambda tasks: [COMPLETE]])
    supervisor, _ = _supervisor((vfs, index), disabled)  # replan_enabled stays False
    results = supervisor.delegate("other-run", _todos(), allow_replan=True)

    assert "ProposeReplan" not in _system_prompt(disabled)
    assert disabled.tool_replies()[0] == "Unknown tool 'ProposeReplan'."
    assert results.replan is None
    assert [todo.status for todo, _, _ in results] == [TodoStatus.COMPLETED] * 2


def test_graph_routes_a_supervisor_replan_to_human_approval_of_a_replacement_plan(stores):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from intelligence_collector.agents.report_judge import ReportJudge
    from intelligence_collector.agents.report_writer import ReportWriter
    from intelligence_collector.agents.scope import ScopeAgent
    from intelligence_collector.graph.build import build_graph
    from intelligence_collector.graph.nodes import ResearchNodes
    from tests.fixtures.fake_models import FakeAcceptanceModel, FakeReportModel
    from tests.integration.test_full_graph_workflow import (
        REPORT,
        REQUEST,
        URL,
        SequencedModel,
        _scope_responses,
    )

    vfs, index = stores
    proposed = []

    def propose_once(tasks):
        if proposed:
            return [COMPLETE]
        proposed.append(True)
        reason = "The question assumes a feature that was removed."
        return [("ProposeReplan", {"reason": reason, "todo_ids": [tasks[0][0]]})]

    research_model = ScriptedSupervisorModel(
        [lambda tasks: [conduct(todo_id, q) for todo_id, q in tasks], propose_once]
    )
    replacement = {
        "questions": ["Which cache storage mechanisms does the current release document?"],
        "evidence_needed": ["current cache documentation"],
        "report_shape": "A short direct answer, then sources.",
    }
    nodes = ResearchNodes(
        vfs,
        index,
        ResearchAgent(
            ModelRouter(research_model),
            lambda: ControlledMCP([{"url": URL, "content": "A cache stores reusable data."}]),
        ),
        report_writer=ReportWriter(ModelRouter(FakeReportModel([REPORT]))),
        report_judge=ReportJudge(
            ModelRouter(
                FakeAcceptanceModel(
                    [{"results": [{"criterion": "all", "met": True, "justification": "Ok."}]}]
                )
            )
        ),
        faithfulness_evaluator=lambda q, r, c: {"score": 0.96, "unsupported": []},
        require_faithfulness=True,
    )
    graph = build_graph(
        scope_agent=ScopeAgent(ModelRouter(SequencedModel(_scope_responses() + [replacement]))),
        vfs=vfs,
        research_nodes=nodes,
        checkpointer=MemorySaver(),
        with_approval=True,
    )
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"run_id": "r", "thread_id": "t", "request": REQUEST}, config)

    paused = graph.invoke(Command(resume={"reply": "approve", "actor": "tester"}), config)

    interrupts = [i.value for task in graph.get_state(config).tasks for i in task.interrupts]
    assert [i["kind"] for i in interrupts] == ["plan_approval"]
    assert paused["plan"]["kind"] == "replacement" and paused["research_replan"] is None
    original, added = paused["todos"]
    assert original["status"] == "completed"  # its evidence is kept
    assert added["research_question"] == replacement["questions"][0]
    assert added["status"] == "planned"
    assert "ProposeReplan" in str(research_model.supervisor_prompts[0][0].content)

    # The replacement plan is approved: its task is researched, and the supervisor can no
    # longer propose another replan.
    done = graph.invoke(Command(resume={"reply": "approve", "actor": "tester"}), config)

    assert done["todos"][1]["status"] == "completed"
    assert "ProposeReplan" not in str(research_model.supervisor_prompts[-1][0].content)
