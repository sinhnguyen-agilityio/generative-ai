from collections.abc import Callable
import json

from intelligence_collector.graph.state import GraphState


def node_wrapper(
    name: str,
    fn: Callable[[GraphState], GraphState],
    tracer=None,
    checkpoint_store=None,
    lifecycle=None,
):
    """Normalize node execution so every graph node returns a serializable state update."""
    def wrapped(state: GraphState) -> GraphState:
        if lifecycle is not None:
            lifecycle(name, "running")
        current = dict(state)
        thread_id = current.get("thread_id", "")
        if checkpoint_store is not None and thread_id:
            persisted = checkpoint_store.load(thread_id) or {}
            current = {**persisted, **current}
        try:
            if tracer is None:
                result = fn(current)
            else:
                with tracer.span(
                    f"graph.{name}",
                    current.get("run_id", ""),
                    current.get("thread_id", ""),
                    input=json.dumps({
                        "stage": current.get("stage"),
                        "request": current.get("request", ""),
                        "messages": [
                            {"role": getattr(message, "type", "message"), "content": getattr(
                                message, "content", str(message))}
                            for message in current.get("messages", [])
                        ],
                        "scope_assessment": current.get("scope_assessment"),
                        "research_brief": current.get("scoped_request"),
                    }, ensure_ascii=False, indent=2, default=str),
                    node=name,
                ) as span:
                    result = fn(current)
                    span.output = json.dumps({
                        "stage": result.get("stage"),
                        "status": result.get("status"),
                        "scope_assessment": result.get("scope_assessment"),
                        "research_brief": result.get("scoped_request"),
                        "todos": result.get("todos"),
                    }, ensure_ascii=False, indent=2, default=str)
        except Exception:
            if lifecycle is not None:
                lifecycle(name, "failure")
            raise
        if lifecycle is not None:
            lifecycle(name, "success")
        if not isinstance(result, dict):
            raise TypeError(f"{name} must return a mapping state update")
        if checkpoint_store is not None:
            result_thread_id = result.get("thread_id", thread_id)
            if result_thread_id:
                checkpoint_store.save(result_thread_id, result)
        return result
    wrapped.__name__ = name
    return wrapped


def build_graph(
    *,
    checkpointer=None,
    checkpoint_store=None,
    with_approval: bool = False,
    scope_agent=None,
    research_nodes=None,
    vfs=None,
    tracer=None,
    lifecycle=None,
):
    """Build the typed MVP graph with optional resumable approval interrupt."""
    from langgraph.graph import END, START, StateGraph

    # Node names are kept in one place so the workflow reads like a diagram and
    # integrations (UI progress, tracing, and tests) share stable identifiers.
    CLARIFY = "clarify_with_user"
    WRITE_BRIEF = "write_research_brief"
    PLAN = "planning"
    RESEARCH = "researching"
    ASSESS_RESEARCH = "research_complete"
    SYNTHESIZE = "synthesizing"
    VERIFY = "verifying"
    APPROVE = "approval"

    graph = StateGraph(GraphState)

    def add_node(name, fn):
        graph.add_node(
            name,
            node_wrapper(name, fn, tracer, checkpoint_store, lifecycle),
        )

    def stage(name):
        def advance(state: GraphState) -> GraphState:
            return {**state, "stage": name}
        return advance

    def approval(state: GraphState) -> GraphState:
        from langgraph.types import interrupt
        decision = interrupt(
            {"kind": "publication_approval", "run_id": state.get("run_id")})
        return {**state, "approval": decision, "stage": "Published" if decision.get("approved") else "Awaiting Approval"}

    # ===== SCOPE AND PLANNING =====

    if scope_agent is not None:
        from intelligence_collector.agents.planner import plan
        from intelligence_collector.contracts import ResearchBrief, ScopeAssessment
        from langchain_core.messages import AIMessage, HumanMessage

        def clarify(state: GraphState) -> GraphState:
            request = state.get("request", "")
            history = list(state.get("messages", []))
            assessment = scope_agent.clarify_with_user(
                request,
                history,
                run_id=state.get("run_id", ""),
                thread_id=state.get("thread_id", ""),
            )
            messages = history + [HumanMessage(content=request)]
            if assessment.need_clarification:
                messages.append(AIMessage(content=assessment.question))
            else:
                messages.append(AIMessage(content=assessment.verification))
            return {
                **state,
                "stage": "Scoping",
                "messages": messages,
                "scope_assessment": assessment.model_dump(mode="json"),
            }

        def write_brief(state: GraphState) -> GraphState:
            assessment = ScopeAssessment.model_validate(
                state["scope_assessment"])
            brief = scope_agent.write_research_brief(
                state.get("request", ""),
                assessment,
                list(state.get("messages", [])),
                run_id=state.get("run_id", ""),
                thread_id=state.get("thread_id", ""),
                vfs=vfs,
            )
            return {
                **state,
                "stage": "Planning",
                "scoped_request": brief.model_dump(mode="json"),
            }

        def planning(state: GraphState) -> GraphState:
            if vfs is None:
                raise RuntimeError("Scope Planning requires an injected VFS")
            brief = ResearchBrief.model_validate(state["scoped_request"])
            todos = plan(vfs, state["run_id"], [brief.research_question])
            return {
                **state,
                "stage": "Planning",
                "todos": [todo.model_dump(mode="json") for todo in todos],
            }

        def scope_route(state: GraphState):
            needs_clarification = state.get("scope_assessment", {}).get(
                "need_clarification", True
            )
            return END if needs_clarification else WRITE_BRIEF

        add_node(CLARIFY, clarify)
        add_node(WRITE_BRIEF, write_brief)
        add_node(PLAN, planning)

        graph.add_edge(START, CLARIFY)
        graph.add_conditional_edges(
            CLARIFY,
            scope_route,
            {WRITE_BRIEF: WRITE_BRIEF, END: END},
        )
        graph.add_edge(WRITE_BRIEF, PLAN)
    else:
        # Lightweight stage-only graph used by the approval interrupt contract.
        fallback_nodes = ["scope", PLAN]
        if research_nodes is None:
            fallback_nodes.extend((RESEARCH, SYNTHESIZE, VERIFY))
        for name in fallback_nodes:
            add_node(name, stage(name))

        graph.add_edge(START, "scope")
        graph.add_edge("scope", PLAN)

    # ===== RESEARCH PIPELINE =====

    if research_nodes is not None:
        add_node(RESEARCH, research_nodes.researching)
        add_node(ASSESS_RESEARCH, research_nodes.research_complete)
        add_node(SYNTHESIZE, research_nodes.synthesizing)
        add_node(VERIFY, research_nodes.verifying)

        graph.add_edge(PLAN, RESEARCH)
        graph.add_edge(RESEARCH, ASSESS_RESEARCH)
        graph.add_edge(ASSESS_RESEARCH, SYNTHESIZE)
        graph.add_edge(SYNTHESIZE, VERIFY)
    elif scope_agent is None:
        graph.add_edge(PLAN, RESEARCH)
        graph.add_edge(RESEARCH, SYNTHESIZE)
        graph.add_edge(SYNTHESIZE, VERIFY)
    else:
        # Scope-only mode intentionally stops once a durable plan exists.
        graph.add_edge(PLAN, END)

    # ===== APPROVAL AND COMPLETION =====

    if scope_agent is not None and research_nodes is None:
        return graph.compile(checkpointer=checkpointer)

    if with_approval:
        add_node(APPROVE, approval)
        if research_nodes is None:
            graph.add_edge(VERIFY, APPROVE)
        else:
            graph.add_conditional_edges(
                VERIFY,
                lambda state: (
                    APPROVE
                    if state.get("verification", {}).get("passed")
                    else END
                ),
                {APPROVE: APPROVE, END: END},
            )
        graph.add_edge(APPROVE, END)
    else:
        graph.add_edge(VERIFY, END)

    return graph.compile(checkpointer=checkpointer)
