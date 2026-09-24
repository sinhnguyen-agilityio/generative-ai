from collections.abc import Callable
import json

from intelligence_collector.graph.state import GraphState


def node_wrapper(
    name: str,
    fn: Callable[[GraphState], GraphState],
    tracer=None,
    checkpoint_store=None,
):
    """Normalize node execution so every graph node returns a serializable state update."""
    def wrapped(state: GraphState) -> GraphState:
        current = dict(state)
        thread_id = current.get("thread_id", "")
        if checkpoint_store is not None and thread_id:
            persisted = checkpoint_store.load(thread_id) or {}
            current = {**persisted, **current}
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
    vfs=None,
    tracer=None,
):
    """Build the typed MVP graph with optional resumable approval interrupt."""
    from langgraph.graph import END, START, StateGraph

    def stage(name):
        def advance(state: GraphState) -> GraphState:
            return {**state, "stage": name}
        return node_wrapper(name, advance)

    def approval(state: GraphState) -> GraphState:
        from langgraph.types import interrupt
        decision = interrupt(
            {"kind": "publication_approval", "run_id": state.get("run_id")})
        return {**state, "approval": decision, "stage": "Published" if decision.get("approved") else "Awaiting Approval"}

    graph = StateGraph(GraphState)
    if scope_agent is not None:
        from intelligence_collector.agents.planner import plan
        from intelligence_collector.contracts import ResearchBrief, ScopeDecision, ScopeAssessment
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
            if assessment.decision == ScopeDecision.NEEDS_CLARIFICATION:
                messages.append(
                    AIMessage(content=assessment.clarification_question or ""))
            elif assessment.decision == ScopeDecision.SCOPE_CLEAR:
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
            decision = state.get("scope_assessment", {}).get("decision")
            return "write_research_brief" if decision == ScopeDecision.SCOPE_CLEAR.value else "__end__"

        graph.add_node("clarify_with_user", node_wrapper(
            "clarify_with_user", clarify, tracer, checkpoint_store))
        graph.add_node("write_research_brief", node_wrapper(
            "write_research_brief", write_brief, tracer, checkpoint_store))
        graph.add_node("planning", node_wrapper(
            "planning", planning, tracer, checkpoint_store))
        graph.add_edge(START, "clarify_with_user")
        graph.add_conditional_edges("clarify_with_user", scope_route, {
            "write_research_brief": "write_research_brief", "__end__": END,
        })
        graph.add_edge("write_research_brief", "planning")
        graph.add_edge("planning", END)
    else:
        graph.add_edge(START, "scope")
    stage_names = ("scope", "researching", "synthesizing", "verifying")
    if scope_agent is None:
        stage_names = ("scope", "planning", "researching",
                       "synthesizing", "verifying")
    for name in stage_names:
        graph.add_node(name, node_wrapper(
            name, stage(name), tracer, checkpoint_store))
    if scope_agent is None:
        graph.add_edge("scope", "planning")
    if scope_agent is None:
        graph.add_edge("planning", "researching")
    graph.add_edge("researching", "synthesizing")
    graph.add_edge("synthesizing", "verifying")
    if with_approval:
        graph.add_node("approval", node_wrapper(
            "approval", approval, tracer, checkpoint_store))
        graph.add_edge("verifying", "approval")
        graph.add_edge("approval", END)
    else:
        graph.add_edge("verifying", END)
    return graph.compile(checkpointer=checkpointer)
