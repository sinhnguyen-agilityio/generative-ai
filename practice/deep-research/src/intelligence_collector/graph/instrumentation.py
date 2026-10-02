"""Cross-cutting behaviour applied uniformly to every graph node.

Nodes stay pure ``state -> state update`` functions; this module layers on
lifecycle notifications (UI progress) and tracing. State persistence between turns is
LangGraph's own checkpointer (``SqliteSaver`` in the app), not this wrapper.
"""

from collections.abc import Callable
import json

from langgraph.errors import GraphInterrupt

from intelligence_collector.contracts import Stage, WorkflowStatus
from intelligence_collector.graph.state import GraphState

# State fields exported to Langfuse as trace scores: {state key: score name}.
SCORE_FIELDS = {
    "faithfulness_score": "faithfulness",
    "context_precision_score": "context_precision",
    "response_relevancy_score": "response_relevancy",
}

NodeFn = Callable[[GraphState], GraphState]
Lifecycle = Callable[[str, str], None]


def _to_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _trace_input(state: GraphState) -> str:
    return _to_json({
        "stage": state.get("stage"),
        "request": state.get("request", ""),
        "messages": [
            {
                "role": getattr(message, "type", "message"),
                "content": getattr(message, "content", str(message)),
            }
            for message in state.get("messages", [])
        ],
        "scope_assessment": state.get("scope_assessment"),
        "research_brief": state.get("scoped_request"),
    })


def _trace_output(result: GraphState) -> str:
    return _to_json({
        "stage": result.get("stage"),
        "status": result.get("status"),
        "scope_assessment": result.get("scope_assessment"),
        "research_brief": result.get("scoped_request"),
        "todos": result.get("todos"),
        # Present only on the quality node, so its results are readable in the trace itself.
        **{
            key: result[key]
            for key in (
                "faithfulness_score", "faithfulness_error",
                "context_precision_score", "context_precision_error", "context_precision_details",
                "response_relevancy_score", "response_relevancy_error",
            )
            if result.get(key) not in (None, "")
        },
    })


def node_wrapper(
    name: str,
    fn: NodeFn,
    tracer=None,
    lifecycle: Lifecycle | None = None,
    run_registry=None,
) -> NodeFn:
    """Wrap a graph node with UI lifecycle notifications and a Langfuse span.

    `lifecycle(node, status)` is called with "running", then "success" or "failure"; a
    ``GraphInterrupt`` (the human-review pause) is re-raised without being marked failed. With a
    `tracer`, the node runs inside a ``graph.<name>`` span and any score fields in its result
    (see ``SCORE_FIELDS``) are attached to the trace. Raises ``TypeError`` if the node returns
    something other than a mapping.
    """
    notify = lifecycle or (lambda _node, _status: None)

    def register(state: GraphState) -> None:
        if run_registry is None:
            return
        run_registry.register_run(
            state.get("run_id", ""),
            state.get("thread_id", ""),
            request_summary=state.get("request", ""),
        )

    def update(state: GraphState, status: WorkflowStatus) -> None:
        if run_registry is None:
            return
        run_registry.update_run(
            state.get("run_id", ""),
            state.get("thread_id", ""),
            workflow_status=status,
            current_stage=state.get("stage", ""),
        )

    def result_status(result: GraphState) -> WorkflowStatus:
        stage = result.get("stage")
        if stage == Stage.PUBLISHED.value:
            return WorkflowStatus.PUBLISHED
        if stage == Stage.NEEDS_REVIEW.value:
            return WorkflowStatus.WAITING_FOR_RECOVERY
        if stage == Stage.AWAITING_APPROVAL.value:
            return WorkflowStatus.WAITING_FOR_APPROVAL
        assessment = result.get("scope_assessment") or {}
        if stage == Stage.SCOPING.value and assessment.get("need_clarification"):
            return WorkflowStatus.WAITING_FOR_CLARIFICATION
        return WorkflowStatus.RUNNING

    def interrupt_status() -> WorkflowStatus:
        if name == "approval":
            return WorkflowStatus.WAITING_FOR_APPROVAL
        if name == "await_recovery":
            return WorkflowStatus.WAITING_FOR_RECOVERY
        return WorkflowStatus.WAITING_FOR_CLARIFICATION

    def run(state: GraphState) -> GraphState:
        """Run the node, inside a Langfuse span when tracing is on."""
        if tracer is None:
            return fn(state)
        with tracer.span(
            f"graph.{name}",
            state.get("run_id", ""),
            state.get("thread_id", ""),
            input=_trace_input(state),
            node=name,
        ) as span:
            result = fn(state)
            span.output = _trace_output(result)
            for key, score_name in SCORE_FIELDS.items():
                if result.get(key) is not None:
                    tracer.score(score_name, float(result[key]))
            return result

    def wrapped(state: GraphState) -> GraphState:
        """Report lifecycle status around the node and check that it returns a mapping."""
        register(state)
        notify(name, "running")
        try:
            result = run(dict(state))
        except GraphInterrupt:
            # An intentional pause for a human decision (see graph/nodes.py's ApprovalNodes),
            # not a failure — GraphInterrupt is a real Exception subclass (so a bare `except
            # Exception` below would catch it too), and LangGraph's own resume mechanism
            # handles it; don't mark this node's lifecycle activity as failed.
            update(state, interrupt_status())
            raise
        except Exception:
            update(state, WorkflowStatus.FAILED)
            notify(name, "failure")
            raise
        notify(name, "success")

        if not isinstance(result, dict):
            raise TypeError(f"{name} must return a mapping state update")
        update(result, result_status(result))
        return result

    wrapped.__name__ = name
    return wrapped
