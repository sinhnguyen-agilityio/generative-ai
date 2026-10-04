"""Cross-cutting behaviour applied uniformly to every graph node.

Nodes stay pure ``state -> state update`` functions; this module layers on lifecycle
notifications (UI progress), the run registry, the synthesis operation journal and trace
annotations. The node's Langfuse observation itself is opened by the LangChain callback that
``RunTracer.execution`` passes in the graph config (see ``observability/tracing.py``); this
wrapper only adds low-cardinality metadata and scores to it, so each node appears once.
State persistence between turns is LangGraph's own checkpointer, not this wrapper.
"""

from collections.abc import Callable

from langgraph.errors import GraphInterrupt

from intelligence_collector.contracts import Stage, WorkflowStatus
from intelligence_collector.graph.state import GraphState
from intelligence_collector.graph.migration import normalize_checkpoint
from intelligence_collector.observability.tracing import annotate_current

# State fields exported to Langfuse as trace scores: {state key: (score name, producing node)}.
# Only the node that computes a score exports it; later nodes carry the field forward unchanged.
SCORE_FIELDS = {
    "faithfulness_score": ("faithfulness", "verifying"),
    "context_precision_score": ("context_precision", "evaluating_quality"),
    "response_relevancy_score": ("response_relevancy", "evaluating_quality"),
}

NodeFn = Callable[[GraphState], GraphState]
Lifecycle = Callable[[str, str], None]


def _annotate(name: str, result: GraphState) -> None:
    """Correlation fields of the node run: low-cardinality identifiers only, never payloads."""
    annotate_current(
        node=name,
        run_id=result.get("run_id"),
        stage=result.get("stage"),
        draft_version=result.get("draft_version") or None,
        improvement_cycle=result.get("recovery_attempt") or None,
        plan_version=result.get("plan_version") or None,
        verification_outcome=result.get("verification_outcome"),
    )


def node_wrapper(
    name: str,
    fn: NodeFn,
    tracer=None,
    lifecycle: Lifecycle | None = None,
    run_registry=None,
    operation_journal=None,
) -> NodeFn:
    """Wrap a graph node with UI lifecycle notifications, run-registry updates and trace
    annotations.

    `lifecycle(node, status)` is called with "running", then "success" or "failure"; a
    ``GraphInterrupt`` (the human-review pause) reports "waiting" and is re-raised without being
    marked failed. With a
    `tracer`, the node's observation gets correlation metadata and any score fields in its
    result (see ``SCORE_FIELDS``) are attached to the trace. Raises ``TypeError`` if the node
    returns something other than a mapping.
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
        if result.get("status") in {"cancelled", "exported_incomplete", "incomplete"}:
            return WorkflowStatus(result["status"])
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
        if name == "await_plan_approval":
            return WorkflowStatus.WAITING_FOR_PLAN_APPROVAL
        return WorkflowStatus.WAITING_FOR_CLARIFICATION

    def run(state: GraphState) -> GraphState:
        """Run the node; with a `tracer`, annotate its observation and export score fields."""
        result = fn(state)
        if tracer is not None and isinstance(result, dict):
            _annotate(name, result)
            for key, (score_name, producer) in SCORE_FIELDS.items():
                if name == producer and result.get(key) is not None:
                    tracer.score(score_name, float(result[key]))
        return result

    def wrapped(state: GraphState) -> GraphState:
        """Report lifecycle status around the node and check that it returns a mapping."""
        state = normalize_checkpoint(state)
        register(state)
        notify(name, "running")
        try:
            journal = operation_journal if name == "synthesizing" else None
            result = journal.read(name, state) if journal is not None else None
            if result is None:
                result = run(dict(state))
                if not isinstance(result, dict):
                    raise TypeError(f"{name} must return a mapping state update")
                if journal is not None:
                    journal.write(name, state, result)
        except GraphInterrupt:
            # An intentional pause for a human decision (see graph/nodes.py's ApprovalNodes),
            # not a failure — GraphInterrupt is a real Exception subclass (so a bare `except
            # Exception` below would catch it too), and LangGraph's own resume mechanism
            # handles it; don't mark this node's lifecycle activity as failed. It still needs a
            # terminal lifecycle event, or the UI keeps showing the node as running.
            update(state, interrupt_status())
            notify(name, "waiting")
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
