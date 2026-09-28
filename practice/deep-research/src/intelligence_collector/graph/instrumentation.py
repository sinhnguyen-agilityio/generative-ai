"""Cross-cutting behaviour applied uniformly to every graph node.

Nodes stay pure ``state -> state update`` functions; this module layers on
lifecycle notifications (UI progress), checkpoint load/save, and tracing.
"""

from collections.abc import Callable
import json

from langgraph.errors import GraphInterrupt

from intelligence_collector.graph.state import GraphState

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
    })


def node_wrapper(
    name: str,
    fn: NodeFn,
    tracer=None,
    checkpoint_store=None,
    lifecycle: Lifecycle | None = None,
) -> NodeFn:
    """Normalize node execution so every graph node returns a serializable state update."""
    notify = lifecycle or (lambda _node, _status: None)

    def run(state: GraphState) -> GraphState:
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
            return result

    def wrapped(state: GraphState) -> GraphState:
        notify(name, "running")
        current = dict(state)
        thread_id = current.get("thread_id", "")
        if checkpoint_store is not None and thread_id:
            persisted = checkpoint_store.load(thread_id) or {}
            current = {**persisted, **current}

        try:
            result = run(current)
        except GraphInterrupt:
            # An intentional pause for a human decision (see graph/nodes.py's ApprovalNodes),
            # not a failure — GraphInterrupt is a real Exception subclass (so a bare `except
            # Exception` below would catch it too), and LangGraph's own resume mechanism
            # handles it; don't mark this node's lifecycle activity as failed.
            raise
        except Exception:
            notify(name, "failure")
            raise
        notify(name, "success")

        if not isinstance(result, dict):
            raise TypeError(f"{name} must return a mapping state update")
        if checkpoint_store is not None:
            result_thread_id = result.get("thread_id", thread_id)
            if result_thread_id:
                checkpoint_store.save(result_thread_id, result)
        return result

    wrapped.__name__ = name
    return wrapped
