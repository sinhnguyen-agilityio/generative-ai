"""Graph node implementations, one module per workflow phase.

Every node is a ``state -> state update`` callable. Nodes that need collaborators (LLM agents,
VFS, index, MCP) are grouped into small classes so dependencies are injected once and the
wiring in ``build.py`` stays declarative.

* ``common``     node identifiers, workflow limits, the activity-log helper
* ``interrupts`` human-in-the-loop pauses (clarification, plan approval, report approval, recovery)
* ``scope``      clarification, research brief, planning
* ``research``   delegation to research sub-agents, readiness assessment, gap research
* ``report``     build -> verify -> fix, and the ``extend`` evidence lookup
* ``review``     reviewer reply classification and publishing
* ``recovery``   applying a human recovery decision
* ``stubs``      stage-only placeholders
"""

from intelligence_collector.graph.nodes.common import (
    DEFAULT_RECOVERY_BUDGET,
    EXTENSION_PASSAGES,
    FAITHFULNESS_THRESHOLD,
    MAX_STALLED_ATTEMPTS,
    NODE_AGENT,
    Node,
    node_agent,
)
from intelligence_collector.graph.nodes.interrupts import (
    approval,
    await_clarification,
    await_plan_approval,
    await_recovery,
)
from intelligence_collector.graph.nodes.recovery import RecoveryNodes
from intelligence_collector.graph.nodes.report import BVFNodes
from intelligence_collector.graph.nodes.research import ResearchNodes
from intelligence_collector.graph.nodes.review import ApprovalNodes
from intelligence_collector.graph.nodes.scope import PlannerNodes, ScopeNodes
from intelligence_collector.graph.nodes.stubs import stage_only

__all__ = [
    "DEFAULT_RECOVERY_BUDGET",
    "EXTENSION_PASSAGES",
    "FAITHFULNESS_THRESHOLD",
    "MAX_STALLED_ATTEMPTS",
    "NODE_AGENT",
    "Node",
    "node_agent",
    "approval",
    "await_clarification",
    "await_plan_approval",
    "await_recovery",
    "RecoveryNodes",
    "BVFNodes",
    "ResearchNodes",
    "ApprovalNodes",
    "PlannerNodes",
    "ScopeNodes",
    "stage_only",
]
