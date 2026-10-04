"""Human review of a verified report: reply classification and publishing."""

from __future__ import annotations

import hashlib

from langchain_core.messages import AIMessage, HumanMessage

from intelligence_collector.contracts import (
    ApprovalDecision,
    FeedbackDecision,
    Stage,
    VerificationResult,
)
from intelligence_collector.graph.nodes.common import (
    DEFAULT_RECOVERY_BUDGET,
    _activity,
)
from intelligence_collector.graph.state import GraphState


class ApprovalNodes:
    """The human-in-the-loop review, reached once a report has fully passed verification (and
    faithfulness scoring):

    * ``approval`` interrupts the graph until the reviewer answers the verified report. The normal
      answer is a free-text chat reply (``{"reply": ...}``), always handed to
      ``classify_feedback``. An explicit ``{"approved": True}`` goes straight to ``publishing``;
      an explicit ``{"approved": False, "feedback": ...}`` is classified but can never be read as
      an approval.
    * ``classify_feedback`` asks an LLM (``feedback_router``) what the reply means:
      - ``approve``: the reply accepts the report as-is -> ``publishing``;
      - ``revise``: change the report using the same evidence -> ``synthesizing`` rewrites it
        with the reviewer's request, and it goes through Verify again before coming back here;
      - ``extend``: add a section within the same question -> ``locate_extension_evidence``
        uses stored evidence when it is enough and researches only what is missing, then the
        report is rewritten with the section and verified again;
      - ``scope``: the change needs different evidence -> ``clarify_with_user``, with the
        feedback as a new user turn, so the scope agent re-assesses it (it may ask a clarifying
        question first), rewrites the research brief, and the planner and researchers run again.
      Keeping the LLM call in its own node means a failed classification is retried on its
      own, never re-prompting the reviewer for a decision already given.
    * ``publishing`` writes the approved, verified report to the permanent output location.

    Approvals always bind to the hash of the exact verified report shown. The reviewer's replies
    and the agent's acknowledgements are appended to ``messages`` so the review reads as part of
    the chat conversation.
    """

    def __init__(self, vfs, output_dir: str, feedback_router=None):
        self.vfs = vfs
        self.output_dir = output_dir
        self.feedback_router = feedback_router

    @staticmethod
    def _approval_for(state: GraphState, approved: bool) -> dict:
        """The reviewer decides on the verified report shown, so the approval binds to it."""
        from intelligence_collector.publication import approval_for

        verification = VerificationResult.model_validate(state["verification"])
        return approval_for(
            hashlib.sha256(state.get("draft", "").encode()).hexdigest(),
            verification.verification_id,
            approved,
            state.get("reviewer") or "local-user",
        ).model_dump(mode="json")

    def approval(self, state: GraphState) -> GraphState:
        """Human review: pause (``interrupt``) for the reviewer's reply. An explicit approval goes
        to ``publishing``; a chat reply or a rejection goes to ``classify_feedback``.
        """
        from langgraph.types import interrupt

        decision = interrupt({"kind": "publication_approval", "run_id": state.get("run_id")})
        from intelligence_collector.contracts import ApprovalResumeDecision

        decision = ApprovalResumeDecision.model_validate(decision).model_dump(mode="json")
        state = {**state, "reviewer": str(decision.get("actor") or "local-user")}
        if decision.get("approved") is True:
            return {
                **state,
                "approval": self._approval_for(state, True),
                "current_action": "Reviewer approved — publishing",
                "activity": _activity(state, "Reviewer approved the report"),
            }
        explicit_reject = decision.get("approved") is False
        return {
            **state,
            # A chat reply is undecided until classify_feedback reads it.
            "approval": self._approval_for(state, False) if explicit_reject else None,
            "human_feedback": str(decision.get("reply") or decision.get("feedback") or "").strip(),
            "feedback_route": "",
            "current_action": "Reading the reviewer's reply",
            "activity": _activity(state, "Reviewer replied to the report"),
        }

    def publishing(self, state: GraphState) -> GraphState:
        """Write the approved report, hash-bound to the approval and the passing verification."""
        from intelligence_collector.publication import publish

        run_id = state["run_id"]
        destination = f"{self.output_dir.rstrip('/')}/{run_id}.md"
        record = publish(
            state.get("draft", ""),
            destination,
            ApprovalDecision.model_validate(state["approval"]),
            VerificationResult.model_validate(state["verification"]),
        )
        ref = self.vfs.write_json(
            run_id,
            "publication/record.json",
            record.model_dump(mode="json"),
            "publication_record",
        )
        outcome = (
            f"Published to {destination}"
            if record.published
            else f"Publication failed: {record.error}"
        )
        return {
            **state,
            "publication": record.model_dump(mode="json"),
            "stage": (Stage.PUBLISHED if record.published else Stage.NEEDS_REVIEW).value,
            "approval": state.get("approval") if record.published else None,
            "verification": state.get("verification") if record.published else None,
            "last_error": "" if record.published else outcome,
            "verification_issue": "" if record.published else "publication_failure",
            "current_action": outcome,
            "messages": [*state.get("messages", []), AIMessage(content=outcome)],
            "activity": _activity(
                state, "Publishing report", "success" if record.published else "failure"
            ),
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }

    def classify_feedback(self, state: GraphState) -> GraphState:
        """LLM step: read the reviewer's chat reply as an approval, a revision request, an
        added section or a scope change."""
        feedback = state.get("human_feedback", "")
        if self.feedback_router is not None and feedback:
            verdict = self.feedback_router.route(
                feedback,
                state.get("draft", ""),
                run_id=state.get("run_id", ""),
                thread_id=state.get("thread_id", ""),
            )
        else:
            # No classifier configured, or no reply text: default to a revision rather than
            # publishing unasked or discarding the whole research plan.
            verdict = FeedbackDecision(
                route="revise", reason="No reply to classify; defaulting to revise."
            )
        explicit_reject = (state.get("approval") or {}).get("approved") is False
        explicit_accept = feedback.strip().casefold().rstrip(".! ") in {
            "approve",
            "approved",
            "i approve",
            "i agree",
            "đồng ý",
            "phê duyệt",
            "looks good, publish it",
        }
        if verdict.route == "approve" and (explicit_reject or not explicit_accept):
            verdict = FeedbackDecision(
                route="revise", reason="Publication requires an explicit, unconditional approval."
            )

        history = [
            *state.get("bvf_history", []),
            {
                "attempt": state.get("recovery_attempt", 0),
                "kind": "HUMAN_FEEDBACK",
                "route": verdict.route,
                "feedback": feedback,
                "reason": verdict.reason,
            },
        ]
        update = {
            "feedback_route": verdict.route,
            "current_action": f"Reviewer feedback routed to {verdict.route}: {verdict.reason or feedback[:120]}",
            "activity": _activity(state, f"Reviewer feedback routed to {verdict.route}"),
            "bvf_history": history,
            "recovery_budget": state.get("recovery_budget", DEFAULT_RECOVERY_BUDGET),
        }
        # clarify_with_user records the reply itself on the scope path.
        replied = (
            [*state.get("messages", []), HumanMessage(content=feedback)]
            if feedback
            else list(state.get("messages", []))
        )
        if verdict.route == "approve":
            return {
                **state,
                **update,
                "messages": replied,
                "human_feedback": "",
                "approval": self._approval_for(state, True),
            }
        if verdict.route == "revise":
            # synthesizing consumes human_feedback and rewrites the reviewed report; the request
            # joins reviewer_requests so later BVF fixes and the judge keep honouring it. A human
            # revision starts a fresh Build-Verify-Fix cycle, so the attempt/stall counters restart.
            requests = (
                [*state.get("reviewer_requests", []), feedback]
                if feedback
                else list(state.get("reviewer_requests", []))
            )
            return {
                **state,
                **update,
                "messages": [
                    *replied,
                    AIMessage(
                        content="Revising the report with your feedback; it will be verified again."
                    ),
                ],
                "human_feedback": feedback or "(no specific request)",
                "reviewer_requests": requests,
                "verification": None,
                "approval": None,
                "publication": None,
                "recovery_attempt": 0,
                "stall_count": 0,
                "progress_fingerprint": "",
                "stage": Stage.SYNTHESIZING.value,
            }
        if verdict.route == "extend":
            # Keep the report, the brief and the plan; locate_extension_evidence looks for the
            # new section's evidence in the run's index and researches only what is missing.
            # The request joins reviewer_requests, so the coverage gate checks the new section.
            requests = [*state.get("reviewer_requests", []), feedback]
            return {
                **state,
                **update,
                "messages": [
                    *replied,
                    AIMessage(
                        content=(
                            "Adding that to the report: checking the evidence already collected first, "
                            "then researching only what is missing. The updated report will be "
                            "verified again."
                        )
                    ),
                ],
                "human_feedback": feedback,
                "reviewer_requests": requests,
                "extension": {"topic": (verdict.topic or feedback).strip(), "request": feedback},
                "verification": None,
                "approval": None,
                "publication": None,
                "recovery_attempt": 0,
                "stall_count": 0,
                "progress_fingerprint": "",
                "stage": Stage.RESEARCHING.value,
            }
        # Scope change: the feedback becomes the next user turn of the scope conversation. The
        # new brief starts a fresh research + Build-Verify-Fix cycle, so earlier reviewer requests
        # and the attempt/stall counters restart.
        return {
            **state,
            **update,
            "request": feedback,
            "human_feedback": "",
            "reviewer_requests": [],
            "verification": None,
            "approval": None,
            "publication": None,
            "recovery_attempt": 0,
            "stall_count": 0,
            "progress_fingerprint": "",
            "stage": Stage.SCOPING.value,
        }
