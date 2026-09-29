from __future__ import annotations

import json

from intelligence_collector.agents.prompts import ROLE_TEMPLATES
from intelligence_collector.contracts import (DraftAssessment, FixDecision, SupervisorAssessment,
                                              TodoPlan, VerificationAssessment, WorkerSearchPlan)


class SemanticRoles:
    """Explicit LLM paths for every semantic role; graph nodes inject the router."""

    def __init__(self, router): self.router = router

    def _invoke(self, role, value, schema, key, *, run_id="", thread_id=""):
        messages = ROLE_TEMPLATES[role].format_messages(**{key: value if isinstance(value, str) else json.dumps(value, default=str)})
        return self.router.invoke_messages(
            role,
            messages,
            schema,
            run_id=run_id,
            thread_id=thread_id,
        )

    def planner(self, brief, *, run_id="", thread_id=""):
        return self._invoke("planner", brief, TodoPlan, "brief", run_id=run_id, thread_id=thread_id)

    def supervisor(self, context, *, run_id="", thread_id=""):
        return self._invoke("supervisor", context, SupervisorAssessment, "context", run_id=run_id, thread_id=thread_id)

    def worker(self, question, *, run_id="", thread_id=""):
        return self._invoke("worker", question, WorkerSearchPlan, "question", run_id=run_id, thread_id=thread_id)

    def synthesizer(self, evidence, *, run_id="", thread_id=""):
        return self._invoke("synthesizer", evidence, DraftAssessment, "evidence", run_id=run_id, thread_id=thread_id)

    def verifier(self, draft, *, run_id="", thread_id=""):
        return self._invoke("verifier", draft, VerificationAssessment, "draft", run_id=run_id, thread_id=thread_id)

    def fixer(self, feedback, *, run_id="", thread_id=""):
        return self._invoke("fixer", feedback, FixDecision, "feedback", run_id=run_id, thread_id=thread_id)
