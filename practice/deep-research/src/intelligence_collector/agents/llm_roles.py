from __future__ import annotations

import json

from intelligence_collector.agents.prompts import (
    FIXER_PROMPT_VERSION,
    PLANNER_PROMPT_VERSION,
    ROLE_TEMPLATES,
    SUPERVISOR_PROMPT_VERSION,
    SYNTHESIS_PROMPT_VERSION,
    VERIFIER_PROMPT_VERSION,
    WORKER_PROMPT_VERSION,
)
from intelligence_collector.contracts import (DraftAssessment, FixDecision, SupervisorAssessment,
                                              TodoPlan, VerificationAssessment, WorkerSearchPlan)


class SemanticRoles:
    """Explicit LLM paths for every semantic role; graph nodes inject the router."""

    def __init__(self, router): self.router = router

    def _invoke(self, role, value, schema, version, key, *, run_id="", thread_id=""):
        messages = ROLE_TEMPLATES[role].format_messages(**{key: value if isinstance(value, str) else json.dumps(value, default=str)})
        return self.router.invoke_messages(
            role,
            messages,
            schema,
            prompt_version=version,
            run_id=run_id,
            thread_id=thread_id,
        )

    def planner(self, brief, *, run_id="", thread_id=""):
        return self._invoke("planner", brief, TodoPlan, PLANNER_PROMPT_VERSION, "brief", run_id=run_id, thread_id=thread_id)

    def supervisor(self, context, *, run_id="", thread_id=""):
        return self._invoke("supervisor", context, SupervisorAssessment, SUPERVISOR_PROMPT_VERSION, "context", run_id=run_id, thread_id=thread_id)

    def worker(self, question, *, run_id="", thread_id=""):
        return self._invoke("worker", question, WorkerSearchPlan, WORKER_PROMPT_VERSION, "question", run_id=run_id, thread_id=thread_id)

    def synthesizer(self, evidence, *, run_id="", thread_id=""):
        return self._invoke("synthesizer", evidence, DraftAssessment, SYNTHESIS_PROMPT_VERSION, "evidence", run_id=run_id, thread_id=thread_id)

    def verifier(self, draft, *, run_id="", thread_id=""):
        return self._invoke("verifier", draft, VerificationAssessment, VERIFIER_PROMPT_VERSION, "draft", run_id=run_id, thread_id=thread_id)

    def fixer(self, feedback, *, run_id="", thread_id=""):
        return self._invoke("fixer", feedback, FixDecision, FIXER_PROMPT_VERSION, "feedback", run_id=run_id, thread_id=thread_id)
