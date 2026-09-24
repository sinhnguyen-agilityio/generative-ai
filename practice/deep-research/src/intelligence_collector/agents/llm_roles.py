from __future__ import annotations

from intelligence_collector.agents.prompts import (FIXER_PROMPT_VERSION, PLANNER_PROMPT_VERSION,
                                                   SUPERVISOR_PROMPT_VERSION, SYNTHESIS_PROMPT_VERSION,
                                                   VERIFIER_PROMPT_VERSION, WORKER_PROMPT_VERSION)
from intelligence_collector.contracts import (DraftAssessment, FixDecision, SupervisorAssessment,
                                              TodoPlan, VerificationAssessment, WorkerSearchPlan)


class SemanticRoles:
    """Explicit LLM paths for every semantic role; graph nodes inject the router."""

    def __init__(self, router): self.router = router

    def planner(self, brief, *, run_id="", thread_id=""):
        return self.router.invoke_structured("planner", brief, TodoPlan, prompt_version=PLANNER_PROMPT_VERSION, run_id=run_id, thread_id=thread_id)

    def supervisor(self, context, *, run_id="", thread_id=""):
        return self.router.invoke_structured("supervisor", context, SupervisorAssessment, prompt_version=SUPERVISOR_PROMPT_VERSION, run_id=run_id, thread_id=thread_id)

    def worker(self, question, *, run_id="", thread_id=""):
        return self.router.invoke_structured("worker", question, WorkerSearchPlan, prompt_version=WORKER_PROMPT_VERSION, run_id=run_id, thread_id=thread_id)

    def synthesizer(self, evidence, *, run_id="", thread_id=""):
        return self.router.invoke_structured("synthesizer", evidence, DraftAssessment, prompt_version=SYNTHESIS_PROMPT_VERSION, run_id=run_id, thread_id=thread_id)

    def verifier(self, draft, *, run_id="", thread_id=""):
        return self.router.invoke_structured("verifier", draft, VerificationAssessment, prompt_version=VERIFIER_PROMPT_VERSION, run_id=run_id, thread_id=thread_id)

    def fixer(self, feedback, *, run_id="", thread_id=""):
        return self.router.invoke_structured("fixer", feedback, FixDecision, prompt_version=FIXER_PROMPT_VERSION, run_id=run_id, thread_id=thread_id)
