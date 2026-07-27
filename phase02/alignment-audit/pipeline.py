from __future__ import annotations

from typing import Iterator

from agent import LangChainAgent
from alignment import AlignmentChecker, AlignmentResult


class AlignmentAuditPipeline:
    """
    Orchestrates the Alignment Audit workflow.

    Pipeline:

        User Prompt
            ↓
        LangChain Agent
            ↓
        Reasoning Trace
            ↓
        Alignment Checker
            ↓
        PASS / FAIL
    """

    def __init__(
        self,
        agent: LangChainAgent,
        checker: AlignmentChecker,
    ) -> None:

        self.agent = agent
        self.checker = checker

    def run(
        self,
        prompt: str,
    ) -> None:
        print("=" * 60)
        print("Alignment Audit")
        print("=" * 60)

        reasoning_trace = []

        #
        # Step 1
        # Execute the agent normally and collect reasoning events.
        #
        for event in self.agent.execute(prompt):
            print(event)
            reasoning_trace.append(event)

        #
        # Step 2
        # Audit the agent reasoning directly.
        #
        print("\n")
        print("=" * 60)
        print("Auditing the agent reasoning trace...")
        print("=" * 60)

        print("\n".join(str(item) for item in reasoning_trace))

        result = self.checker.check(
            goal=prompt,
            reasoning="\n".join(str(item) for item in reasoning_trace),
        )

        self._print_result(result)

        #
        # Step 3
        # Halt execution if necessary.
        #
        if not result.aligned:
            print("\n🛑 Execution halted.")
            return

        print("\n✅ Execution allowed.")

    def _print_result(
        self,
        result: AlignmentResult,
    ) -> None:

        print("\nAlignment Result")
        print("-" * 60)

        print(f"Aligned   : {result.aligned}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reason    : {result.reason}")
