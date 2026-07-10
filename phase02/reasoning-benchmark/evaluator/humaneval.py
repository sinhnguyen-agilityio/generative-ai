from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

from evaluator.base import Evaluator
from models.evaluation_result import EvaluationResult
from models.reasoning_result import ReasoningResult
from models.task import ReasoningTask


class HumanEvalEvaluator(Evaluator):
    """
    Evaluates generated Python code using the HumanEval test cases.
    """

    def evaluate(
        self,
        task: ReasoningTask,
        reasoning: ReasoningResult,
    ) -> EvaluationResult:
        start_time = time.perf_counter()

        try:
            namespace = self._build_namespace()

            #
            # Execute the generated solution.
            #
            exec(reasoning.generated_code, namespace)

            #
            # Load the HumanEval test.
            #
            exec(task.test, namespace)

            #
            # Execute the HumanEval check().
            #
            self._run_test(
                namespace=namespace,
                entry_point=task.entry_point,
            )

            execution_time = time.perf_counter() - start_time

            return EvaluationResult(
                passed=True,
                execution_time=execution_time,
                error_type=None,
                error_message=None,
            )

        except Exception as ex:

            execution_time = time.perf_counter() - start_time

            return EvaluationResult(
                passed=False,
                execution_time=execution_time,
                error_type=type(ex).__name__,
                error_message=str(ex),
            )

    def _build_namespace(self) -> dict[str, Any]:
        """
        Create a namespace for executing generated code.

        This method exists so that future versions can inject
        helper functions or use a restricted execution environment.
        """
        return {}

    def _run_test(
        self,
        namespace: dict[str, Any],
        entry_point: str,
    ) -> None:
        """
        Execute HumanEval's check(candidate).
        """

        candidate = namespace.get(entry_point)

        if candidate is None:
            raise ValueError(
                f"Entry point '{entry_point}' was not found."
            )

        check = namespace.get("check")

        if check is None:
            raise ValueError(
                "HumanEval test does not define 'check'."
            )

        candidate_fn = cast(Callable[..., Any], candidate)
        check_fn = cast(Callable[[Callable[..., Any]], None], check)

        check_fn(candidate_fn)
