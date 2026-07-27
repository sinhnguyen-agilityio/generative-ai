from dataclasses import dataclass

from .evaluation_result import EvaluationResult
from .reasoning_result import ReasoningResult
from .task import ReasoningTask


@dataclass(slots=True)
class BenchmarkResult:
    """
    Final benchmark record.
    """

    task: ReasoningTask
    reasoning: ReasoningResult
    evaluation: EvaluationResult
