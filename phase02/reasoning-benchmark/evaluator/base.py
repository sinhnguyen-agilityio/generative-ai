from abc import ABC, abstractmethod

from models.evaluation_result import EvaluationResult
from models.reasoning_result import ReasoningResult
from models.task import ReasoningTask


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        task: ReasoningTask,
        reasoning: ReasoningResult,
    ) -> EvaluationResult:
        raise NotImplementedError
