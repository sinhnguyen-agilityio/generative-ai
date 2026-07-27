from __future__ import annotations

from typing import Sequence

from dataloader.loader import DatasetLoader
from models.reasoning_result import ReasoningResult
from models.benchmark_result import BenchmarkResult
from evaluator.humaneval import HumanEvalEvaluator
from evaluator.base import Evaluator
from reasoning.base import ReasoningStrategy


class BenchmarkRunner:
    """
    Execute one or more reasoning strategies on one or more tasks.
    """

    def __init__(
        self,
        loader: DatasetLoader,
        evaluator: Evaluator,
    ):
        self.loader = loader
        self.evaluator = evaluator

    def run(
        self,
        strategies: list[ReasoningStrategy],
        limit: int = 1,
    ) -> list[BenchmarkResult]:

        tasks = self.loader.load(limit)
        results = []

        for task in tasks:
            for strategy in strategies:
                reasoning = strategy.solve(task)
                evaluation = self.evaluator.evaluate(
                    task,
                    reasoning,
                )

                results.append(
                    BenchmarkResult(
                        task=task,
                        reasoning=reasoning,
                        evaluation=evaluation,
                    )
                )

        return results
