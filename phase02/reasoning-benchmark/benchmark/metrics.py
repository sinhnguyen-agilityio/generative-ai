from __future__ import annotations

from collections import defaultdict

from models.benchmark_metric import BenchmarkMetrics
from models.benchmark_result import BenchmarkResult


class MetricsCalculator:
    """
    Calculate aggregate benchmark metrics for each reasoning strategy.
    """

    def calculate(
        self,
        results: list[BenchmarkResult],
    ) -> list[BenchmarkMetrics]:

        grouped: dict[str, list[BenchmarkResult]] = defaultdict(list)

        #
        # Group results by strategy.
        #
        for result in results:
            grouped[result.reasoning.strategy].append(result)

        metrics: list[BenchmarkMetrics] = []

        #
        # Calculate metrics for each strategy.
        #
        for strategy, strategy_results in grouped.items():
            metrics.append(
                self._calculate_strategy(
                    strategy,
                    strategy_results,
                )
            )

        return metrics

    def _calculate_strategy(
        self,
        strategy: str,
        results: list[BenchmarkResult],
    ) -> BenchmarkMetrics:
        total_tasks = len(results)
        passed_tasks = sum(
            result.evaluation.passed
            for result in results
        )
        failed_tasks = total_tasks - passed_tasks
        pass_rate = (
            passed_tasks / total_tasks
            if total_tasks
            else 0.0
        )

        avg_input_tokens = (
            sum(
                r.reasoning.input_tokens
                for r in results
            )
            / total_tasks
        )

        avg_output_tokens = (
            sum(
                r.reasoning.output_tokens
                for r in results
            )
            / total_tasks
        )

        avg_latency = (
            sum(
                r.reasoning.latency
                for r in results
            )
            / total_tasks
        )

        total_tokens = sum(
            r.reasoning.total_tokens
            for r in results
        )

        tokens_per_pass = (
            total_tokens / passed_tasks
            if passed_tasks
            else 0.0
        )

        estimated_cost = 0.0

        error_counts: dict[str, int] = {}

        for result in results:
            error = result.evaluation.error_type

            if error is None:
                continue

            error_counts[error] = (
                error_counts.get(error, 0)
                + 1
            )

        return BenchmarkMetrics(
            strategy=strategy,
            total_tasks=total_tasks,
            passed_tasks=passed_tasks,
            failed_tasks=failed_tasks,
            pass_rate=pass_rate,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            avg_total_tokens=avg_input_tokens + avg_output_tokens,
            avg_latency=avg_latency,
            estimated_cost=estimated_cost,
            tokens_per_pass=tokens_per_pass,
            error_counts=error_counts,
        )
