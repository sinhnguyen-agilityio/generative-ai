from abc import ABC, abstractmethod

from models.benchmark_metric import BenchmarkMetrics


class Reporter(ABC):

    @abstractmethod
    def report(
        self,
        metrics: list[BenchmarkMetrics],
    ) -> None:
        raise NotImplementedError


class ConsoleReporter(Reporter):

    def report(
        self,
        metrics: list[BenchmarkMetrics],
    ) -> None:

        print()
        print("=" * 80)
        print("REASONING BENCHMARK REPORT")
        print("=" * 80)

        for metric in metrics:

            self._print_summary(metric)

            self._print_errors(metric)

    def _print_summary(
        self,
        metric: BenchmarkMetrics,
    ) -> None:
        print()
        print(f"Strategy : {metric.strategy}")
        print("-" * 60)
        print(f"Tasks               : {metric.total_tasks}")
        print(f"Passed             : {metric.passed_tasks}")
        print(f"Failed             : {metric.failed_tasks}")
        print(f"Accuracy           : {metric.pass_rate:.2%}")
        print()
        print(f"Avg Input Tokens   : {metric.avg_input_tokens:.1f}")
        print(f"Avg Output Tokens  : {metric.avg_output_tokens:.1f}")
        print(f"Avg Total Tokens   : {metric.avg_total_tokens:.1f}")
        print()
        print(f"Avg Latency (sec)  : {metric.avg_latency:.2f}")
        print()
        print(f"Tokens / Pass      : {metric.tokens_per_pass:.1f}")

    def _print_errors(
        self,
        metric: BenchmarkMetrics,
    ) -> None:

        print()
        print("Errors")
        print("-" * 20)

        if not metric.error_counts:
            print("None")
            return

        for error, count in metric.error_counts.items():
            print(f"{error:<20}{count}")
