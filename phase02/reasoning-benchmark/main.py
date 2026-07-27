from benchmark.runner import BenchmarkRunner
from dataloader.humaneval import HumanEvalLoader
from evaluator.humaneval import HumanEvalEvaluator
from llm import LLMClient
from reasoning.cod import ChainOfDraft
from reasoning.cot import ChainOfThought
from benchmark.metrics import MetricsCalculator
from benchmark.reporter import ConsoleReporter


def main():

    loader = HumanEvalLoader()
    evaluator = HumanEvalEvaluator()
    llm = LLMClient()
    strategies = [
        ChainOfThought(llm),
        ChainOfDraft(llm),
    ]

    runner = BenchmarkRunner(
        loader=loader,
        evaluator=evaluator,
    )
    results = runner.run(
        strategies=strategies,
        limit=3,
    )

    calculator = MetricsCalculator()

    metrics = calculator.calculate(
        results
    )
    reporter = ConsoleReporter()
    reporter.report(
        metrics
    )


if __name__ == "__main__":
    main()
