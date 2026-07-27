from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BenchmarkMetrics:
    """
    Aggregated benchmark metrics for a single reasoning strategy.
    """

    #
    # Strategy
    #
    strategy: str

    #
    # Task statistics
    #
    total_tasks: int
    passed_tasks: int
    failed_tasks: int

    #
    # Accuracy
    #
    pass_rate: float

    #
    # Token statistics
    #
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float

    #
    # Performance
    #
    avg_latency: float

    #
    # Cost
    #
    estimated_cost: float

    #
    # Efficiency
    #
    tokens_per_pass: float
    error_counts: dict[str, int]
