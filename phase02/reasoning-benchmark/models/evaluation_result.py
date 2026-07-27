from dataclasses import dataclass


@dataclass(slots=True)
class EvaluationResult:
    """
    Result of executing the generated solution.
    """

    passed: bool
    execution_time: float
    error_type: str | None = None
    error_message: str | None = None
