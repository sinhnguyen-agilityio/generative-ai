from dataclasses import dataclass


@dataclass(slots=True)
class ReasoningTask:
    """
    A reasoning problem loaded from a benchmark dataset.
    """
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str
