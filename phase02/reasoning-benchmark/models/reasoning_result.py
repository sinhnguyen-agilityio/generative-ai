from dataclasses import dataclass


@dataclass(slots=True)
class ReasoningResult:
    """
    Output produced by a reasoning strategy.
    """
    strategy: str
    generated_code: str
    raw_response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency: float
    model: str
