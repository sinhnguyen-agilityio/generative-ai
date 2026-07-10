import time
from abc import ABC, abstractmethod

from llm import LLMClient
from models.reasoning_result import ReasoningResult
from models.task import ReasoningTask
from parsers.code_parser import CodeParser


class ReasoningStrategy(ABC):
    """
    Base class for all reasoning strategies.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        raise NotImplementedError

    def solve(self, task: ReasoningTask) -> ReasoningResult:
        """Solve a reasoning task."""
        prompt = self.prompt_template.format(
            task=task.prompt
        )

        response = self.llm.generate(
            prompt=prompt,
        )

        start = time.perf_counter()
        response = self.llm.generate(prompt)

        print("Response: ", response.text)
        latency = time.perf_counter() - start

        generated_code = CodeParser.extract_python_code(response.text)

        return ReasoningResult(
            strategy=self.name,
            generated_code=generated_code,
            raw_response=response.text,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            latency=latency,
            model=response.model,
        )

    @property
    @abstractmethod
    def prompt_template(self) -> str:
        raise NotImplementedError
