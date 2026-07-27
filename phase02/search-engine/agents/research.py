from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from models.agent_response import AgentResponse
from models.react_step import ReActStep
from tools.base import BaseTool


class ResearchAgent:
    def __init__(
        self,
        prompt: ChatPromptTemplate,
        llm: BaseChatModel | None,
        search_tool: BaseTool,
        max_iterations: int = 5,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> None:
        """Create the agent and connect it to the model, prompt, and search tool."""
        self.prompt = prompt
        self.llm = llm
        self.search_tool = search_tool
        self.max_iterations = max_iterations
        self.callbacks = list(callbacks or [])
        self.react_llm = llm.with_structured_output(
            ReActStep) if llm is not None else None
        self.chain = prompt | self.react_llm if self.react_llm is not None else None

    def _coerce_react_step(self, output: Any) -> ReActStep:
        """Convert the model output into a consistent ReActStep object."""
        if isinstance(output, ReActStep):
            return output

        if isinstance(output, dict):
            return ReActStep(**output)

        if hasattr(output, "model_dump"):
            return ReActStep.model_validate(output.model_dump())

        if hasattr(output, "dict"):
            return ReActStep.model_validate(output.dict())

        raise TypeError(
            f"Unsupported structured output type: {type(output)!r}")

    def _run_step(self, question: str, context: str) -> ReActStep:
        """Ask the model for one reasoning step using the current question and context."""
        if self.chain is None:
            raise RuntimeError(
                "A language model is required to run the research agent.")

        invoke_config: RunnableConfig | None = None
        if self.callbacks:
            invoke_config = {"callbacks": list(self.callbacks)}

        output = self.chain.invoke(
            {
                "question": question,
                "context": context,
            },
            config=invoke_config,
        )
        return self._coerce_react_step(output)

    def _format_search_results(self, query: str, results: list) -> str:
        """Turn search results into a short, readable block for the prompt."""
        if not results:
            return f"Search query: {query}\nNo results found."

        formatted = [
            f"- {result.title}: {result.snippet} ({result.url})" for result in results]
        return f"Search query: {query}\n" + "\n".join(formatted)

    def run(self, question: str) -> AgentResponse:
        """Run the full research loop until the agent gives an answer or stops."""
        search_results: list = []
        thoughts: list[str] = []
        context_blocks: list[str] = []
        steps: list[ReActStep] = []
        last_step: ReActStep | None = None

        for iteration in range(1, self.max_iterations + 1):
            step = self._run_step(question, "\n\n".join(context_blocks))
            thoughts.append(step.thought)
            last_step = step

            if step.final_answer:
                steps.append(step)
                return AgentResponse(
                    answer=step.final_answer,
                    search_results=search_results,
                    thoughts=thoughts,
                    steps=steps,
                    iterations=iteration,
                )

            if step.action and step.action_input and self.search_tool:
                tool_name = getattr(self.search_tool, "name", None)
                if tool_name is not None and step.action.lower() != tool_name.lower():
                    break

                tool_results = self.search_tool.run(step.action_input)
                search_results.extend(tool_results)
                observation = self._format_search_results(
                    step.action_input, tool_results)
                context_blocks.append(observation)
                steps.append(step.model_copy(
                    update={"observation": observation}))
                continue

            steps.append(step)
            break

        fallback_answer = (
            last_step.final_answer
            if last_step and last_step.final_answer
            else "I could not determine a reliable answer from the available information."
        )

        return AgentResponse(
            answer=fallback_answer,
            search_results=search_results,
            thoughts=thoughts,
            steps=steps,
            iterations=max(1, iteration),
        )
