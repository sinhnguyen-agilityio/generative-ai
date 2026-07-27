from __future__ import annotations

import sys

from agents.research import ResearchAgent
from llm import OpenAIClient
from observability.langfuse import get_langfuse_handler
from prompts.react_prompt import research_prompt
from tools.duckduckgo import DuckDuckGoSearchTool


def build_agent() -> ResearchAgent:
    """Create the research agent and wire it to the LLM, search tool, and tracing."""
    llm_client = OpenAIClient()
    langfuse_handler = get_langfuse_handler()
    search_tool = DuckDuckGoSearchTool()
    callbacks = [langfuse_handler] if langfuse_handler else None
    return ResearchAgent(
        prompt=research_prompt,
        llm=llm_client.llm,
        search_tool=search_tool,
        callbacks=callbacks,
    )


def main() -> None:
    """Run the app from the command line and print the final research answer."""

    # Get the question from parameter
    question = " ".join(sys.argv[1:]).strip()

    # If the question does not exist. Input the question
    if not question:
        question = input("What do you want to research? ").strip()

    if not question:
        raise SystemExit("Please enter a question.")

    print("Researching your question...")
    agent = build_agent()
    response = agent.run(question)

    if response.steps:
        print("\nReAct steps:")
        for step in response.steps:
            if step.thought:
                print(f"Thought: {step.thought}")
            if step.action:
                print(f"Action: {step.action}")
            if step.observation:
                print(f"Observation: {step.observation}")
            if step.final_answer:
                print(f"Final Answer: {step.final_answer}")
            print()

    print(f"\nAnswer:\n{response.answer}")
    print(f"\nCompleted in {response.iterations} step(s).")


if __name__ == "__main__":
    main()
