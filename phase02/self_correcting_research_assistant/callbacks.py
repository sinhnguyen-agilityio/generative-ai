from typing import Any


class ReActLogger:
    """
    Pretty printer for agent execution.

    This logger formats streamed agent events into a ReAct-style trace.

    It does NOT expose the model's internal reasoning (chain of thought).
    Instead, it displays observable execution events such as tool calls,
    observations, and the final answer.
    """

    def on_event(self, event: dict[str, Any]) -> None:
        """
        Process a streamed event from `agent.stream()`.
        """
        for node, data in event.items():
            messages = data.get("messages", [])

            for message in messages:
                message_type = message.__class__.__name__

                if message_type == "AIMessage":
                    self._print_ai(message)

                elif message_type == "ToolMessage":
                    self._print_tool(message)

    def _print_ai(self, message) -> None:

        # Final answer
        if message.content:
            print("\n==============================")
            print("Final Answer")
            print("==============================")
            print(message.content)

        # Tool call
        for tool_call in getattr(message, "tool_calls", []):

            print("\n------------------------------")
            print("Action")
            print("------------------------------")
            print(tool_call["name"])

            print("\nAction Input")
            print("------------------------------")
            print(tool_call["args"])

    def _print_tool(self, message) -> None:
        print("\nObservation")
        print("------------------------------")
        print(message.content)
