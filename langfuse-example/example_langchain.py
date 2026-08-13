from langchain.agents import create_agent
from trace_client import langfuse_handler


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together and return the result."""
    return a + b


# Create the agent
agent = create_agent(
    model="openai:gpt-5.4-nano",
    tools=[add_numbers],
    system_prompt="You are a helpful math tutor who can do calculations using the provided tools.",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is 42 + 58?"}]},
    config={"callbacks": [langfuse_handler]}
)
