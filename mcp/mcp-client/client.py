import asyncio
import sys
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_types import TextContent


load_dotenv()

MODEL = "gpt-5"

openai = OpenAI()


def server_params(server_script_path: str) -> StdioServerParameters:
    """Describe the subprocess that runs an MCP server."""

    if server_script_path.endswith(".py"):
        command = "python"
    elif server_script_path.endswith(".js"):
        command = "node"
    else:
        raise ValueError("Server script must be a .py or .js file")

    return StdioServerParameters(
        command=command,
        args=[server_script_path],
    )


def convert_mcp_tools_to_openai(tools: list[Any]) -> list[dict]:
    """Convert MCP tools to OpenAI function tools."""

    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        }
        for tool in tools
    ]


async def process_query(client: Client, query: str) -> str:
    """Process a query using OpenAI and available MCP tools."""

    tool_list = await client.list_tools()

    available_tools = convert_mcp_tools_to_openai(tool_list.tools)

    input_messages: list[dict] = [
        {
            "role": "user",
            "content": query,
        }
    ]

    final_text: list[str] = []

    while True:
        response = openai.responses.create(
            model=MODEL,
            input=input_messages,
            tools=available_tools,
        )

        # Add model output to the conversation.
        input_messages += response.output

        tool_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        # No tool calls -> final answer
        if not tool_calls:
            final_text.append(response.output_text)
            break

        # Execute every requested MCP tool
        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_args = tool_call.arguments

            print(
                f"\n[Calling tool {tool_name} with args {tool_args}]"
            )

            import json

            arguments = json.loads(tool_args)

            result = await client.call_tool(
                tool_name,
                arguments,
            )

            tool_output = "\n".join(
                block.text
                for block in result.content
                if isinstance(block, TextContent)
            )

            input_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": tool_output,
                }
            )

    return "\n".join(final_text)


async def chat_loop(client: Client) -> None:
    """Run an interactive chat loop."""

    print("\nMCP Client Started!")
    print("Type your queries or 'quit' to exit.")

    while True:
        try:
            query = (
                await asyncio.to_thread(input, "\nQuery: ")
            ).strip()
        except EOFError:
            break

        if query.lower() == "quit":
            break

        if not query:
            continue

        try:
            response = await process_query(client, query)
            print("\n" + response)
        except Exception as e:
            print(f"\nError: {e}")


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    server_script = sys.argv[1]

    async with Client(
        stdio_client(server_params(server_script))
    ) as client:

        tool_list = await client.list_tools()

        tool_names = [
            tool.name
            for tool in tool_list.tools
        ]

        print(
            "\nConnected to server with tools:",
            tool_names,
        )

        await chat_loop(client)


if __name__ == "__main__":
    asyncio.run(main())
