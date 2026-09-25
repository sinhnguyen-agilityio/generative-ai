import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class FakeResearchModel:
    """Tool-calling research sub-agent fake.

    Stateless per call (decisions derive from the messages), so it is safe to share across
    parallel sub-agents. For each research question it issues the scripted ``tavily_search``
    queries one at a time, then a final note; compression returns the scripted findings.
    """

    model_name = "fake-researcher"

    def __init__(self, queries=None, findings=None, fail_questions=()):
        self.queries = queries or {}
        self.findings = findings or {}
        self.fail_questions = set(fail_questions)
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    @staticmethod
    def _question(messages) -> str:
        for message in messages:
            if isinstance(message, HumanMessage) and "Research question:\n" in message.content:
                return message.content.split("Research question:\n", 1)[1].split("\n\n", 1)[0]
        return ""

    def invoke(self, messages, config=None):
        question = self._question(messages)
        if question in self.fail_questions:
            raise RuntimeError(f"forced research failure for {question}")
        if "research-compress.v1" in messages[0].content:
            return AIMessage(content=json.dumps(self.findings.get(question, {
                "summary": f"Findings for {question} [https://controlled.test]",
                "sufficient": True,
                "unresolved_gaps": [],
            })))
        done = sum(isinstance(message, ToolMessage) for message in messages)
        queries = self.queries.get(question, [question])
        if done < len(queries):
            return AIMessage(content="", tool_calls=[{
                "name": "tavily_search",
                "args": {"query": queries[done]},
                "id": f"call-{done}",
            }])
        return AIMessage(content="I have enough evidence to answer.")


class FakeModel:
    """A deterministic role-aware model substitute for tests."""
    def __init__(self, responses=None, error_roles=None):
        self.responses = responses or {}
        self.error_roles = set(error_roles or ())
        self.calls = []

    def __call__(self, role, prompt):
        self.calls.append((role, prompt))
        if role in self.error_roles:
            raise RuntimeError(f"forced model failure for {role}")
        return self.responses.get(role, prompt)

