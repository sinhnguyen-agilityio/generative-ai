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
        self.tool_choices = []

    def bind_tools(self, tools, tool_choice=None):
        self.bound_tools = tools
        self.tool_choices.append(tool_choice)
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


class FakeReportModel:
    """Scripted report writer + BVF reflection fake that records every prompt it receives.

    ``drafts`` and ``reflections`` are consumed in order; the last item repeats once exhausted.
    """

    model_name = "fake-report-writer"

    def __init__(self, drafts, reflections=None):
        self.drafts = list(drafts)
        self.reflections = list(reflections or [{
            "diagnosis": "A sentence had no citation.",
            "lessons": ["End every body sentence with a catalog citation such as [S1]."],
            "strategy": "Add the missing citations and keep wording close to the excerpts.",
        }])
        self.calls: list[tuple[str, list]] = []

    @staticmethod
    def _next(items):
        return items.pop(0) if len(items) > 1 else items[0]

    def invoke(self, messages, config=None):
        if "bvf-reflect.v1" in messages[0].content:
            self.calls.append(("reflect", messages))
            return AIMessage(content=json.dumps(self._next(self.reflections)))
        self.calls.append(("write", messages))
        return AIMessage(content=json.dumps({"report_markdown": self._next(self.drafts)}))

    def prompts(self, kind: str) -> list[str]:
        return ["\n".join(str(m.content) for m in messages) for k, messages in self.calls if k == kind]


def cited_report(claims: list[str], sources: dict[str, str], *, limitations: list[str] = ()) -> str:
    """Build a report in the source-verification format: claims already carry [S#] markers."""
    parts = ["# Research report", "", "## Scope", "This report answers the research brief.", "",
             "## Findings", *claims, ""]
    if limitations:
        parts += ["## Limitations", *(f"- {gap}" for gap in limitations), ""]
    parts += ["## Sources", *(f"- [{sid}] Source - {url}" for sid, url in sources.items())]
    return "\n".join(parts)


class FakeAcceptanceModel:
    """Scripted acceptance-verdict fake; records every prompt it receives."""

    model_name = "fake-acceptance-verifier"

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls: list[list] = []

    @staticmethod
    def _next(items):
        return items.pop(0) if len(items) > 1 else items[0]

    def invoke(self, messages, config=None):
        self.calls.append(messages)
        return AIMessage(content=json.dumps(self._next(self.verdicts)))

    def prompts(self) -> list[str]:
        return ["\n".join(str(m.content) for m in messages) for messages in self.calls]


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

