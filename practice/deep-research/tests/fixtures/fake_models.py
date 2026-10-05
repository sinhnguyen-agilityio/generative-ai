import json
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

_TASK = re.compile(r"- todo_id: (\S+)\n  question: (.*)")
_PAGE = re.compile(r"<untrusted_page>\n(.*)\n</untrusted_page>", re.S)


def fake_supervisor_turn(messages) -> AIMessage:
    """A well-behaved research supervisor: think, delegate every pending task in one parallel
    batch, think again, then ResearchComplete."""
    conversation = [message for message in messages if message.type != "system"]
    decisions = [message for message in conversation if isinstance(message, AIMessage)]
    tasks = _TASK.findall(conversation[0].content)
    if not decisions:
        calls = [("think_tool", {"reflection": f"Plan: research all {len(tasks)} tasks."})]
    elif len(decisions) == 1:
        calls = [
            ("ConductResearch", {"todo_id": todo_id, "research_topic": question})
            for todo_id, question in tasks
        ]
    elif len(decisions) == 2:
        calls = [("think_tool", {"reflection": "Every task returned findings."})]
    else:
        calls = [("ResearchComplete", {})]
    turn = len(decisions)
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"supervisor-{turn}-{n}"}
            for n, (name, args) in enumerate(calls)
        ],
    )


def fake_page_summary(messages) -> AIMessage:
    """Summarize a page as its first line, quoting that line verbatim plus one invented quote
    (which the summarizer must drop)."""
    page = _PAGE.search(messages[-1].content).group(1)
    first = page.strip().splitlines()[0].strip()
    return AIMessage(
        content=json.dumps(
            {"summary": f"Summary: {first}", "key_excerpts": [first, "An invented quote."]}
        )
    )


class FakeResearchModel:
    """Tool-calling research sub-agent fake, also playing the research supervisor and the page
    summarizer that share its router.

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
        system = str(messages[0].content)
        if system.startswith("You are the research supervisor"):
            return fake_supervisor_turn(messages)
        if system.startswith("You summarize one web page"):
            return fake_page_summary(messages)
        question = self._question(messages)
        if question in self.fail_questions:
            raise RuntimeError(f"forced research failure for {question}")
        if "compress raw web search results" in messages[0].content:
            return AIMessage(
                content=json.dumps(
                    self.findings.get(
                        question,
                        {
                            "summary": f"Findings for {question} [https://controlled.test]",
                            "sufficient": True,
                            "unresolved_gaps": [],
                        },
                    )
                )
            )
        done = sum(isinstance(message, ToolMessage) for message in messages)
        queries = self.queries.get(question, [question])
        if done < len(queries):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tavily_search",
                        "args": {"query": queries[done]},
                        "id": f"call-{done}",
                    }
                ],
            )
        return AIMessage(content="I have enough evidence to answer.")


class ScriptedSupervisorModel(FakeResearchModel):
    """FakeResearchModel whose supervisor turns follow a script.

    Each script step maps the pending tasks (``[(todo_id, question), ...]``) to the tool calls
    of that turn, as ``[(tool_name, args), ...]``; the last step repeats. Every supervisor prompt
    is recorded, so tests can read the tool replies the supervisor received.
    """

    def __init__(self, script, **kwargs):
        super().__init__(**kwargs)
        self.script = list(script)
        self.supervisor_prompts: list[list] = []

    def invoke(self, messages, config=None):
        if not str(messages[0].content).startswith("You are the research supervisor"):
            return super().invoke(messages, config)
        self.supervisor_prompts.append(messages)
        conversation = [message for message in messages if message.type != "system"]
        turn = sum(isinstance(message, AIMessage) for message in conversation)
        step = self.script[min(turn, len(self.script) - 1)]
        calls = step(_TASK.findall(conversation[0].content))
        return AIMessage(
            content="" if calls else "Done.",
            tool_calls=[
                {"name": name, "args": args, "id": f"s{turn}-{n}"}
                for n, (name, args) in enumerate(calls)
            ],
        )

    def tool_replies(self) -> list[str]:
        """Every tool reply the supervisor saw, in order (from its latest prompt)."""
        latest = self.supervisor_prompts[-1] if self.supervisor_prompts else []
        return [str(m.content) for m in latest if isinstance(m, ToolMessage)]


class FakeReportModel:
    """Scripted report writer + BVF reflection fake that records every prompt it receives.

    ``drafts`` and ``reflections`` are consumed in order; the last item repeats once exhausted.
    """

    model_name = "fake-report-writer"

    def __init__(self, drafts, reflections=None):
        self.drafts = list(drafts)
        self.reflections = list(
            reflections
            or [
                {
                    "diagnosis": "A sentence had no citation.",
                    "lessons": ["End every body sentence with a catalog citation such as [S1]."],
                    "strategy": "Add the missing citations and keep wording close to the excerpts.",
                }
            ]
        )
        self.calls: list[tuple[str, list]] = []

    @staticmethod
    def _next(items):
        return items.pop(0) if len(items) > 1 else items[0]

    def invoke(self, messages, config=None):
        if "Fix planner in a Build-Verify-Fix loop" in messages[0].content:
            self.calls.append(("reflect", messages))
            return AIMessage(content=json.dumps(self._next(self.reflections)))
        self.calls.append(("write", messages))
        return AIMessage(content=json.dumps({"report_markdown": self._next(self.drafts)}))

    def prompts(self, kind: str) -> list[str]:
        return [
            "\n".join(str(m.content) for m in messages) for k, messages in self.calls if k == kind
        ]


def cited_report(claims: list[str], sources: dict[str, str], *, limitations: list[str] = ()) -> str:
    """Build a report in the source-verification format: claims already carry [S#] markers."""
    parts = [
        "# Research report",
        "",
        "## Scope",
        "This report answers the research brief.",
        "",
        "## Findings",
        *claims,
        "",
    ]
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
        scripted = self._next(self.verdicts)
        # Scripted verdicts describe selected failures; fill the actual prompt checklist.
        # Production rejects omissions, so this test fake must obey that same contract.
        prompt = str(messages[-1].content)
        block = prompt.split("Criteria to judge:\n", 1)[1].split("\n\n", 1)[0]
        criteria = [line[2:] for line in block.splitlines() if line.startswith("- ")]
        selected = []
        for item in scripted["results"]:
            label = item["criterion"]
            index = (
                1
                if label == "Fairly represents the gathered evidence"
                else 2
                if label == "Explains uncertainty and evidence gaps"
                else int(label[2:-1]) - 1
                if label.startswith("[C") and label.endswith("]")
                else 0
            )
            matches = [c for c in criteria if label in c]
            criterion = matches[0] if matches else criteria[min(index, len(criteria) - 1)]
            selected.append({**item, "criterion": criterion})
        used = {item["criterion"] for item in selected}
        selected.extend(
            {"criterion": criterion, "met": True, "justification": "Satisfied."}
            for criterion in criteria
            if criterion not in used
        )
        return AIMessage(content=json.dumps({"results": selected}))

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
