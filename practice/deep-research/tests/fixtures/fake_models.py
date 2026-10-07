import json
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

_TASK = re.compile(r"- todo_id: (\S+)\n  question: (.*)")
_RESULT_ROW = re.compile(r"^\[(\d+)\] ", re.MULTILINE)
_POINT_ID = re.compile(r"^- (E\d+):", re.MULTILINE)


def _source_blocks(text: str) -> list[tuple[str, str]]:
    """``(source_id, passages)`` per "Source <id>: title / URL: ... / passages" block."""
    blocks = []
    for block in ("\n" + text.strip()).split("\nSource ")[1:]:
        header, _, rest = block.partition("\n")
        _url, _, passages = rest.partition("\n")
        blocks.append((header.split(":", 1)[0].strip(), passages))
    return blocks


def rephrase_points(messages) -> dict:
    """Query-rewriter answer: one extra phrasing per evidence point."""
    points = _POINT_ID.findall(str(messages[-1].content))
    return {"points": [{"point_id": p, "phrasings": [f"{p} alternative wording"]} for p in points]}


def quote_report(messages, *, answered: bool = True) -> dict:
    """Searcher-report answer: every point answered (or missing) with a verbatim quote of the
    first source's first words, and one finding per source."""
    data = str(messages[-1].content)
    points_part, _, sources_part = data.partition("Stored sources data:")
    points = _POINT_ID.findall(points_part)
    sources = [(sid, text.strip()) for sid, text in _source_blocks(sources_part)]
    support = []
    if sources and answered:
        sid, text = sources[0]
        first_line = text.splitlines()[0] if text else ""
        quote = " ".join(first_line.split()[:6])
        if quote:
            support = [{"source_id": sid, "quote": quote}]
    return {
        "points": [
            {"point_id": p, "status": "answered" if support else "missing", "support": support}
            for p in points
        ],
        "findings": [
            {"text": f"Finding from {sid}.", "source_ids": [sid]} for sid, _ in sources[:5]
        ],
    }


def accept_all_results(messages) -> dict:
    """Result-evaluator answer that keeps every result and marks every evidence point covered
    (the behavior of searchers before result evaluation existed)."""
    data = str(messages[-1].content)
    points_part, _, rows_part = data.partition("Search results data:")
    points = _POINT_ID.findall(points_part)
    return {
        "rows": [
            {"index": int(n), "decision": "accepted", "covers": points}
            for n in _RESULT_ROW.findall(rows_part)
        ]
    }


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


class FakeResearchModel:
    """Tool-calling research sub-agent fake, also playing the research supervisor that shares its
    router.

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
        if system.startswith("You evaluate web search results"):
            return AIMessage(content=json.dumps(accept_all_results(messages)))
        if system.startswith("You write lookup queries"):
            return AIMessage(content=json.dumps(rephrase_points(messages)))
        question = self._question(messages)
        if question in self.fail_questions:
            raise RuntimeError(f"forced research failure for {question}")
        if system.startswith("You write the short structured report"):
            scripted = self.findings.get(question) or {}
            return AIMessage(
                content=json.dumps(
                    quote_report(messages, answered=scripted.get("sufficient", True))
                )
            )
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


_CLAIM = re.compile(r"^(K\d+): (.*)$", re.M)
_CITED = re.compile(r"^  \[(S\d+)\]", re.M)


def judge_claims(messages, *, verdicts=None) -> dict:
    """Claim-judge answer: every sentence supported by each of its cited sources, unless
    `verdicts` maps a sentence substring to (verdict, states_fact, unsupported source ids)."""
    data = str(messages[-1].content)
    claims = []
    blocks = re.split(r"\n\n(?=K\d+: )", data.split("\n", 1)[1] if "\n" in data else data)
    for block in blocks:
        head = _CLAIM.search(block)
        if not head:
            continue
        claim_id, sentence = head.groups()
        cited = _CITED.findall(block)
        # Uncited sentences are framing unless a verdict is scripted for them.
        verdict, states_fact, bad = "supported", bool(cited), set()
        for needle, scripted in (verdicts or {}).items():
            if needle in sentence:
                verdict, states_fact, bad = scripted
        claims.append({
            "claim_id": claim_id,
            "states_fact": states_fact,
            "verdict": verdict,
            "per_source": [{"source_id": sid, "supports_part": sid not in bad} for sid in cited],
            "unsupported_parts": [] if verdict == "supported" else [sentence[:60]],
        })
    return {"claims": claims}


class FakeClaimModel:
    """Scripted claim judge (role ``claim_judge``); see ``judge_claims``. Uncited sentences are
    framing (states_fact false) unless scripted otherwise."""

    model_name = "fake-claim-judge"

    def __init__(self, verdicts=None, fail=False):
        self.verdicts = verdicts or {}
        self.fail = fail
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("claim judge unavailable")
        return AIMessage(content=json.dumps(judge_claims(messages, verdicts=self.verdicts)))
