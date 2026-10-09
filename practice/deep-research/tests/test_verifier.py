"""The verifier judges each sentence against the evidence of the sources it cites, sends the report
back to the writer at most twice, then deletes the unsupported sentences left."""

from itertools import count

from deep_research.agents.verifier import VerifierAgent, parse, remove
from deep_research.agents.writer import with_sources
from deep_research.config import ModelSpec, Settings
from deep_research.state import CriterionVerdict, ReportCheck, SentenceVerdict
from tests.fakes import FakeTier, make_app

SETTINGS = Settings(reasoning=ModelSpec("fake"), standard=ModelSpec("fake"))
CONFIG = {"configurable": {"thread_id": "t"}}
NOTES = [
    {
        "report": "Gemini costs $0.30 [1]. A blog says it is cheap [2].",
        "sources": [
            {"id": 1, "title": "Gemini pricing", "url": "https://a", "full_text": True, "passages": ["Gemini costs $0.30."]},
            {"id": 2, "title": "Blog", "url": "https://b", "full_text": False, "passages": []},
        ],
    }
]
SOURCES = {1: ("Gemini pricing", "https://a"), 2: ("Blog", "https://b")}
CHECK = ReportCheck(
    sentences=[
        SentenceVerdict(sentence=1, verdict="supported", problem=""),
        SentenceVerdict(sentence=2, verdict="unsupported", problem="[2] has no evidence."),
        SentenceVerdict(sentence=9, verdict="unsupported", problem="no such sentence"),
    ],
    criteria=[CriterionVerdict(criterion=2, met=False, problem="Speed is not covered.")],
)


def state(fix_rounds):
    """Build report verification state at the requested correction round.

    Args:
        fix_rounds: Number of writer corrections already attempted.

    Returns:
        Brief, criteria, report, source notes, and fix counter.
    """
    report = with_sources("# Costs\nGemini is $0.30 [1]. It is cheap [2].", SOURCES)
    return {"brief": "Compare costs.", "criteria": ["Price?", "Speed?"], "report": report, "notes": NOTES, "fix_rounds": fix_rounds}


def verifier(check, prompts=None):
    """Build a verifier with a scripted judge and optional prompt recording.

    Args:
        check: Structured verdict or exception to return from the judge.
        prompts: Optional list receiving judge prompts.

    Returns:
        VerifierAgent backed by a fake model tier.
    """
    def judge(prompt):
        """Capture the verification prompt and return or raise the configured verdict."""
        if prompts is not None:
            prompts.append(prompt)
        if isinstance(check, Exception):
            raise check
        return check

    return VerifierAgent(FakeTier({"ReportCheck": judge}), SETTINGS)


def test_parse_and_remove_keep_markup_and_drop_emptied_sections():
    """Verify that parse and remove keep markup and drop emptied sections."""
    body = "\n".join(
        [
            "# Title", "## Costs", "- Gemini is $0.30 [1]. It is cheap [2].", "| A | B |", "|---|---|",
            "| x | y [1] |", "```", "# not a heading", "x = 1. y = 2.", "```", "## Speed", "It is fast [2].",
            "## Limitations", "Speed was not established.",
        ]
    )
    lines = parse(body)
    judged = [s for _, sentences in lines if sentences for s in sentences]
    assert judged == [
        "Gemini is $0.30 [1].", "It is cheap [2].", "| A | B |", "| x | y [1] |", "It is fast [2].", "Speed was not established."
    ]
    assert remove(lines, {(2, 1), (11, 0)}) == "\n".join(
        [
            "# Title", "## Costs", "- Gemini is $0.30 [1].", "| A | B |", "|---|---|", "| x | y [1] |", "```",
            "# not a heading", "x = 1. y = 2.", "```", "## Limitations", "Speed was not established.",
        ]
    )


def test_problems_go_back_to_the_writer_with_the_evidence_of_the_cited_sources_only():
    """Verify that problems go back to the writer with the evidence of the cited sources only."""
    prompts = []
    result = verifier(CHECK, prompts).verify(state(fix_rounds=0), CONFIG)
    assert result == {
        "verification": {
            "issues": ['S2 "It is cheap [2].": [2] has no evidence.', "Criterion not answered: Speed? (Speed is not covered.)"]
        },
        "fix_rounds": 1,
    }
    assert "S1: Gemini is $0.30 [1].\nS2: It is cheap [2]." in prompts[0]
    assert "[1] Gemini pricing\nGemini costs $0.30." in prompts[0]
    assert "[2] Blog\n(no passage of this source was read)" in prompts[0]
    assert "C1: Price?\nC2: Speed?" in prompts[0]


def test_after_two_fix_rounds_unsupported_sentences_are_deleted_and_sources_rebuilt():
    """Verify that after two fix rounds unsupported sentences are deleted and sources rebuilt."""
    result = verifier(CHECK).verify(state(fix_rounds=2), CONFIG)
    assert result == {
        "report": "# Costs\nGemini is $0.30 [1].\n\n## Sources\n[1] Gemini pricing: https://a",
        "verification": {"support": 0.5, "rounds": 2, "removed": ["It is cheap [2]."], "unmet": ["Speed?"]},
    }


def test_a_passing_report_is_kept_as_it_is():
    """Verify that a passing report is kept as it is."""
    passed = ReportCheck(sentences=[SentenceVerdict(sentence=1, verdict="no_claim", problem="")], criteria=[])
    result = verifier(passed).verify(state(fix_rounds=0), CONFIG)
    assert result == {"report": state(0)["report"], "verification": {"support": 1.0, "rounds": 0, "removed": [], "unmet": []}}


def test_a_failed_check_is_recorded_not_raised():
    """Verify that a failed check is recorded not raised."""
    assert verifier(RuntimeError("judge down")).verify(state(0), CONFIG) == {"verification": {"error": "RuntimeError: judge down"}}


def test_the_graph_fixes_twice_then_hands_the_report_to_review(tmp_path):
    """Verify that the graph fixes twice then hands the report to review."""
    prompts, reports = [], count(1)

    def writer(prompt):
        """Record writer prompts and return a newly numbered report draft."""
        prompts.append(prompt)
        return f"# Report v{next(reports)}"

    unmet = ReportCheck(sentences=[], criteria=[CriterionVerdict(criterion=1, met=False, problem="missing")])
    app, _ = make_app(tmp_path, check=unmet, writer=writer)
    list(app.ask("t", "Research X"))
    nodes = [node for node, _ in app.resume("t", {"approved": True})]
    assert nodes[-7:] == ["write_report", "verify", "write_report", "verify", "write_report", "verify", "evaluate"]
    assert "Criterion not answered: What is X? (missing)" in prompts[1]
    pending = app.view("t").pending
    assert pending["report"] == "# Report v3"
    assert pending["verification"] == {"support": 1.0, "rounds": 2, "removed": [], "unmet": ["What is X?"]}

    # Reviewer feedback starts a new report with a fresh fix budget.
    list(app.resume("t", {"approved": False, "feedback": "shorter"}))
    assert app.view("t").pending["report"] == "# Report v6"
