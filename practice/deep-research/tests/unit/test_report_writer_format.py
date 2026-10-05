import json

from langchain_core.messages import AIMessage

from intelligence_collector.agents.report_writer import ReportWriter
from intelligence_collector.contracts import ResearchBrief
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.source_verification import SourceCatalog


class _RecordingModel:
    model_name = "recording"

    def __init__(self, report):
        self.report = report
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append("\n".join(str(m.content) for m in messages))
        return AIMessage(content=json.dumps({"report_markdown": self.report}))


def test_write_applies_reviewer_requests_to_the_reviewed_report():
    model = _RecordingModel("One line [S1].")
    writer = ReportWriter(ModelRouter(model))
    brief = ResearchBrief(research_question="A sufficiently long research question here.")

    result = writer.write(
        brief,
        SourceCatalog(bundles=[]),
        reviewer_requests=["Format the report in 1 line"],
        reviewed_report="# Title\n\n## Body\nA fact [S1].",
    )

    assert result == "One line [S1]."
    prompt = model.prompts[0]
    assert "Format the report in 1 line" in prompt and "A fact [S1]." in prompt
    # The request wins over the layout rules, never over the evidence rules.
    assert "take precedence over the layout rules" in prompt
    assert "keeps its [S<n>] citation" in prompt


def test_write_without_reviewer_requests_has_no_reviewer_block():
    model = _RecordingModel("# T\n\n## Scope\nS.\n")
    writer = ReportWriter(ModelRouter(model))
    brief = ResearchBrief(research_question="A sufficiently long research question here.")

    writer.write(brief, SourceCatalog(bundles=[]))

    assert "human reviewer asked" not in model.prompts[0]


def test_write_shows_the_operator_guidelines_to_the_writer():
    model = _RecordingModel("# T\n\n## Scope\nS.\n")
    writer = ReportWriter(ModelRouter(model))
    brief = ResearchBrief(research_question="A sufficiently long research question here.")

    writer.write(
        brief, SourceCatalog(bundles=[]), guidelines=["The report MUST be written in English."]
    )

    assert "The report MUST be written in English." in model.prompts[0]


DUPLICATED_ANSWER = """# What is LangGraph?

## Answer
LangGraph is an orchestration framework for stateful agents [S1].

## Answer
It is used for long-running, resumable agent workflows [S1].

## Limitations
- Subsection headings could not be established from the collected sources.

## Sources
- [S1] LangGraph README - https://github.com/langchain-ai/langgraph
"""


def test_one_answer_section_per_report_even_when_the_model_repeats_it():
    from intelligence_collector.agents.report_writer import merge_repeated_sections

    merged = merge_repeated_sections(DUPLICATED_ANSWER)

    assert merged.count("## Answer") == 1
    assert merged == (
        "# What is LangGraph?\n\n"
        "## Answer\n"
        "LangGraph is an orchestration framework for stateful agents [S1].\n\n"
        "It is used for long-running, resumable agent workflows [S1].\n\n"
        "## Limitations\n"
        "- Subsection headings could not be established from the collected sources.\n\n"
        "## Sources\n"
        "- [S1] LangGraph README - https://github.com/langchain-ai/langgraph\n"
    )


def test_a_repeated_section_joins_the_first_one_and_keeps_its_place():
    from intelligence_collector.agents.report_writer import merge_repeated_sections

    report = "# T\n\n## Answer\nA [S1].\n\n## Comparison\n| x |\n\n## answer ##\nB [S2].\n"
    assert merge_repeated_sections(report) == (
        "# T\n\n## Answer\nA [S1].\n\nB [S2].\n\n## Comparison\n| x |\n"
    )


def test_reports_without_repeats_and_headings_in_code_are_left_untouched():
    from intelligence_collector.agents.report_writer import merge_repeated_sections

    plain = "# T\n\n## Answer\nA [S1].\n\n\n## Sources\n- [S1] x - https://x.test\n"
    assert merge_repeated_sections(plain) == plain  # even its spacing
    fenced = "## Answer\nA [S1].\n\n```md\n## Answer\n```\n"
    assert merge_repeated_sections(fenced) == fenced
    assert merge_repeated_sections("No headings at all.") == "No headings at all."


def test_writer_returns_a_single_answer_section_and_asks_for_one():
    model = _RecordingModel(DUPLICATED_ANSWER)
    writer = ReportWriter(ModelRouter(model))
    brief = ResearchBrief(
        research_question="What is LangGraph and what is it used for?",
        required_answers=["What is LangGraph?", "What is LangGraph used for?"],
    )

    report = writer.write(brief, SourceCatalog(bundles=[]))

    assert report.count("## Answer") == 1
    assert "It is used for long-running, resumable agent workflows [S1]." in report
    prompt = model.prompts[0]
    assert "exactly ONE '## Answer' section" in prompt
    assert "answer all of them inside the single '## Answer' section" in prompt
