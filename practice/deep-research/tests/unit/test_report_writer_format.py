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
        brief, SourceCatalog(bundles=[]),
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

    writer.write(brief, SourceCatalog(bundles=[]), guidelines=["The report MUST be written in English."])

    assert "The report MUST be written in English." in model.prompts[0]
