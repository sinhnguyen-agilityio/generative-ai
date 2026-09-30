import json

from langchain_core.messages import AIMessage

from intelligence_collector.agents.feedback_router import MAX_REPORT_EXCERPT_CHARS, FeedbackRouter
from intelligence_collector.models.router import ModelRouter


class _RecordingModel:
    model_name = "recording"

    def __init__(self):
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append("\n".join(str(m.content) for m in messages))
        return AIMessage(content=json.dumps({"route": "revise", "reason": "Only formatting."}))


def test_only_a_bounded_excerpt_of_the_report_is_sent_to_the_classifier():
    model = _RecordingModel()
    report = "HEAD " + "x" * (MAX_REPORT_EXCERPT_CHARS * 3) + " TAIL"

    decision = FeedbackRouter(ModelRouter(model)).route("Use bullet points.", report)

    assert decision.route == "revise"
    assert "Use bullet points." in model.prompts[0]
    assert "HEAD" in model.prompts[0] and "TAIL" not in model.prompts[0]
