"""One source numbering for the whole run: merging topic reports, and the writer's Sources section."""

from deep_research.agents.utils import merge_notes
from deep_research.agents.writer import WriterAgent
from tests.fakes import FakeTier

TOPIC_A = """## Status: complete
## Findings
Gemini input costs $0.30 [1]. The free tier is 15 RPM [2].
## Citations
2 sources, 1 domains, 0 snippet only
[1] Gemini pricing: https://ai.google.dev/pricing
[2] Rate limits: https://ai.google.dev/rate-limits"""

TOPIC_B = """## Status: partial
## Findings
GPT-4o mini costs $0.15 [1]. Gemini limits are lower [2].
## Citations
2 sources, 2 domains, 0 snippet only
[1] OpenAI pricing: https://openai.com/api/pricing
[2] Rate limits: https://ai.google.dev/rate-limits"""

FAILED = "## Status: failed\n## Coverage: minimal\n## Missing parts\n- The whole topic (essential: down)"


def test_merge_notes_gives_each_url_one_number():
    notes, sources = merge_notes([TOPIC_A, TOPIC_B, FAILED])
    assert "Gemini input costs $0.30 [1]. The free tier is 15 RPM [2]." in notes
    assert "GPT-4o mini costs $0.15 [3]. Gemini limits are lower [2]." in notes
    assert "## Citations" not in notes and "## Status: failed" in notes
    assert sources == {
        1: ("Gemini pricing", "https://ai.google.dev/pricing"),
        2: ("Rate limits", "https://ai.google.dev/rate-limits"),
        3: ("OpenAI pricing", "https://openai.com/api/pricing"),
    }


def test_writer_adds_sources_for_the_cited_numbers_only():
    text = "# Costs\nGemini is $0.30 [1] and GPT-4o mini $0.15 [3]. Invented [7].\n\n## Sources\n[1] made up"
    writer = WriterAgent(FakeTier({}, plain=lambda _: text))
    report = writer.write({"brief": "Compare costs.", "notes": [TOPIC_A, TOPIC_B]})["report"]
    assert report == (
        "# Costs\nGemini is $0.30 [1] and GPT-4o mini $0.15 [3]. Invented ."
        "\n\n## Sources\n[1] Gemini pricing: https://ai.google.dev/pricing\n[3] OpenAI pricing: https://openai.com/api/pricing"
    )
