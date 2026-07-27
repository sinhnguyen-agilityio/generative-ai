import unittest

from agents.research import ResearchAgent
from models.react_step import ReActStep
from models.search_result import SearchResult
from prompts.react_prompt import research_prompt


class DummySearchTool:
    def __init__(self):
        self.calls = []

    def run(self, query: str):
        self.calls.append(query)
        return [SearchResult(title="Example", url="https://example.com", snippet="A useful result")]


class DummyResearchAgent(ResearchAgent):
    def __init__(self):
        super().__init__(prompt=research_prompt, llm=None, search_tool=DummySearchTool())
        self.steps = [
            ReActStep(
                thought="Need more information.",
                action="duckduckgo",
                action_input="Python history",
                final_answer=None,
            ),
            ReActStep(
                thought="Now I can answer.",
                action=None,
                action_input=None,
                final_answer="Python was created by Guido van Rossum.",
            ),
        ]

    def _run_step(self, question: str, context: str) -> ReActStep:
        return self.steps.pop(0)


class ResearchAgentTests(unittest.TestCase):
    def test_run_uses_search_results_before_returning_final_answer(self):
        agent = DummyResearchAgent()

        response = agent.run("Who invented Python?")

        self.assertEqual(
            response.answer, "Python was created by Guido van Rossum.")
        self.assertEqual(len(response.search_results), 1)
        self.assertEqual(agent.search_tool.calls, ["Python history"])
        self.assertEqual(response.iterations, 2)
        self.assertEqual(len(response.steps), 2)
        self.assertIn("Need more information.", response.thoughts)


if __name__ == "__main__":
    unittest.main()
