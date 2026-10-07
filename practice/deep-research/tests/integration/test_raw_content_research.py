"""Raw page content end to end: Tavily's full page goes to the VFS and the vector index, while
every LLM prompt only ever sees the page's selected passages."""

import json

from langchain_core.messages import AIMessage, ToolMessage

from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.worker import run_worker
from intelligence_collector.contracts import ResearchTodo, TodoStatus
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.storage.evidence_index import ensure_run_indexed, reindex_run
from intelligence_collector.storage.vector_index import LocalVectorIndex
from intelligence_collector.tools.mcp import ControlledMCP
from intelligence_collector.agents.passage_selector import MAX_PASSAGES_PER_PAGE, PASSAGE_SEPARATOR
from intelligence_collector.tools.page_content import CHUNK_CHARS, source_id_for
from tests.fixtures.fake_models import FakeResearchModel

URL = "https://docs.test/langgraph/persistence"
DEEP_FACT = "DEEP-FACT: the SqliteSaver writes every checkpoint into a local SQLite file."
PAGE = "\n\n".join(
    [
        "# LangGraph persistence",
        *(
            f"## Part {n}\n" + " ".join(f"Persistence detail {n}.{i}." for i in range(70))
            for n in range(4)
        ),
        DEEP_FACT,
    ]
)
ROW = {"url": URL, "title": "Persistence", "content": "Tavily snippet.", "raw_content": PAGE}


class RecordingModel(FakeResearchModel):
    """FakeResearchModel that searches once, reflects with think_tool, then stops; records
    every prompt so tests can check what the LLMs saw."""

    def __init__(self):
        super().__init__()
        self.prompts: list[str] = []

    def invoke(self, messages, config=None):
        self.prompts.append("\n".join(str(message.content) for message in messages))
        system = str(messages[0].content)
        if system.startswith("You are a research agent"):
            done = sum(isinstance(message, ToolMessage) for message in messages)
            if done == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "think_tool",
                            "args": {"reflection": "The persistence page answers the question."},
                            "id": "think-1",
                        }
                    ],
                )
        return super().invoke(messages, config)


def _research(stores, model=None):
    vfs, index = stores
    model = model or RecordingModel()
    todo = ResearchTodo(todo_id="t1", research_question="How does LangGraph persist state?")
    mcp = ControlledMCP([ROW])
    result = run_worker(vfs, index, "run", todo, ResearchAgent(ModelRouter(model), lambda: mcp))
    return model, mcp, result


def test_llms_read_selected_passages_never_the_whole_page(stores):
    model, _, (todo, _, _) = _research(stores)

    assert todo.status == TodoStatus.COMPLETED
    # No LLM call reads the page itself: there is no per-page summarizer, and the passage the
    # question does not ask about never reaches a prompt.
    assert model.prompts and all(DEEP_FACT not in prompt for prompt in model.prompts)
    researcher_view = next(p for p in model.prompts if "Search results for" in p)
    assert "# LangGraph persistence" in researcher_view  # verbatim page text
    assert "Tavily snippet." not in researcher_view


def test_think_tool_reflection_is_recorded_but_is_not_evidence(stores):
    model, mcp, (todo, _, _) = _research(stores)
    vfs, _ = stores

    assert mcp.calls == 1  # a reflection does not spend the search budget
    final_turn = next(
        p for p in reversed(model.prompts) if p.startswith("You are a research agent")
    )
    assert "Reflection recorded: The persistence page answers the question." in final_turn
    report = next(p for p in model.prompts if p.startswith("You write the short structured report"))
    assert "Reflection recorded" not in report
    bundle = json.loads(vfs.read("run", "research/t1/evidence/result.json"))
    assert bundle["search_calls"] == 1


def test_the_full_page_is_stored_in_the_vfs_and_indexed_as_chunks(stores):
    vfs, index = stores
    _, _, (_, refs, chunks) = _research(stores)

    page = vfs.read("run", "research/t1/raw/0.md").decode()
    assert page.startswith("# LangGraph persistence") and DEEP_FACT in page
    capture = json.loads(vfs.read("run", "research/t1/raw/0.json"))
    assert capture["raw_path"] == "research/t1/raw/0.md"
    assert capture["source"]["source_id"] == source_id_for(URL)
    assert capture["content"].startswith("# LangGraph persistence")
    assert len(capture["content"]) <= MAX_PASSAGES_PER_PAGE * (CHUNK_CHARS + len(PASSAGE_SEPARATOR))
    assert "raw_content" not in capture["source"]["provenance"]
    assert {ref.kind for ref in refs} >= {"raw_page", "raw_capture", "evidence_bundle"}

    # One record per chunk, each with the page title as context and a stable id.
    assert len(chunks) > 3
    assert [chunk.chunk_id for chunk in chunks] == [
        f"run:{source_id_for(URL)}:{n}" for n in range(len(chunks))
    ]
    assert all(chunk.text.startswith("Persistence\n\n") for chunk in chunks)
    hit = index.retrieve("SqliteSaver writes every checkpoint SQLite file", limit=1, run_id="run")
    assert DEEP_FACT in hit[0].text  # a passage selection left out is still retrievable

    bundle = json.loads(vfs.read("run", "research/t1/evidence/result.json"))
    source = bundle["sources"][0]
    # The writer reads the selected passages, plus any looked-up passage behind a report quote.
    assert source["excerpt"].startswith(capture["content"])
    assert source["raw_path"] == "research/t1/raw/0.md"
    assert all("text" not in chunk for chunk in bundle["indexed_chunks"])


def test_a_new_index_is_rebuilt_once_from_the_stored_pages(stores, tmp_path):
    vfs, _ = stores
    _, _, (_, _, chunks) = _research(stores)
    new_model_index = LocalVectorIndex(tmp_path / "other-embedding-model.json")

    assert ensure_run_indexed(vfs, new_model_index, "run") == len(chunks)
    assert ensure_run_indexed(vfs, new_model_index, "run") == 0  # marked: not rebuilt again
    rebuilt = new_model_index.retrieve("SqliteSaver SQLite file", limit=1, run_id="run")
    assert DEEP_FACT in rebuilt[0].text
    assert [c.chunk_id for c in new_model_index.retrieve("", limit=99, run_id="run")] == [
        c.chunk_id for c in chunks
    ]
    assert reindex_run(vfs, new_model_index, "run") == len(chunks)  # same ids: an upsert


def test_a_page_too_short_to_use_keeps_the_snippet_and_stores_no_page(stores):
    vfs, index = stores
    todo = ResearchTodo(todo_id="t2", research_question="Short page?")
    mcp = ControlledMCP([{**ROW, "raw_content": "Access denied."}])

    _, _, chunks = run_worker(
        vfs, index, "run", todo, ResearchAgent(ModelRouter(FakeResearchModel()), lambda: mcp)
    )

    capture = json.loads(vfs.read("run", "research/t2/raw/0.json"))
    assert capture["content"] == "Tavily snippet." and capture["raw_path"] is None
    assert not (vfs.root / "run" / "research" / "t2" / "raw" / "0.md").exists()
    assert [chunk.text for chunk in chunks] == ["Persistence\n\nTavily snippet."]
