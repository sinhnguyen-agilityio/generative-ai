"""Crash boundaries, operation retries and concurrent URL sharing."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from intelligence_collector.tools.url_tracking import UrlClaims
from intelligence_collector.services.workflow import (
    DurableOperations,
    OperationFailure,
    RepairDiagnosis,
)
from intelligence_collector.source_verification import catalog_from_bundles
from intelligence_collector.agents.supervisor import ResearchSupervisor
from intelligence_collector.agents.researcher import ResearchOutcome
from intelligence_collector.contracts import ResearchTodo, ResearchFindings, IndexedChunk
from intelligence_collector.storage.mcp_vector_index import MCPVectorIndex
from tests.fixtures.workflow import workflow, approve_plan


def test_concurrent_workers_share_one_fetch_and_restore_before_checkpoint(stores):
    vfs, _ = stores
    claims = UrlClaims(vfs=vfs, run_id="run")
    calls, lock = [], threading.Lock()

    def fetch():
        with lock:
            calls.append("fetch")
        time.sleep(0.02)
        return {"url": "https://source.test/page", "content": "original"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(claims.capture, "https://source.test/page#fragment", str(i), fetch)
            for i in range(2)
        ]
        assert [f.result()["content"] for f in futures] == ["original", "original"]
    assert calls == ["fetch"]
    # No graph state/checkpoint is needed to recognize the complete fetch.
    restored = UrlClaims(vfs=vfs, run_id="run")
    assert (
        restored.capture(
            "https://source.test/page", "resumed", lambda: pytest.fail("must reuse original")
        )["content"]
        == "original"
    )


def test_failed_claim_allows_legitimate_retry(stores):
    vfs, _ = stores
    claims = UrlClaims(vfs=vfs, run_id="run")

    def fail():
        raise TimeoutError("temporary")

    with pytest.raises(TimeoutError):
        claims.capture("https://source.test/page", "first", fail)
    assert claims.owner("https://source.test/page") is None
    result = claims.capture(
        "https://source.test/page",
        "second",
        lambda: {"url": "https://source.test/page", "content": "saved"},
    )
    assert result["content"] == "saved"
    assert claims.owner("https://source.test/page") == "second"


def test_durable_retry_has_two_attempts_and_resume_grants_no_more(stores):
    vfs, _ = stores
    calls = []

    def fail():
        calls.append(1)
        raise TimeoutError("temporary")

    operations = DurableOperations(vfs)
    with pytest.raises(OperationFailure):
        operations.call("run", "fetch", {"url": "a"}, fail)
    with pytest.raises(OperationFailure):
        DurableOperations(vfs).call("run", "fetch", {"url": "a"}, fail)
    assert len(calls) == 2
    records = [
        json.loads(p.read_text()) for p in (vfs.root / "run" / "operations/policy").glob("*.json")
    ]
    assert records[0]["attempts"] == 2


def test_invalid_arguments_are_not_retried(stores):
    vfs, _ = stores
    calls = []

    def fail():
        calls.append(1)
        raise ValueError("invalid argument")

    with pytest.raises(OperationFailure):
        DurableOperations(vfs).call("run", "tool", {}, fail)
    assert len(calls) == 1


def test_sufficient_vfs_evidence_avoids_retrieval_and_missing_evidence_targets_research(stores):
    vfs, index = stores
    catalog = catalog_from_bundles(
        [
            {
                "task_id": "cache",
                "question": "Cache behavior",
                "summary": "Known cache evidence",
                "sources": [{"url": "https://source.test", "content": "Cache behavior"}],
            }
        ]
    )
    state = {
        "run_id": "run",
        "draft": "cache",
        "verification": {
            "issues": [
                {
                    "question": "cache",
                    "location": "Findings",
                    "reason": "Missing cache synthesis",
                    "route": "",
                    "evidence_refs": [],
                }
            ]
        },
    }
    diagnosis = RepairDiagnosis(vfs, index)
    assert diagnosis.diagnose(state, catalog)[0] == "vfs"
    assert diagnosis.diagnose(state, catalog_from_bundles([]))[0] == "research"
    state["verification"]["issues"][0]["reason"] = "Wrong report language and formatting"
    assert diagnosis.diagnose(state, catalog)[0] == "presentation"


def test_chroma_hits_are_read_back_from_original_vfs(stores):
    vfs, index = stores
    vfs.write("run", "raw/page.json", "original provenance", "raw_capture")
    index.index(
        [
            IndexedChunk(
                chunk_id="c",
                source_id="s",
                artifact_id="a",
                text="missing latency",
                metadata={"run_id": "run", "vfs_path": "raw/page.json"},
            )
        ]
    )
    state = {
        "run_id": "run",
        "draft": "draft",
        "verification": {
            "issues": [
                {
                    "question": "latency",
                    "location": "Findings",
                    "reason": "Missing latency",
                    "evidence_refs": [],
                    "route": "",
                }
            ]
        },
    }
    route, issues = RepairDiagnosis(vfs, index).diagnose(state, catalog_from_bundles([]))
    assert route == "chroma"
    assert issues[0]["evidence_refs"] == ["raw/page.json"]


def test_embedding_output_is_reused_after_failed_upsert(stores):
    vfs, _ = stores

    class Transport:
        def __init__(self):
            self.calls = []
            self.fail = True

        def call(self, tool, args):
            self.calls.append(tool)
            if tool == "embed":
                return {"embeddings": [[1.0, 0.0]]}
            if self.fail:
                raise TimeoutError("temporary")
            return {"chunk_ids": [c["chunk_id"] for c in args["chunks"]]}

    transport = Transport()
    index = MCPVectorIndex(transport, vfs, model="configured", provider="configured")
    chunks = [
        IndexedChunk(
            chunk_id="c",
            source_id="s",
            artifact_id="a",
            text="original",
            metadata={"run_id": "run", "vfs_path": "raw.json"},
        )
    ]
    with pytest.raises(OperationFailure):
        index.index(chunks)
    transport.fail = False
    # The consumed failed upsert cannot silently be retried; embedding is still reused.
    with pytest.raises(OperationFailure):
        MCPVectorIndex(transport, vfs, model="configured", provider="configured").index(chunks)
    assert transport.calls.count("embed") == 1
    assert transport.calls.count("chroma_upsert") == 2


def test_failure_stops_admission_and_retains_active_sibling_evidence(stores):
    vfs, index = stores

    class Researcher:
        def __init__(self):
            self.calls = []

        def research(self, question, **kwargs):
            self.calls.append(question)
            if question == "broken":
                time.sleep(0.01)
                raise ValueError("programming failure")
            time.sleep(0.04)
            return ResearchOutcome(
                question=question,
                search_budget=5,
                mcp_server="controlled",
                mcp_tool="search",
                sources=[{"url": f"https://source.test/{question}", "content": "original"}],
                findings=ResearchFindings(summary="Complete", sufficient=True),
            )

    researcher = Researcher()
    todos = [
        ResearchTodo(todo_id=q, research_question=q)
        for q in ("broken", "active-a", "active-b", "not-admitted")
    ]
    supervisor = ResearchSupervisor(researcher, vfs, index, max_concurrent_researchers=99)
    assert supervisor.max_concurrent_researchers == 3
    with pytest.raises(RuntimeError):
        supervisor.delegate("run", todos)
    assert "not-admitted" not in researcher.calls
    for q in ("active-a", "active-b"):
        assert vfs.read("run", f"research/{q}/evidence/result.json")
    assert json.loads(vfs.read("run", "research/worker_errors.json"))[0]["todo_id"] == "broken"


def test_resume_verifies_saved_revision_within_same_cycle(tmp_path):
    calls = []

    def evaluator(*args):
        calls.append(args[1])
        if len(calls) == 2:
            # Simulate process death after attempt intent, before response/checkpoint.
            raise SystemExit("crash")
        return 0.99

    graph, config, initial, nodes, writer, judge, researcher, diagnosis = workflow(
        tmp_path, failures=1, score=evaluator
    )
    with pytest.raises(SystemExit):
        approve_plan(graph, config, initial)
    saved = graph.get_state(config).values
    assert "version 2" in saved["draft"]
    assert saved["active_cycle_id"] == "initial:1"
    result = graph.invoke(None, config)
    assert result["verification"]["outcome"] == "PASS"
    assert result["batch_cycles"] == result["cumulative_cycles"] == 1
    assert len(writer.calls) == 2
    assert len(judge.calls) == 2  # successful coverage is reused


def test_overwritten_evidence_invalidates_previous_approval_binding(tmp_path):
    from intelligence_collector.services.workflow import versions

    graph, config, initial, nodes, writer, judge, researcher, diagnosis = workflow(tmp_path)
    result = approve_plan(graph, config, initial)
    ref = result["evidence"][0]
    relative = str(__import__("pathlib").Path(ref["path"]).relative_to(nodes.vfs.root / "run"))
    old = versions(result, nodes.vfs)
    nodes.vfs.write("run", relative, '{"changed": true}', "evidence_bundle")
    assert versions(result, nodes.vfs)["evidence_hash"] != old["evidence_hash"]


def test_resume_reembeds_completed_siblings_without_research(stores):
    from intelligence_collector.agents.worker import persist_research

    vfs, previous_index = stores
    todo = ResearchTodo(todo_id="saved", research_question="cache")
    outcome = ResearchOutcome(
        question="cache",
        search_budget=1,
        mcp_server="controlled",
        mcp_tool="search",
        sources=[{"url": "https://source.test/cache", "content": "saved raw"}],
        findings=ResearchFindings(summary="Complete", sufficient=True),
    )
    persist_research(vfs, previous_index, "run", todo, outcome)

    class Transport:
        def __init__(self):
            self.calls = []

        def call(self, tool, args):
            self.calls.append((tool, args))
            if tool == "embed":
                return {"embeddings": [[1.0, 0.0]]}
            return {"chunk_ids": [c["chunk_id"] for c in args["chunks"]]}

    class Researcher:
        def research(self, *args, **kwargs):
            pytest.fail("Completed evidence must not be searched again")

    transport = Transport()
    index = MCPVectorIndex(transport, vfs, model="text-embedding-3-small", provider="openai")
    supervisor = ResearchSupervisor(Researcher(), vfs, index)
    supervisor.delegate("run", [todo])
    supervisor.delegate("run", [todo])
    assert [tool for tool, _ in transport.calls] == ["embed", "chroma_upsert"]
    assert transport.calls[1][1]["model"] == "text-embedding-3-small"
    assert transport.calls[1][1]["provider"] == "openai"


def test_long_openai_research_is_lossless_bounded_and_resumable(stores):
    import tiktoken

    vfs, _ = stores
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    text = "Nghiên cứu dữ liệu 🌍 <|endoftext|>\n" * 10000
    chunk = IndexedChunk(
        chunk_id="long",
        source_id="source",
        artifact_id="raw",
        text=text,
        metadata={"run_id": "run", "vfs_path": "raw.json"},
    )
    operations = DurableOperations(vfs)
    old_inputs = {
        "rows": [chunk.model_dump(mode="json")],
        "model": "text-embedding-3-small",
        "provider": "openai",
    }

    def reject():
        raise ValueError("maximum input length is 8192 tokens")

    with pytest.raises(OperationFailure):
        operations.call("run", "embedding", old_inputs, reject)

    class Transport:
        def __init__(self):
            self.calls, self.rows = [], []

        def call(self, tool, args):
            self.calls.append(tool)
            if tool == "embed":
                assert len(args["texts"]) <= 64
                assert all(
                    len(encoding.encode(s, disallowed_special=())) <= 2000 for s in args["texts"]
                )
                return {"embeddings": [[1.0, 0.0] for _ in args["texts"]]}
            self.rows.extend(args["chunks"])
            return {"chunk_ids": [row["chunk_id"] for row in args["chunks"]]}

    transport = Transport()
    index = MCPVectorIndex(transport, vfs, model="text-embedding-3-small", provider="openai")
    ids = index.index([chunk])
    assert "".join(row["text"] for row in transport.rows) == text
    assert len(ids) > 64
    assert len(ids) == len(set(ids))
    assert all(
        row["source_id"] == "source"
        and row["artifact_id"] == "raw"
        and row["metadata"]["vfs_path"] == "raw.json"
        for row in transport.rows
    )
    before = list(transport.calls)
    assert index.index([chunk]) == ids
    assert transport.calls == before


def test_resume_bypasses_old_upsert_failure_and_reuses_embedding(stores):
    vfs, _ = stores
    chunk = IndexedChunk(
        chunk_id="c", source_id="s", artifact_id="a", text="saved", metadata={"run_id": "run"}
    )

    class Transport:
        def call(self, tool, args):
            assert tool == "chroma_upsert", "Saved embeddings must be reused"
            return {"chunk_ids": ["c"]}

    index = MCPVectorIndex(Transport(), vfs, model="text-embedding-3-small", provider="openai")
    inputs = {
        "storage_contract": 3,
        "rows": [chunk.model_dump(mode="json")],
        "model": index.model,
        "provider": index.provider,
    }
    index.operations.call("run", "embedding", inputs, lambda: {"embeddings": [[1.0, 0.0]]})

    def fail():
        raise RuntimeError("MCP storage tool chroma_upsert failed")

    with pytest.raises(OperationFailure):
        index.operations.call(
            "run", "chroma_upsert", {**inputs, "index_identity": index.identity}, fail
        )
    assert index.index([chunk]) == ["c"]
