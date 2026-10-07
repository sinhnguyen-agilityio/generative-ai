"""One redaction for every error and log line that may echo a credential."""

import importlib.util
import json
import logging
from pathlib import Path

from intelligence_collector.agents.passage_selector import PassageSelector
from intelligence_collector.contracts import FAITHFULNESS_THRESHOLD, VerificationCheck
from intelligence_collector.contracts import VerificationResult
from intelligence_collector.models.redaction import redact_error
from intelligence_collector.ui.html import verification_gates

SECRET = "tvly-dev-0123456789abcdef"


def test_redact_error_masks_mcp_endpoint_keys_and_bearer_tokens():
    exc = ConnectionError(
        f"GET https://mcp.tavily.com/mcp/?tavilyApiKey={SECRET} failed; "
        f"Authorization: Bearer {SECRET}"
    )
    message = redact_error(exc)
    assert message.startswith("ConnectionError: ")
    assert SECRET not in message
    assert "[REDACTED]" in message


def test_embedding_failure_log_is_redacted_and_has_no_traceback(caplog):
    class Embeddings:
        def embed_documents(self, texts):
            raise RuntimeError(f"POST https://embeddings.example/v1?key=x&api_key={SECRET}")

        def embed_query(self, text):
            return [0.0]

    selector = PassageSelector(Embeddings())
    with caplog.at_level(logging.WARNING):
        scores = selector._scores("status codes", ["401 means unauthenticated"])

    assert len(scores) == 1  # fell back to word overlap
    assert caplog.records, "the fallback is logged"
    assert all(SECRET not in record.getMessage() for record in caplog.records)
    assert all(record.exc_info is None for record in caplog.records)


def test_ui_faithfulness_gate_follows_the_claim_judge_not_the_ragas_threshold():
    passed_coverage = VerificationResult(
        artifact_hash="h",
        passed=False,
        check_results=[VerificationCheck(name="c", passed=True, gate="coverage")],
    )
    # A share above the RAGAS threshold still fails: every factual sentence must be supported.
    assert verification_gates(passed_coverage, FAITHFULNESS_THRESHOLD + 0.05)[-1]["passed"] is False
    assert verification_gates(passed_coverage, 1.0)[-1]["passed"] is True


# ----- spec 002: every new failure path (T061) -----

LEAK = f"provider down; Authorization: Bearer {SECRET}; api_key={SECRET}"


class Leaky:
    """A model provider whose errors echo the credential."""

    model_name = "leaky"

    def bind_tools(self, tools, tool_choice=None):
        return self

    def invoke(self, messages, config=None):
        raise RuntimeError(LEAK)


def _todo():
    from intelligence_collector.contracts import EvidencePoint, ResearchTodo

    return ResearchTodo(
        todo_id="t1",
        research_question="SQLite writers",
        evidence_points=[EvidencePoint(point_id="E1", text="Writers at once", required_answers=["w"])],
    )


def _no_secret_logged(caplog):
    assert all(SECRET not in record.getMessage() for record in caplog.records)
    assert all(
        SECRET not in str(record.exc_info[1]) for record in caplog.records if record.exc_info
    )


def test_result_evaluation_failure_is_stored_without_the_key(caplog):
    from intelligence_collector.agents.result_evaluation import evaluate_results
    from intelligence_collector.models.router import ModelRouter

    caplog.set_level(logging.DEBUG)
    outcome = evaluate_results(
        [{"url": "https://a.test", "content": "c", "raw_content": "page", "title": "A"}],
        todo=_todo(), question="q", query="q", search_index=1, seen_urls=set(),
        router=ModelRouter(Leaky()),
    )
    assert outcome.failed and SECRET not in outcome.error and "[REDACTED]" in outcome.error
    _no_secret_logged(caplog)


def test_passage_lookup_failure_is_stored_without_the_key(caplog):
    from intelligence_collector.storage.passage_lookup import PassageLookup

    class Down:
        def retrieve(self, *args, **kwargs):
            raise ConnectionError(LEAK)

        def documents(self, run_id):
            raise ConnectionError(LEAK)

    caplog.set_level(logging.DEBUG)
    result = PassageLookup(Down()).lookup(["writers"], run_id="run", allowed_source_ids=None)
    assert result.error and SECRET not in result.error
    _no_secret_logged(caplog)


def test_searcher_report_failure_is_stored_without_the_key(caplog):
    from intelligence_collector.agents.searcher_report import StoredSource, write_report
    from intelligence_collector.agents.worker import failed_outcome
    from intelligence_collector.models.router import ModelRouter

    caplog.set_level(logging.DEBUG)
    todo = _todo()
    report = write_report(
        todo, "q", {"S1": StoredSource(source_id="S1", url="https://a.test", title="A", text="t")},
        ModelRouter(Leaky()), outcome=failed_outcome(todo, "x"), lookup_used=set(),
    )
    assert report.report_status.value == "unreadable"
    assert SECRET not in json.dumps(report.model_dump(mode="json"))
    _no_secret_logged(caplog)


def test_gap_analysis_failure_is_stored_without_the_key(caplog):
    from intelligence_collector.agents.gap_analysis import GapAnalyzer
    from intelligence_collector.contracts import SearcherReport
    from intelligence_collector.models.router import ModelRouter

    caplog.set_level(logging.DEBUG)
    todo = _todo()
    report = SearcherReport.model_validate({
        "todo_id": "t1", "points": [{"point_id": "E1", "status": "missing"}],
        "stop_reason": "no_progress",
    })
    result = GapAnalyzer(ModelRouter(Leaky()), None).analyze(
        run_id="run", todos=[todo], reports={"t1": report}, checklist=["w"], round_number=1,
        previous=None, stall_count=0, allowed_source_ids=set(), stored_text=lambda sid: "",
    )
    stored = json.dumps(result.analysis.model_dump(mode="json"))
    assert "model call failed" in stored and SECRET not in stored
    _no_secret_logged(caplog)


def test_claim_judge_failure_is_stored_and_logged_without_the_key(stores, caplog):
    from tests.fixtures.fake_models import FakeClaimModel
    from tests.integration.test_bvf_loop import GOOD, FakeReportModel, _run

    class LeakyJudge(FakeClaimModel):
        def invoke(self, messages, config=None):
            raise RuntimeError(LEAK)

    caplog.set_level(logging.DEBUG)
    vfs, _ = stores
    result = _run(stores, FakeReportModel([GOOD]), claim_model=LeakyJudge())
    assert result["verification_outcome"] == "ERROR"
    assert "Claim evaluation failed" in result["verifier_error"]
    assert SECRET not in result["verifier_error"]
    for path in Path(vfs.root).rglob("*"):
        if path.is_file():
            assert SECRET not in path.read_text(encoding="utf-8", errors="ignore"), path
    _no_secret_logged(caplog)


def _benchmark_script():
    path = Path(__file__).parents[2] / "scripts" / "benchmark_supervisor_searcher.py"
    spec = importlib.util.spec_from_file_location("benchmark_supervisor_searcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_files_never_contain_a_credential(tmp_path):
    from intelligence_collector.models.redaction import redact_text, redact_value

    script = _benchmark_script()
    target = tmp_path / "benchmark_results.json"
    script.write_result(target, {
        "runs": [{"id": "q1", "error": f"RuntimeError: {LEAK}"}],
        "components": {"claim_judge": {"error": LEAK}},
        "settings": {"tavily_api_key": SECRET},
    }, redact_value)
    assert SECRET not in target.read_text(encoding="utf-8")
    # The committed baseline was written the same way.
    baseline = (Path(__file__).parents[2] / "evals/supervisor_searcher/baseline.json").read_text(
        encoding="utf-8"
    )
    assert redact_text(baseline) == baseline and "tvly-" not in baseline and "sk-" not in baseline
