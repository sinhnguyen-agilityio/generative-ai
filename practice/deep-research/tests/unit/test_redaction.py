"""One redaction for every error and log line that may echo a credential."""

import logging

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


def test_ui_faithfulness_gate_uses_the_workflow_threshold():
    passed_coverage = VerificationResult(
        artifact_hash="h",
        passed=False,
        check_results=[VerificationCheck(name="c", passed=True, gate="coverage")],
    )
    at_threshold = verification_gates(passed_coverage, FAITHFULNESS_THRESHOLD)[-1]
    above = verification_gates(passed_coverage, FAITHFULNESS_THRESHOLD + 0.01)[-1]
    # Strictly greater passes, exactly as in BVFNodes._faithfulness.
    assert at_threshold["passed"] is False
    assert above["passed"] is True
