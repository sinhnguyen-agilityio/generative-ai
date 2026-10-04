import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from intelligence_collector.config import LangfuseSettings, Settings
from intelligence_collector.guardrails import validate_request
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.storage.vfs import VFS


class Decision(BaseModel):
    answer: str


def test_vfs_rejects_run_and_relative_path_escape(tmp_path):
    vfs = VFS(tmp_path / "vfs")

    with pytest.raises(ValueError, match="cross-run"):
        vfs.write("../outside", "artifact.json", "bad", "test")
    with pytest.raises(ValueError, match="cross-run"):
        vfs.write("run", "../outside.json", "bad", "test")
    with pytest.raises(ValueError, match="invalid VFS relative"):
        vfs.write("run", str(tmp_path / "absolute.json"), "bad", "test")

    assert not (tmp_path / "outside" / "artifact.json").exists()
    assert not (tmp_path / "outside.json").exists()


def test_vfs_redacts_credential_shaped_json_fields(tmp_path):
    vfs = VFS(tmp_path / "vfs")
    secret = "do-not-persist"
    vfs.write_json(
        "run",
        "artifact.json",
        {"query": "safe", "nested": {"api_key": secret, "access_token": secret}},
        "test",
    )

    payload = json.loads(vfs.read("run", "artifact.json"))
    assert payload["query"] == "safe"
    assert payload["nested"] == {
        "api_key": "[REDACTED]",
        "access_token": "[REDACTED]",
    }
    assert secret not in repr(payload)


def test_guardrail_rejects_credentials_before_model_or_artifact_use():
    decision = validate_request("Research this with api_key=do-not-share")

    assert decision.allowed is False
    assert "credentials" in decision.reason
    assert "do-not-share" not in decision.reason


def test_configuration_budgets_are_positive_and_capped(monkeypatch):
    for name, value in (
        ("INTELLIGENCE_MAX_SEARCH_CALLS", "0"),
        ("INTELLIGENCE_MAX_SEARCH_CALLS", "6"),
        ("INTELLIGENCE_RECOVERY_ATTEMPTS", "0"),
        ("INTELLIGENCE_RECOVERY_ATTEMPTS", "11"),
    ):
        monkeypatch.setenv(name, value)
        other = (
            "INTELLIGENCE_RECOVERY_ATTEMPTS"
            if name == "INTELLIGENCE_MAX_SEARCH_CALLS"
            else "INTELLIGENCE_MAX_SEARCH_CALLS"
        )
        monkeypatch.delenv(other, raising=False)
        with pytest.raises(ValueError, match="must be between"):
            Settings.from_env()


def test_model_budget_and_trace_redaction_are_enforced():
    with pytest.raises(ValueError, match="max_attempts"):
        ModelRouter(lambda role, prompt: {}, max_attempts=0)
    with pytest.raises(ValueError, match="max_attempts"):
        ModelRouter(lambda role, prompt: {}, max_attempts=4)

    events = []
    secret = "do-not-trace"
    router = ModelRouter(
        lambda role, prompt: {"answer": f"token={secret}"},
        max_attempts=1,
        trace=lambda **event: events.append(event),
    )
    router.invoke_messages(
        "test", [HumanMessage(content=f"authorization=Bearer-{secret}")], Decision
    )

    assert secret not in repr(events)


def test_secret_settings_repr_and_dependency_manifest_are_safe():
    settings = LangfuseSettings(public_key="public-secret", secret_key="private-secret")
    assert "public-secret" not in repr(settings)
    assert "private-secret" not in repr(settings)

    manifest = Path("pyproject.toml").read_text(encoding="utf-8").lower()
    source = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in Path("src").rglob("*.py")
    )
    prohibited = "deep" + "agents"
    assert prohibited not in manifest.replace("-", "").replace("_", "")
    assert f"import {prohibited}" not in source.replace("_", "")
