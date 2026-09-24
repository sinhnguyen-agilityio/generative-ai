from pathlib import Path

from intelligence_collector.config import load_local_env


def test_local_env_does_not_override_process_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SCOPE_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("SCOPE_MODEL", "from-process")

    load_local_env()

    assert __import__("os").environ["SCOPE_MODEL"] == "from-process"


def test_env_example_contains_placeholders_not_credentials():
    example = Path(".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in example
    assert "SCOPE_MODEL=" in example
    assert "SCOPE_PROVIDER=" in example
    assert "RUN_LIVE_LLM=" in example
    assert "sk-" not in example
