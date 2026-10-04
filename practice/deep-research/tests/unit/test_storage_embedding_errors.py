def test_embedding_error_preserves_redacted_provider_cause(monkeypatch):
    # Never load the developer's local dotenv or contact a provider.
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    from intelligence_collector.storage import mcp_server

    class ProviderError(Exception):
        code = 400

    def fail(*args):
        raise ProviderError(
            "Invalid model; api_key=private-key https://provider.test/?key=private-key"
        )

    monkeypatch.setattr(mcp_server, "_embed", fail)
    error = mcp_server.embed(["original"], "configured-model", "google_genai")["error"]
    assert error["type"] == "ProviderError"
    assert error["status_code"] == 400
    assert not error["retryable"]
    assert "Invalid model" in error["message"]
    assert "private-key" not in error["message"]


def test_google_embedding_inputs_remain_separate_contents(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    from intelligence_collector.storage import mcp_server
    from contextlib import nullcontext
    import langchain_google_genai.embeddings as integration

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    seen = []

    def embed_content(*, model, contents, config):
        assert model == "gemini-embedding-2"
        assert config.task_type == "TASK_TYPE_UNSPECIFIED"
        assert config.output_dimensionality is None
        assert config.http_options.timeout == 60000
        assert config.http_options.retry_options.attempts == 1
        seen.append(contents)
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[float(i), 1.0]) for i, _ in enumerate(contents)]
        )

    class FakeClient(nullcontext):
        def __init__(self, **kwargs):
            super().__init__(self)
            self.models = SimpleNamespace(embed_content=embed_content)

    monkeypatch.setattr(integration, "Client", FakeClient)
    result = mcp_server._embed(["first page", "second page"], "gemini-embedding-2", "google_genai")
    assert result["embeddings"] == [[0.0, 1.0], [1.0, 1.0]]
    assert [c["parts"][0]["text"] for c in seen[0]] == ["first page", "second page"]


def test_langchain_provider_error_keeps_retryable_status(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import langchain_google_genai.embeddings as integration

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from intelligence_collector.storage import mcp_server

    class ProviderError(Exception):
        code = 429

    def fail(**kwargs):
        raise ProviderError("Rate limited; api_key=private-key")

    class FakeClient(nullcontext):
        def __init__(self, **kwargs):
            super().__init__(self)
            self.models = SimpleNamespace(embed_content=fail)

    monkeypatch.setattr(integration, "Client", FakeClient)
    error = mcp_server.embed(["text"], "gemini-embedding-2", "google_genai")["error"]
    assert error["type"] == "ProviderError"
    assert error["status_code"] == 429
    assert error["retryable"]
    assert "private-key" not in error["message"]


def test_google_embedding_rejects_partial_results(monkeypatch):
    import pytest
    from contextlib import nullcontext
    from types import SimpleNamespace
    import langchain_google_genai.embeddings as integration

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from intelligence_collector.storage import mcp_server

    class FakeClient(nullcontext):
        def __init__(self, **kwargs):
            super().__init__(self)
            self.models = SimpleNamespace(
                embed_content=lambda **kwargs: SimpleNamespace(embeddings=[])
            )

    monkeypatch.setattr(integration, "Client", FakeClient)
    with pytest.raises(ValueError, match="Partial embeddings"):
        mcp_server._embed(["text"], "gemini-embedding-2", "google_genai")


def test_openai_embedding_preserves_input_order(monkeypatch):
    from types import SimpleNamespace
    import openai

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    from intelligence_collector.storage import mcp_server

    def create(**kwargs):
        assert kwargs == {"model": "text-embedding-3-small", "input": ["first", "second"]}
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[2.0]),
                SimpleNamespace(index=0, embedding=[1.0]),
            ]
        )

    def client(**kwargs):
        assert kwargs == {"max_retries": 0, "timeout": 60}
        return SimpleNamespace(embeddings=SimpleNamespace(create=create))

    monkeypatch.setattr(openai, "OpenAI", client)
    assert mcp_server._embed(["first", "second"], "text-embedding-3-small", "openai") == {
        "embeddings": [[1.0], [2.0]]
    }


def test_model_collections_isolate_incompatible_vectors(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("INTELLIGENCE_CHROMA_PATH", str(tmp_path / "chroma"))
    from intelligence_collector.storage import mcp_server

    google = mcp_server.collection("gemini-embedding-2", "google_genai")
    openai = mcp_server.collection("text-embedding-3-small", "openai")
    google.upsert(ids=["same"], embeddings=[[1.0, 0.0, 0.0]])
    openai.upsert(ids=["same"], embeddings=[[1.0, 0.0]])
    assert google.name != openai.name
    assert google.get(include=["embeddings"])["embeddings"].shape == (1, 3)
    assert openai.get(include=["embeddings"])["embeddings"].shape == (1, 2)


def test_chroma_upsert_preserves_dimension_error(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    from intelligence_collector.storage import mcp_server

    def fail(*args):
        raise ValueError(
            "Collection expecting embedding with dimension of 3072, got 1536; api_key=private-key"
        )

    monkeypatch.setattr(mcp_server, "_chroma_upsert", fail)
    error = mcp_server.chroma_upsert("run", [], [], "text-embedding-3-small", "openai")["error"]
    assert error["type"] == "ValueError"
    assert "3072" in error["message"] and "1536" in error["message"]
    assert "private-key" not in error["message"]
    assert not error["retryable"]
