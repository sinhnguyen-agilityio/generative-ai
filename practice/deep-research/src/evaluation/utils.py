"""Configure RAGAS judge and embedding interfaces."""

from __future__ import annotations

TIMEOUT_MS = 120_000


def _ragas_llm(model: str):
    """A Gemini judge on an async google-genai client (reads GOOGLE_API_KEY).

    ragas 0.4.3's ``llm_factory`` wraps a google-genai client for sync calls only, and the metrics
    call ``agenerate``, so the async Instructor client is built here. Instructor drops ragas'
    1024-token output cap for Gemini, so a long claim list is not truncated.

    Args:
        model: Gemini judge model identifier.

    Returns:
        Async InstructorLLM using the google-genai client and GOOGLE_API_KEY.
    """
    import instructor
    from google import genai
    from google.genai import types
    from ragas.llms.base import InstructorLLM

    client = genai.Client(http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    return InstructorLLM(client=instructor.from_genai(client, use_async=True), model=model, provider="google")


def _ragas_embeddings(embeddings):
    """The project's LangChain embeddings behind the interface ragas' collection metrics expect.

    Args:
        embeddings: LangChain embedding client with synchronous and asynchronous query methods.

    Returns:
        Adapter implementing the RAGAS text-embedding interface.
    """
    from evaluation.ragas_embeddings import RagasEmbeddings

    return RagasEmbeddings(embeddings)
