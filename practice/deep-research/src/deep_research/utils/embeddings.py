"""Create embedding clients for the configured provider."""

from __future__ import annotations


def _embeddings(model: str, provider: str):
    """Initialize the embedding client for the configured provider.

    Args:
        model: Embedding model identifier.
        provider: openai selects OpenAI; other values select Google GenAI.

    Returns:
        LangChain embedding client.
    """
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=model)
