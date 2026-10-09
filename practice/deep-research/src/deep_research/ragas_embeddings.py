"""LangChain embedding adapter for RAGAS metrics."""

from ragas.embeddings import BaseRagasEmbedding


class RagasEmbeddings(BaseRagasEmbedding):
    """Adapt LangChain query embeddings to the RAGAS embedding interface."""

    def __init__(self, embeddings):
        """Retain the embedding client used for synchronous and async queries.

        Args:
            embeddings: LangChain client providing embed_query and aembed_query.
        """
        self.embeddings = embeddings

    def embed_text(self, text: str, **kwargs) -> list[float]:
        """Embed one query through the wrapped LangChain client.

        Args:
            text: Text to embed.
            kwargs: RAGAS interface options; not forwarded to the client.

        Returns:
            Embedding vector.
        """
        return self.embeddings.embed_query(text)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        """Embed one query asynchronously through the wrapped LangChain client.

        Args:
            text: Text to embed.
            kwargs: RAGAS interface options; not forwarded to the client.

        Returns:
            Embedding vector.
        """
        return await self.embeddings.aembed_query(text)
