"""Compatibility shim so ``ragas`` can be imported in this environment.

``ragas`` 0.4.x hard-imports ``ChatVertexAI`` from
``langchain_community.chat_models.vertexai`` at module load time, purely to add it to a
static ``isinstance()`` list of chat models that support n-completions. Recent
``langchain-community`` releases removed that module (Vertex AI integrations moved to the
standalone ``langchain-google-vertexai`` package), so ``import ragas`` crashes with
``ModuleNotFoundError`` before this project -- which never uses Vertex AI -- gets a chance
to do anything.

Pinning an older, compatible ``langchain-community`` is not viable: it would drag in an
older ``langchain-core`` that conflicts with the modern ``langchain``/``langchain-openai``/
``langchain-google-genai``/``langchain-groq`` versions every other model integration in
this project depends on. Instead, register a harmless stub module in ``sys.modules`` before
``ragas`` imports it, leaving every installed package version untouched.
"""

from __future__ import annotations

import sys
import types


def ensure_ragas_importable() -> None:
    """Call once, before the first ``import ragas``, to work around the missing module."""
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401

        return
    except ImportError:
        pass

    stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # never instantiated: this project uses google_genai, not Vertex AI
        pass

    stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = stub
