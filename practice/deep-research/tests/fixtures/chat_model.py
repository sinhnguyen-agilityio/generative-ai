"""A real LangChain ``BaseChatModel`` around the duck-typed fakes in ``fake_models.py``.

The fakes decide what to answer; this adapter makes the call go through LangChain's chat-model
machinery, so callbacks see ``on_chat_model_start`` and Langfuse records a ``generation`` with
the model name and token usage, exactly as with a provider model.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

USAGE = {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17}


class FakeChatModel(BaseChatModel):
    """Answers with `delegate.invoke(messages)`, or with the next scripted payload."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "fake-chat-model"
    max_tokens: int | None = None  # lets ModelRouter apply its per-role output caps
    delegate: Any = None
    scripted: list[Any] = Field(default_factory=list)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _calls: list[int] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    @property
    def calls(self) -> int:
        return len(self._calls)

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        if self.delegate is not None and hasattr(self.delegate, "bind_tools"):
            self.delegate.bind_tools(tools, tool_choice=tool_choice)
        return self

    def with_structured_output(self, schema, **kwargs):
        return self  # the router parses the JSON text answer

    def _answer(self, messages) -> AIMessage:
        if self.delegate is not None:
            reply = self.delegate.invoke(messages)
        else:
            with self._lock:
                reply = self.scripted.pop(0) if len(self.scripted) > 1 else self.scripted[0]
        if isinstance(reply, AIMessage):
            return reply
        return AIMessage(content=reply if isinstance(reply, str) else json.dumps(reply))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        with self._lock:
            self._calls.append(1)
        reply = self._answer(messages)
        message = AIMessage(
            content=reply.content,
            tool_calls=reply.tool_calls,
            usage_metadata=USAGE,
            response_metadata={"model_name": self.model_name},
        )
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"model_name": self.model_name},
        )
