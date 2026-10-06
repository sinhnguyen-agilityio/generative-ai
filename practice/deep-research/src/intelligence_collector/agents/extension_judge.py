"""Extension judge: decides whether a reviewer's requested section can be written from evidence
already stored for the run (see ``BVFNodes.locate_extension_evidence``).

One structured LLM call (role ``extension_judge``) over the passages the vector index found for
the requested topic. It only judges; it never writes the report or searches.
"""

from __future__ import annotations

from intelligence_collector.ports import ModelRouterPort
from intelligence_collector.agents.prompts import EXTENSION_JUDGE_TEMPLATE
from intelligence_collector.contracts import ExtensionEvidenceVerdict
from intelligence_collector.source_verification import CatalogSource

# Enough of each page to judge whether it explains the topic, while 8 passages stay small.
MAX_PASSAGE_CHARS = 1500


def _format_passages(passages: list[CatalogSource]) -> str:
    return "\n\n".join(
        f"[{source.source_id}] {source.title or 'Untitled'}\nURL: {source.url}\n"
        f"{source.excerpt[:MAX_PASSAGE_CHARS]}"
        for source in passages
    )


class ExtensionJudge:
    """Reads stored passages with an LLM and returns an ``ExtensionEvidenceVerdict``."""

    def __init__(self, router: ModelRouterPort):
        self.router = router

    def judge(
        self,
        question: str,
        request: str,
        topic: str,
        passages: list[CatalogSource],
        *,
        run_id: str = "",
        thread_id: str = "",
    ) -> ExtensionEvidenceVerdict:
        messages = EXTENSION_JUDGE_TEMPLATE.format_messages(
            question=question,
            request=request,
            topic=topic,
            passages=_format_passages(passages),
        )
        return self.router.invoke_messages(
            "extension_judge",
            messages,
            ExtensionEvidenceVerdict,
            run_id=run_id,
            thread_id=thread_id,
        )
