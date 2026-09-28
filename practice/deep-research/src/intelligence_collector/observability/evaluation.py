from __future__ import annotations

from intelligence_collector.observability._ragas_compat import ensure_ragas_importable

ensure_ragas_importable()


def faithfulness_score(answer: str, contexts: list[str]) -> float:
    """Deterministic faithfulness baseline used by the repeatable local evaluation."""
    if not answer.strip() or not contexts:
        return 0.0
    words = {word.strip(".,:;!?()[]").lower() for word in answer.split() if len(word) > 3}
    supported = {word.strip(".,:;!?()[]").lower() for text in contexts for word in text.split()}
    return round(len(words & supported) / max(len(words), 1), 4)

