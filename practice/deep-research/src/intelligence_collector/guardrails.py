"""Input guardrails: deterministic checks on every chat message (empty, too long, prompt injection,
credentials) and the contracts of the LLM content-safety check.
"""

from dataclasses import dataclass
import re
import unicodedata

from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class GuardrailDecision:
    """Result of the deterministic request check."""

    allowed: bool
    reason: str = ""


class GuardrailViolation(Exception):
    """Raised by a pre-model guardrail to block a request before it reaches an LLM."""


class ContentSafetyVerdict(BaseModel):
    """Structured output of the pre-model content-safety classifier (see agents/scope.py)."""

    model_config = ConfigDict(extra="forbid")
    allowed: bool
    reason: str = ""


_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")
_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")

# Lookalike letters (Cyrillic, Greek, other confusable scripts) commonly used to evade
# keyword/regex filters by swapping in a visually identical character from another script.
# NFKC alone does not fold these: they are distinct, non-compatibility code points that
# merely render the same as their Latin counterparts.
_CONFUSABLES = str.maketrans(
    {
        "а": "a",
        "А": "A",
        "е": "e",
        "Е": "E",
        "о": "o",
        "О": "O",
        "р": "p",
        "Р": "P",
        "с": "c",
        "С": "C",
        "у": "y",
        "У": "Y",
        "х": "x",
        "Х": "X",
        "і": "i",
        "І": "I",
        "ѕ": "s",
        "Ѕ": "S",
        "ј": "j",
        "Ј": "J",
        "ԁ": "d",
        "ѡ": "w",
        "ѵ": "v",
        "α": "a",
        "Α": "A",
        "β": "b",
        "Β": "B",
        "ο": "o",
        "Ο": "O",
        "ρ": "p",
        "Ρ": "P",
        "κ": "k",
        "Κ": "K",
        "τ": "t",
        "Τ": "T",
        "υ": "u",
        "Υ": "Y",
        "χ": "x",
        "Χ": "X",
        "ⅰ": "i",
        "ⅼ": "l",
        "ⅽ": "c",
        "ⅾ": "d",
        "ⅿ": "m",
    }
)


# Prompt-injection patterns. Each allows a few filler words ("all previous", "your", "the above")
# between the verb and its target so trivial rewording cannot slip past.
_INJECTION_RE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\b(?:\s+\w+){0,4}?\s+"
    r"(?:instructions?|prompts?|rules|directives|guidelines)\b"
    r"|\b(?:reveal|show|print|display|repeat|output|leak|tell\s+me|give\s+me|what\s+(?:is|are))\b"
    r"(?:\s+\w+){0,4}?\s+(?:system|hidden|initial|original|developer)\s+"
    r"(?:prompt|instructions?|message)s?\b"
    r"|\bjailbreak|\bDAN\s+mode\b|\bdeveloper\s+mode\b",
    re.I,
)


def normalize_text(text: str) -> str:
    """Fold homoglyphs, invisible characters, and Unicode compatibility forms to a
    canonical form before any guardrail check, storage, or model call sees the text.

    This closes three attack surfaces on raw user input: evasion (lookalike characters
    slipping banned words past keyword/regex filters), poisoning (zero-width or control
    characters hiding payloads that later get indexed into the vector store), and privacy
    (disguised PII patterns that would otherwise dodge detection regexes).
    """
    if not isinstance(text, str):
        return text
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(_CONFUSABLES)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    normalized = _CONTROL_RE.sub("", normalized)
    return normalized


def validate_request(text: str) -> GuardrailDecision:
    """Reject empty, oversized, prompt-injection or credential-bearing input before it reaches any
    model.
    """
    if not isinstance(text, str) or not text.strip():
        return GuardrailDecision(False, "Request cannot be empty.")
    text = normalize_text(text)
    if len(text) > 20_000:
        return GuardrailDecision(False, "Request is too long.")
    credential = (
        r"\b(?:api[_-]?key|secret|password|token|authorization)\s*[:=]\s*\S+"
        r"|\bbearer\s+[a-z0-9._-]{8,}|\bsk-[a-z0-9_-]{8,}"
    )
    if re.search(credential, text, re.I):
        return GuardrailDecision(
            False,
            "Remove credentials from the request and configure them through the environment.",
        )
    if _INJECTION_RE.search(text):
        return GuardrailDecision(False, "Request contains an unsafe instruction pattern.")
    if re.search(r"\b(kill|murder|bomb|terrorist attack)\b", text, re.I):
        return GuardrailDecision(False, "This request is not supported.")
    return GuardrailDecision(True)
