from dataclasses import dataclass
import re


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    reason: str = ""


def validate_request(text: str) -> GuardrailDecision:
    if not isinstance(text, str) or not text.strip():
        return GuardrailDecision(False, "Request cannot be empty.")
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
    injection = r"ignore\s+(all|previous)\s+instructions|reveal\s+(the\s+)?system\s+prompt|jailbreak"
    if re.search(injection, text, re.I):
        return GuardrailDecision(False, "Request contains an unsafe instruction pattern.")
    if re.search(r"\b(kill|murder|bomb|terrorist attack)\b", text, re.I):
        return GuardrailDecision(False, "This request is not supported.")
    return GuardrailDecision(True)
