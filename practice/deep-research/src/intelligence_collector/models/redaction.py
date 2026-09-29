"""Secret redaction for anything the model layer sends to traces or error messages."""

import re

# Dict keys whose value is always replaced, whatever the value looks like.
SECRET_MARKERS = ("api_key", "apikey", "secret", "password", "token", "authorization")

_KEY_VALUE = re.compile(r"(?i)\b(api[_-]?key|secret|password|token|authorization)\s*[:=]\s*([^\s,;]+)")
_BEARER = re.compile(r"(?i)\bbearer\s+[a-z0-9._-]{8,}")


def redact_text(value: object) -> str:
    """Stringify `value` and mask `key=value` secrets and `Bearer <token>` credentials."""
    text = _KEY_VALUE.sub(r"\1=[REDACTED]", str(value))
    return _BEARER.sub("Bearer [REDACTED]", text)


def redact_value(value):
    """Recursively redact a JSON-like structure: secret-named dict keys are masked outright,
    and every string is scrubbed with `redact_text`."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if any(marker in key.lower() for marker in SECRET_MARKERS)
            else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_value(item) for item in value]
    return redact_text(value) if isinstance(value, str) else value
