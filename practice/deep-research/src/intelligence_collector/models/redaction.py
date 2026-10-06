"""Secret redaction shared by every error message, log line and trace that may echo a
credential: model and compression errors, MCP (Tavily) errors, embedding failures and traces."""

import re

# Dict keys whose value is always replaced, whatever the value looks like.
SECRET_MARKERS = ("api_key", "apikey", "secret", "password", "token", "authorization")

_KEY_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|secret|password|token|authorization)\s*[:=]\s*([^\s,;]+)"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[a-z0-9._-]{8,}")


# URL query parameters whose name contains a secret marker, e.g. Tavily's `?tavilyApiKey=...`.
_QUERY_SECRET = re.compile(
    r"(?i)([?&][\w.-]*(?:api[_-]?key|secret|password|token)[\w.-]*=)[^&\s'\"]+"
)


def redact_text(value: object) -> str:
    """Stringify `value` and mask `key=value` secrets, secret URL query parameters and
    `Bearer <token>` credentials."""
    text = _QUERY_SECRET.sub(r"\1[REDACTED]", str(value))
    # Bearer first: in "Authorization: Bearer <token>" the key/value rule would mask only the
    # word "Bearer" and leave the token.
    text = _BEARER.sub("Bearer [REDACTED]", text)
    return _KEY_VALUE.sub(r"\1=[REDACTED]", text)


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


def redact_error(exc: BaseException) -> str:
    """``"<ExceptionType>: <message>"`` with secrets masked, for error fields and log lines."""
    return f"{type(exc).__name__}: {redact_text(exc)}"
