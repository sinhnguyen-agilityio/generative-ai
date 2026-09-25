"""URL identity and run-level tracking so each web page is captured and indexed once per run.

* ``normalize_url`` gives one identity to URL spellings of the same page (case of scheme/host,
  default ports, fragments, trailing slashes, tracking parameters, query order).
* ``UrlRegistry`` records which normalized URLs a run has already captured, persisted in the VFS
  at ``research/url_registry.json`` so parallel tasks, gap research, and resumed runs reuse the
  first capture instead of storing and indexing the page again.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REGISTRY_PATH = "research/url_registry.json"
_DEFAULT_PORTS = {"http": 80, "https": 443}
_TRACKING_PARAMS = frozenset({"gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ref_src"})


def _is_tracking(name: str) -> bool:
    return name.lower().startswith("utm_") or name.lower() in _TRACKING_PARAMS


def normalize_url(url: str) -> str:
    """Canonical identity of a URL; path case is preserved because paths are case-sensitive."""
    text = str(url).strip()
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return text.rstrip("/")
    if not parts.scheme or not parts.hostname:
        return text.rstrip("/")
    scheme = parts.scheme.lower()
    netloc = parts.hostname.lower()
    if port and port != _DEFAULT_PORTS.get(scheme):
        netloc += f":{port}"
    query = urlencode(sorted(
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking(name)
    ))
    return urlunsplit((scheme, netloc, parts.path.rstrip("/"), query, ""))


class UrlRegistry:
    """Run-scoped record of captured URLs; mutate it from one thread (the supervisor's)."""

    def __init__(self, entries: dict[str, dict] | None = None):
        self._entries = dict(entries or {})
        self._dirty = False

    @classmethod
    def load(cls, vfs, run_id: str) -> "UrlRegistry":
        try:
            return cls(json.loads(vfs.read(run_id, REGISTRY_PATH)))
        except FileNotFoundError:
            return cls()

    def get(self, url: str) -> dict | None:
        return self._entries.get(normalize_url(url))

    def register(self, url: str, entry: dict) -> None:
        key = normalize_url(url)
        if key not in self._entries:
            self._entries[key] = entry
            self._dirty = True

    def __len__(self) -> int:
        return len(self._entries)

    def save(self, vfs, run_id: str):
        """Persist only when something new was registered; returns the artifact ref or None."""
        if not self._dirty:
            return None
        self._dirty = False
        return vfs.write_json(run_id, REGISTRY_PATH, self._entries, "url_registry")
