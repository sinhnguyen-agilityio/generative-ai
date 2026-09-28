"""URL identity and run-level tracking so each web page is captured and indexed once per run.

* ``normalize_url`` gives one identity to URL spellings of the same page (case of scheme/host,
  default ports, fragments, trailing slashes, tracking parameters, query order).
* ``UrlRegistry`` records which normalized URLs a run has already captured, persisted in the VFS
  at ``research/url_registry.json`` so parallel tasks, gap research, and resumed runs reuse the
  first capture instead of storing and indexing the page again.
"""

from __future__ import annotations

import json
import threading
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

    def owners(self) -> dict[str, str]:
        """Normalized URL -> id of the task that first captured it."""
        return {key: str(entry.get("task_id", "")) for key, entry in self._entries.items()}

    def save(self, vfs, run_id: str):
        """Persist only when something new was registered; returns the artifact ref or None."""
        if not self._dirty:
            return None
        self._dirty = False
        return vfs.write_json(run_id, REGISTRY_PATH, self._entries, "url_registry")


class UrlClaims:
    """Which research task first retrieved each URL, shared by concurrently running sub-agents.

    ``UrlRegistry`` only dedupes at persistence time, after every sub-agent has finished, so a page
    parallel sub-agents both found would take a source slot in each of them. Claiming at search
    time means the first task to retrieve a page owns it; any other task links to it instead of
    spending its own source budget on it (``UrlRegistry`` still stores and indexes it only once).
    Seeded from the persisted registry, so gap research and resumed runs treat pages captured
    earlier in the run the same way.
    """

    def __init__(self, owners: dict[str, str] | None = None):
        self._owners = dict(owners or {})
        self._lock = threading.Lock()

    def claim(self, url: str, task_id: str) -> str:
        """Atomically claim a page; returns its owner (``task_id`` itself if it was unclaimed)."""
        key = normalize_url(url)
        with self._lock:
            return self._owners.setdefault(key, task_id)

    def owner(self, url: str) -> str | None:
        """Current owner without claiming — for a task with no budget left to keep the page."""
        key = normalize_url(url)
        with self._lock:
            return self._owners.get(key)
