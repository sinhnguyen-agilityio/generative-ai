"""One research task end to end: lookup → search → persist → lookup → report → evidence bundle.

``SearchTask`` splits the pipeline into steps so ``ResearchSupervisor`` can run the model steps
(``phrasings``, ``search``, ``report``) on worker threads and the storage steps (lookups, which
read the index; persistence, which writes the VFS, URL registry and index) on its own thread,
never overlapping. ``run`` executes every step in order on the calling thread.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

from intelligence_collector.agents.researcher import ResearchAgent, ResearchOutcome
from intelligence_collector.agents.searcher_report import (
    DEFAULT_REPORT_MAX_WORDS,
    LookupFindings,
    bundle_from_report,
    lookup_points,
    passages_block,
    phrasings_for,
    stored_sources,
    task_points,
    write_report,
)
from intelligence_collector.agents.worker import (
    PersistedPages,
    failed_outcome,
    persist_pages,
    research_prefix,
    write_evidence_bundle,
    write_task_failure,
)
from intelligence_collector.contracts import ResearchFindings, ResearchTodo, SearcherReport
from intelligence_collector.models.redaction import redact_error
from intelligence_collector.ports import ArtifactStore, VectorIndex
from intelligence_collector.storage.passage_lookup import PassageLookup
from intelligence_collector.tools.url_tracking import UrlClaims, UrlRegistry


@dataclass
class SearchTask:
    """State of one task moving through the pipeline."""

    vfs: ArtifactStore
    index: VectorIndex | None
    researcher: ResearchAgent
    run_id: str
    todo: ResearchTodo
    question: str
    active_task_ids: set[str]
    thread_id: str = ""
    lookup: PassageLookup | None = None
    report_max_words: int = DEFAULT_REPORT_MAX_WORDS
    phrasings: dict[str, list[str]] = field(default_factory=dict)
    phrasing_failed: bool = True
    found: LookupFindings = field(default_factory=LookupFindings)
    outcome: ResearchOutcome | None = None
    stored: PersistedPages | None = None
    report: SearcherReport | None = None

    @property
    def prefix(self) -> str:
        return research_prefix(self.todo)

    # ----- resume -----

    def restore(self) -> bool:
        """Reload a search finished before an interruption; True when it exists."""
        try:
            payload = json.loads(self.vfs.read(self.run_id, f"{self.prefix}/retrieved_outcome.json"))
        except FileNotFoundError:
            return False
        if payload.get("findings") is not None:
            payload["findings"] = ResearchFindings.model_validate(payload["findings"])
        self.outcome = ResearchOutcome(**payload)
        return True

    def _saved_report(self) -> SearcherReport | None:
        try:
            return SearcherReport.model_validate_json(
                self.vfs.read(self.run_id, f"{self.prefix}/report.json")
            )
        except FileNotFoundError:
            return None

    # ----- steps -----

    def allowed_sources(self, registry: UrlRegistry | None = None) -> set[str]:
        """Sources stored so far for the run's active tasks (FR-013b)."""
        registry = registry or UrlRegistry.load(self.vfs, self.run_id)
        return registry.source_ids(self.active_task_ids)

    def prepare_phrasings(self) -> None:
        """Model step: alternative phrasings for every evidence point (FR-013d)."""
        self.phrasings, self.phrasing_failed = phrasings_for(
            task_points(self.todo),
            self.question,
            [],
            getattr(self.researcher, "router", None),
            run_id=self.run_id,
            thread_id=self.thread_id,
        )

    def lookup_before_search(self, registry: UrlRegistry | None = None) -> None:
        """Storage step: look every point up in pages finished tasks stored, before any web
        search is spent on it."""
        self.found.merge(
            lookup_points(
                task_points(self.todo),
                todo=self.todo,
                question=self.question,
                lookup=self.lookup,
                run_id=self.run_id,
                allowed_source_ids=self.allowed_sources(registry),
                phrasings=self.phrasings,
                phrasing_failed=self.phrasing_failed,
                requested_by=f"searcher:{self.todo.todo_id}",
            )
        )

    def search(self, claims: UrlClaims | None) -> None:
        """Model step: the searcher's web search. Points the lookup already found passages for
        are shown to it; when every point has some, its first search is no longer forced."""
        points = {p.point_id for p in task_points(self.todo)}
        try:
            self.outcome = self.researcher.search(
                self.question,
                todo=self.todo,
                run_id=self.run_id,
                thread_id=self.thread_id,
                task_id=self.todo.todo_id,
                claims=claims,
                known_passages=passages_block(self.found),
                force_first_search=not points <= self.found.points,
            )
        except Exception as exc:
            self.outcome = failed_outcome(
                self.todo, f"Research sub-agent failed: {redact_error(exc)}"
            )

    def save_outcome(self) -> None:
        """Resume copy of the search; pages live in raw/<n>.md."""
        payload = {
            **self.outcome.__dict__,
            "findings": self.outcome.findings.model_dump(mode="json")
            if self.outcome.findings
            else None,
            "sources": [
                {key: value for key, value in source.items() if key != "raw_content"}
                for source in self.outcome.sources
            ],
        }
        self.vfs.write_json(
            self.run_id,
            f"{self.prefix}/retrieved_outcome.json",
            payload,
            "retrieved_outcome",
            task_id=self.todo.todo_id,
        )

    def persist(self, registry: UrlRegistry | None = None) -> None:
        """Storage step: evaluations, accepted pages and their chunks (FR-009, FR-012)."""
        self.save_outcome()
        self.stored = persist_pages(
            self.vfs, self.index, self.run_id, self.todo, self.outcome, registry
        )

    def lookup_before_report(self, registry: UrlRegistry | None = None) -> None:
        """Storage step: look up the points the search did not cover, now including this task's
        own stored pages."""
        uncovered = [
            p for p in task_points(self.todo) if p.point_id not in set(self.outcome.covered_points)
        ]
        self.found.merge(
            lookup_points(
                uncovered,
                todo=self.todo,
                question=self.question,
                lookup=self.lookup,
                run_id=self.run_id,
                allowed_source_ids=self.allowed_sources(registry),
                phrasings=self.phrasings,
                phrasing_failed=self.phrasing_failed,
                requested_by=f"searcher:{self.todo.todo_id}",
            )
        )

    def _save_report(self):
        return self.vfs.write_json(
            self.run_id,
            f"{self.prefix}/report.json",
            self.report.model_dump(mode="json"),
            "searcher_report",
            task_id=self.todo.todo_id,
        )

    def write(self) -> None:
        """Model step: draft and validate the report, saved at once so a resume never drafts
        it again (reused when a saved one exists). Each task writes only its own file."""
        self.report = self._saved_report()
        if self.report is not None:
            return
        sources = stored_sources(
            self.vfs, self.index, self.run_id, self.stored.evidence_sources, self.found
        )
        self.report = write_report(
            self.todo,
            self.question,
            sources,
            getattr(self.researcher, "report_router", None)
            or getattr(self.researcher, "router", None),
            outcome=self.outcome,
            lookup_used=set(self.found.points),
            max_words=self.report_max_words,
            run_id=self.run_id,
            thread_id=self.thread_id,
        )
        self._save_report()

    def finish(self):
        """Storage step: lookups, report and evidence bundle; returns ``(todo, refs, chunks)``.
        The bundle is written last, so it marks the task done."""
        refs = list(self.stored.refs)
        refs.append(
            self.vfs.write_json(
                self.run_id,
                f"{self.prefix}/lookups.json",
                [record.model_dump(mode="json") for record in self.found.records],
                "passage_lookups",
                task_id=self.todo.todo_id,
            )
        )
        refs.append(self._save_report())
        sources = stored_sources(
            self.vfs, self.index, self.run_id, self.stored.evidence_sources, self.found
        )
        fields = bundle_from_report(self.report, self.todo, sources, self.stored.evidence_sources)
        stored = replace(self.stored, refs=refs, evidence_sources=fields.evidence_sources)
        if not fields.evidence_sources:
            return write_task_failure(self.vfs, self.run_id, self.todo, self.outcome, stored)
        return write_evidence_bundle(
            self.vfs,
            self.run_id,
            self.todo,
            self.outcome,
            stored,
            summary=fields.summary,
            sufficient=fields.sufficient,
            unresolved_gaps=fields.unresolved_gaps,
            extra={"report_status": self.report.report_status.value},
        )

    def run(self, claims: UrlClaims | None = None, registry: UrlRegistry | None = None):
        """Every step on the calling thread."""
        if not self.restore():
            self.prepare_phrasings()
            self.lookup_before_search(registry)
            self.search(claims)
        self.persist(registry)
        self.lookup_before_report(registry)
        self.write()
        return self.finish()
