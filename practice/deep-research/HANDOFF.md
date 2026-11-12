# Deep Research: handoff for the next coding agent

State as of 2026-10-08, branch `feat/deep-research`, last commit `a1c1da7` plus the uncommitted
report verifier (section 6b) and the two kinds of report feedback. 42 tests pass, ruff is clean,
10 commits are ahead of `gitlab/feat/deep-research` (not pushed). `src/deep_research/search.py` has an **uncommitted formatting-only
change made by the user**: leave it alone and never stage it with your own work.

All paths below are relative to `practice/deep-research/`.

## 1. What this is

A multi-agent deep-research app on LangGraph with a Streamlit UI. A user asks a question; the app
clarifies it, writes a research question (the "brief") with acceptance criteria, plans topics, lets
a human approve the plan and criteria, researches each topic with web search (Tavily MCP) and
retrieval (Chroma), writes a cited report, verifies it sentence by sentence against the cited
sources (build-verify-fix), scores it with RAGAS, and shows it as the chat answer. The user then
publishes it (saved as Markdown, the run ends) or sends a follow-up message, as in a chat.

```
guardrail → clarify → write_brief → planner → review_plan ⏸ → supervisor ⇄ supervisor_tools
          → write_report ⇄ verify → evaluate → review_report ⏸ → publish
                                                  └─ follow-up → triage → write_report | supervisor
⏸ = LangGraph interrupt (waits for the user). Plan feedback loops to planner. verify sends a report
with problems back to write_report at most twice. triage sends a follow-up that only changes the
presentation to write_report, and one that needs new facts to supervisor first.
```

## 2. Run, test, lint

```bash
uv sync                                                         # once
uv run --no-sync --with pytest --with ruff python -m pytest tests -q -p no:cacheprovider
uv run --no-sync --with ruff ruff check src tests app.py
uv run --no-sync --with playwright python .claude/skills/run-deep-research/driver.py offline   # UI, scripted, no keys
uv run streamlit run app.py                                     # live, needs .env
```

- `pytest` and `ruff` are not in `.venv`; always use `--with`.
- A warning about `VIRTUAL_ENV=...deep_research_from_scratch\.venv` is harmless (another venv is active).
- The skill `.claude/skills/run-deep-research/SKILL.md` documents the UI driver (offline and live).
- Streamlit caches the app (`@st.cache_resource`): restart it after code or prompt changes.
- **Never read, print or grep `.env`.** Load settings with `Settings.from_env()` in code.

## 3. Code map

| File | Role |
|---|---|
| `app.py`, `src/deep_research/ui.py` | Streamlit UI: chat, plan review, the report as the answer (source check, RAGAS scores, Publish), follow-ups in the chat box |
| `runner.py` | `DeepResearch`: builds the graph (SQLite checkpointer, Langfuse config), `ask` / `resume` / `view`; input goes through `check_input` |
| `orchestrator.py` | `build_graph(settings, reasoning, standard, search, store, checkpointer=None, scorer=None)` wires every node |
| `state.py` | `ResearchState`, `ResearcherState`, structured outputs, tool schemas |
| `prompts.py` | Every prompt, grouped by agent |
| `config.py` | `Settings` / `ModelSpec` from env (`INTELLIGENCE_*`) |
| `models.py` | `Tier`: primary model plus optional fallback; `.structured()` (Groq: strict `json_schema`, since gpt-oss renames forced tools), `.tools()`, `.plain()` |
| `search.py` | `TavilySearch` (MCP), `normalize_url`, `clean`, `prepare_pages`, `format_passages`, `PageStore` (Chroma, cosine) |
| `tracing.py` | Langfuse: one trace per thread, scores |
| `agents/guardrail.py` | Deterministic `check_input` plus LLM safety check (first node) |
| `agents/scope.py` | `clarify`, `write_brief` (brief plus acceptance criteria), `triage` (follow-up: revise or research) |
| `agents/planner.py` | Plan topics; revises on reviewer feedback |
| `agents/human.py` | `review_plan` (topics and criteria), `review_report` (publish or follow-up) (interrupts) |
| `agents/supervisor.py` | Plan round, follow-up rounds, parallel researchers, research requested by a follow-up message |
| `agents/researcher.py` | Per-topic research subgraph, topic note (`finalize`, `render_note`, `failed_note`) |
| `agents/writer.py` | Final report, fixes and revisions; `with_sources` builds the Sources section in code |
| `agents/verifier.py` | Build-verify-fix: per-sentence citation check and acceptance criteria (section 6b) |
| `agents/evaluator.py` | RAGAS scoring of the final report |
| `agents/publisher.py` | Writes `exports/<timestamp>-<thread>.md` |
| `agents/utils.py` | `CITATION`, `SENTENCE_END`, `merge_notes`, `passages_by_url`, `today`, `bullets`, `thread_id` |
| `tests/fakes.py` | Scripted tiers, search, store, `fake_scorer`, `make_app` (used by tests and the offline UI) |

Model tiers: **reasoning** = clarify, brief, planner, supervisor, verifier, triage; **standard** =
guardrail, researcher, topic report, writer. The user's live `.env` currently runs `gpt-5.4-nano`
(OpenAI) for both tiers; `.env.example` shows other providers.

## 4. How a run works

### Supervisor (`agents/supervisor.py`)
- **Plan round, no model call**: on the first turn `supervisor` creates one `ConductResearch` call
  per approved topic (ids `plan-1`…) and sets `supervisor_iterations = 0`.
- `supervisor_tools` runs every requested topic in a `ThreadPoolExecutor(max_workers=max_concurrent_researchers)`.
  The pool is a queue (a free worker takes the next topic); the node returns when the whole round is
  done (barrier), so the model reviews all reports of a round together.
- Later turns: the model reads the topic reports and delegates follow-ups or calls `ResearchComplete`.
- `next_step` stops on: no tool calls, `ResearchComplete`, or `supervisor_iterations >= 6`
  (`max_supervisor_iterations`, hard-coded). A `Think`-only turn counts as an iteration. On the 6th
  turn any delegated topics are dropped (the turn is wasted).
- **Follow-up that needs research**: `triage` sets `research_request` and resets
  `supervisor_iterations` to 0. The next `supervisor` turn answers the tool calls that closed the
  research with ToolMessages (OpenAI rejects unanswered tool calls), adds the request as a
  HumanMessage (`SUPERVISOR_MORE`), and lets the model delegate. Then the writer revises with the
  request as feedback and the new notes.

### Researcher (`agents/researcher.py`), one subgraph per topic
`researcher ⇄ researcher_tools → report`. The model calls `WebSearch(query)` or `Think`; it stops by
itself or after `max_researcher_searches` (5). Each `WebSearch`:
1. Tavily search (5 results, `include_raw_content=True`, keeps `score`).
2. `prepare_pages`: normalize URL, `clean()` page noise, fall back to Tavily's snippet when the
   cleaned page is under 500 chars (`full_text=False`), drop results under 200 chars, merge results
   that normalize to the same URL.
3. Source registry `sources[url] = {id, title, full_text, passages}`: a new page gets the next `[n]`;
   a page seen again keeps its number.
4. `PageStore.add` (chunk 1000/100, embed, upsert; chunk id = `sha1(run_id|url|i)`, no topic in the id).
5. Results under 8000 chars are shown whole; otherwise `PageStore.query` returns the top `retrieval_k`
   (6) chunks of this search's URLs, at most 2 per URL. Indexing/retrieval errors fall back to snippets.
6. The passages shown are appended to `sources[url]["passages"]` and rendered by `format_passages`
   as `[n] Title — URL (full text|snippet only)`.

`report` node: one structured call (`TopicReport`: findings, answered, missing_parts, coverage),
then `finalize` in code:
- drops lines that look like headings or source lines (they echo web content);
- drops sentences that cite only unknown numbers; removes unknown numbers from the others;
- an answered item with no valid citation becomes an essential missing part;
- dedupes missing parts (case/space-insensitive; essential wins);
- coverage cannot be `full` with an essential gap; `minimal` when nothing is answered;
- **status is computed in code**: `failed` with no valid citation, `complete` when coverage is full
  and no essential gap is left, else `partial`.

The topic note is data: `{"report": text, "sources": [{id, title, url, full_text, passages}]}`
(cited sources only). `render_note` adds the Citations section for the supervisor's ToolMessage.
Failures become `failed_note(reason)`, never exceptions.

### Writer, verifier and evaluator
- `merge_notes` gives each URL one run-wide number from the notes' `sources` data and rewrites the
  notes' citations. The writer cites those numbers; **code** removes unknown numbers, cuts any
  Sources list the model wrote, and appends `## Sources` with only the cited sources (`with_sources`).
- The writer sees the acceptance criteria. It rewrites the previous report (without Sources) in two
  cases: the verifier's issues (`REPORT_FIX`, takes precedence) or the user's follow-up
  (`REPORT_REVISION`, findings only; facts the findings lack go to Limitations).
- `verify` (section 6b) sends the report back at most twice, then cleans it in code.
- `evaluate` (RAGAS, see section 6) never changes the route; a failure is stored as `{"error": ...}`.

### After the report: publish or follow up (`agents/human.py`, `agents/scope.py`, `ui.py`)
- The verified report is shown as the assistant's chat answer with the source check, RAGAS scores
  and a **Publish report** button. Publishing is the only permanent write (HITL gate) and ends the run.
- Any message in the chat box is a follow-up. `review_report` adds the report and the message to
  `messages` (the chat history) and resets `fix_rounds`, so the new report gets two fixes.
- `triage` (reasoning tier, `FollowUp`) reads the brief, the researched topics, the report and the
  message: **revise** (format, structure, length, tone, language, more detail on what the report
  covers) goes to `write_report`, findings only, and `verify` deletes anything added without
  support; **research** (facts the report lacks) goes to `supervisor` first (see above).
- Unmet criteria are shown under the answer, suggesting a research follow-up.

## 5. Invariants: do not break these

1. **Sources travel as data, never parsed back from rendered text.** Topic reports quote web content;
   three security reviews found citation spoofing whenever text was parsed (`merge_notes` once read
   the `## Citations` section). Keep numbering and URLs in `note["sources"]`.
2. Page titles are collapsed to one line (`prepare_pages`) because they are rendered into source lines.
3. Status and citations are decided by code, not by the model.
4. The Tavily MCP URL carries the API key: `TavilySearch.search` re-raises a clean error `from None`.
5. Researchers, the verifier and the evaluator never raise into the graph; they return failed notes
   or error dicts.
6. `max_concurrent_researchers` means concurrent workers, not "topics per turn".

## 6. RAGAS evaluator (`agents/evaluator.py`), added in `26f9b05`

- Metrics: `Faithfulness` (report vs the passages behind every cited source of the notes, deduped)
  and `AnswerRelevancy` (report vs brief, project embeddings through an adapter). The report is
  scored without its Sources section and `[n]` markers. Scores go to state and Langfuse as
  `faithfulness`, `answer_relevancy`.
- **Judge: Gemini through google-genai** (`INTELLIGENCE_EVALUATION_MODEL`, default
  `gemini-flash-latest`, reads `GOOGLE_API_KEY`), apart from the research tiers so the models that
  write the report do not grade it. Embeddings for AnswerRelevancy are the project's
  (`INTELLIGENCE_EMBEDDING_*`).
- `_ragas_llm` builds `InstructorLLM` on `instructor.from_genai(client, use_async=True)` itself:
  ragas 0.4.3's `llm_factory` wraps a google-genai client for sync calls only, and the metrics call
  `agenerate`. Needs `instructor[google-genai]` (pulls `jsonref`). Instructor ignores ragas' 1024
  `max_output_tokens` for Gemini, so the model's own output limit applies (no truncation). The
  old JAILBREAK safety-setting bug is fixed in instructor 1.17.
- `langchain-community==0.4.1` stays pinned in `pyproject.toml` (later versions break `import ragas`).
  A new client per call (async clients cannot be reused across `asyncio.run` loops).
- Smoke test 2026-10-08 (2 claims, 1 supported): `gemini-3.1-flash-lite` and `gemini-flash-latest`
  gave faithfulness 0.5 (correct), 26 s and 84 s. `gemini-2.5-flash` returns 404 for new users,
  `gemini-3.8-flash` returned 503 (high demand), `gemini-3.5-flash` 429 (no quota on this key).
  `gemini-flash-latest` is an alias that moves; pin a version for trend comparisons once one works.
- Measured with the old OpenAI judge (`gpt-5.4-nano`) on one report: 9 LLM calls, about $0.022, about
  106 s; faithfulness 0.87, answer relevancy 0.59 (lower on long reports by construction; use it for
  trends only). Not measured with Gemini yet; scores across judges are not comparable.
- RAGAS faithfulness ignores citations (a claim supported by any passage counts), so it cannot catch
  a claim attributed to the wrong `[n]`. The verifier (6b) does; RAGAS stays as the measurement.

## 6b. Verifier: build-verify-fix (`agents/verifier.py`), uncommitted

- Acceptance criteria: 2-5 questions written with the brief (`ResearchBrief.criteria`), editable at
  plan review, given to the writer and the verifier.
- `parse` splits the report body into sentences (table rows whole; headings, table rules and code
  blocks skipped). One `ReportCheck` call on the **reasoning** tier gets the numbered sentences
  `S<n>`, the criteria `C<n>`, and the passages of **only the sources the sentences cite**
  (`passages_by_url`; a cited source with no passages shows "(no passage of this source was read)",
  which covers T1 at report level). Verdicts: supported / unsupported / no_claim; criteria met or not.
- Code decides: any unsupported sentence or unmet criterion while `fix_rounds < MAX_FIX_ROUNDS` (2)
  → `verification = {"issues": [...]}` and back to `write_report`. Otherwise the unsupported
  sentences left are deleted (`remove`; emptied lines and sections go too), Sources are rebuilt, and
  `verification = {"support", "rounds", "removed", "unmet"}`. Unmet criteria cannot be fixed in code;
  they are shown under the answer.
- Langfuse scores: `citation_support` per draft (supported / claims), `sentences_removed`.
- Cost, not measured yet: one reasoning call per draft, up to 3 per report. The writer fixes from the
  findings, not from the passages, so a finding that misstates its source cannot be fixed by
  rewording; it is deleted after the second round.

## 7. Configuration

Env (see `.env.example`): `INTELLIGENCE_{REASONING,STANDARD}_{MODEL,PROVIDER,KWARGS,FALLBACK_MODEL,FALLBACK_PROVIDER,TIMEOUT}`,
`TAVILY_API_KEY`, `TAVILY_MCP_URL`, `INTELLIGENCE_MAX_CONCURRENT_RESEARCHERS` (3),
`INTELLIGENCE_MAX_SEARCH_CALLS` (5), `INTELLIGENCE_RETRIEVAL_K` (6), `INTELLIGENCE_CHROMA_PATH`,
`INTELLIGENCE_EMBEDDING_{MODEL,PROVIDER}`, `INTELLIGENCE_EVALUATION_MODEL`, `INTELLIGENCE_CHECKPOINT_PATH`, `INTELLIGENCE_OUTPUT_DIR`,
`INTELLIGENCE_LANGFUSE_ENABLED`, `LANGFUSE_*`.

Constants in code: `max_supervisor_iterations` 6, `search_max_results` 5 (`config.py`);
`MIN_FULL_TEXT` 500, `MIN_TEXT` 200, `MAX_CHUNKS_PER_URL` 2 (`search.py`); `SMALL_RESULTS_CHARS` 8000
(`researcher.py`); `TIMEOUT_MS` 120000 (`evaluator.py`); `MAX_FIX_ROUNDS` 2 (`verifier.py`).

## 8. Tests

- `tests/test_deep_research.py`: end-to-end graph on fakes (plan/report review, plan round runs all
  topics before the model, criteria edits, follow-ups triaged to rewrite or research, clarification, guardrail).
- `tests/test_verifier.py`: sentence parsing and removal, issues and evidence sent to the judge,
  clean-up after two rounds, error handling, the fix loop in the graph.
- `tests/test_researcher.py`: WebSearch pipeline, source numbering, fallbacks, `finalize` rules.
- `tests/test_writer.py`: `merge_notes`, Sources section, spoofing cases.
- `tests/test_evaluator.py`: evaluator inputs, error handling, the Gemini judge client (fake scorer).
- `tests/test_models.py`: Groq structured output uses strict `json_schema`, not a tool call.
- `FakeTier.structured` answers by schema name; a callable value answers from the prompt.
- RAGAS itself is not exercised by tests; it needs a real model.

## 9. Measuring a live run

Every run is one Langfuse trace (session = thread id). Generations carry `metadata.langgraph_node`.
Useful patterns, used for every analysis so far:

```python
from deep_research.config import Settings; Settings.from_env()
from langfuse import get_client; lf = get_client()
lf.api.trace.list(limit=5)                                     # latest traces, total_cost
r = lf.api.observations.get_many(trace_id=TID, type="GENERATION", limit=100, cursor=None,
                                 fields="core,basic,metadata,model,usage,io",
                                 expand_metadata="langgraph_node")  # page with r.meta.cursor
# per node: o.metadata["langgraph_node"], o.usage_details (input, input_cache_read, output, output_reasoning), o.total_cost
lf.api.scores.get_many(trace_id=TID)                           # faithfulness / answer_relevancy
```

State of a thread: `SqliteSaver(sqlite3.connect(settings.checkpoint_path)).get_tuple({"configurable": {"thread_id": TID}}).checkpoint["channel_values"]`
(`plan`, `criteria`, `supervisor_messages`, `supervisor_iterations`, `notes`, `report`, `verification`, `fix_rounds`, `evaluation`).
Rough `gpt-5.4-nano` rates derived from trace costs: input about $0.20/M, output about $1.24/M.

Baselines (question "Explain how the LangGraph Functional API differs from the Graph API"):

| Version | Cost | Tokens | Researchers | Supervisor turns | Faithfulness / relevance (old LLM judge) |
|---|---|---|---|---|---|
| before `f77f6a4` | $0.058 | 206,763 | 12 | 6/6 | 0.82 / 0.68 |
| `f77f6a4`, long plan topics | $0.081 | 315,303 | 18 | 6/6 | 0.86 / 0.70 |
| `f77f6a4` + short topics | $0.029 | 69,890 | 5 | 2 | 0.86 / 0.74 |

`26f9b05` (RAGAS), `07cbb9d` / `a1c1da7` (brief, planner and report prompts) and the verifier have
**not** been measured live yet. Do that first: cost and time of `verify`, fix rounds used,
`citation_support` per draft, sentences removed, and whether RAGAS faithfulness rises above 0.9.

## 10. Known issues and backlog (evidence-based, highest value first)

1. **Measure the latest prompts live** (brief as one question, planner rules by question type,
   limitations section). Offline tries gave 1 topic for "what is X / best practices for X", 2 topics
   for "compare A and B", but a run of "What is langgraph?" still showed 4 dated topics; it may have
   run before Streamlit was restarted. Check plan length, supervisor turns and cost.
2. **Cited-but-unseen sources are accepted in topic notes** ("T1"). `finalize` accepts any `[n]` in
   the registry, including pages whose passages were never shown. Seen in real data (2 of 51 claims).
   The verifier now catches such claims in the final report; the supervisor still sees them.
   Fix if needed: accept only sources with non-empty `passages` in `finalize`.
3. **Claims not supported by their cited source** ("T4"). A per-claim judge on one run: 59%
   supported, 22% partial, 14% unsupported, 4% unseen source; several were fabricated "quotes".
   Handled for the final report by the verifier (6b); topic notes are not checked. RAGAS on topic
   reports was tried offline and rejected (scores ~0.97, citation-blind, +70% cost). Embedding
   similarity cannot decide alone (supported median 0.65 vs unsupported 0.44; any threshold drops
   35%+ of good claims).
4. **Gap-chasing spiral** is held back only by prompts. Follow-ups written by the supervisor ask for
   "exact …", researchers then mark that exactness essential. Safety net if it returns: cap research
   rounds in `next_step` (count AIMessages with `ConductResearch`) and route to `write_report` without
   another supervisor call. The user chose prompt-only so far.
5. **Chroma has little value today**: about 8% of embedded chunks are ever shown, and nothing reads
   the store after the search that wrote it. Options: in-memory selection and drop Chroma, or give it
   real readers (look up before declaring a part missing, per-claim checks, cross-topic reuse).
6. Cost drivers: the `report` node output (findings, 30–48% of a run) and supervisor reasoning
   tokens; RAGAS now adds about 60%.
7. Old checkpoints store `notes` as strings; resuming such a thread to `write_report` fails. New
   threads are fine.
8. `Think`-only supervisor turns count as iterations; researchers have no cap on `Think` turns other
   than `recursion_limit=50`.

## 11. Decisions already made: do not relitigate without new data

- Keep the researcher tool loop (`WebSearch`/`Think`), not a fixed query pipeline.
- Rounds with a barrier (as in langchain-ai/deep_research_from_scratch); no continuous lanes, no asyncio.
- Code researches the approved plan; the model only plans follow-ups.
- Rejected: sending only the short report to the supervisor (saves about 3–5%), similarity
  thresholds on retrieval (scores sit in a narrow 0.5–0.7 band; top-k already selects), RAGAS on
  topic reports, URL dedup across topics (deferred; Tavily bills per request, not per page).

## 12. Working conventions

- Commit messages: one-line subject only, no body, no `Co-Authored-By` trailer. Commit only when asked.
- Lean changes: delete what a change makes obsolete in the same change; no speculative options.
- Docs in English. `specs/` was deleted on purpose; there is no spec for the current code.
- Measure with Langfuse before and after changes that affect cost or behavior.
- Shell quirk on this machine: inserting code through `python - <<'PY'` with `"\n"` inside triple
  quotes writes a real newline into the target file and breaks string literals. Prefer the editor
  tool for strings that contain `\n`.
