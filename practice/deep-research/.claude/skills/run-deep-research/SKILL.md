---
name: run-deep-research
description: Run, start, drive and screenshot the deep-research Streamlit app (multi-agent research with plan review, a verified report and chat follow-ups), offline on scripted models or live with .env. Use when asked to run the app, take a screenshot of its UI, check a UI or graph change end-to-end, or run its tests.
---

A Streamlit app (`app.py` → `src/deep_research/ui.py`). Drive it with
`.claude/skills/run-deep-research/driver.py`: it launches Streamlit on the project `.venv`, drives
headless Chromium through Playwright, and screenshots every step. Prefer **offline** mode: same
UI and graph, scripted models/search/store from `tests/fakes.py`, no keys, no cost, ~10 s.

All paths are relative to `practice/deep-research/`. Verified on Windows 11 (Git Bash) with uv.

## Prerequisites

- `uv` on PATH and the project `.venv` (created with `uv sync`; not re-run in this session).
- Playwright's Chromium in the local cache. It was already present here; on a clean machine
  install it once with `uv run --with playwright python -m playwright install chromium`.
- Live mode only: a filled-in `.env` (see `.env.example`). Never print or read it in a session.

## Run (agent path)

Offline, full flow: ask → approve plan → report → follow-up → publish, one screenshot per step:

```bash
uv run --no-sync --with playwright python .claude/skills/run-deep-research/driver.py offline
```

Expected tail: `saw 'Report published to'`, `console errors: none`, `streamlit exceptions: none`,
exit 0. Screenshots: `.claude/skills/run-deep-research/shots/{1-landing,2-plan-review,3-report,4-follow-up,5-published}.png` (offline also sends one follow-up, "Make it shorter"; live skips it).
Streamlit log: `.claude/skills/run-deep-research/streamlit-8599.log`. Both are git-ignored.

Live app with `.env`, landing page only (free: builds models, Tavily client and Chroma, sends nothing):

```bash
uv run --no-sync --with playwright python .claude/skills/run-deep-research/driver.py live --shots .claude/skills/run-deep-research/shots/live
```

`live --question "..."` runs the same full flow on real models, Tavily and Langfuse. It costs
money, needs the user's go-ahead, and has **not** been verified yet (step timeout 900 s).

| flag | effect |
|---|---|
| `offline` / `live` | entry point: `offline_app.py` (fakes) or `app.py` (real) |
| `--question TEXT` | live: also run the full flow with this question |
| `--port N` | default 8599; the driver refuses a port that already serves Streamlit |
| `--shots DIR` | screenshot directory |
| `--timeout S` | seconds per step (offline 30, live 900) |

To change what the offline models answer (topics, report text, evaluation scores, a blocked
request), edit `make_app` in `tests/fakes.py`; tests and driver share it.

## Run (human path)

```bash
.venv/Scripts/python.exe -m streamlit run app.py --server.headless true --server.port 8599 --browser.gatherUsageStats false
curl -s http://localhost:8599/_stcore/health   # -> ok
```

Open http://localhost:8599. Stop it by killing the port's listener (next section).

## Stop a leftover server (Windows)

```bash
PID=$(netstat -ano | grep ':8599 ' | grep LISTENING | awk '{print $5}' | head -1); taskkill //PID $PID //T //F
```

## Test

```bash
uv run --no-sync --with pytest --with ruff python -m pytest tests -q -p no:cacheprovider
uv run --no-sync --with ruff ruff check src tests app.py
```

Expected: `11 passed`, `All checks passed!`.

## Gotchas

- **No `chromium-cli` or `tmux` on this host** → Playwright Python in a uv overlay
  (`uv run --with playwright`). The project `.venv` is not touched; the IDE's mypy therefore
  reports `Cannot find ... playwright.sync_api` in `driver.py`. Expected.
- **Two Pythons.** The driver runs in the overlay; Streamlit runs on `.venv/Scripts/python.exe`
  (the driver starts it). Running Streamlit from the overlay would miss project deps.
- **`pytest`/`ruff` are not in `.venv`** (it was synced without the `test` extra) → always run
  them through `uv run --with pytest --with ruff`.
- **`warning: VIRTUAL_ENV=...deep_research_from_scratch\.venv does not match`** — the shell has
  another project's venv activated. uv ignores it and uses `./.venv`; harmless.
- **A stale server on the port would be driven silently** (the health check passes). The driver
  now exits with `Port 8599 already serves a Streamlit app`; stop it as above.
- **Python exceptions render in the page, not the console.** The driver also checks
  `[data-testid='stException']`; trust that, not only `console errors: none`.
- **Offline state lives in memory** (`MemorySaver`): threads vanish when the server restarts, and
  published reports go to a fresh `%TEMP%/deep-research-offline-*` directory.
- **Chat input** is found by its placeholder `What should I research?`; `fill` then `press Enter`.

## Troubleshooting

- **`No module named pytest`** with `.venv/Scripts/python.exe -m pytest`: use the `uv run --with`
  command above.
- **`Port 8599 already serves a Streamlit app; stop it or pass --port`**: a previous run is still
  up; kill it (Stop section) or use `--port 8600`.
