# Quality evaluation results (2026-10-04)

Cases and rubrics: `evals/cases.json`. Runner: `scripts/evaluate_quality.py --run-live --label <x>`
(real models, Tavily MCP, Chroma, RAGAS; the clarification reply and the initial plan approval are
automated, report approval is never given). Rubric outcomes below are a human reading of each
saved report against the evidence collected in that run (`evidence.json` next to it). The
workflow's own verdict is a separate column: a gate PASS is not a rubric pass.

Models: reasoning `openai/gpt-oss-120b` (Groq), standard `gemini-3.5-flash-lite`, Faithfulness
judge `gpt-4.1-mini`. One run per case per label, so run-to-run variance is not measured; the two
"after" iterations below already disagree on two cases.

## Runs

| Label | Code | Folder |
|---|---|---|
| head-mcp-transport-error | HEAD with `.venv` on `mcp 1.30` (lock says 2.2.0): every Tavily call failed, all cases ended in NEEDS_REVIEW with no draft | `runs/head-mcp-transport-error/` |
| head-defects | HEAD, comparison case: crashed on the URL-registry redaction bug | `runs/head-defects/` |
| before | HEAD, locked `mcp`, redaction fix only for the comparison case | `runs/before/` |
| after-iteration1 | First version of the changes (kept for the defects it exposed) | `runs/after-iteration1/` |
| after-iteration2 | + brief-template, title/gap scoring and source-title fixes | `runs/after-iteration2/` |
| after | + gap sentences uncited, `Assessment:` label, no glosses (final code) | `runs/after/` |

## Outcomes

| Case | Label | Workflow result | Rubric (human) | Why |
|---|---|---|---|---|
| simple_factual | before | PASS v1 → approval | Pass, with narration | "Python 3.14.8 … Sept. 30, 2026 [S1]"; Scope says "uses only the provided official excerpts" |
| | after-it2 | FAIL after 2 cycles → review | Pass (content) | "3.14.8 [S1]" correct; "(general availability)" gloss judged unsupported (Faithfulness 0.50) and the writer kept it |
| | after | PASS v2 → approval | **Fail** | "3.14 — 2025-10-07": no patch version, date is 3.14.0's; coverage judge accepted it (false pass) |
| simple_comparison | before | PASS, ended "Incomplete" **without approval** | **Fail** | Every section "INCOMPLETE"; 403 meaning and decision rule never stated although S6/S14 held them (false pass) |
| | after-it2 | PASS v1 → approval | Pass (WWW-Authenticate only in Limitations) | Defines 401, 403 and the decision rule with citations |
| | after | FAIL after 2 cycles → review | Partial | v1 dropped Sources entries (fixed by repair); Faithfulness then correctly flagged "401 … invalid or insufficient credentials" (not in S1) and the writer kept it |
| explicit_subquestions | before | ERROR ×3 → review, never shown | Pass (content), narration | Judge did not echo criteria verbatim → ERROR on a correct report |
| | after-it2 | PASS v1 → approval | Partial | (1)(2)(3) answered; "section 7.2" is wrong (7.2 is Security Considerations; 429 is section 4) |
| | after | FAIL after 3 cycles → review | Partial | v1 had no Sources (repaired); Faithfulness correctly flagged "section 4.2" (excerpt reads "April 20124. 429…") and the writer kept it |
| uncertain_no_evidence | before | ERROR → review, never shown | **Fail** | "INCOMPLETE:" labels, no plain statement, no proxy, research-process notes as Limitations |
| | after-it2 | FAIL after 3 cycles → review | Pass (content) | Plain "could not be established" + PyPI-download proxy; judge demanded a citation on the gap sentence (rule conflict, fixed) |
| | after | PASS v1 → approval | Partial | Correct gap statement and proxy, but "~34.5M … in September 2026": the source says "2026" only |

## Cost and latency (sum over the four cases)

| Label | Model calls | Input / output tokens* | Tavily searches | Chroma retrievals | RAGAS runs | Wall time |
|---|---|---|---|---|---|---|
| before | 83 | 209 370 / 58 881 | 21 | 3 | 1 | 422 s |
| after-iteration2 | 57 | 107 103 / 16 876 | 6 | 0 | 6 | 179 s |
| after | 69 | 161 821 / 20 030 | 10 | 0 | 8 | 232 s |

\* LangChain-reported tokens of workflow models; RAGAS judge tokens are not captured. Per-role
counts are in each `summary.json`. Main savings: one task for simple questions, no research
assessor call, no pre-draft gap loop when evidence exists, deterministic diagnosis instead of a
reflection call; first drafts that pass stop with zero cycles.

## Conclusions

- Fixed and observed live: brief no longer inflates the question; plans pause before
  retrieval; honest gap reports reach report approval instead of ending "Incomplete"; judge
  criterion mismatches no longer produce ERROR on correct reports; a passing first draft stops
  without a cycle; repairs target the failing item (presentation repairs restored a missing
  Sources section twice).
- Not fixed: the gates do not reliably separate good from bad reports on this snippet-only
  evidence. Coverage passed "3.14" as the latest version; Faithfulness passed "September 2026"
  for a "2026" figure and "section 7.2", but correctly failed "section 4.2". When Faithfulness
  flags a statement, the standard-tier writer often keeps it, so the run spends its cycles and
  goes to human review with the latest draft.
- Retrieval limit: Tavily returns ~130–1 500 character snippets here; full-page content was not
  obtainable (`include_raw_content` empty, `tavily_extract` "404 page not found").
