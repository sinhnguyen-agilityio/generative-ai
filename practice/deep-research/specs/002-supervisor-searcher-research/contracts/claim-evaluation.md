# Contract: Claim Evaluation (faithfulness gate)

Module: `agents/claim_judge.py`; called by `BVFNodes.verifying` after coverage and structural
citation checks pass. Replaces RAGAS Faithfulness as the gate (RAGAS moves to
`evaluating_quality`, informational).

## Inputs

- The draft (Markdown).
- The draft's evidence snapshot `synthesis/evidence.v{N}.json` — the only text the judge may read.

## Segmentation (code)

1. Scored body = `observability.evaluation.report_sections(draft)` (drops Scope, Limitations,
   Sources sections, the title, gap-marker sentences and `Assessment:` sentences).
2. Sentences: split on `.`, `?`, `!` followed by whitespace and an uppercase letter or end of
   text, never inside `[S…]` citations or numbers; each Markdown list item is one sentence;
   headings and table header rows are skipped; table rows count as one sentence each.
3. Citations per sentence: every `[S<n>]` (comma/semicolon lists allowed) inside the sentence.

## Judge call (`claim_judge` role, reasoning tier, structured output)

Batches of ≤ 10 sentences, ≤ 4 calls in flight. Per sentence the model receives `claim_id`, the
sentence, and for each cited source its snapshot text.

```json
{"claims": [
  {"claim_id": "K3", "states_fact": true, "verdict": "supported",
   "per_source": [{"source_id": "S2", "supports_part": true},
                  {"source_id": "S9", "supports_part": true}],
   "unsupported_parts": []}
]}
```

Missing claim ids get one supplemental call; still missing → ERROR (`verifier_failure`).

## Rules (code)

| Condition | Failure code |
| --- | --- |
| `states_fact` and no citation | `uncited` |
| Cited id not in the snapshot | `unknown_source` |
| Any cited source with `supports_part = false` | `source_supports_nothing` (names the source) |
| `verdict = partly_supported` or `unsupported` | `not_fully_supported` (lists unsupported parts) |
| `verdict = contradicted` | `contradicted` |

Sentences with `states_fact = false` are recorded but not scored. Draft passes the gate when no
factual sentence has a failure. `supported_share` = supported factual sentences ÷ factual
sentences (1.0 when there are none).

## Output

- VFS: `verification/claims.{verification_id}.json` (`ClaimEvaluation`, see data-model.md).
- `VerificationCheck(name="claims", gate="faithfulness", passed, missing_points=[failing
  sentences, max 8], details=[one line per failure], affected_ids=["claim:K3", ...])`.
- State: `claim_support_score`, `claim_evaluation_ref`.
- Repair: `repair.diagnose` routes gate `faithfulness` to `vfs` (existing).
