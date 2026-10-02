# Cost controls

The router caps output before binding tools or structured schemas. Both primary and fallback
models get the same role cap; a lower provider-configured cap is preserved. Shared model
objects are cloned rather than mutated, so parallel research calls cannot change each
other's settings.

| Role / node | Maximum generated tokens per call |
| --- | ---: |
| Safety guardrail | 128 |
| Scope / brief | 1536 |
| Planner / replan | 1024 |
| Research tool decision | 512 |
| Research compression | 1536 |
| Synthesis / revision | 2048 |
| Fix reflection | 768 |
| Report verification | 3072 |
| Research assessment | 2048 |
| Feedback classification | 256 |
| Optional quality judge | 2048 |

Override router caps using the JSON `INTELLIGENCE_NODE_OUTPUT_TOKENS` environment variable.
Set `INTELLIGENCE_MODEL_MAX_ATTEMPTS` (1–3) to control structured-output attempts; live
routers default to one attempt. Provider transport retries default to zero. An explicitly
configured fallback still runs if the primary fails. Research nodes make multiple calls
per todo; these are per-call caps, not aggregate node or run quotas.

The example environment now uses two searches and three sources per task, one draft repair,
low reasoning effort, and no optional RAGAS scores. Copy these settings into your local `.env`
and restart the application to apply them. Existing local settings are not overwritten.
Reasoning settings depend on the chosen provider; keep unsupported settings out of its kwargs.

## The $0.02 target

A token limit is not a dollar limit. Total cost includes input tokens, generated tokens
(including hidden reasoning when billed by the provider), fallback/retry calls, all research
tasks, embeddings, search and optional evaluation. Model prices and search billing differ;
these limits do not guarantee a complete run below $0.02. Check actual usage in the existing
provider/Langfuse traces before choosing caps or judging whether this target is achievable.
There is no new durable dollar-spend ledger or automatic $0.02 stop in this change.

Official OpenAI documentation confirms that generated-token caps include non-visible tokens:
[Counting tokens](https://developers.openai.com/api/docs/guides/token-counting).
Small caps with high reasoning effort can therefore leave no room for a JSON verdict.

Verification still sees the complete report and evidence. Nothing is truncated or marked
passed to save cost. Missing criteria or invalid/capped verdicts take the bounded verifier
error path and eventually require human review. A synthesis output that cannot be parsed
remains an operational error with its checkpoint available for recovery.
