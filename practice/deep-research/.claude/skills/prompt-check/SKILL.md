---
name: prompt-check
description: Audit a prompt in src/deep_research/prompts.py against its callers, schema and conventions. Use after editing a prompt.
disable-model-invocation: true
argument-hint: "[PROMPT_NAME]"
---

Audit the prompt named in the arguments (default: every prompt changed in `git diff`) in `src/deep_research/prompts.py`.

1. **Placeholders**: every `{name}` in the prompt matches the `.format()` call in its caller under `src/deep_research/agents/` (find it with grep). Literal braces must be doubled.
2. **Schema**: fields the prompt mentions match the Pydantic model in `src/deep_research/schemas/`. The prompt must not repeat the JSON schema, since structured output already enforces it.
3. **Untrusted text**: user or fetched text is declared as data, not instructions (compare `GUARDRAIL_PROMPT`, `TRIAGE_PROMPT`).
4. **Language**: the output language is stated, as in the other prompts.
5. **Routing**: loops are bounded and consistent with `src/deep_research/utils/routing.py` and the agent docstring.
6. Run `uv run pytest` and report failures.

Report findings as `file:line - problem - suggested fix`, most severe first. Do not edit files unless asked.
