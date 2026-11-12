---
name: trace-debug
description: Fetch and summarize a Langfuse trace for a research run (node timings, tokens, errors). Use when debugging a run or asking why an agent behaved a certain way.
argument-hint: "[--trace-id ID | --thread-id ID] [--hours N]"
---

1. Run `python scripts/fetch_langfuse_trace.py $ARGUMENTS` (no arguments fetches the latest trace). It reads Langfuse credentials itself; never open or print `.env`.
2. Full inputs and outputs are saved under `.data/traces`; read the file the script reports instead of re-fetching.
3. Summarize: nodes in order with duration, token use, errors or retries, and which node produced the problem. Quote only the relevant prompt or output.
4. If the thread is still running or older than the window, say so and suggest a larger `--hours`.
