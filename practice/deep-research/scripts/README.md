# Manual live checks

These scripts are not part of the offline pytest suite. Importing them does not invoke
providers. Running them requires `--run-live` and can incur API costs.

```text
.venv\Scripts\python.exe scripts/check_model_fallback.py --run-live
.venv\Scripts\python.exe scripts/check_faithfulness.py --run-live
```

The fallback check uses `INTELLIGENCE_STANDARD_FALLBACK_MODEL` and its provider. The
faithfulness check uses `INTELLIGENCE_FAITHFULNESS_MODEL` and its provider, with the
corresponding `OPENAI_API_KEY` or `GOOGLE_API_KEY`. Configure a model compatible with
that provider. Both load the repository's local `.env` without overriding process settings.

Former root scripts: `test_groq_langchain.py` and `test_gemini_faithfulness.py`.

## Drawing the workflow graph

`draw_graph.py` builds the same graph the app runs, with placeholder collaborators (no model,
Tavily or Langfuse configuration, no calls), and writes its Mermaid source.

```text
.venv\Scripts\python.exe scripts/draw_graph.py          # exports/graph.mmd
.venv\Scripts\python.exe scripts/draw_graph.py --png    # also exports/graph.png via mermaid.ink
```

In a notebook started from the repository root:

```python
from IPython.display import Image, display
from scripts.draw_graph import build_app_graph

display(Image(build_app_graph().get_graph().draw_mermaid_png()))
```
