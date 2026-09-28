import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

load_dotenv()

import os

from intelligence_collector.config import Settings
from intelligence_collector.observability.evaluation import make_faithfulness_evaluator

api_key = os.getenv("GOOGLE_API_KEY", "").strip()
if not api_key:
    print("GOOGLE_API_KEY is not set in the environment — nothing to check.")
    sys.exit(1)

settings = Settings.from_env()
print(f"Checking RAGAS Faithfulness judge: model={settings.faithfulness_model!r}, provider=google")

evaluate = make_faithfulness_evaluator(model=settings.faithfulness_model, api_key=api_key)

question = "What does the cache do?"
report = "The cache stores data in memory for fast repeated reads [S1]."
contexts = ["The cache stores data in memory to speed up repeated reads."]

try:
    score = evaluate(question, report, contexts)
except Exception as exc:
    print(f"FAILED: judge LLM call raised {type(exc).__name__}: {exc}")
    sys.exit(1)

if score is None:
    print("FAILED: evaluator returned None for a non-empty report/contexts — unexpected.")
    sys.exit(1)

print(f"OK — GOOGLE_API_KEY works, faithfulness score: {score:.2%}")
