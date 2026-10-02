"""Manual, opt-in smoke check of the configured RAGAS faithfulness provider."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

from intelligence_collector.config import Settings
from intelligence_collector.observability.evaluation import make_faithfulness_evaluator

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true", help="Allow potentially paid judge calls")
    args = parser.parse_args()
    if not args.run_live:
        parser.error("Pass --run-live to authorize live judge calls")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    settings = Settings.from_env()
    key_names = {"google_genai": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}
    key_name = key_names.get(settings.faithfulness_provider)
    if key_name is None:
        parser.error("Configure a supported INTELLIGENCE_FAITHFULNESS_PROVIDER")
    api_key = os.getenv(key_name, "").strip()
    if not api_key:
        parser.error(f"{key_name} is not set")
    evaluate = make_faithfulness_evaluator(
        model=settings.faithfulness_model, api_key=api_key,
        provider=settings.faithfulness_provider,
    )
    try:
        score = evaluate(
            "What does the cache do?",
            "The cache stores data in memory for fast repeated reads [S1].",
            ["The cache stores data in memory to speed up repeated reads."],
        )
    except Exception as exc:
        # Provider errors can contain request data or credentials; show only the type.
        print(f"FAILED: judge call raised {type(exc).__name__}")
        return 1
    if score is None:
        print("FAILED: evaluator returned no score for non-empty inputs")
        return 1
    print(f"OK — faithfulness score: {score:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
