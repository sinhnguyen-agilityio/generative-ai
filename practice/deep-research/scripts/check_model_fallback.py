"""Manual, opt-in smoke check of the configured standard-tier fallback model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

from intelligence_collector.config import RoleModelSettings
from intelligence_collector.models.factory import ModelConfig, ModelFactory

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true", help="Allow a potentially paid API call")
    args = parser.parse_args()
    if not args.run_live:
        parser.error("Pass --run-live to authorize a live model call")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    settings = RoleModelSettings.from_env()
    if not settings.standard_fallback_model:
        parser.error("Configure INTELLIGENCE_STANDARD_FALLBACK_MODEL first")
    model = ModelFactory().create(ModelConfig(
        model=settings.standard_fallback_model, provider=settings.standard_fallback_provider,
        kwargs={"max_tokens": 256},
    ))
    response = model.invoke("Explain why fast inference is critical for reasoning models")
    print(response.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
