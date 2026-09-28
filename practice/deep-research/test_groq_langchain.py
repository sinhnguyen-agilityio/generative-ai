import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from intelligence_collector.config import RoleModelSettings
from intelligence_collector.models.factory import ModelConfig, ModelFactory

settings = RoleModelSettings.from_env()

groq_model = ModelFactory().create(
    ModelConfig(model=settings.standard_fallback_model, provider=settings.standard_fallback_provider)
)

response = groq_model.invoke("Explain why fast inference is critical for reasoning models")
print(response.content)
