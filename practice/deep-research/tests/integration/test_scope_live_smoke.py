import os
import pytest
from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.models.router import ModelRouter

@pytest.mark.skipif(os.getenv("RUN_LIVE_LLM") != "1", reason="opt-in live LLM smoke test")
def test_scope_live_llm_smoke():
    pytest.skip("Configure a real provider adapter in the environment before running live smoke")
