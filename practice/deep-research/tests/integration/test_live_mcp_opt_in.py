import os
import pytest
from intelligence_collector.agents.worker import ControlledMCP

@pytest.mark.skipif(os.getenv("RUN_LIVE_MCP") != "1", reason="opt-in live MCP acceptance run")
def test_live_mcp_opt_in():
    # The configured MCP adapter is supplied by the deployment environment.
    worker = ControlledMCP([{"url": "configured", "content": "configured source"}])
    assert worker.search("controlled acceptance")
