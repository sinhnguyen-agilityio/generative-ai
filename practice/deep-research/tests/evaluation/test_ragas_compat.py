"""Regression coverage for the ragas/langchain-community import crash and its fix.

``ragas`` 0.4.x hard-imports ``ChatVertexAI`` from a ``langchain_community`` submodule that
newer ``langchain-community`` releases removed (see ``observability/_ragas_compat.py``).
Both tests run in a fresh subprocess so Python's module cache can't hide the real behavior.
"""

import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).parents[2] / "src")


def _run(code: str, *, with_src_on_path: bool = False) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if with_src_on_path:
        env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )


def test_ragas_is_not_importable_without_the_compat_shim():
    result = _run("import ragas")
    assert result.returncode != 0
    assert "langchain_community.chat_models.vertexai" in result.stderr


def test_ragas_is_importable_with_the_compat_shim():
    code = (
        "from intelligence_collector.observability._ragas_compat import ensure_ragas_importable\n"
        "ensure_ragas_importable()\n"
        "import ragas\n"
        "from ragas.metrics import Faithfulness\n"
        "from ragas.llms import LangchainLLMWrapper\n"
        "from ragas.dataset_schema import SingleTurnSample\n"
    )
    result = _run(code, with_src_on_path=True)
    assert result.returncode == 0, result.stderr


def test_importing_the_evaluation_module_applies_the_shim_as_a_side_effect():
    code = (
        "import intelligence_collector.observability.evaluation\n"
        "import ragas\n"
        "from ragas.metrics import Faithfulness\n"
    )
    result = _run(code, with_src_on_path=True)
    assert result.returncode == 0, result.stderr
