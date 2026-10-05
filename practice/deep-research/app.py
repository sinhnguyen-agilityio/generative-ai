"""Streamlit entry point: ``streamlit run app.py``. The UI lives in ``intelligence_collector.ui``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from intelligence_collector.ui.main import main  # noqa: E402

main()
