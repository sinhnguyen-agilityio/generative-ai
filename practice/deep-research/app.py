"""Streamlit entry point: ``streamlit run app.py``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from deep_research.ui import main

main()
