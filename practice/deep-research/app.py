"""Streamlit entry point: ``streamlit run app.py``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ui.app import main

main()
