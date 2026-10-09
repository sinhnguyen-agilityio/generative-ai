"""The real Streamlit UI on scripted models, search and store (tests/fakes.py): no API keys,
no cost, deterministic. Run with: streamlit run .claude/skills/run-deep-research/offline_app.py"""

import sys
import tempfile
from pathlib import Path

UNIT = Path(__file__).resolve().parents[3]  # run-deep-research -> skills -> .claude -> unit
sys.path[:0] = [str(UNIT / "src"), str(UNIT)]

import streamlit as st  # noqa: E402

from ui import app as ui  # noqa: E402
from tests.fakes import make_app  # noqa: E402


@st.cache_resource
def offline_app():
    return make_app(Path(tempfile.mkdtemp(prefix="deep-research-offline-")))[0]


ui.get_app = offline_app
ui.main()
