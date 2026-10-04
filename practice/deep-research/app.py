from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from intelligence_collector.app import render

render()
