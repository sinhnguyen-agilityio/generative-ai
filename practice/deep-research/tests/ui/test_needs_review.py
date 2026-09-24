from pathlib import Path
from streamlit.testing.v1 import AppTest
def test_needs_review_message_is_renderable():
    app = AppTest.from_file(Path(__file__).parents[2] / "app.py")
    app.run()
    assert not app.exception
