from pathlib import Path

from streamlit.testing.v1 import AppTest

from intelligence_collector.contracts import Stage


def test_needs_review_message_is_renderable():
    app = AppTest.from_file(Path(__file__).parents[2] / "app.py")
    app.run()
    assert not app.exception


def test_needs_review_banner_shows_when_the_graph_stops_for_review():
    """A run that ends at the needs_review node (recovery budget exhausted / no progress) must
    surface a banner in the UI — this state has no automatic approval path, so the banner is
    the only signal the human operator gets that the run stalled."""
    app = AppTest.from_file(Path(__file__).parents[2] / "app.py")
    app.session_state["scope_state"] = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "stage": Stage.NEEDS_REVIEW.value,
        "current_action": "Needs review: recovery budget 3 exhausted after 3 fix attempt(s)",
    }

    app.run()

    assert not app.exception
    assert any(
        "needs human review" in element.value for element in app.error
    )
    assert any(
        "recovery budget 3 exhausted" in element.value for element in app.caption
    )
