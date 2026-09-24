from __future__ import annotations
from intelligence_collector.agents.scope import missing_scope_fields, narrowing_suggestions
from intelligence_collector.guardrails import validate_request
def render():
    try:
        import streamlit as st
    except ImportError:
        return None
    st.title("Intelligence Collector")
    st.session_state.setdefault("stage", "Scoping")
    st.session_state.setdefault("progress", "No research started")
    st.write(f"Stage: **{st.session_state.stage}**")
    st.write(f"Progress: {st.session_state.progress}")
    if st.session_state.get("needs_review"):
        st.error("This run needs human review before it can be treated as verified or published.")
        st.caption(st.session_state.get("review_feedback", "Unresolved verification feedback is stored with the run."))
    text = st.chat_input("What should we research?")
    if text:
        decision = validate_request(text)
        if not decision.allowed:
            st.error(decision.reason)
            st.session_state.progress = "Request rejected; no research started"
        else:
            st.session_state.progress = "Scope accepted; clarification/planning pending"
            st.info("Request accepted. Scope clarification begins.")
            st.session_state.scope_turns = st.session_state.get("scope_turns", []) + [text]
            st.warning("Please define the research boundaries, success criteria, and deliverable format before Planning.")
            suggestions = narrowing_suggestions({"objective": text})
            for suggestion in suggestions:
                st.caption(suggestion)
    return st
if __name__ == "__main__": render()
