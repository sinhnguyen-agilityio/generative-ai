from __future__ import annotations
from intelligence_collector.guardrails import validate_request
def render():
    try:
        import streamlit as st
    except ImportError:
        return None
    st.title("Intelligence Collector")
    text = st.chat_input("What should we research?")
    if text:
        decision = validate_request(text)
        if not decision.allowed: st.error(decision.reason)
        else: st.info("Request accepted. Scope clarification begins.")
    return st
if __name__ == "__main__": render()

