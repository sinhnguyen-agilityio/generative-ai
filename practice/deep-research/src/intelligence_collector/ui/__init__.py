"""Streamlit UI of the research workflow.

    main.py            page composition; answers a reply with one graph turn
    state.py           session-state keys, defaults and every change to them
    resources.py       per-session dependencies (tracer, models, graph, VFS, recovery service)
    turn.py            one chat turn: graph streaming, lifecycle log, live supervisor decisions
    run_status.py      what the UI derives from a run snapshot (pure)
    evidence.py        stored research read for display (pure)
    html.py            escaped HTML fragments; theme.css holds their styles
    components/        header, chat (cards and input), panel (tabs), recovery

Run it with ``streamlit run app.py`` from the project root.
"""
