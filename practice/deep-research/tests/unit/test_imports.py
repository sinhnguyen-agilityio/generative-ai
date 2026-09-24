def test_core_modules_import():
    import intelligence_collector.contracts  # noqa: F401
    import intelligence_collector.guardrails  # noqa: F401
    import intelligence_collector.publication  # noqa: F401
    import intelligence_collector.storage.vfs  # noqa: F401
    import intelligence_collector.storage.checkpoints  # noqa: F401
    import intelligence_collector.storage.vector_index  # noqa: F401
    import intelligence_collector.tools.write_todos  # noqa: F401
    import intelligence_collector.graph.routing  # noqa: F401
