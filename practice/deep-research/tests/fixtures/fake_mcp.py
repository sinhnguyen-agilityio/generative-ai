class FakeMCP:
    """Deterministic MCP substitute with observable call count and provenance."""
    def __init__(self, sources=None):
        self.sources = sources or [{"url": "https://fixture.test/source", "content": "fixture evidence"}]
        self.calls = []

    def search(self, query, *, server="fixture", tool="search"):
        self.calls.append({"query": query, "server": server, "tool": tool})
        return list(self.sources)

