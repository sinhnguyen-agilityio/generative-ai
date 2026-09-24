from intelligence_collector.storage.artifacts import ArtifactStore
def test_manifest_versions_and_lists(stores):
    store = ArtifactStore(stores[0])
    first = store.write("run", "draft", "reports/draft.v1.md", "one")
    second = store.write("run", "draft", "reports/draft.v2.md", "two")
    assert (first.version, second.version) == (1, 2)
    assert len(store.list("run", "draft")) == 2
