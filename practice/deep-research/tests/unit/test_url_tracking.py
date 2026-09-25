from intelligence_collector.tools.url_tracking import UrlRegistry, normalize_url


def test_normalize_url_unifies_spellings_of_the_same_page():
    canonical = normalize_url("https://docs.test/Guide/Cache")
    assert normalize_url("HTTPS://Docs.Test:443/Guide/Cache/") == canonical
    assert normalize_url("https://docs.test/Guide/Cache#section-2") == canonical
    assert normalize_url("https://docs.test/Guide/Cache?utm_source=x&fbclid=y") == canonical
    assert normalize_url("https://docs.test/guide/cache") != canonical  # path case is significant
    assert normalize_url("https://docs.test/a?b=2&a=1") == normalize_url("https://docs.test/a?a=1&b=2")
    assert normalize_url("http://docs.test:8080/a") == "http://docs.test:8080/a"


def test_registry_persists_first_capture_and_saves_only_when_changed(stores):
    vfs, _ = stores
    registry = UrlRegistry.load(vfs, "run")
    assert registry.save(vfs, "run") is None  # nothing new, nothing written

    registry.register("https://docs.test/cache/", {"task_id": "first"})
    registry.register("https://DOCS.test/cache#top", {"task_id": "second"})  # same page: ignored
    assert registry.save(vfs, "run").kind == "url_registry"

    reloaded = UrlRegistry.load(vfs, "run")
    assert len(reloaded) == 1
    assert reloaded.get("https://docs.test/cache")["task_id"] == "first"
