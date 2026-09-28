import threading

from intelligence_collector.tools.url_tracking import UrlClaims


def test_first_task_to_claim_a_page_owns_it_across_url_spellings():
    claims = UrlClaims()

    assert claims.claim("https://docs.test/page", "task-1") == "task-1"
    assert claims.claim("https://DOCS.test/page/#section", "task-2") == "task-1"
    assert claims.claim("https://docs.test/page", "task-1") == "task-1"


def test_owner_lookup_never_claims():
    claims = UrlClaims()

    assert claims.owner("https://docs.test/unclaimed") is None
    assert claims.claim("https://docs.test/unclaimed", "task-2") == "task-2"


def test_claims_are_seeded_from_pages_captured_earlier_in_the_run():
    claims = UrlClaims({"https://docs.test/earlier": "task-0"})

    assert claims.claim("https://docs.test/earlier", "gap-task") == "task-0"


def test_parallel_sub_agents_racing_for_one_page_get_exactly_one_owner():
    claims = UrlClaims()
    owners, start = [], threading.Barrier(8)

    def race(task_id):
        start.wait()
        owners.append(claims.claim("https://docs.test/contended", task_id))

    threads = [threading.Thread(target=race, args=(f"task-{n}",)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(set(owners)) == 1
