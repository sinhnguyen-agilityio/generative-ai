from intelligence_collector.ui.run_status import failure_stage_label as _failure_stage_label


def test_failure_stage_label_names_the_last_active_node():
    activity = [
        {"id": "a:scope", "label": "Evaluating scope", "status": "complete"},
        {"id": "a:verify", "label": "Verifying report with LLM judge", "status": "running"},
    ]

    assert _failure_stage_label(activity) == "Verifying report with LLM judge"


def test_failure_stage_label_falls_back_to_request_when_nothing_ran_yet():
    assert _failure_stage_label([]) == "Request"


def test_failure_stage_label_falls_back_to_request_when_label_is_missing():
    assert _failure_stage_label([{"id": "a:x", "status": "running"}]) == "Request"
