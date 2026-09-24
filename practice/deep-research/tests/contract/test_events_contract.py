from intelligence_collector.observability.events import EventRecorder, WorkflowError, WorkflowEvent
def test_events_preserve_failure_guidance_and_artifacts():
    recorder = EventRecorder()
    recorder.record(WorkflowEvent("artifact.write", "run", "Research", artifact_ids=("a",)))
    error = recorder.record(WorkflowError("timeout", "run", "Research", retryable=True, can_resume=True, artifact_ids=("a",)))
    assert error.can_resume and error.artifact_ids == ("a",) and len(recorder.events) == 2
