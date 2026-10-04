"""Bounded semantic verifier retries and explicit resume must bypass failed caches."""

import json
import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from intelligence_collector.agents.report_judge import ReportJudge
from intelligence_collector.graph.nodes import Node
from intelligence_collector.models.router import ModelRouter
from tests.fixtures.workflow import workflow, approve_plan


class VerdictModel:
    model_name = "verdict-test"

    def __init__(self, invalid_calls=1, defect="missing"):
        self.invalid_calls = invalid_calls
        self.defect = defect
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append(messages)
        prompt = str(messages[-1].content)
        checklist = prompt.split("Criteria to judge:\n", 1)[1].split("\n\n", 1)[0]
        criteria = [line[2:] for line in checklist.splitlines() if line.startswith("- ")]
        results = [{"criterion": c, "met": True, "justification": "Covered"} for c in criteria]
        if len(self.calls) <= self.invalid_calls:
            if self.defect == "missing":
                results = results[:-1]
            elif self.defect == "duplicate":
                results[-1] = results[0]
            elif self.defect == "extra":
                results.append(
                    {"criterion": "Invented criterion", "met": True, "justification": "Extra"}
                )
            elif self.defect == "json":
                return AIMessage(content="not JSON")
        return AIMessage(content=json.dumps({"results": results}))


def attach_model(nodes, model):
    nodes.bvf.verification_service.judge = ReportJudge(ModelRouter(model, max_attempts=1))


@pytest.mark.parametrize("defect", ["missing", "duplicate", "extra"])
def test_complete_unresolved_criteria_inside_one_verification(tmp_path, defect):
    graph, config, initial, nodes, writer, judge, researcher, diagnosis = workflow(tmp_path)
    model = VerdictModel(defect=defect)
    attach_model(nodes, model)
    result = approve_plan(graph, config, initial)
    assert result["verification"]["outcome"] == "PASS"
    assert result["verifier_attempt"] == 1
    expected_calls = 1 if defect == "extra" else 2
    assert len(model.calls) == expected_calls
    assert len(writer.calls) == len(researcher.calls) == 1
    assert not diagnosis.calls and result["batch_cycles"] == 0
    if defect != "extra":
        assert "unresolved criteria" in str(model.calls[1][0].content)
    assert not any(
        e.source == Node.VERIFY and e.target == Node.VERIFY for e in graph.get_graph().edges
    )


def test_json_error_pauses_without_reverifying_or_blind_rewrite(tmp_path):
    graph, config, initial, nodes, writer, judge, researcher, diagnosis = workflow(tmp_path)
    model = VerdictModel(defect="json")
    attach_model(nodes, model)
    result = approve_plan(graph, config, initial)
    assert result["verification"]["outcome"] == "ERROR"
    assert result["verifier_attempt"] == len(model.calls) == 1
    assert len(writer.calls) == 1 and not diagnosis.calls
    assert graph.get_state(config).next == (Node.TECHNICAL_RECOVERY,)


def test_incomplete_judge_with_known_defect_fixes_and_builds_new_draft(tmp_path):
    from intelligence_collector.services.workflow import VerificationOutputError

    graph, config, initial, nodes, writer, judge, researcher, diagnosis = workflow(tmp_path)
    delegate = nodes.bvf.verification_service.judge

    class PartialJudge:
        calls = 0

        def verify(self, brief, criteria, report, evidence, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise VerificationOutputError(
                    "Missing criterion",
                    partial_results=[
                        {
                            "criterion": criteria[0],
                            "met": False,
                            "justification": "Explain cache reuse with the stored evidence",
                            "issue_type": "draft_defect",
                        }
                    ],
                )
            return delegate.verify(brief, criteria, report, evidence, **kwargs)

    partial = PartialJudge()
    nodes.bvf.verification_service.judge = partial
    result = approve_plan(graph, config, initial)
    assert [v["outcome"] for v in result["verification_history"]] == ["ERROR", "PASS"], result.get(
        "technical_failure"
    )
    assert len(writer.calls) == partial.calls == 2
    assert len(diagnosis.calls) == 1 and result["batch_cycles"] == 1
    assert "Explain cache reuse" in writer.calls[1]["revision"].failures
    assert "version 2" in result["draft"]


def test_known_defects_consume_only_three_fix_build_verify_cycles(tmp_path):
    from intelligence_collector.services.workflow import VerificationOutputError

    graph, config, initial, nodes, writer, judge, researcher, diagnosis = workflow(tmp_path)

    class PartialJudge:
        calls = 0

        def verify(self, brief, criteria, report, evidence, **kwargs):
            self.calls += 1
            raise VerificationOutputError(
                "Missing criterion",
                partial_results=[
                    {
                        "criterion": criteria[0],
                        "met": False,
                        "justification": "Explain cache reuse with evidence",
                    }
                ],
            )

    partial = PartialJudge()
    nodes.bvf.verification_service.judge = partial
    result = approve_plan(graph, config, initial)
    assert partial.calls == len(writer.calls) == 4
    assert len(diagnosis.calls) == 3
    assert result["batch_cycles"] == result["cumulative_cycles"] == 3
    assert result["verification"]["outcome"] == "ERROR"
    assert graph.get_state(config).next == (Node.TECHNICAL_RECOVERY,)
    assert len(researcher.calls) == 1


def test_resume_bypasses_error_cache_without_resetting_fix_budget(tmp_path):
    from intelligence_collector.graph.recovery import WorkflowRecoveryService
    from intelligence_collector.storage.run_registry import RunRegistry

    graph, config, initial, nodes, writer, judge, researcher, diagnosis = workflow(tmp_path)
    model = VerdictModel(invalid_calls=99)
    attach_model(nodes, model)
    result = approve_plan(graph, config, initial)
    assert result["verification"]["outcome"] == "ERROR" and len(model.calls) == 2
    graph.invoke(None, config)
    assert len(model.calls) == 2
    model.invalid_calls = 0
    registry = RunRegistry(tmp_path / "registry.sqlite")
    registry.register_run("run", "run")
    resumed = WorkflowRecoveryService(graph, registry).resume_interrupt(
        "run", {"action": "resume_saved_step"}
    )
    assert resumed["verification"]["outcome"] == "PASS"
    assert resumed["verifier_retry_epoch"] == 1
    assert resumed["draft"] == result["draft"]
    assert len(model.calls) == 3 and len(writer.calls) == 1
    assert resumed["batch_cycles"] == result["batch_cycles"] == 0


def test_invalid_cached_coverage_cannot_pass_or_trigger_a_blind_fix(tmp_path):
    from intelligence_collector.services.workflow import digest

    graph, config, initial, nodes, writer, judge, researcher, diagnosis = workflow(tmp_path)
    model = VerdictModel(invalid_calls=0)
    attach_model(nodes, model)
    original_call = nodes.bvf.verification_service.operations.call
    injected = False

    def call(run_id, operation, inputs, invoke, **kwargs):
        nonlocal injected
        if operation == "coverage" and not injected:
            injected = True
            key = digest({"operation": operation, "inputs": inputs})
            nodes.vfs.write_json(
                run_id,
                f"operations/policy/{key}.json",
                {
                    "operation": "coverage",
                    "attempts": 1,
                    "status": "complete",
                    "result": {"results": []},
                },
                "legacy_cache",
            )
        return original_call(run_id, operation, inputs, invoke, **kwargs)

    nodes.bvf.verification_service.operations.call = call
    result = approve_plan(graph, config, initial)
    assert result["verification"]["outcome"] == "ERROR"
    assert not model.calls and len(writer.calls) == 1
    resumed = graph.invoke(Command(resume={"action": "resume_saved_step"}), config)
    assert resumed["verification"]["outcome"] == "PASS"
    assert len(model.calls) == 1 and len(writer.calls) == 1


def test_supplement_preserves_negative_assessment_and_causes_fix(tmp_path):
    graph, config, initial, nodes, writer, judge, researcher, diagnosis = workflow(tmp_path)

    class NegativeModel(VerdictModel):
        def invoke(self, messages, config=None):
            response = super().invoke(messages, config)
            if len(self.calls) == 1:
                value = json.loads(response.content)
                value["results"][0].update(
                    met=False, justification="Explain cache reuse from evidence"
                )
                return AIMessage(content=json.dumps(value))
            return response

    model = NegativeModel(defect="missing")
    attach_model(nodes, model)
    result = approve_plan(graph, config, initial)
    assert [v["outcome"] for v in result["verification_history"]] == ["FAIL", "PASS"]
    assert (
        len(model.calls) == 3
    )  # first verdict, missing-criterion supplement, revised draft verdict
    assert len(writer.calls) == 2 and len(diagnosis.calls) == 1
    assert result["batch_cycles"] == 1
    assert "Explain cache reuse" in writer.calls[1]["revision"].failures
