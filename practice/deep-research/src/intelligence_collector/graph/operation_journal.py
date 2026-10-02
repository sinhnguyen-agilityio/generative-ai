"""Reuse completed node outputs across the side-effect/checkpoint crash boundary."""

import hashlib
import json

from langchain_core.messages import messages_from_dict, messages_to_dict


class OperationJournal:
    """JSON only: never deserialize executable objects from run artifacts.

    Interrupt nodes are intentionally excluded by the caller. An in-flight provider call
    can still repeat if the process dies before its output is durably recorded.
    """

    def __init__(self, vfs, context=""):
        self.vfs = vfs
        self.context = context

    @staticmethod
    def encode(state):
        value = dict(state)
        if "messages" in value:
            value["messages"] = messages_to_dict(value["messages"])
        return value

    def path(self, name, state):
        from intelligence_collector.agents.prompts import REPORT_WRITER_PROMPT, REVIEWER_BLOCK

        payload = json.dumps(self.encode(state), sort_keys=True, default=str)
        digest = hashlib.sha256((self.context + REPORT_WRITER_PROMPT + REVIEWER_BLOCK + payload)
                                .encode()).hexdigest()
        return f"operations/v1/{name}/{digest}.json"

    def read(self, name, state):
        try:
            result = json.loads(self.vfs.read(state["run_id"], self.path(name, state)))
        except FileNotFoundError:
            return None
        if result.get("run_id") != state["run_id"] or result.get("thread_id") != state.get("thread_id"):
            raise ValueError("Operation journal identity mismatch")
        if "messages" in result:
            result["messages"] = messages_from_dict(result["messages"])
        return result

    def write(self, name, state, result):
        self.vfs.write_json(state["run_id"], self.path(name, state), self.encode(result),
                            "operation_result")
