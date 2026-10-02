import json
import unittest

from local_model_app.context_checkpoint import build_semantic_checkpoint, render_context_pack


class ContextCheckpointTests(unittest.TestCase):
    def test_repeated_compaction_merges_semantic_state_without_nesting_previous_pack(self) -> None:
        first_messages = [
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": {"query": "memory"}},
            }]},
            {"role": "tool", "tool_call_id": "call-1", "name": "search", "content": json.dumps({
                "isError": False,
                "url": "https://example.test/paper",
                "artifact_path": r"C:\research\paper.md",
            })},
        ]
        first = build_semantic_checkpoint(
            objective="Research memory",
            plan="Search, inspect, and finish when the evidence target is met.",
            messages=first_messages,
            structured_state={"next_step": "fetch the paper"},
            previous=None,
            checkpoint_index=1,
            trigger="token_pressure",
            prompt_tokens=3800,
            context_window=4096,
            reserved_output_tokens=256,
        )
        second_messages = [
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call-2",
                "type": "function",
                "function": {"name": "fetch", "arguments": {"url": "https://example.test/paper"}},
            }]},
            {"role": "tool", "tool_call_id": "call-2", "name": "fetch", "content": json.dumps({
                "isError": False,
                "source_id": "source-1",
            })},
        ]
        second = build_semantic_checkpoint(
            objective="Research memory",
            plan="Search, inspect, and finish when the evidence target is met.",
            messages=second_messages,
            structured_state={"next_step": "analyze source-1"},
            previous=first,
            checkpoint_index=2,
            trigger="call_interval",
        )

        self.assertEqual([row["tool"] for row in second["completed_actions"]], ["search", "fetch"])
        self.assertEqual(second["exact_next_action"], "analyze source-1")
        self.assertTrue(any(row["value"] == "source-1" for row in second["artifacts_and_identifiers"]))
        self.assertEqual(second["checkpoint_history"][0]["checkpoint_id"], first["checkpoint_id"])
        self.assertNotIn("prior_checkpoint", second)
        rendered = json.loads(render_context_pack(second))
        self.assertEqual(rendered["objective"], "Research memory")
        self.assertEqual(len(rendered["completed_actions"]), 2)

    def test_failures_and_acceptance_plan_survive_checkpoint(self) -> None:
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call-bad", "type": "function",
                "function": {"name": "fetch", "arguments": {"url": "https://example.test/missing"}},
            }]},
            {"role": "tool", "tool_call_id": "call-bad", "name": "fetch", "content": json.dumps({
                "isError": True, "error": "not_found",
            })},
        ]
        checkpoint = build_semantic_checkpoint(
            objective="Produce a report",
            plan="Completion test: a verified DOCX exists.",
            messages=messages,
            structured_state={},
            previous=None,
            checkpoint_index=1,
            trigger="token_pressure",
        )

        self.assertEqual(checkpoint["acceptance_and_execution_plan"], "Completion test: a verified DOCX exists.")
        self.assertEqual(checkpoint["failed_actions"][0]["tool"], "fetch")
        self.assertIn("Do not claim completion", " ".join(checkpoint["invariants"]))


if __name__ == "__main__":
    unittest.main()
