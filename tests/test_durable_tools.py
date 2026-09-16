import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from local_model_app.durable_tools import DurableToolExecutor


class DurableToolExecutorTests(unittest.TestCase):
    def test_background_tool_turn_uses_shared_inference_lock(self) -> None:
        lock = asyncio.Lock()
        observed = []

        class FakeCoordinator:
            def __init__(self, *args, **kwargs):
                pass

            async def respond(self, prompt, history, plugin_ids):
                observed.append(lock.locked())
                return "Evidence: https://example.com/source"

        model = SimpleNamespace(settings=SimpleNamespace(max_tool_calls_per_step=256))
        with tempfile.TemporaryDirectory() as temporary:
            executor = DurableToolExecutor(
                model,
                SimpleNamespace(),
                SimpleNamespace(),
                Path(temporary),
                SimpleNamespace(),
                lock,
            )
            task = {
                "id": "task-1",
                "definition": {
                    "goal": "Research a topic",
                    "metadata": {"plugin_ids": ["local.web-research"], "chat_id": "chat-1"},
                },
            }
            item = {
                "id": "item-1",
                "title": "Collect sources",
                "instructions": "Search and preserve URLs.",
                "completion_check": "Sources are recorded.",
                "result": None,
            }
            with patch("local_model_app.durable_tools.ToolCoordinator", FakeCoordinator):
                outcome = asyncio.run(executor.execute_work_item(task, item, []))

        self.assertEqual(observed, [True])
        self.assertFalse(lock.locked())
        self.assertEqual(outcome.outcome, "completed")


if __name__ == "__main__":
    unittest.main()
