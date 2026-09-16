import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from local_model_app.chat_store import ChatStore
from local_model_app.model import AssistantReply
from local_model_app.server import Runtime


class FakeApprovalManager:
    def __init__(self) -> None:
        self.calls = []

    async def ensure_started(self, plugin_ids):
        self.started = list(plugin_ids)

    async def call_tool(self, name, arguments, *, approved=False):
        self.calls.append((name, arguments, approved))
        return {"isError": False, "structuredContent": {"profiles": ["TPL-001"]}, "content": []}


class FakeApprovalModel:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(max_new_tokens=4096, tool_temperature=0.0)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        tool_message = next(message for message in messages if message["role"] == "tool")
        assert "TPL-001" in tool_message["content"]
        return AssistantReply(content="The approved regulatory catalog contains TPL-001.", tool_calls=[])


class ApprovalTests(unittest.TestCase):
    def test_pending_approval_executes_and_resumes_exactly_once(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            store = ChatStore(root / "data" / "chats.sqlite3")
            chat = store.create_chat()
            store.set_plugin_selected(chat["id"], "grid-workshop.powerworld", selected=True)
            store.append_exchange(chat["id"], "List regulatory tests", "Approval is required.")
            activity = root / "data" / "tool_activity" / f"{chat['id']}.jsonl"
            state_path = activity.with_suffix(".state.json")
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({
                "status": "waiting_for_approval",
                "plugin_id": "grid-workshop.powerworld",
                "exposed_tool": "grid__regulatory_list_tests",
                "arguments": {},
            }), encoding="utf-8")

            runtime = Runtime.__new__(Runtime)
            runtime.store = store
            runtime.mcp = FakeApprovalManager()
            runtime.model = FakeApprovalModel()
            runtime._inference_lock = asyncio.Lock()

            async def scenario():
                answer = await runtime.approve_tool_call(chat["id"])
                with self.assertRaisesRegex(RuntimeError, "already been consumed"):
                    await runtime.approve_tool_call(chat["id"])
                return answer

            with patch("local_model_app.server.ROOT", root):
                answer = asyncio.run(scenario())

            self.assertIn("TPL-001", answer)
            self.assertEqual(runtime.mcp.calls, [("grid__regulatory_list_tests", {}, True)])
            self.assertEqual(runtime.model.calls[0][1]["max_new_tokens"], 4096)
            self.assertEqual(json.loads(state_path.read_text(encoding="utf-8"))["status"], "complete")
            self.assertEqual(store.messages(chat["id"])[-1]["content"], answer)
            store.close()


if __name__ == "__main__":
    unittest.main()
