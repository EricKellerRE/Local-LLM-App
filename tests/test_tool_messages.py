import json
import unittest
from types import SimpleNamespace

from local_model_app.model import TransformersModel


class ToolMessageTests(unittest.TestCase):
    def test_openai_tool_arguments_are_deserialized_for_gemma_template(self) -> None:
        messages = [
            {"role": "user", "content": "Route this request."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "powerworld_route_request",
                            "arguments": json.dumps({"request_text": "compare two cases"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "powerworld_route_request", "content": '{"ok": true}'},
        ]

        converted = TransformersModel._processor_messages(messages)

        self.assertEqual(
            converted[1]["tool_calls"][0]["function"]["arguments"],
            {"request_text": "compare two cases"},
        )
        self.assertEqual(converted[1]["content"], [])
        self.assertEqual(converted[2]["tool_call_id"], "call_1")

    def test_context_limit_honors_user_setting_and_native_model_limit(self) -> None:
        adapter = TransformersModel(SimpleNamespace(context_window=16_384, reasoning_budget=None))
        adapter.model = SimpleNamespace(config=SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=8_192),
        ))

        self.assertEqual(adapter.native_context_window, 8_192)
        self.assertEqual(adapter.effective_context_window, 8_192)

    def test_reasoning_budget_is_passed_to_compatible_chat_templates(self) -> None:
        adapter = TransformersModel(SimpleNamespace(context_window=None, reasoning_budget=1_024))

        self.assertEqual(adapter._template_reasoning_options(512), {
            "enable_thinking": True,
            "thinking_budget": 512,
            "reasoning_budget": 512,
        })
        adapter.settings.reasoning_budget = 0
        self.assertEqual(adapter._template_reasoning_options(512), {"enable_thinking": False})

    def test_context_trimming_removes_an_old_turn_but_keeps_system_and_latest(self) -> None:
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new question"},
        ]

        shortened = TransformersModel._without_oldest_turn(messages)

        self.assertEqual(shortened, [messages[0], messages[3]])


if __name__ == "__main__":
    unittest.main()
