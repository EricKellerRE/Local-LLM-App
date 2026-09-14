import json
import unittest

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


if __name__ == "__main__":
    unittest.main()
