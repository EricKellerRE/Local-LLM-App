import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from local_model_app.coordinator import Coordinator
from local_model_app.scratchpad import Scratchpad


class FakeModel:
    def __init__(self) -> None:
        self.loaded = False
        self.settings = type("Settings", (), {"model_id": "test-model"})()
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        self.loaded = True
        return "Plan: clarify and answer." if len(self.calls) == 1 else "Here is the answer."


class CoordinatorTests(unittest.TestCase):
    def test_plans_answers_and_persists_visible_notes(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            model = FakeModel()
            coordinator = Coordinator(model, Scratchpad(Path(directory) / "scratchpad.jsonl"))
            self.assertEqual(coordinator.respond("What is local inference?"), "Here is the answer.")
            self.assertEqual(len(model.calls), 2)
            notes = coordinator.scratchpad.render()
            self.assertIn("[plan] Plan: clarify and answer.", notes)
            self.assertIn("[answer_summary] Here is the answer.", notes)

    def test_long_chat_creates_and_reuses_semantic_context_checkpoint(self) -> None:
        class PressureModel(FakeModel):
            effective_context_window = 1024

            def __init__(self) -> None:
                super().__init__()
                self.settings = type("Settings", (), {
                    "model_id": "test-model",
                    "context_window": 1024,
                    "max_new_tokens": 256,
                })()

            def count_prompt_tokens(self, messages, tools=None):
                return 900 if len(messages) > 6 else 200

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            model = PressureModel()
            scratchpad = Scratchpad(Path(directory) / "scratchpad.jsonl")
            coordinator = Coordinator(model, scratchpad)
            coordinator.history = [
                {"role": "user" if index % 2 == 0 else "assistant", "content": f"message {index}"}
                for index in range(14)
            ]

            coordinator.respond("What remains?")

            checkpoint_path = scratchpad.path.with_suffix(".context.json")
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["compacted_message_count"], 10)
            self.assertEqual(checkpoint["trigger"], "token_pressure")
            self.assertIn("message 0", checkpoint["user_requirements"])
            self.assertIn("message 1", checkpoint["decisions"])
            self.assertTrue(any("Persisted conversation checkpoint" in call[0]["content"] for call in model.calls))


if __name__ == "__main__":
    unittest.main()
