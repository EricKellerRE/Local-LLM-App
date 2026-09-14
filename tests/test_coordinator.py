import unittest
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


if __name__ == "__main__":
    unittest.main()
