import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from local_model_app.scratchpad import Scratchpad


class ScratchpadTests(unittest.TestCase):
    def test_records_and_renders_notes(self) -> None:
        # Keep temporary test data inside the repository so it also works in
        # restricted Windows environments where the system temp folder is unavailable.
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            pad = Scratchpad(Path(directory) / "nested" / "scratchpad.jsonl")
            pad.add("plan", "Identify the question")
            pad.add("answer_summary", "Respond concisely")
            self.assertEqual([item["kind"] for item in pad.recent()], ["plan", "answer_summary"])
            self.assertIn("[plan] Identify the question", pad.render())


if __name__ == "__main__":
    unittest.main()
