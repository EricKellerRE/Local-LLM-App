"""Manual end-to-end smoke test for the configured local model."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_model_app.config import Settings
from local_model_app.coordinator import Coordinator
from local_model_app.model import TransformersModel
from local_model_app.scratchpad import Scratchpad


def main() -> None:
    root = ROOT
    coordinator = Coordinator(
        model=TransformersModel(Settings.from_environment(root)),
        scratchpad=Scratchpad(root / "data" / "scratchpad.jsonl"),
    )
    task = (
        "Create a five-step plan for organizing a one-day workshop. First verify that it has exactly five steps "
        "and covers goal, audience, agenda, logistics, and follow-up. Return only the corrected final five-step plan."
    )
    print(coordinator.respond(task))


if __name__ == "__main__":
    main()
