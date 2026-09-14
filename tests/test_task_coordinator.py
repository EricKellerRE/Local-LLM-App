import asyncio
import unittest

from local_model_app.task_coordinator import ModelTaskCoordinator


class SequencedGenerator:
    def __init__(self, replies):
        self.replies = list(replies)

    async def __call__(self, messages):
        return self.replies.pop(0)


class TaskCoordinatorTests(unittest.TestCase):
    def test_model_assembles_valid_definition_from_natural_language(self) -> None:
        generator = SequencedGenerator(
            [
                """```json
                {
                  "title": "Literature review",
                  "goal": "Review the literature",
                  "success_criteria": ["100 sources are verified"],
                  "deliverables": ["Review"],
                  "constraints": ["Use primary sources"],
                  "execution": {
                    "deadline": null,
                    "max_steps_per_episode": 8,
                    "maximum_attempts": null,
                    "retry_initial_seconds": 60,
                    "retry_maximum_seconds": 3600,
                    "resume_after_restart": true
                  },
                  "metadata": {"task_type": "literature_review"}
                }
                ```"""
            ]
        )
        coordinator = ModelTaskCoordinator(generator)

        definition = asyncio.run(coordinator.propose("Review 100 papers"))

        self.assertEqual(definition.title, "Literature review")
        self.assertIsNone(definition.execution.maximum_attempts)
        self.assertTrue(definition.execution.resume_after_restart)


if __name__ == "__main__":
    unittest.main()
