import asyncio
import unittest

from local_model_app.task_coordinator import ModelTaskCoordinator


class SequencedGenerator:
    def __init__(self, replies):
        self.replies = list(replies)

    async def __call__(self, messages):
        self.messages = messages
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

    def test_durable_tool_plan_uses_capabilities_and_finishes_with_synthesis(self) -> None:
        generator = SequencedGenerator([
            '{"work_items": ['
            '{"key":"baseline","kind":"tool","title":"Measure baseline","instructions":"Inspect case",'
            '"completion_check":"Metric recorded","priority":10,"depends_on":[]},'
            '{"key":"report","kind":"synthesis","title":"Write report","instructions":"Compare results",'
            '"completion_check":"Report complete","priority":0,"depends_on":["baseline"]}'
            ']}'
        ])

        async def catalog(plugin_ids):
            self.assertEqual(plugin_ids, ["grid-workshop.powerworld"])
            return [{"name": "grid__case_overview", "description": "Inspect a case"}]

        coordinator = ModelTaskCoordinator(generator, capability_catalog=catalog)
        task = {"definition": {
            "goal": "Improve the grid",
            "success_criteria": ["Metric improves"],
            "metadata": {"mode": "durable_tools", "plugin_ids": ["grid-workshop.powerworld"]},
        }}

        items = asyncio.run(coordinator.plan_task(task))

        self.assertEqual([item.kind for item in items], ["tool", "synthesis"])
        self.assertIn("grid__case_overview", generator.messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
