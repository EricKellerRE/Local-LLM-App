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

    def test_web_research_uses_fast_deterministic_evidence_plan(self) -> None:
        generator = SequencedGenerator([])
        coordinator = ModelTaskCoordinator(generator)
        task = {"definition": {
            "goal": "Research long-term memory formation and maintenance",
            "success_criteria": ["A sourced report is complete"],
            "metadata": {
                "mode": "durable_tools",
                "plugin_ids": ["local.web-research"],
                "workflow_skill_id": "scholarly-research-report",
                "workflow_skill_version": "1.1.0",
                "skill_inputs": {"topic": "long-term memory formation and maintenance"},
            },
        }}

        items = asyncio.run(coordinator.plan_task(task))

        self.assertEqual(len(items), 9)
        self.assertEqual(sum(item.kind == "research_discovery" for item in items), 1)
        self.assertEqual(sum(item.kind == "research_notes" for item in items), 0)
        self.assertEqual(sum(item.kind == "section" for item in items), 7)
        self.assertEqual(items[-1].kind, "synthesis")
        self.assertEqual(
            set(items[-1].depends_on),
            {item.key for item in items if item.kind == "section"},
        )
        self.assertEqual(generator.replies, [])

    def test_web_research_audit_is_host_enforced_without_another_model_turn(self) -> None:
        generator = SequencedGenerator([])
        coordinator = ModelTaskCoordinator(generator)
        task = {"definition": {
            "success_criteria": ["A complete report is produced."],
            "metadata": {
                "plugin_ids": ["local.web-research"],
                "workflow_skill_id": "scholarly-research-report",
                "workflow_skill_version": "1.1.0",
                "skill_inputs": {"topic": "memory"},
                "report_min_words": 6000,
                "report_min_sources": 12,
            },
        }}
        items = [
            {"kind": "research_discovery", "status": "completed", "result": {}},
            *[{"kind": "section", "status": "completed", "result": {}} for _ in range(7)],
            {"kind": "synthesis", "status": "completed", "result": {
                "completion_evidence": ["word_count=7000", "source_url_count=20"],
                "artifacts": [{
                    "media_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                }],
            }},
        ]

        audit = asyncio.run(coordinator.audit_completion(task, items))

        self.assertTrue(audit.passed)
        self.assertEqual(generator.replies, [])


if __name__ == "__main__":
    unittest.main()
