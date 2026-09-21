import json
import tempfile
import unittest
from pathlib import Path

from local_model_app.workflow_skills import WorkflowSkillRegistry, research_topic


class WorkflowSkillTests(unittest.TestCase):
    def test_research_topic_separates_subject_from_workflow_instructions(self) -> None:
        request = (
            "Produce a comprehensive research report on the biological mechanisms for long-term memory "
            "formation and maintenance. Search paginated results, follow references, and export a document."
        )

        self.assertEqual(
            research_topic(request),
            "the biological mechanisms for long-term memory formation and maintenance",
        )

    def test_scholarly_skill_builds_a_concise_definition_and_templated_graph(self) -> None:
        registry = WorkflowSkillRegistry.default()
        request = (
            "Produce a comprehensive research report on biological mechanisms of long-term memory formation "
            "and maintenance. Keep working overnight and export it as a document."
        )
        skill = registry.select(request, ["local.web-research"])

        self.assertIsNotNone(skill)
        definition = skill.task_definition(
            request,
            ["local.web-research"],
            metadata={"chat_id": "chat-1", "research_seed_sources": 12},
        )
        items = skill.instantiate_work_items(definition.metadata["skill_inputs"])

        self.assertEqual(definition.goal, "biological mechanisms of long-term memory formation and maintenance")
        self.assertEqual(definition.metadata["workflow_skill_id"], "scholarly-research-report")
        self.assertEqual(definition.metadata["workflow_skill_version"], "1.2.0")
        self.assertEqual(definition.metadata["chat_id"], "chat-1")
        self.assertIn("scholarly_article", definition.metadata["source_policy"]["core_types"])
        self.assertIn("tertiary_overview", definition.metadata["source_policy"]["supplemental_types"])
        self.assertFalse(definition.metadata["source_policy"].get("supplemental_counts_toward_target", False))
        self.assertEqual(len(items), 9)
        self.assertEqual(items[0].key, "source-corpus")
        self.assertFalse(any(item.kind == "research_notes" for item in items))
        self.assertIn(definition.goal, items[0].instructions)
        self.assertNotIn("{topic}", json.dumps([item.model_dump() for item in items]))

    def test_skill_requires_its_plugin_and_matching_request(self) -> None:
        registry = WorkflowSkillRegistry.default()

        self.assertIsNone(registry.select("Write a research report about memory", []))
        self.assertIsNone(registry.select("Explain this paragraph", ["local.web-research"]))
        self.assertIsNotNone(registry.select(
            "Review the literature on memory",
            ["local.web-research"],
        ))

    def test_invalid_manifest_fails_fast_with_its_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "broken.json"
            path.write_text('{"id":"broken"}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "broken.json"):
                WorkflowSkillRegistry(Path(directory))


if __name__ == "__main__":
    unittest.main()
