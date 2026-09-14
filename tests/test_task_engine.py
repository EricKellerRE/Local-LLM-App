import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from local_model_app.task_engine import UniversalTaskEngine
from local_model_app.task_models import (
    CompletionAudit,
    CriterionAudit,
    ProposedWorkItem,
    TaskDefinition,
    WorkItemOutcome,
)
from local_model_app.task_store import TaskStore


class FakeCoordinator:
    async def propose(self, request):
        return TaskDefinition(title="Proposed", goal=request, success_criteria=["Done"])

    async def plan_task(self, task):
        return [
            ProposedWorkItem(
                title="First step",
                instructions="Complete a bounded step",
                completion_check="The step has a result",
            )
        ]

    async def execute_work_item(self, task, item, completed_items):
        return WorkItemOutcome(outcome="completed", summary="Step complete", result="Evidence")

    async def audit_completion(self, task, work_items):
        return CompletionAudit(
            passed=True,
            summary="All criteria passed",
            criteria=[CriterionAudit(criterion="Done", satisfied=True, evidence="Evidence")],
        )


class WaitingCoordinator(FakeCoordinator):
    async def execute_work_item(self, task, item, completed_items):
        return WorkItemOutcome(
            outcome="waiting_for_tools",
            summary="A literature search tool is required.",
        )


class TaskEngineTests(unittest.TestCase):
    def test_bounded_episode_completes_and_audits_task(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            store = TaskStore(Path(directory) / "tasks.sqlite3")
            task = store.create_task(
                "Do this",
                TaskDefinition(title="Task", goal="Do this", success_criteria=["Done"]),
            )
            store.start_task(task["id"])
            engine = UniversalTaskEngine(store, FakeCoordinator(), poll_seconds=0.01)

            self.assertTrue(asyncio.run(engine.run_pending_once()))

            detail = store.get_task(task["id"], include_details=True)
            self.assertEqual(detail["status"], "completed")
            self.assertEqual(detail["work_items"][0]["status"], "completed")
            self.assertTrue(detail["latest_audit"]["passed"])
            store.close()

    def test_missing_capability_waits_without_claiming_success(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            store = TaskStore(Path(directory) / "tasks.sqlite3")
            task = store.create_task(
                "Research this",
                TaskDefinition(title="Research", goal="Research this", success_criteria=["Sources verified"]),
            )
            store.start_task(task["id"])
            engine = UniversalTaskEngine(store, WaitingCoordinator(), poll_seconds=0.01)

            asyncio.run(engine.run_pending_once())

            detail = store.get_task(task["id"], include_details=True)
            self.assertEqual(detail["status"], "waiting_for_tools")
            self.assertEqual(detail["work_items"][0]["status"], "blocked")
            self.assertIn("literature search", detail["waiting_reason"])
            store.close()


if __name__ == "__main__":
    unittest.main()
