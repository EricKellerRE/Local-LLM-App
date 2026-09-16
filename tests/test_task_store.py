import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from local_model_app.task_models import ProposedWorkItem, TaskDefinition, TaskStatus
from local_model_app.task_store import TaskStore


class TaskStoreTests(unittest.TestCase):
    def test_task_lifecycle_and_work_item_history(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            store = TaskStore(Path(directory) / "tasks.sqlite3")
            definition = TaskDefinition(
                title="Durable task",
                goal="Finish something",
                success_criteria=["The result is recorded"],
            )
            task = store.create_task("Please finish this", definition)
            self.assertEqual(task["status"], "draft")
            store.start_task(task["id"])
            claimed = store.claim_task("worker-one")
            self.assertEqual(claimed["status"], "running")

            created = store.add_work_items(
                task["id"],
                [
                    ProposedWorkItem(
                        title="Do the work",
                        instructions="Produce the result",
                        completion_check="A result exists",
                    )
                ],
            )
            item = store.claim_work_item(task["id"])
            self.assertEqual(item["id"], created[0]["id"])
            checkpointed = store.checkpoint_work_item(
                item["id"], {"pending_mcp_task": {"task_id": "task-123"}}
            )
            self.assertEqual(checkpointed["status"], "running")
            self.assertEqual(
                checkpointed["result"]["pending_mcp_task"]["task_id"], "task-123"
            )
            store.complete_work_item(item["id"], {"summary": "done"})
            store.record_audit(task["id"], {"passed": True, "summary": "verified"})
            finished = store.release_task(task["id"], TaskStatus.COMPLETED, summary="verified")

            self.assertEqual(finished["status"], "completed")
            detail = store.get_task(task["id"], include_details=True)
            self.assertEqual(detail["work_items"][0]["result"]["summary"], "done")
            self.assertTrue(detail["latest_audit"]["passed"])
            self.assertIn("work_item_completed", [event["event"] for event in detail["events"]])
            self.assertEqual([item["id"] for item in store.undelivered_terminal_tasks()], [task["id"]])
            store.mark_delivered(task["id"])
            self.assertEqual(store.undelivered_terminal_tasks(), [])
            store.close()

    def test_work_item_dependencies_are_enforced(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            store = TaskStore(Path(directory) / "tasks.sqlite3")
            task = store.create_task(
                "Run in order",
                TaskDefinition(title="Ordered", goal="Run in order", success_criteria=["Both complete"]),
            )
            store.start_task(task["id"])
            store.claim_task("worker")
            store.add_work_items(
                task["id"],
                [
                    ProposedWorkItem(
                        key="synthesize",
                        title="Synthesize",
                        instructions="Use the collection",
                        completion_check="Synthesis exists",
                        priority=10,
                        depends_on=["collect"],
                    ),
                    ProposedWorkItem(
                        key="collect",
                        title="Collect",
                        instructions="Collect inputs",
                        completion_check="Inputs exist",
                    ),
                ],
            )
            first = store.claim_work_item(task["id"])
            self.assertEqual(first["key"], "collect")
            store.complete_work_item(first["id"], {"summary": "collected"})
            self.assertEqual(store.claim_work_item(task["id"])["key"], "synthesize")
            store.close()

    def test_pause_requeues_in_flight_work(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            store = TaskStore(Path(directory) / "tasks.sqlite3")
            task = store.create_task(
                "Pause safely",
                TaskDefinition(title="Pause", goal="Pause safely", success_criteria=["Done"]),
            )
            store.start_task(task["id"])
            store.claim_task("worker")
            created = store.add_work_items(
                task["id"],
                [ProposedWorkItem(title="Work", instructions="Work", completion_check="Done")],
            )
            store.claim_work_item(task["id"])

            store.pause_task(task["id"])
            self.assertEqual(store.get_work_item(created[0]["id"])["status"], "pending")
            store.resume_task(task["id"])
            self.assertEqual(store.claim_task("worker")["id"], task["id"])
            store.close()

    def test_expired_run_is_recovered_and_scheduled_wait_is_promoted(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            store = TaskStore(Path(directory) / "tasks.sqlite3")
            definition = TaskDefinition(title="Recover", goal="Recover", success_criteria=["Recovered"])
            task = store.create_task("Recover", definition)
            store.start_task(task["id"])
            store.claim_task("old-worker", lease_seconds=1)
            with store._lock:
                store._connection.execute(
                    "UPDATE tasks SET lease_expires_at = ? WHERE id = ?",
                    ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), task["id"]),
                )
                store._connection.commit()
            self.assertEqual(store.recover_expired_leases(), 1)
            self.assertEqual(store.claim_task("new-worker")["id"], task["id"])

            retry_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            store.release_task(task["id"], TaskStatus.WAITING, next_run_at=retry_at)
            self.assertEqual(store.claim_task("new-worker")["id"], task["id"])
            store.close()


if __name__ == "__main__":
    unittest.main()
