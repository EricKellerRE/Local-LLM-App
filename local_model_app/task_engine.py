from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

from local_model_app.task_coordinator import TaskCoordinatorProtocol, WorkItemExecutorProtocol
from local_model_app.task_models import CompletionAudit, TaskStatus
from local_model_app.task_store import TaskStore


class UniversalTaskEngine:
    """Runs resumable tasks in bounded episodes and checkpoints every transition."""

    def __init__(
        self,
        store: TaskStore,
        coordinator: TaskCoordinatorProtocol,
        *,
        executors: dict[str, WorkItemExecutorProtocol] | None = None,
        poll_seconds: float = 5.0,
        lease_seconds: int = 1800,
    ) -> None:
        self.store = store
        self.coordinator = coordinator
        self.executors: dict[str, WorkItemExecutorProtocol] = dict(executors or {"model": coordinator})
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
        self.worker_id = f"local-{uuid.uuid4()}"
        self._wake_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None

    def register_executor(self, kind: str, executor: WorkItemExecutorProtocol) -> None:
        """Register a task-specific executor, such as a future MCP adapter."""
        self.executors[kind] = executor

    async def start(self) -> None:
        if self._runner and not self._runner.done():
            return
        # This application owns one local worker. Any persisted running lease at
        # process startup belongs to the previous process and is therefore orphaned.
        self.store.recover_expired_leases(recover_all=True)
        self._stop_event.clear()
        self._runner = asyncio.create_task(self._run(), name="universal-task-worker")

    async def stop(self) -> None:
        self.request_stop()
        if self._runner:
            await self._runner
            self._runner = None

    def request_stop(self) -> None:
        """Stop after the current checkpointable episode finishes."""
        self._stop_event.set()
        self._wake_event.set()

    def wake(self) -> None:
        self._wake_event.set()

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            worked = await self.run_pending_once()
            if worked:
                continue
            self._wake_event.clear()
            try:
                await asyncio.wait_for(self._wake_event.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass

    async def run_pending_once(self) -> bool:
        task = self.store.claim_task(self.worker_id, lease_seconds=self.lease_seconds)
        if task is None:
            return False
        await self._run_episode(task)
        return True

    def _retry_at(self, task: dict, attempt: int, requested_wait: int | None = None) -> datetime:
        policy = task["definition"]["execution"]
        if requested_wait is None:
            initial = int(policy["retry_initial_seconds"])
            maximum = int(policy["retry_maximum_seconds"])
            wait_seconds = min(maximum, initial * (2 ** max(0, attempt - 1)))
        else:
            wait_seconds = min(int(policy["retry_maximum_seconds"]), requested_wait)
        return datetime.now(timezone.utc) + timedelta(seconds=wait_seconds)

    async def _run_episode(self, task: dict) -> None:
        task_id = task["id"]
        policy = task["definition"]["execution"]
        deadline = policy.get("deadline")
        if deadline and datetime.fromisoformat(deadline) <= datetime.now(timezone.utc):
            self.store.release_task(
                task_id,
                TaskStatus.FAILED,
                summary="The configured task deadline passed before completion.",
                error="deadline_exceeded",
            )
            return

        try:
            items = self.store.work_items(task_id)
            if not items:
                planned = await self.coordinator.plan_task(task)
                self.store.add_work_items(task_id, planned)

            max_steps = int(policy["max_steps_per_episode"])
            for _ in range(max_steps):
                current_task = self.store.get_task(task_id)
                if current_task["status"] != TaskStatus.RUNNING.value:
                    return
                item = self.store.claim_work_item(task_id)
                if item is None:
                    await self._finish_or_extend(task_id)
                    return

                completed = [
                    candidate
                    for candidate in self.store.work_items(task_id)
                    if candidate["status"] == "completed"
                ]
                executor = self.executors.get(item["kind"])
                if executor is None:
                    reason = f"No executor is registered for work-item kind '{item['kind']}'."
                    self.store.block_work_item(item["id"], reason)
                    self.store.release_task(
                        task_id,
                        TaskStatus.WAITING_FOR_TOOLS,
                        summary=reason,
                        waiting_reason=reason,
                    )
                    return
                try:
                    outcome = await executor.execute_work_item(task, item, completed)
                except Exception as exc:
                    await self._handle_execution_error(task, item, exc)
                    return

                if self.store.get_task(task_id)["status"] != TaskStatus.RUNNING.value:
                    return
                if outcome.outcome == "completed":
                    self.store.complete_work_item(item["id"], outcome.model_dump(mode="json"))
                    self.store.add_work_items(task_id, outcome.follow_up_items)
                    await asyncio.sleep(0)
                    continue
                if outcome.outcome == "retry":
                    retry_at = self._retry_at(task, item["attempts"], outcome.wait_seconds)
                    self.store.retry_work_item(item["id"], outcome.summary, retry_at)
                    self.store.release_task(
                        task_id,
                        TaskStatus.WAITING,
                        summary=outcome.summary,
                        waiting_reason="retry_scheduled",
                        next_run_at=retry_at,
                    )
                    return
                if outcome.outcome == "waiting_for_input":
                    self.store.block_work_item(item["id"], outcome.summary)
                    self.store.release_task(
                        task_id,
                        TaskStatus.WAITING_FOR_INPUT,
                        summary=outcome.summary,
                        waiting_reason=outcome.summary,
                    )
                    return
                if outcome.outcome == "waiting_for_tools":
                    self.store.block_work_item(item["id"], outcome.summary)
                    self.store.release_task(
                        task_id,
                        TaskStatus.WAITING_FOR_TOOLS,
                        summary=outcome.summary,
                        waiting_reason=outcome.summary,
                    )
                    return
                self.store.fail_work_item(item["id"], outcome.summary)
                self.store.release_task(
                    task_id,
                    TaskStatus.FAILED,
                    summary=outcome.summary,
                    error=outcome.summary,
                )
                return

            self.store.release_task(
                task_id,
                TaskStatus.RUNNABLE,
                summary="Episode call budget reached; work was checkpointed for the next episode.",
                next_run_at=datetime.now(timezone.utc),
            )
        except Exception as exc:
            self.store.release_task(
                task_id,
                TaskStatus.FAILED,
                summary="The task engine could not continue this episode.",
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _handle_execution_error(self, task: dict, item: dict, exc: Exception) -> None:
        policy = task["definition"]["execution"]
        maximum_attempts = policy.get("maximum_attempts")
        reason = f"{type(exc).__name__}: {exc}"
        if maximum_attempts is not None and item["attempts"] >= int(maximum_attempts):
            self.store.fail_work_item(item["id"], reason)
            self.store.release_task(
                task["id"], TaskStatus.FAILED, summary="Retry limit reached.", error=reason
            )
            return
        retry_at = self._retry_at(task, item["attempts"])
        self.store.retry_work_item(item["id"], reason, retry_at)
        self.store.release_task(
            task["id"],
            TaskStatus.WAITING,
            summary="A transient task error was checkpointed and scheduled for retry.",
            waiting_reason="retry_scheduled",
            next_run_at=retry_at,
            error=reason,
        )

    async def _finish_or_extend(self, task_id: str) -> None:
        task = self.store.get_task(task_id)
        items = self.store.work_items(task_id)
        failed = [item for item in items if item["status"] == "failed"]
        blocked = [item for item in items if item["status"] == "blocked"]
        pending_later = [item for item in items if item["status"] == "pending"]
        if failed:
            self.store.release_task(
                task_id,
                TaskStatus.FAILED,
                summary="One or more work items failed.",
                error=failed[-1].get("error") or "work_item_failed",
            )
            return
        if blocked:
            self.store.release_task(
                task_id,
                TaskStatus.WAITING_FOR_INPUT,
                summary="Work is blocked and requires attention.",
                waiting_reason=blocked[-1].get("error"),
            )
            return
        if pending_later:
            run_times = [item["run_after"] for item in pending_later if item.get("run_after")]
            if run_times:
                self.store.release_task(
                    task_id,
                    TaskStatus.WAITING,
                    summary="Waiting until the next work item is eligible to run.",
                    waiting_reason="scheduled",
                    next_run_at=datetime.fromisoformat(min(run_times)),
                )
            else:
                self.store.release_task(
                    task_id,
                    TaskStatus.WAITING_FOR_INPUT,
                    summary="No pending work item has satisfiable dependencies.",
                    waiting_reason="A work-item dependency is missing or cyclic.",
                )
            return

        audit: CompletionAudit = await self.coordinator.audit_completion(task, items)
        self.store.record_audit(task_id, audit.model_dump(mode="json"))
        if audit.passed:
            self.store.release_task(task_id, TaskStatus.COMPLETED, summary=audit.summary)
            return
        if audit.follow_up_items:
            self.store.add_work_items(task_id, audit.follow_up_items)
            self.store.release_task(
                task_id,
                TaskStatus.RUNNABLE,
                summary=audit.summary,
                next_run_at=datetime.now(timezone.utc),
            )
            return
        self.store.release_task(
            task_id,
            TaskStatus.WAITING_FOR_INPUT,
            summary=audit.summary,
            waiting_reason="Completion audit found unresolved criteria without an automatic remedy.",
        )
