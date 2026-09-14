from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from local_model_app.task_models import (
    TERMINAL_TASK_STATUSES,
    ProposedWorkItem,
    TaskDefinition,
    TaskStatus,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    return json.loads(value)


class TaskStore:
    """Durable state for long-running, resumable tasks."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                request TEXT NOT NULL,
                definition_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                next_run_at TEXT,
                lease_owner TEXT,
                lease_expires_at TEXT,
                episode_count INTEGER NOT NULL DEFAULT 0,
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                current_summary TEXT NOT NULL DEFAULT '',
                waiting_reason TEXT,
                last_error TEXT
            );
            CREATE INDEX IF NOT EXISTS tasks_status_next_run ON tasks(status, next_run_at);

            CREATE TABLE IF NOT EXISTS task_work_items (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                item_key TEXT,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                instructions TEXT NOT NULL,
                completion_check TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                sequence INTEGER NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                run_after TEXT,
                result_json TEXT,
                depends_on_json TEXT NOT NULL DEFAULT '[]',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS task_work_items_ready
                ON task_work_items(task_id, status, priority DESC, sequence);

            CREATE TABLE IF NOT EXISTS task_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                work_item_id TEXT,
                event TEXT NOT NULL,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS task_events_task_id_id ON task_events(task_id, id);

            CREATE TABLE IF NOT EXISTS task_audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                passed INTEGER NOT NULL,
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        work_columns = {
            row["name"] for row in self._connection.execute("PRAGMA table_info(task_work_items)").fetchall()
        }
        if "item_key" not in work_columns:
            self._connection.execute("ALTER TABLE task_work_items ADD COLUMN item_key TEXT")
            self._connection.execute("UPDATE task_work_items SET item_key = id WHERE item_key IS NULL")
        if "depends_on_json" not in work_columns:
            self._connection.execute(
                "ALTER TABLE task_work_items ADD COLUMN depends_on_json TEXT NOT NULL DEFAULT '[]'"
            )
        self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS task_work_items_task_key ON task_work_items(task_id, item_key)"
        )
        self._connection.commit()

    def _event(
        self,
        task_id: str,
        event: str,
        data: dict[str, Any] | None = None,
        work_item_id: str | None = None,
    ) -> None:
        self._connection.execute(
            "INSERT INTO task_events(task_id, work_item_id, event, data_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (task_id, work_item_id, event, json.dumps(data or {}, ensure_ascii=False), _timestamp()),
        )

    @staticmethod
    def _task_row(row: sqlite3.Row) -> dict[str, Any]:
        task = dict(row)
        task["definition"] = _loads(task.pop("definition_json"), {})
        return task

    @staticmethod
    def _work_item_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["result"] = _loads(item.pop("result_json"), None)
        item["depends_on"] = _loads(item.pop("depends_on_json"), [])
        item["key"] = item.pop("item_key")
        return item

    def create_task(self, request: str, definition: TaskDefinition) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        now = _timestamp()
        with self._lock:
            self._connection.execute(
                """INSERT INTO tasks(
                    id, request, definition_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (task_id, request, definition.model_dump_json(), TaskStatus.DRAFT.value, now, now),
            )
            self._event(task_id, "task_created", {"title": definition.title})
            self._connection.commit()
        return self.get_task(task_id)

    def get_task(self, task_id: str, *, include_details: bool = False) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            task = self._task_row(row)
            if include_details:
                task["work_items"] = self.work_items(task_id)
                task["events"] = self.events(task_id, limit=100)
                audit = self._connection.execute(
                    "SELECT passed, report_json, created_at FROM task_audits WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                    (task_id,),
                ).fetchone()
                task["latest_audit"] = None if audit is None else {
                    "passed": bool(audit["passed"]),
                    "report": _loads(audit["report_json"], {}),
                    "created_at": audit["created_at"],
                }
            return task

    def list_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM tasks ORDER BY updated_at DESC"
            ).fetchall()
        return [self._task_row(row) for row in rows]

    def start_task(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self.get_task(task_id)
            if task["status"] != TaskStatus.DRAFT.value:
                raise ValueError("Only a draft task can be started.")
            now = _timestamp()
            self._connection.execute(
                "UPDATE tasks SET status = ?, started_at = ?, next_run_at = ?, updated_at = ? WHERE id = ?",
                (TaskStatus.RUNNABLE.value, now, now, now, task_id),
            )
            self._event(task_id, "task_started")
            self._connection.commit()
        return self.get_task(task_id)

    def pause_task(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self.get_task(task_id)
            if task["status"] in TERMINAL_TASK_STATUSES or task["status"] == TaskStatus.DRAFT.value:
                raise ValueError("This task cannot be paused in its current state.")
            self._connection.execute(
                """UPDATE tasks SET status = ?, lease_owner = NULL, lease_expires_at = NULL,
                   updated_at = ? WHERE id = ?""",
                (TaskStatus.PAUSED.value, _timestamp(), task_id),
            )
            self._connection.execute(
                "UPDATE task_work_items SET status = 'pending', updated_at = ? WHERE task_id = ? AND status = 'running'",
                (_timestamp(), task_id),
            )
            self._event(task_id, "task_paused")
            self._connection.commit()
        return self.get_task(task_id)

    def resume_task(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self.get_task(task_id)
            allowed = {
                TaskStatus.PAUSED.value,
                TaskStatus.WAITING.value,
                TaskStatus.WAITING_FOR_INPUT.value,
                TaskStatus.WAITING_FOR_TOOLS.value,
                TaskStatus.FAILED.value,
            }
            if task["status"] not in allowed:
                raise ValueError("This task cannot be resumed in its current state.")
            now = _timestamp()
            self._connection.execute(
                """UPDATE tasks SET status = ?, next_run_at = ?, waiting_reason = NULL,
                   last_error = NULL, completed_at = NULL, consecutive_failures = 0,
                   updated_at = ? WHERE id = ?""",
                (TaskStatus.RUNNABLE.value, now, now, task_id),
            )
            self._connection.execute(
                "UPDATE task_work_items SET status = ?, updated_at = ? WHERE task_id = ? AND status IN ('blocked', 'failed', 'running')",
                ("pending", now, task_id),
            )
            self._event(task_id, "task_resumed")
            self._connection.commit()
        return self.get_task(task_id)

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self.get_task(task_id)
            if task["status"] in TERMINAL_TASK_STATUSES:
                return task
            now = _timestamp()
            self._connection.execute(
                """UPDATE tasks SET status = ?, completed_at = ?, lease_owner = NULL,
                   lease_expires_at = NULL, updated_at = ? WHERE id = ?""",
                (TaskStatus.CANCELLED.value, now, now, task_id),
            )
            self._connection.execute(
                "UPDATE task_work_items SET status = ?, updated_at = ? WHERE task_id = ? AND status IN ('pending', 'running', 'blocked')",
                ("cancelled", now, task_id),
            )
            self._event(task_id, "task_cancelled")
            self._connection.commit()
        return self.get_task(task_id)

    def recover_expired_leases(self, *, recover_all: bool = False) -> int:
        now = _timestamp()
        with self._lock:
            if recover_all:
                rows = self._connection.execute(
                    "SELECT id FROM tasks WHERE status = ?", (TaskStatus.RUNNING.value,)
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT id FROM tasks WHERE status = ? AND (lease_expires_at IS NULL OR lease_expires_at <= ?)",
                    (TaskStatus.RUNNING.value, now),
                ).fetchall()
            for row in rows:
                self._connection.execute(
                    """UPDATE tasks SET status = ?, next_run_at = ?, lease_owner = NULL,
                       lease_expires_at = NULL, updated_at = ? WHERE id = ?""",
                    (TaskStatus.RUNNABLE.value, now, now, row["id"]),
                )
                self._connection.execute(
                    "UPDATE task_work_items SET status = 'pending', updated_at = ? WHERE task_id = ? AND status = 'running'",
                    (now, row["id"]),
                )
                self._event(row["id"], "lease_recovered")
            self._connection.commit()
        return len(rows)

    def claim_task(self, worker_id: str, *, lease_seconds: int = 900) -> dict[str, Any] | None:
        now_dt = _now()
        now = _timestamp(now_dt)
        lease_expires = _timestamp(now_dt + timedelta(seconds=lease_seconds))
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    """UPDATE tasks SET status = ?, updated_at = ?
                       WHERE status = ? AND next_run_at IS NOT NULL AND next_run_at <= ?""",
                    (TaskStatus.RUNNABLE.value, now, TaskStatus.WAITING.value, now),
                )
                row = self._connection.execute(
                    """SELECT id FROM tasks
                       WHERE status = ? AND (next_run_at IS NULL OR next_run_at <= ?)
                       ORDER BY COALESCE(next_run_at, created_at), created_at LIMIT 1""",
                    (TaskStatus.RUNNABLE.value, now),
                ).fetchone()
                if row is None:
                    self._connection.commit()
                    return None
                updated = self._connection.execute(
                    """UPDATE tasks SET status = ?, lease_owner = ?, lease_expires_at = ?,
                       episode_count = episode_count + 1, updated_at = ?
                       WHERE id = ? AND status = ?""",
                    (
                        TaskStatus.RUNNING.value,
                        worker_id,
                        lease_expires,
                        now,
                        row["id"],
                        TaskStatus.RUNNABLE.value,
                    ),
                )
                if updated.rowcount != 1:
                    self._connection.rollback()
                    return None
                self._event(row["id"], "episode_started", {"worker_id": worker_id})
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_task(row["id"])

    def release_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        summary: str = "",
        waiting_reason: str | None = None,
        next_run_at: datetime | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            current = self.get_task(task_id)
            if current["status"] != TaskStatus.RUNNING.value:
                return current
            now = _timestamp()
            completed_at = now if status.value in TERMINAL_TASK_STATUSES else None
            failures = current["consecutive_failures"] + 1 if error else 0
            self._connection.execute(
                """UPDATE tasks SET status = ?, current_summary = ?, waiting_reason = ?,
                   next_run_at = ?, lease_owner = NULL, lease_expires_at = NULL,
                   completed_at = COALESCE(?, completed_at), last_error = ?,
                   consecutive_failures = ?, updated_at = ? WHERE id = ?""",
                (
                    status.value,
                    summary,
                    waiting_reason,
                    _timestamp(next_run_at) if next_run_at else None,
                    completed_at,
                    error,
                    failures,
                    now,
                    task_id,
                ),
            )
            self._event(
                task_id,
                "episode_finished",
                {"status": status.value, "summary": summary, "reason": waiting_reason, "error": error},
            )
            self._connection.commit()
        return self.get_task(task_id)

    def add_work_items(self, task_id: str, items: list[ProposedWorkItem]) -> list[dict[str, Any]]:
        if not items:
            return []
        now = _timestamp()
        created_ids: list[str] = []
        with self._lock:
            self.get_task(task_id)
            existing_keys = {
                row["item_key"]
                for row in self._connection.execute(
                    "SELECT item_key FROM task_work_items WHERE task_id = ?", (task_id,)
                ).fetchall()
            }
            row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS maximum FROM task_work_items WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            sequence = int(row["maximum"])
            for item in items:
                sequence += 1
                item_id = str(uuid.uuid4())
                item_key = item.key or f"item-{sequence}"
                if item_key in existing_keys:
                    continue
                existing_keys.add(item_key)
                created_ids.append(item_id)
                self._connection.execute(
                    """INSERT INTO task_work_items(
                        id, task_id, item_key, kind, title, instructions, completion_check,
                        priority, sequence, status, depends_on_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
                    (
                        item_id,
                        task_id,
                        item_key,
                        item.kind,
                        item.title,
                        item.instructions,
                        item.completion_check,
                        item.priority,
                        sequence,
                        json.dumps(item.depends_on),
                        now,
                        now,
                    ),
                )
                self._event(task_id, "work_item_created", {"title": item.title, "kind": item.kind}, item_id)
            self._connection.commit()
        return [self.get_work_item(item_id) for item_id in created_ids]

    def get_work_item(self, work_item_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM task_work_items WHERE id = ?", (work_item_id,)
            ).fetchone()
        if row is None:
            raise KeyError(work_item_id)
        return self._work_item_row(row)

    def work_items(self, task_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM task_work_items WHERE task_id = ? ORDER BY sequence", (task_id,)
            ).fetchall()
        return [self._work_item_row(row) for row in rows]

    def claim_work_item(self, task_id: str) -> dict[str, Any] | None:
        now = _timestamp()
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM task_work_items
                   WHERE task_id = ? AND status = 'pending' AND (run_after IS NULL OR run_after <= ?)
                   ORDER BY priority DESC, sequence""",
                (task_id, now),
            ).fetchall()
            statuses = {
                row["item_key"]: row["status"]
                for row in self._connection.execute(
                    "SELECT item_key, status FROM task_work_items WHERE task_id = ?", (task_id,)
                ).fetchall()
            }
            row = next(
                (
                    candidate
                    for candidate in rows
                    if all(statuses.get(key) == "completed" for key in _loads(candidate["depends_on_json"], []))
                ),
                None,
            )
            if row is None:
                return None
            self._connection.execute(
                "UPDATE task_work_items SET status = 'running', attempts = attempts + 1, updated_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            self._event(task_id, "work_item_started", {"attempt": row["attempts"] + 1}, row["id"])
            self._connection.commit()
        return self.get_work_item(row["id"])

    def complete_work_item(self, work_item_id: str, result: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            item = self.get_work_item(work_item_id)
            now = _timestamp()
            self._connection.execute(
                """UPDATE task_work_items SET status = 'completed', result_json = ?, error = NULL,
                   completed_at = ?, updated_at = ? WHERE id = ?""",
                (json.dumps(result, ensure_ascii=False), now, now, work_item_id),
            )
            self._event(item["task_id"], "work_item_completed", {"summary": result.get("summary", "")}, work_item_id)
            self._connection.commit()
        return self.get_work_item(work_item_id)

    def block_work_item(self, work_item_id: str, reason: str) -> dict[str, Any]:
        with self._lock:
            item = self.get_work_item(work_item_id)
            self._connection.execute(
                "UPDATE task_work_items SET status = 'blocked', error = ?, updated_at = ? WHERE id = ?",
                (reason, _timestamp(), work_item_id),
            )
            self._event(item["task_id"], "work_item_blocked", {"reason": reason}, work_item_id)
            self._connection.commit()
        return self.get_work_item(work_item_id)

    def retry_work_item(self, work_item_id: str, reason: str, run_after: datetime) -> dict[str, Any]:
        with self._lock:
            item = self.get_work_item(work_item_id)
            self._connection.execute(
                """UPDATE task_work_items SET status = 'pending', error = ?, run_after = ?,
                   updated_at = ? WHERE id = ?""",
                (reason, _timestamp(run_after), _timestamp(), work_item_id),
            )
            self._event(item["task_id"], "work_item_retry_scheduled", {"reason": reason}, work_item_id)
            self._connection.commit()
        return self.get_work_item(work_item_id)

    def fail_work_item(self, work_item_id: str, reason: str) -> dict[str, Any]:
        with self._lock:
            item = self.get_work_item(work_item_id)
            self._connection.execute(
                "UPDATE task_work_items SET status = 'failed', error = ?, updated_at = ? WHERE id = ?",
                (reason, _timestamp(), work_item_id),
            )
            self._event(item["task_id"], "work_item_failed", {"reason": reason}, work_item_id)
            self._connection.commit()
        return self.get_work_item(work_item_id)

    def record_audit(self, task_id: str, report: dict[str, Any]) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO task_audits(task_id, passed, report_json, created_at) VALUES (?, ?, ?, ?)",
                (task_id, int(bool(report.get("passed"))), json.dumps(report, ensure_ascii=False), _timestamp()),
            )
            self._event(task_id, "completion_audited", {"passed": bool(report.get("passed"))})
            self._connection.commit()

    def events(self, task_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT id, work_item_id, event, data_json, created_at FROM (
                   SELECT * FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT ?
                   ) ORDER BY id""",
                (task_id, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "work_item_id": row["work_item_id"],
                "event": row["event"],
                "data": _loads(row["data_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._connection.close()
