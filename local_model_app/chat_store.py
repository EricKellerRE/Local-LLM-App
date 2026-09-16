from __future__ import annotations

import gzip
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatStore:
    """Small SQLite repository shared by the server and the future UI."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.archive_dir = path.parent / "archives"
        self.scratchpad_dir = path.parent / "scratchpads"
        self.tool_activity_dir = path.parent / "tool_activity"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.scratchpad_dir.mkdir(parents=True, exist_ok=True)
        self.tool_activity_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_chat_id_id ON messages(chat_id, id);
            CREATE TABLE IF NOT EXISTS chat_plugins (
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                plugin_id TEXT NOT NULL,
                enabled_at TEXT NOT NULL,
                PRIMARY KEY(chat_id, plugin_id)
            );
            CREATE INDEX IF NOT EXISTS chat_plugins_plugin_id ON chat_plugins(plugin_id);
            """
        )
        columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(chats)").fetchall()}
        if "archived_at" not in columns:
            self._connection.execute("ALTER TABLE chats ADD COLUMN archived_at TEXT")
        self._connection.commit()

    def create_chat(self, title: str = "New chat") -> dict[str, str]:
        chat_id = str(uuid.uuid4())
        timestamp = _now()
        with self._lock:
            self._connection.execute(
                "INSERT INTO chats(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (chat_id, title.strip() or "New chat", timestamp, timestamp),
            )
            self._connection.commit()
        return self.get_chat(chat_id)

    def get_chat(self, chat_id: str) -> dict[str, str]:
        with self._lock:
            row = self._connection.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if row is None:
            raise KeyError(chat_id)
        return dict(row)

    def list_chats(self, *, archived: bool = False) -> list[dict[str, str | None]]:
        if archived:
            chats: list[dict[str, str | None]] = []
            for archive_path in self.archive_dir.glob("*.json.gz"):
                try:
                    chats.append(dict(self._read_archive_path(archive_path)["chat"]))
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    continue
            return sorted(chats, key=lambda chat: str(chat.get("updated_at") or ""), reverse=True)
        with self._lock:
            rows = self._connection.execute("SELECT * FROM chats WHERE archived_at IS NULL ORDER BY updated_at DESC").fetchall()
        return [dict(row) for row in rows]

    def _archive_path(self, chat_id: str) -> Path:
        try:
            normalized = str(uuid.UUID(chat_id))
        except ValueError as exc:
            raise KeyError(chat_id) from exc
        if normalized != chat_id:
            raise KeyError(chat_id)
        return self.archive_dir / f"{chat_id}.json.gz"

    @staticmethod
    def _read_archive_path(path: Path) -> dict:
        with gzip.open(path, "rt", encoding="utf-8") as archive_file:
            return json.load(archive_file)

    def _read_archive(self, chat_id: str) -> dict:
        path = self._archive_path(chat_id)
        if not path.exists():
            raise KeyError(chat_id)
        return self._read_archive_path(path)

    def get_archived_chat(self, chat_id: str) -> dict[str, str | None]:
        return dict(self._read_archive(chat_id)["chat"])

    def archived_messages(self, chat_id: str) -> list[dict[str, str]]:
        return list(self._read_archive(chat_id).get("messages", []))

    def archived_plugins(self, chat_id: str) -> list[str]:
        return [str(plugin_id) for plugin_id in self._read_archive(chat_id).get("plugins", [])]

    def messages(self, chat_id: str) -> list[dict[str, str]]:
        self.get_chat(chat_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def append_exchange(self, chat_id: str, user_content: str, assistant_content: str) -> None:
        chat = self.get_chat(chat_id)
        timestamp = _now()
        title = chat["title"]
        if title == "New chat":
            title = " ".join(user_content.strip().split())[:72] or title
        with self._lock:
            self._connection.executemany(
                "INSERT INTO messages(chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                [
                    (chat_id, "user", user_content, timestamp),
                    (chat_id, "assistant", assistant_content, timestamp),
                ],
            )
            self._connection.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?", (title, timestamp, chat_id)
            )
            self._connection.commit()

    def append_message(self, chat_id: str, role: str, content: str) -> None:
        self.get_chat(chat_id)
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat role: {role}")
        timestamp = _now()
        with self._lock:
            self._connection.execute(
                "INSERT INTO messages(chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, role, content, timestamp),
            )
            self._connection.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?", (timestamp, chat_id)
            )
            self._connection.commit()

    def selected_plugins(self, chat_id: str) -> list[str]:
        self.get_chat(chat_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT plugin_id FROM chat_plugins WHERE chat_id = ? ORDER BY enabled_at", (chat_id,)
            ).fetchall()
        return [str(row["plugin_id"]) for row in rows]

    def set_plugin_selected(self, chat_id: str, plugin_id: str, *, selected: bool) -> list[str]:
        self.get_chat(chat_id)
        with self._lock:
            if selected:
                self._connection.execute(
                    "INSERT OR IGNORE INTO chat_plugins(chat_id, plugin_id, enabled_at) VALUES (?, ?, ?)",
                    (chat_id, plugin_id, _now()),
                )
            else:
                self._connection.execute(
                    "DELETE FROM chat_plugins WHERE chat_id = ? AND plugin_id = ?", (chat_id, plugin_id)
                )
            self._connection.commit()
        return self.selected_plugins(chat_id)

    def plugin_selection_count(self, plugin_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM chat_plugins WHERE plugin_id = ?", (plugin_id,)
            ).fetchone()
        return int(row["count"])

    def delete_chat(self, chat_id: str) -> None:
        try:
            self.get_chat(chat_id)
        except KeyError:
            archive_path = self._archive_path(chat_id)
            if not archive_path.exists():
                raise
            archive_path.unlink()
        else:
            with self._lock:
                self._connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
                self._connection.commit()

    def archive_chat(self, chat_id: str, *, archived: bool = True) -> dict[str, str | None]:
        if not archived:
            return self._restore_chat(chat_id)

        chat = self.get_chat(chat_id)
        archived_at = _now()
        chat["archived_at"] = archived_at
        scratchpad_path = self.scratchpad_dir / f"{chat_id}.jsonl"
        activity_path = self.tool_activity_dir / f"{chat_id}.jsonl"
        payload = {
            "format": "local-model-chat-archive",
            "version": 1,
            "archived_at": archived_at,
            "chat": chat,
            "messages": self.messages(chat_id),
            "plugins": self.selected_plugins(chat_id),
            "scratchpad": scratchpad_path.read_text(encoding="utf-8") if scratchpad_path.exists() else None,
            "tool_activity": activity_path.read_text(encoding="utf-8") if activity_path.exists() else None,
        }
        archive_path = self._archive_path(chat_id)
        temporary_path = Path(f"{archive_path}.tmp")
        try:
            with gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=9) as archive_file:
                json.dump(payload, archive_file, ensure_ascii=False, separators=(",", ":"))
            os.replace(temporary_path, archive_path)
            with self._lock:
                self._connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
                self._connection.commit()
        except Exception:
            temporary_path.unlink(missing_ok=True)
            archive_path.unlink(missing_ok=True)
            raise
        scratchpad_path.unlink(missing_ok=True)
        activity_path.unlink(missing_ok=True)
        return dict(chat)

    def _restore_chat(self, chat_id: str) -> dict[str, str | None]:
        archive_path = self._archive_path(chat_id)
        payload = self._read_archive(chat_id)
        chat = dict(payload["chat"])
        chat["archived_at"] = None
        messages = list(payload.get("messages", []))
        plugins = [str(plugin_id) for plugin_id in payload.get("plugins", [])]
        restored_at = _now()
        scratchpad = payload.get("scratchpad")
        tool_activity = payload.get("tool_activity")
        scratchpad_path = self.scratchpad_dir / f"{chat_id}.jsonl"
        scratchpad_temporary_path = self.scratchpad_dir / f"{chat_id}.jsonl.tmp"
        activity_path = self.tool_activity_dir / f"{chat_id}.jsonl"
        activity_temporary_path = self.tool_activity_dir / f"{chat_id}.jsonl.tmp"
        if isinstance(scratchpad, str):
            scratchpad_temporary_path.write_text(scratchpad, encoding="utf-8")
        if isinstance(tool_activity, str):
            activity_temporary_path.write_text(tool_activity, encoding="utf-8")
        try:
            with self._lock:
                self._connection.execute(
                    "INSERT INTO chats(id, title, created_at, updated_at, archived_at) VALUES (?, ?, ?, ?, NULL)",
                    (chat["id"], chat["title"], chat["created_at"], restored_at),
                )
                self._connection.executemany(
                    "INSERT INTO messages(chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    [(chat_id, message["role"], message["content"], message["created_at"]) for message in messages],
                )
                self._connection.executemany(
                    "INSERT INTO chat_plugins(chat_id, plugin_id, enabled_at) VALUES (?, ?, ?)",
                    [(chat_id, plugin_id, restored_at) for plugin_id in plugins],
                )
                self._connection.commit()
        except Exception:
            scratchpad_temporary_path.unlink(missing_ok=True)
            activity_temporary_path.unlink(missing_ok=True)
            raise
        if isinstance(scratchpad, str):
            os.replace(scratchpad_temporary_path, scratchpad_path)
        if isinstance(tool_activity, str):
            os.replace(activity_temporary_path, activity_path)
        archive_path.unlink()
        return self.get_chat(chat_id)

    def close(self) -> None:
        with self._lock:
            self._connection.close()
