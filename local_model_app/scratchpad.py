from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Scratchpad:
    """Append-only, locally stored planning notes visible to the user."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def add(self, kind: str, content: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"at": datetime.now(timezone.utc).isoformat(), "kind": kind, "content": content}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 8) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        entries: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def render(self, limit: int = 8) -> str:
        return "\n".join(f"[{item['kind']}] {item['content']}" for item in self.recent(limit))
