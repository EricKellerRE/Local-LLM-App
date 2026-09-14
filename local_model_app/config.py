from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    model_id: str
    model_kind: str
    device: str
    cpu_memory_gb: int | None
    offload_dir: str
    dtype: str
    max_new_tokens: int
    temperature: float
    top_p: float
    planner_max_new_tokens: int
    tool_action_max_new_tokens: int
    tool_final_max_new_tokens: int
    tool_temperature: float
    trust_remote_code: bool

    @classmethod
    def from_environment(cls, project_root: Path) -> "Settings":
        _load_dotenv(project_root / ".env")
        return cls(
            model_id=os.getenv("LOCAL_MODEL_ID", "").strip(),
            model_kind=os.getenv("LOCAL_MODEL_KIND", "auto").strip().lower(),
            device=os.getenv("LOCAL_MODEL_DEVICE", "auto").strip().lower(),
            cpu_memory_gb=(int(os.environ["LOCAL_MODEL_CPU_MEMORY_GB"]) if os.getenv("LOCAL_MODEL_CPU_MEMORY_GB") else None),
            offload_dir=os.getenv("LOCAL_MODEL_OFFLOAD_DIR", "").strip(),
            dtype=os.getenv("LOCAL_MODEL_DTYPE", "auto").strip().lower(),
            max_new_tokens=int(os.getenv("LOCAL_MODEL_MAX_NEW_TOKENS", "512")),
            temperature=float(os.getenv("LOCAL_MODEL_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("LOCAL_MODEL_TOP_P", "0.9")),
            planner_max_new_tokens=int(os.getenv("LOCAL_PLANNER_MAX_NEW_TOKENS", "192")),
            tool_action_max_new_tokens=int(os.getenv("LOCAL_TOOL_ACTION_MAX_NEW_TOKENS", "96")),
            tool_final_max_new_tokens=int(os.getenv("LOCAL_TOOL_FINAL_MAX_NEW_TOKENS", "256")),
            tool_temperature=float(os.getenv("LOCAL_TOOL_TEMPERATURE", "0")),
            trust_remote_code=os.getenv("LOCAL_MODEL_TRUST_REMOTE_CODE", "false").lower() in {"1", "true", "yes"},
        )
