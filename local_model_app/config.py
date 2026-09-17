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
    context_window: int | None
    max_new_tokens: int
    reasoning_budget: int | None
    temperature: float
    top_p: float
    task_planner_max_new_tokens: int
    planner_max_new_tokens: int
    tool_action_max_new_tokens: int
    post_tool_decision_max_new_tokens: int
    section_max_new_tokens: int
    synthesis_max_new_tokens: int
    research_classifier_max_new_tokens: int
    research_notes_max_new_tokens: int
    research_seed_sources: int
    research_depth_passes: int
    research_max_sources: int
    research_references_per_source: int
    research_notes_batch_size: int
    max_tool_calls_per_step: int
    tool_temperature: float
    router_model_id: str
    router_device: str
    router_semantic_weight: float
    trust_remote_code: bool

    @classmethod
    def from_environment(cls, project_root: Path) -> "Settings":
        _load_dotenv(project_root / ".env")
        response_budget = int(os.getenv("LOCAL_MODEL_MAX_NEW_TOKENS", "8192"))
        return cls(
            model_id=os.getenv("LOCAL_MODEL_ID", "").strip(),
            model_kind=os.getenv("LOCAL_MODEL_KIND", "auto").strip().lower(),
            device=os.getenv("LOCAL_MODEL_DEVICE", "auto").strip().lower(),
            cpu_memory_gb=(int(os.environ["LOCAL_MODEL_CPU_MEMORY_GB"]) if os.getenv("LOCAL_MODEL_CPU_MEMORY_GB") else None),
            offload_dir=os.getenv("LOCAL_MODEL_OFFLOAD_DIR", "").strip(),
            dtype=os.getenv("LOCAL_MODEL_DTYPE", "auto").strip().lower(),
            context_window=(int(os.environ["LOCAL_MODEL_CONTEXT_WINDOW"]) if os.getenv("LOCAL_MODEL_CONTEXT_WINDOW") else None),
            max_new_tokens=response_budget,
            reasoning_budget=(int(os.environ["LOCAL_MODEL_REASONING_BUDGET"]) if os.getenv("LOCAL_MODEL_REASONING_BUDGET") else None),
            temperature=float(os.getenv("LOCAL_MODEL_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("LOCAL_MODEL_TOP_P", "0.9")),
            task_planner_max_new_tokens=int(os.getenv("LOCAL_TASK_PLANNER_MAX_NEW_TOKENS", "1024")),
            planner_max_new_tokens=int(os.getenv("LOCAL_PLANNER_MAX_NEW_TOKENS", "192")),
            tool_action_max_new_tokens=int(os.getenv("LOCAL_TOOL_ACTION_MAX_NEW_TOKENS", "1024")),
            post_tool_decision_max_new_tokens=int(
                os.getenv("LOCAL_POST_TOOL_DECISION_MAX_NEW_TOKENS", str(response_budget))
            ),
            section_max_new_tokens=int(os.getenv("LOCAL_SECTION_MAX_NEW_TOKENS", "3072")),
            synthesis_max_new_tokens=int(os.getenv("LOCAL_SYNTHESIS_MAX_NEW_TOKENS", str(response_budget))),
            research_classifier_max_new_tokens=int(os.getenv("LOCAL_RESEARCH_CLASSIFIER_MAX_NEW_TOKENS", "512")),
            research_notes_max_new_tokens=int(os.getenv("LOCAL_RESEARCH_NOTES_MAX_NEW_TOKENS", "1536")),
            research_seed_sources=int(os.getenv("LOCAL_RESEARCH_SEED_SOURCES", "12")),
            research_depth_passes=int(os.getenv("LOCAL_RESEARCH_DEPTH_PASSES", "3")),
            research_max_sources=int(os.getenv("LOCAL_RESEARCH_MAX_SOURCES", "80")),
            research_references_per_source=int(os.getenv("LOCAL_RESEARCH_REFERENCES_PER_SOURCE", "12")),
            research_notes_batch_size=int(os.getenv("LOCAL_RESEARCH_NOTES_BATCH_SIZE", "6")),
            max_tool_calls_per_step=int(os.getenv("LOCAL_MAX_TOOL_CALLS_PER_STEP", "256")),
            tool_temperature=float(os.getenv("LOCAL_TOOL_TEMPERATURE", "0")),
            router_model_id=os.getenv(
                "LOCAL_ROUTER_MODEL_ID",
                "sentence-transformers/all-MiniLM-L6-v2",
            ).strip(),
            router_device=os.getenv("LOCAL_ROUTER_DEVICE", "cpu").strip().lower(),
            router_semantic_weight=float(os.getenv("LOCAL_ROUTER_SEMANTIC_WEIGHT", "36")),
            trust_remote_code=os.getenv("LOCAL_MODEL_TRUST_REMOTE_CODE", "false").lower() in {"1", "true", "yes"},
        )
