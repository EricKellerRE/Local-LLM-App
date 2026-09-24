from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_ENV_KEYS = {
    "model_id": "LOCAL_MODEL_ID",
    "model_kind": "LOCAL_MODEL_KIND",
    "device": "LOCAL_MODEL_DEVICE",
    "dtype": "LOCAL_MODEL_DTYPE",
    "cpu_memory_gb": "LOCAL_MODEL_CPU_MEMORY_GB",
    "offload_dir": "LOCAL_MODEL_OFFLOAD_DIR",
    "context_window": "LOCAL_MODEL_CONTEXT_WINDOW",
    "max_new_tokens": "LOCAL_MODEL_MAX_NEW_TOKENS",
    "reasoning_budget": "LOCAL_MODEL_REASONING_BUDGET",
    "task_planner_max_new_tokens": "LOCAL_TASK_PLANNER_MAX_NEW_TOKENS",
    "work_item_planner_max_new_tokens": "LOCAL_PLANNER_MAX_NEW_TOKENS",
    "tool_action_max_new_tokens": "LOCAL_TOOL_ACTION_MAX_NEW_TOKENS",
    "post_tool_decision_max_new_tokens": "LOCAL_POST_TOOL_DECISION_MAX_NEW_TOKENS",
    "section_max_new_tokens": "LOCAL_SECTION_MAX_NEW_TOKENS",
    "synthesis_max_new_tokens": "LOCAL_SYNTHESIS_MAX_NEW_TOKENS",
    "research_classifier_max_new_tokens": "LOCAL_RESEARCH_CLASSIFIER_MAX_NEW_TOKENS",
    "research_notes_max_new_tokens": "LOCAL_RESEARCH_NOTES_MAX_NEW_TOKENS",
    "research_source_dossier_max_new_tokens": "LOCAL_RESEARCH_SOURCE_DOSSIER_MAX_NEW_TOKENS",
    "research_whole_source_max_tokens": "LOCAL_RESEARCH_WHOLE_SOURCE_MAX_TOKENS",
    "research_section_input_tokens": "LOCAL_RESEARCH_SECTION_INPUT_TOKENS",
    "research_source_max_characters": "LOCAL_RESEARCH_SOURCE_MAX_CHARACTERS",
    "research_seed_sources": "LOCAL_RESEARCH_SEED_SOURCES",
    "research_depth_passes": "LOCAL_RESEARCH_DEPTH_PASSES",
    "research_max_sources": "LOCAL_RESEARCH_MAX_SOURCES",
    "research_references_per_source": "LOCAL_RESEARCH_REFERENCES_PER_SOURCE",
    "max_tool_calls_per_step": "LOCAL_MAX_TOOL_CALLS_PER_STEP",
    "temperature": "LOCAL_MODEL_TEMPERATURE",
    "top_p": "LOCAL_MODEL_TOP_P",
    "trust_remote_code": "LOCAL_MODEL_TRUST_REMOTE_CODE",
}


@dataclass(frozen=True)
class AppPaths:
    data_directory: Path
    models_directory: Path


class AppSettingsStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.path = self.root / "config" / "app-settings.json"
        self.env_path = self.root / ".env"

    def paths(self) -> AppPaths:
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            saved = {}
        data_directory = Path(saved.get("data_directory") or self.root / "data").expanduser().resolve()
        models_directory = Path(saved.get("models_directory") or self.root / "models").expanduser().resolve()
        return AppPaths(data_directory=data_directory, models_directory=models_directory)

    def _read_env(self) -> tuple[list[str], dict[str, str]]:
        try:
            lines = self.env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        values: dict[str, str] = {}
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        return lines, values

    def public_settings(self) -> dict[str, Any]:
        _, values = self._read_env()
        paths = self.paths()
        defaults: dict[str, Any] = {
            "model_id": "",
            "model_kind": "auto",
            "device": "auto",
            "dtype": "auto",
            "cpu_memory_gb": None,
            "offload_dir": "",
            "context_window": None,
            "max_new_tokens": 8192,
            "reasoning_budget": None,
            "task_planner_max_new_tokens": 1024,
            "work_item_planner_max_new_tokens": 192,
            "tool_action_max_new_tokens": 192,
            "post_tool_decision_max_new_tokens": 256,
            "section_max_new_tokens": 768,
            "synthesis_max_new_tokens": 8192,
            "research_classifier_max_new_tokens": 256,
            "research_notes_max_new_tokens": 768,
            "research_source_dossier_max_new_tokens": 1536,
            "research_whole_source_max_tokens": 8192,
            "research_section_input_tokens": 6144,
            "research_source_max_characters": 160000,
            "research_seed_sources": 12,
            "research_depth_passes": 3,
            "research_max_sources": 80,
            "research_references_per_source": 12,
            "max_tool_calls_per_step": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "trust_remote_code": False,
        }
        result = {
            key: values.get(env_key, default)
            for key, env_key in MODEL_ENV_KEYS.items()
            for default in [defaults[key]]
        }
        result["cpu_memory_gb"] = int(result["cpu_memory_gb"]) if str(result["cpu_memory_gb"] or "") else None
        result["context_window"] = int(result["context_window"]) if str(result["context_window"] or "") else None
        result["max_new_tokens"] = int(result["max_new_tokens"])
        result["reasoning_budget"] = int(result["reasoning_budget"]) if str(result["reasoning_budget"] or "") else None
        for key in ("synthesis_max_new_tokens",):
            env_key = MODEL_ENV_KEYS[key]
            if not str(values.get(env_key) or "").strip():
                result[key] = result["max_new_tokens"]
        for key in (
            "task_planner_max_new_tokens",
            "work_item_planner_max_new_tokens",
            "tool_action_max_new_tokens",
            "post_tool_decision_max_new_tokens",
            "section_max_new_tokens",
            "synthesis_max_new_tokens",
            "research_classifier_max_new_tokens",
            "research_notes_max_new_tokens",
            "research_source_dossier_max_new_tokens",
            "research_whole_source_max_tokens",
            "research_section_input_tokens",
            "research_source_max_characters",
            "research_seed_sources",
            "research_depth_passes",
            "research_max_sources",
            "research_references_per_source",
            "max_tool_calls_per_step",
        ):
            result[key] = int(result[key])
        result["temperature"] = float(result["temperature"])
        result["top_p"] = float(result["top_p"])
        result["trust_remote_code"] = str(result["trust_remote_code"]).lower() in {"1", "true", "yes"}
        result["data_directory"] = str(paths.data_directory)
        result["models_directory"] = str(paths.models_directory)
        return result

    @staticmethod
    def _validated_directory(value: str, label: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{label} must be an absolute folder path.")
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        paths = self.paths()
        data_directory = self._validated_directory(
            str(values.get("data_directory") or paths.data_directory), "App data directory"
        )
        models_directory = self._validated_directory(
            str(values.get("models_directory") or paths.models_directory), "Models directory"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "data_directory": str(data_directory),
            "models_directory": str(models_directory),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

        lines, _ = self._read_env()
        replacements = {
            env_key: self._serialize(values[key])
            for key, env_key in MODEL_ENV_KEYS.items()
            if key in values
        }
        seen: set[str] = set()
        updated: list[str] = []
        for line in lines:
            if "=" in line and not line.lstrip().startswith("#"):
                key = line.split("=", 1)[0].strip()
                if key in replacements:
                    updated.append(f"{key}={replacements[key]}")
                    seen.add(key)
                    continue
            updated.append(line)
        for key, value in replacements.items():
            if key not in seen:
                updated.append(f"{key}={value}")
        env_temporary = self.env_path.with_suffix(".env.tmp")
        env_temporary.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")
        os.replace(env_temporary, self.env_path)
        return self.public_settings()

    @staticmethod
    def _serialize(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def installed_models(self) -> list[dict[str, str]]:
        root = self.paths().models_directory
        root.mkdir(parents=True, exist_ok=True)
        models: list[dict[str, str]] = []
        if (root / "config.json").is_file():
            models.append({"name": root.name, "path": str(root)})
        for config in root.glob("*/config.json"):
            models.append({"name": config.parent.name.replace("--", "/"), "path": str(config.parent)})
        return sorted(models, key=lambda item: item["name"].lower())
