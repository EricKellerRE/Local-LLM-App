from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from local_model_app.task_models import (
    CompletionAudit,
    CriterionAudit,
    ProposedWorkItem,
    TaskDefinition,
    TaskExecutionPolicy,
)


def research_topic(request: str) -> str:
    """Extract a bounded search topic without treating workflow instructions as query text."""
    normalized = " ".join(str(request).split()).strip()
    patterns = (
        r"(?i)\b(?:report|review)\s+(?:on|about|into)\s+(.+?)(?=[.!?](?:\s|$)|$)",
        r"(?i)\b(?:review|research)\s+(?:the\s+)?literature\s+(?:on|about|into)\s+(.+?)(?=[.!?](?:\s|$)|$)",
        r"(?i)\b(?:deep|scholarly|literature)\s+research\s*(?::|on|about|into)\s*(.+?)(?=[.!?](?:\s|$)|$)",
        r"(?i)\b(?:research|investigate|analyze|analyse)\s+(?:the\s+topic\s+of\s+)?(.+?)(?=[.!?](?:\s|$)|$)",
    )
    topic = ""
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            topic = match.group(1)
            break
    if not topic:
        topic = re.split(r"[.!?](?:\s|$)", normalized, maxsplit=1)[0]
        topic = re.sub(
            r"(?i)^(?:please\s+)?(?:produce|create|write|make)\s+(?:a\s+|an\s+|the\s+)?"
            r"(?:(?:comprehensive|detailed|complete|full)\s+)*(?:research\s+)?(?:report|review)\s*"
            r"(?:on|about|into)?\s*",
            "",
            topic,
        )
    topic = topic.strip(" \t\r\n:;,.-")
    topic = re.sub(r"(?i)^(?:the\s+topic\s+of|the\s+subject\s+of)\s+", "", topic)
    if not topic:
        raise ValueError("The workflow skill could not identify a research topic.")
    if len(topic) > 300:
        topic = topic[:300].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return topic


class SkillTrigger(BaseModel):
    plugin_ids_all: list[str] = Field(default_factory=list)
    request_terms_any: list[str] = Field(default_factory=list)

    def matches(self, request: str, plugin_ids: list[str]) -> bool:
        selected = set(plugin_ids)
        if not set(self.plugin_ids_all).issubset(selected):
            return False
        normalized = request.casefold()
        return not self.request_terms_any or any(term.casefold() in normalized for term in self.request_terms_any)


class SkillInputSpec(BaseModel):
    source: Literal["request"] = "request"
    transform: Literal["identity", "research_topic"] = "identity"
    required: bool = True
    default: Any = None

    def resolve(self, request: str) -> Any:
        value: Any = research_topic(request) if self.transform == "research_topic" else request.strip()
        if (value is None or value == "") and self.required and self.default is None:
            raise ValueError("A required workflow-skill input could not be resolved.")
        return self.default if value is None or value == "" else value


class EvidenceMinimum(BaseModel):
    name: str = Field(min_length=1)
    metadata_key: str = Field(min_length=1)
    default: int = Field(ge=0)


class HostAuditSpec(BaseModel):
    required_completed: dict[str, int] = Field(default_factory=dict)
    final_kind: str = "synthesis"
    minimum_evidence: list[EvidenceMinimum] = Field(default_factory=list)
    required_artifact_media_types: list[str] = Field(default_factory=list)

    @field_validator("required_completed")
    @classmethod
    def positive_required_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 1 for count in value.values()):
            raise ValueError("Required completed-work counts must be positive.")
        return value


def _render(value: Any, inputs: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(inputs)
    if isinstance(value, list):
        return [_render(item, inputs) for item in value]
    if isinstance(value, dict):
        return {key: _render(item, inputs) for key, item in value.items()}
    return value


class WorkflowSkill(BaseModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    version: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1)
    trigger: SkillTrigger
    inputs: dict[str, SkillInputSpec] = Field(default_factory=dict)
    title_template: str = Field(min_length=1)
    goal_template: str = Field(min_length=1)
    success_criteria: list[str] = Field(min_length=1)
    deliverables: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    execution: TaskExecutionPolicy = Field(default_factory=TaskExecutionPolicy)
    metadata_defaults: dict[str, Any] = Field(default_factory=dict)
    work_items: list[ProposedWorkItem] = Field(min_length=1)
    audit: HostAuditSpec

    def matches(self, request: str, plugin_ids: list[str]) -> bool:
        return self.trigger.matches(request, plugin_ids)

    def resolve_inputs(self, request: str) -> dict[str, Any]:
        return {name: spec.resolve(request) for name, spec in self.inputs.items()}

    def task_definition(
        self,
        request: str,
        plugin_ids: list[str],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TaskDefinition:
        inputs = self.resolve_inputs(request)
        merged_metadata = dict(_render(self.metadata_defaults, inputs))
        merged_metadata.update(metadata or {})
        merged_metadata.update({
            "mode": "durable_tools",
            "plugin_ids": list(plugin_ids),
            "workflow_skill_id": self.id,
            "workflow_skill_version": self.version,
            "skill_inputs": inputs,
        })
        title = str(_render(self.title_template, inputs))[:160]
        return TaskDefinition(
            title=title,
            goal=str(_render(self.goal_template, inputs)),
            success_criteria=list(_render(self.success_criteria, inputs)),
            deliverables=list(_render(self.deliverables, inputs)),
            constraints=list(_render(self.constraints, inputs)),
            execution=self.execution,
            metadata=merged_metadata,
        )

    def instantiate_work_items(self, inputs: dict[str, Any]) -> list[ProposedWorkItem]:
        return [ProposedWorkItem.model_validate(_render(item.model_dump(), inputs)) for item in self.work_items]

    def audit_completion(self, task: dict[str, Any], work_items: list[dict[str, Any]]) -> CompletionAudit:
        metadata = task["definition"].get("metadata") or {}
        completed_counts: dict[str, int] = {}
        for item in work_items:
            if item.get("status") == "completed":
                kind = str(item.get("kind") or "")
                completed_counts[kind] = completed_counts.get(kind, 0) + 1

        final_items = [item for item in work_items if item.get("kind") == self.audit.final_kind]
        final_outcome = final_items[-1].get("result") or {} if final_items else {}
        evidence = final_outcome.get("completion_evidence", []) if isinstance(final_outcome, dict) else []

        def metric(name: str) -> int:
            prefix = f"{name}="
            for value in evidence:
                if str(value).startswith(prefix):
                    try:
                        return int(str(value)[len(prefix):])
                    except ValueError:
                        return 0
            return 0

        checks: dict[str, bool] = {
            f"completed {kind}": completed_counts.get(kind, 0) >= count
            for kind, count in self.audit.required_completed.items()
        }
        metric_details: list[str] = []
        for requirement in self.audit.minimum_evidence:
            minimum = int(metadata.get(requirement.metadata_key) or requirement.default)
            actual = metric(requirement.name)
            checks[requirement.name] = actual >= minimum
            metric_details.append(f"{requirement.name}={actual}/{minimum}")

        artifacts = final_outcome.get("artifacts", []) if isinstance(final_outcome, dict) else []
        available_media_types = {str(artifact.get("media_type")) for artifact in artifacts}
        for media_type in self.audit.required_artifact_media_types:
            checks[f"artifact {media_type}"] = media_type in available_media_types

        passed = all(checks.values())
        detail = ", ".join(f"{name}={'passed' if value else 'failed'}" for name, value in checks.items())
        evidence_detail = "; ".join(metric_details)
        summary = (
            f"Workflow skill {self.id}@{self.version} passed host-enforced completion checks. " if passed else
            f"Workflow skill {self.id}@{self.version} did not pass host-enforced completion checks. "
        ) + detail
        return CompletionAudit(
            passed=passed,
            summary=summary,
            criteria=[
                CriterionAudit(
                    criterion=criterion,
                    satisfied=passed,
                    evidence=f"Host-enforced workflow checks: {detail}. {evidence_detail}".strip(),
                )
                for criterion in task["definition"]["success_criteria"]
            ],
        )

    def public_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "required_plugin_ids": self.trigger.plugin_ids_all,
            "input_names": list(self.inputs),
            "work_item_count": len(self.work_items),
        }


class WorkflowSkillRegistry:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._skills: dict[str, WorkflowSkill] = {}
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    skill = WorkflowSkill.model_validate(payload)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"Invalid workflow skill {path}: {exc}") from exc
                if skill.id in self._skills:
                    raise ValueError(f"Duplicate workflow skill id: {skill.id}")
                self._skills[skill.id] = skill

    @classmethod
    def default(cls) -> "WorkflowSkillRegistry":
        return cls(Path(__file__).resolve().parents[1] / "config" / "skills")

    @property
    def skills(self) -> list[WorkflowSkill]:
        return list(self._skills.values())

    def get(self, skill_id: str) -> WorkflowSkill:
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            raise KeyError(f"Unknown workflow skill: {skill_id}") from exc

    def select(self, request: str, plugin_ids: list[str]) -> WorkflowSkill | None:
        matches = [skill for skill in self.skills if skill.matches(request, plugin_ids)]
        if not matches:
            return None
        matches.sort(
            key=lambda skill: (len(skill.trigger.plugin_ids_all), len(skill.trigger.request_terms_any), skill.id),
            reverse=True,
        )
        return matches[0]

    def for_task(self, definition: dict[str, Any]) -> WorkflowSkill | None:
        metadata = definition.get("metadata") or {}
        skill_id = str(metadata.get("workflow_skill_id") or "").strip()
        if skill_id:
            return self.get(skill_id)
        plugin_ids = [str(value) for value in metadata.get("plugin_ids") or []]
        return self.select(str(definition.get("goal") or ""), plugin_ids)

    def catalog(self) -> list[dict[str, Any]]:
        return [skill.public_record() for skill in self.skills]
