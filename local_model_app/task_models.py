from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskStatus(StrEnum):
    DRAFT = "draft"
    RUNNABLE = "runnable"
    RUNNING = "running"
    WAITING = "waiting"
    WAITING_FOR_INPUT = "waiting_for_input"
    WAITING_FOR_TOOLS = "waiting_for_tools"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_TASK_STATUSES = {
    TaskStatus.COMPLETED.value,
    TaskStatus.FAILED.value,
    TaskStatus.CANCELLED.value,
}


class WorkItemStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskExecutionPolicy(BaseModel):
    deadline: datetime | None = None
    max_steps_per_episode: int = Field(default=8, ge=1, le=100)
    maximum_attempts: int | None = Field(default=None, ge=1)
    retry_initial_seconds: int = Field(default=60, ge=1)
    retry_maximum_seconds: int = Field(default=21600, ge=1)
    resume_after_restart: bool = True

    @field_validator("deadline")
    @classmethod
    def make_deadline_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @model_validator(mode="after")
    def validate_retry_window(self) -> "TaskExecutionPolicy":
        if self.retry_maximum_seconds < self.retry_initial_seconds:
            raise ValueError("retry_maximum_seconds must be at least retry_initial_seconds")
        return self


class TaskDefinition(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    goal: str = Field(min_length=1)
    success_criteria: list[str] = Field(min_length=1)
    deliverables: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    execution: TaskExecutionPolicy = Field(default_factory=TaskExecutionPolicy)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProposedWorkItem(BaseModel):
    key: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_.-]*$")
    kind: str = Field(default="model", pattern=r"^[a-z][a-z0-9_.-]*$")
    title: str = Field(min_length=1, max_length=200)
    instructions: str = Field(min_length=1)
    completion_check: str = Field(min_length=1)
    priority: int = Field(default=0, ge=-1000, le=1000)
    depends_on: list[str] = Field(default_factory=list)


class WorkItemOutcome(BaseModel):
    outcome: Literal["completed", "retry", "waiting_for_input", "waiting_for_tools", "failed"]
    summary: str = Field(min_length=1)
    result: str = ""
    follow_up_items: list[ProposedWorkItem] = Field(default_factory=list)
    completion_evidence: list[str] = Field(default_factory=list)
    wait_seconds: int | None = Field(default=None, ge=1)


class CriterionAudit(BaseModel):
    criterion: str
    satisfied: bool
    evidence: str


class CompletionAudit(BaseModel):
    passed: bool
    summary: str
    criteria: list[CriterionAudit]
    follow_up_items: list[ProposedWorkItem] = Field(default_factory=list)
