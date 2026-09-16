from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Protocol

from local_model_app.task_models import (
    CompletionAudit,
    ProposedWorkItem,
    TaskDefinition,
    WorkItemOutcome,
)


AsyncGenerator = Callable[[list[dict[str, Any]]], Awaitable[str]]
CapabilityCatalog = Callable[[list[str]], Awaitable[list[dict[str, Any]]]]


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model did not return a JSON object.")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("The model response must be a JSON object.")
    return value


class TaskCoordinatorProtocol(Protocol):
    async def propose(self, request: str) -> TaskDefinition: ...

    async def plan_task(self, task: dict[str, Any]) -> list[ProposedWorkItem]: ...

    async def audit_completion(
        self,
        task: dict[str, Any],
        work_items: list[dict[str, Any]],
    ) -> CompletionAudit: ...


class WorkItemExecutorProtocol(Protocol):

    async def execute_work_item(
        self,
        task: dict[str, Any],
        item: dict[str, Any],
        completed_items: list[dict[str, Any]],
    ) -> WorkItemOutcome: ...

class ModelTaskCoordinator:
    """Uses the local model for planning while leaving lifecycle control to the host."""

    def __init__(
        self,
        generate: AsyncGenerator,
        capability_catalog: CapabilityCatalog | None = None,
    ) -> None:
        self.generate = generate
        self.capability_catalog = capability_catalog

    async def propose(self, request: str) -> TaskDefinition:
        prompt = (
            "Convert the user's request into a durable long-running task definition. "
            "Do not claim that work has begun. Infer useful defaults, but preserve the user's intent. "
            "Return only JSON with: title, goal, success_criteria (non-empty string array), "
            "deliverables (string array), constraints (string array), execution, and metadata. "
            "execution must contain deadline (ISO-8601 or null), max_steps_per_episode (1-100), "
            "maximum_attempts (positive integer or null), retry_initial_seconds, retry_maximum_seconds, "
            "and resume_after_restart. Use null attempt/deadline values when the user requests indefinite work."
        )
        response = await self.generate(
            [{"role": "system", "content": prompt}, {"role": "user", "content": request}]
        )
        return TaskDefinition.model_validate(_parse_json_object(response))

    async def plan_task(self, task: dict[str, Any]) -> list[ProposedWorkItem]:
        definition = task["definition"]
        metadata = definition.get("metadata") or {}
        if metadata.get("mode") == "durable_tools":
            plugin_ids = [str(value) for value in metadata.get("plugin_ids") or []]
            capabilities = (
                await self.capability_catalog(plugin_ids)
                if self.capability_catalog is not None and plugin_ids
                else []
            )
            prompt = (
                "Plan a durable, resumable tool-driven task. Return only JSON with a work_items array. "
                "Create 3-12 bounded evidence-producing items of kind 'tool', each covering a distinct research "
                "question, engineering stage, metric, scenario, or validation concern. Use dependencies where a "
                "later operation requires a case/session/artifact handle from earlier work. Then create exactly one "
                "kind 'synthesis' item that depends on every evidence item and produces the complete user-facing "
                "deliverable. For iterative optimization, include baseline measurement, candidate changes, repeated "
                "metric evaluation, regression/safety checks, and final comparison. For research, include diverse "
                "search angles, primary-source reading, disagreement/gap analysis, and citation capture. Do not put "
                "all work into one item. Each item has key, kind, title, instructions, completion_check, priority, "
                "and depends_on. Keys must start with a lowercase letter and contain only letters, digits, dot, "
                "underscore, or hyphen."
            )
            response = await self.generate([
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({
                    "definition": definition,
                    "available_capabilities": capabilities,
                }, ensure_ascii=False)},
            ])
            data = _parse_json_object(response)
            return [ProposedWorkItem.model_validate(item) for item in data.get("work_items", [])]
        prompt = (
            "Create the first bounded work items for this durable task. You are planning, not claiming execution. "
            "The host currently supports model reasoning and drafting but has no task-specific external tools. "
            "Create model work only when it can be honestly completed from supplied information. "
            "If external capabilities are necessary, create one item describing exactly what capability is required. "
            "Return only JSON: {\"work_items\": [{\"key\": str, \"kind\": \"model\", \"title\": str, "
            "\"instructions\": str, \"completion_check\": str, \"priority\": int, \"depends_on\": [key]}]}. "
            "Use stable unique keys and dependencies when ordering matters. "
            "Use 1-6 small items."
        )
        response = await self.generate(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(definition, ensure_ascii=False)},
            ]
        )
        data = _parse_json_object(response)
        return [ProposedWorkItem.model_validate(item) for item in data.get("work_items", [])]

    async def execute_work_item(
        self,
        task: dict[str, Any],
        item: dict[str, Any],
        completed_items: list[dict[str, Any]],
    ) -> WorkItemOutcome:
        history = [
            {
                "title": completed["title"],
                "result": completed.get("result"),
            }
            for completed in completed_items[-12:]
        ]
        prompt = (
            "Perform one bounded work item for a durable task. Never claim to search, read files, contact services, "
            "or perform other external actions: no task-specific tools are connected yet. You may reason, outline, "
            "draft, classify supplied material, or transform supplied text. If a required capability is unavailable, "
            "return waiting_for_tools and name it. If the user must decide something, return waiting_for_input. "
            "Return only JSON with: outcome (completed, retry, waiting_for_input, waiting_for_tools, or failed), "
            "summary, result, follow_up_items, completion_evidence, and wait_seconds. Follow-up items use the same "
            "shape as the current item and must be small and necessary. They may depend on existing item keys."
        )
        payload = {
            "task": task["definition"],
            "current_work_item": {
                "title": item["title"],
                "instructions": item["instructions"],
                "completion_check": item["completion_check"],
            },
            "completed_work": history,
        }
        response = await self.generate(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
        return WorkItemOutcome.model_validate(_parse_json_object(response))

    async def audit_completion(
        self,
        task: dict[str, Any],
        work_items: list[dict[str, Any]],
    ) -> CompletionAudit:
        evidence = [
            {
                "title": item["title"],
                "status": item["status"],
                "result": item.get("result"),
            }
            for item in work_items[-40:]
        ]
        durable_tools = (task["definition"].get("metadata") or {}).get("mode") == "durable_tools"
        prompt = (
            "Audit a durable task against every success criterion using only recorded evidence. Be conservative. "
            "A plan or assertion is not evidence that external work occurred. Return only JSON with: passed, summary, "
            "criteria (each containing criterion, satisfied, evidence), and follow_up_items. passed may be true only "
            "when every criterion is satisfied. Create bounded follow-up items for remediable gaps."
            + (
                " For this tool-driven task, follow-up evidence or optimization items must use kind 'tool'. If any "
                "follow-up is created, also create a new kind 'synthesis' item depending on every new follow-up key. "
                "Do not pass an optimization without baseline/final metric evidence, and do not pass research without "
                "source URLs, conclusions, techniques, and explicit gaps or limitations."
                if durable_tools else ""
            )
        )
        response = await self.generate(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"definition": task["definition"], "recorded_work": evidence}, ensure_ascii=False
                    ),
                },
            ]
        )
        return CompletionAudit.model_validate(_parse_json_object(response))
