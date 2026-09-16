from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from local_model_app.mcp_manager import McpPluginManager
from local_model_app.model import TransformersModel
from local_model_app.scratchpad import Scratchpad
from local_model_app.task_models import WorkItemOutcome
from local_model_app.task_store import TaskStore
from local_model_app.tool_coordinator import ToolCoordinator
from local_model_app.tool_router import ToolRouter


AsyncGenerator = Callable[[list[dict[str, Any]]], Awaitable[str]]
URL_PATTERN = re.compile(r"https?://[^\s<>)\]}]+", re.IGNORECASE)


class DurableToolExecutor:
    """Execute one checkpointed work item with the task's selected MCP capabilities."""

    def __init__(
        self,
        model: TransformersModel,
        manager: McpPluginManager,
        router: ToolRouter,
        data_directory: Path,
        store: TaskStore,
    ) -> None:
        self.model = model
        self.manager = manager
        self.router = router
        self.data_directory = data_directory
        self.store = store

    @staticmethod
    def _completed_context(items: list[dict[str, Any]]) -> str:
        records = []
        for item in items[-20:]:
            outcome = item.get("result") or {}
            records.append({
                "key": item.get("key"),
                "title": item.get("title"),
                "summary": outcome.get("summary") if isinstance(outcome, dict) else "",
                "result": outcome.get("result") if isinstance(outcome, dict) else outcome,
            })
        return json.dumps(records, ensure_ascii=False)

    @staticmethod
    def _attach_task_to_approval(path: Path, task_id: str, work_item_id: str) -> None:
        state_path = path.with_suffix(".state.json")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        state.update({"durable_task_id": task_id, "durable_work_item_id": work_item_id})
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(state_path)

    async def execute_work_item(
        self,
        task: dict[str, Any],
        item: dict[str, Any],
        completed_items: list[dict[str, Any]],
    ) -> WorkItemOutcome:
        metadata = task["definition"].get("metadata") or {}
        plugin_ids = [str(value) for value in metadata.get("plugin_ids") or []]
        if not plugin_ids:
            return WorkItemOutcome(
                outcome="waiting_for_tools",
                summary="This durable work item requires a selected MCP capability.",
            )
        chat_id = str(metadata.get("chat_id") or task["id"])
        activity_path = self.data_directory / "tool_activity" / f"{chat_id}.jsonl"
        checkpoint = item.get("result") or {}
        pending_task = checkpoint.get("pending_mcp_task") if isinstance(checkpoint, dict) else None
        if isinstance(pending_task, dict):
            try:
                result = await self.manager.resume_tool_task(pending_task)
            except Exception:
                self.store.checkpoint_work_item(item["id"], None)
                raise
            serialized = json.dumps(result, ensure_ascii=False)
            return WorkItemOutcome(
                outcome="completed",
                summary=f"Completed durable MCP task for checkpoint: {item['title']}",
                result=serialized,
                completion_evidence=[serialized[:2000]],
            )

        async def save_task_handle(handle: dict[str, Any]) -> None:
            self.store.checkpoint_work_item(item["id"], {"pending_mcp_task": handle})

        coordinator = ToolCoordinator(
            self.model,
            self.manager,
            Scratchpad(self.data_directory / "scratchpads" / f"{chat_id}.jsonl"),
            activity_path,
            router=self.router,
            max_calls=int(getattr(self.model.settings, "max_tool_calls_per_step", 256)),
            task_checkpoint=save_task_handle,
        )
        prompt = (
            f"Durable task goal:\n{task['definition']['goal']}\n\n"
            f"Current checkpointed work item:\n{item['title']}\n{item['instructions']}\n\n"
            f"Completion check:\n{item['completion_check']}\n\n"
            "Use the available tools rather than merely describing what could be done. Work iteratively, inspect "
            "results, and preserve exact source URLs, artifact handles, case/session identifiers, metric values, "
            "assumptions, and errors in the answer. Do not claim the completion check passed without evidence.\n\n"
            f"Previously completed checkpoints:\n{self._completed_context(completed_items)}"
        )
        answer = await coordinator.respond(prompt, [], plugin_ids)
        lowered = answer.lower()
        if "approval is required" in lowered:
            self._attach_task_to_approval(activity_path, task["id"], item["id"])
            return WorkItemOutcome(outcome="waiting_for_input", summary=answer)
        if "cannot execute" in lowered or "please provide" in lowered:
            return WorkItemOutcome(outcome="waiting_for_input", summary=answer)
        urls = list(dict.fromkeys(URL_PATTERN.findall(answer)))
        return WorkItemOutcome(
            outcome="completed",
            summary=f"Completed checkpoint: {item['title']}",
            result=answer,
            completion_evidence=urls or [answer[:500]],
        )


class DurableSynthesisExecutor:
    """Compose the user-facing report from durable, persisted checkpoint evidence."""

    def __init__(self, generate: AsyncGenerator) -> None:
        self.generate = generate

    async def execute_work_item(
        self,
        task: dict[str, Any],
        item: dict[str, Any],
        completed_items: list[dict[str, Any]],
    ) -> WorkItemOutcome:
        evidence = []
        for completed in completed_items:
            outcome = completed.get("result") or {}
            evidence.append({
                "key": completed.get("key"),
                "title": completed.get("title"),
                "summary": outcome.get("summary") if isinstance(outcome, dict) else "",
                "result": outcome.get("result") if isinstance(outcome, dict) else outcome,
                "completion_evidence": outcome.get("completion_evidence", []) if isinstance(outcome, dict) else [],
            })
        system = (
            "Write the final deliverable for a durable task from the recorded checkpoint evidence. Produce a "
            "complete standalone report, not a progress summary. Include an executive summary, scope and method, "
            "findings, quantitative results where available, techniques or interventions, conclusions, unresolved "
            "gaps, limitations, and recommended next steps. For research, cite exact source URLs beside supported "
            "claims and distinguish source evidence from inference. For engineering optimization, compare baseline "
            "and final metrics, enumerate changes, preserve artifact/case handles, and state whether each success "
            "criterion passed. Never invent evidence."
        )
        report = await self.generate([
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({
                "task": task["definition"],
                "synthesis_instructions": item["instructions"],
                "recorded_evidence": evidence,
            }, ensure_ascii=False)},
        ])
        return WorkItemOutcome(
            outcome="completed",
            summary="Final report synthesized from persisted checkpoint evidence.",
            result=report,
            completion_evidence=["Final report generated from completed work-item records."],
        )
