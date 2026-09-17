from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from local_model_app.app_settings import AppSettingsStore
from local_model_app.chat_store import ChatStore
from local_model_app.config import Settings, _load_dotenv
from local_model_app.coordinator import Coordinator
from local_model_app.durable_tools import DurableSectionExecutor, DurableSynthesisExecutor, DurableToolExecutor
from local_model_app.model import TransformersModel
from local_model_app.research_pipeline import ResearchDiscoveryExecutor, ResearchNotesExecutor
from local_model_app.huggingface_service import HuggingFaceService
from local_model_app.mcp_manager import McpPluginManager
from local_model_app.mcp_plugins import McpPluginRegistry, PluginConfigurationError
from local_model_app.scratchpad import Scratchpad
from local_model_app.task_coordinator import ModelTaskCoordinator
from local_model_app.task_engine import UniversalTaskEngine
from local_model_app.task_models import TaskDefinition, TaskStatus, WorkItemOutcome
from local_model_app.task_store import TaskStore
from local_model_app.tool_coordinator import FINAL_TOKENS, ToolCoordinator, _compact_result
from local_model_app.tool_router import LocalEmbeddingEncoder, ToolRouter


ROOT = Path(__file__).resolve().parents[1]
_load_dotenv(ROOT / ".env")


def tracked_activity(label: str):
    def decorate(function):
        @wraps(function)
        async def wrapped(self, *args, **kwargs):
            async with self.activity(label):
                return await function(self, *args, **kwargs)

        return wrapped

    return decorate


class CompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)


class NewChatRequest(BaseModel):
    title: str = "New chat"
    project_id: str | None = None


class ChatMessageRequest(BaseModel):
    content: str = Field(min_length=1)


class ArchiveChatRequest(BaseModel):
    archived: bool = True


class PluginSelectionRequest(BaseModel):
    enabled: bool


class McpContentRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class TaskProposalRequest(BaseModel):
    request: str = Field(min_length=1)


class CreateTaskRequest(BaseModel):
    request: str = Field(min_length=1)
    definition: TaskDefinition


class NewProjectRequest(BaseModel):
    path: str = Field(min_length=1)
    name: str | None = None
    plugin_ids: list[str] = Field(default_factory=list)


class MoveChatRequest(BaseModel):
    project_id: str | None = None


class SettingsUpdateRequest(BaseModel):
    model_id: str = ""
    model_kind: str = "auto"
    device: str = "auto"
    dtype: str = "auto"
    cpu_memory_gb: int | None = Field(default=None, ge=1)
    offload_dir: str = ""
    context_window: int | None = Field(default=None, ge=256, le=4_194_304)
    max_new_tokens: int = Field(default=8192, ge=1, le=262_144)
    reasoning_budget: int | None = Field(default=None, ge=0, le=262_144)
    task_planner_max_new_tokens: int = Field(default=1024, ge=1, le=262_144)
    work_item_planner_max_new_tokens: int = Field(default=192, ge=1, le=262_144)
    tool_action_max_new_tokens: int = Field(default=1024, ge=1, le=262_144)
    post_tool_decision_max_new_tokens: int = Field(default=8192, ge=1, le=262_144)
    section_max_new_tokens: int = Field(default=3072, ge=1, le=262_144)
    synthesis_max_new_tokens: int = Field(default=8192, ge=1, le=262_144)
    research_classifier_max_new_tokens: int = Field(default=512, ge=1, le=262_144)
    research_notes_max_new_tokens: int = Field(default=1536, ge=1, le=262_144)
    research_seed_sources: int = Field(default=12, ge=1, le=100)
    research_depth_passes: int = Field(default=3, ge=0, le=10)
    research_max_sources: int = Field(default=80, ge=1, le=1000)
    research_references_per_source: int = Field(default=12, ge=1, le=100)
    research_notes_batch_size: int = Field(default=6, ge=1, le=50)
    max_tool_calls_per_step: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    trust_remote_code: bool = False
    data_directory: str
    models_directory: str

    @model_validator(mode="after")
    def validate_token_budgets(self) -> "SettingsUpdateRequest":
        if self.research_max_sources < self.research_seed_sources:
            raise ValueError("Research total source cap must be at least the seed-source count.")
        if self.context_window is not None and self.max_new_tokens >= self.context_window:
            raise ValueError("Response budget must be smaller than the context window.")
        if self.context_window is not None:
            named_budgets = {
                "Task planner": self.task_planner_max_new_tokens,
                "Work-item planner": self.work_item_planner_max_new_tokens,
                "Tool action": self.tool_action_max_new_tokens,
                "Post-tool decision": self.post_tool_decision_max_new_tokens,
                "Report section": self.section_max_new_tokens,
                "Synthesis": self.synthesis_max_new_tokens,
                "Research relevance": self.research_classifier_max_new_tokens,
                "Research notes": self.research_notes_max_new_tokens,
            }
            invalid = [name for name, value in named_budgets.items() if value >= self.context_window]
            if invalid:
                raise ValueError(f"{', '.join(invalid)} budget must be smaller than the context window.")
        if self.reasoning_budget is not None and self.reasoning_budget > self.max_new_tokens:
            raise ValueError("Reasoning budget cannot exceed the response budget.")
        return self


class ModelDownloadRequest(BaseModel):
    repo_id: str = Field(min_length=3)


class Runtime:
    def __init__(self) -> None:
        self.app_settings = AppSettingsStore(ROOT)
        self.paths = self.app_settings.paths()
        settings = Settings.from_environment(ROOT)
        self.model = TransformersModel(
            settings,
            telemetry_path=self.paths.data_directory / "generation_telemetry.jsonl",
        )
        encoder = (
            LocalEmbeddingEncoder(settings.router_model_id, device=settings.router_device)
            if settings.router_model_id
            else None
        )
        self.tool_router = ToolRouter(encoder, semantic_weight=settings.router_semantic_weight)
        self.store = ChatStore(self.paths.data_directory / "chats.sqlite3")
        self.store.discard_empty_chats()
        self.tasks = TaskStore(self.paths.data_directory / "tasks.sqlite3")
        self.mcp = McpPluginManager(McpPluginRegistry(ROOT / "config" / "mcp.d", project_root=ROOT))
        self.load_error: str | None = None
        self.loading = False
        self.draining = False
        self._active_operations: dict[str, str] = {}
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self.task_coordinator = ModelTaskCoordinator(
            self.task_generate,
            capability_catalog=self.task_capability_catalog,
        )
        self.task_engine = UniversalTaskEngine(
            self.tasks,
            self.task_coordinator,
            executors={
                "model": self.task_coordinator,
                "tool": DurableToolExecutor(
                    self.model,
                    self.mcp,
                    self.tool_router,
                    self.paths.data_directory,
                    self.tasks,
                    self._inference_lock,
                ),
                "research_discovery": ResearchDiscoveryExecutor(
                    self.mcp,
                    self.paths.data_directory,
                    self.classify_references,
                ),
                "research_notes": ResearchNotesExecutor(self.task_analyze_sources, self.paths.data_directory),
                "section": DurableSectionExecutor(self.task_write_section),
                "synthesis": DurableSynthesisExecutor(self.task_synthesize, self.paths.data_directory),
            },
            on_status=self.deliver_task_status,
            poll_seconds=float(os.getenv("LOCAL_TASK_POLL_SECONDS", "5")),
            lease_seconds=int(os.getenv("LOCAL_TASK_LEASE_SECONDS", "1800")),
        )
        self.huggingface = HuggingFaceService(self.app_settings, self.activity)

    @property
    def data_directory(self) -> Path:
        """Return the active data folder, including for lightweight test runtimes."""
        paths = getattr(self, "paths", None)
        if paths is not None:
            return paths.data_directory
        store = getattr(self, "store", None)
        if store is not None and getattr(store, "path", None) is not None:
            return Path(store.path).parent
        return ROOT / "data"

    async def ensure_loaded(self) -> None:
        if self.model.loaded:
            return
        async with self._load_lock:
            if self.model.loaded:
                return
            self.loading = True
            self.load_error = None
            try:
                await asyncio.to_thread(self.model.load)
            except Exception as exc:
                self.load_error = str(exc)
                raise
            finally:
                self.loading = False

    @asynccontextmanager
    async def activity(self, label: str):
        token = uuid.uuid4().hex
        operations = getattr(self, "_active_operations", None)
        if operations is None:
            operations = self._active_operations = {}
        operations[token] = label
        try:
            yield
        finally:
            operations.pop(token, None)

    @tracked_activity("a response is in progress")
    async def chat_completion(self, body: CompletionRequest):
        async with self._inference_lock:
            await self.ensure_loaded()
            return await asyncio.to_thread(
                self.model.chat,
                body.messages,
                tools=body.tools,
                max_new_tokens=body.max_tokens,
                temperature=body.temperature,
                generation_class="api_tool_turn" if body.tools else "ordinary_response",
            )

    @tracked_activity("background work is in progress")
    async def task_generate(self, messages: list[dict[str, Any]]) -> str:
        async with self._inference_lock:
            await self.ensure_loaded()
            settings = self.model.settings
            planner_budget = min(
                settings.max_new_tokens,
                int(getattr(settings, "task_planner_max_new_tokens", max(1024, settings.tool_action_max_new_tokens))),
            )
            return await asyncio.to_thread(
                self.model.generate,
                messages,
                max_new_tokens=planner_budget,
                temperature=settings.tool_temperature,
                generation_class="task_planning",
            )

    @tracked_activity("a final report is being written")
    async def task_synthesize(self, messages: list[dict[str, Any]]) -> str:
        async with self._inference_lock:
            await self.ensure_loaded()
            return await asyncio.to_thread(
                self.model.generate,
                messages,
                max_new_tokens=int(getattr(
                    self.model.settings,
                    "synthesis_max_new_tokens",
                    self.model.settings.max_new_tokens,
                )),
                temperature=self.model.settings.temperature,
                generation_class="final_synthesis",
            )

    @tracked_activity("a report section is being written")
    async def task_write_section(self, messages: list[dict[str, Any]]) -> str:
        async with self._inference_lock:
            await self.ensure_loaded()
            return await asyncio.to_thread(
                self.model.generate,
                messages,
                max_new_tokens=int(getattr(self.model.settings, "section_max_new_tokens", 3072)),
                temperature=self.model.settings.temperature,
                generation_class="report_section",
            )

    @tracked_activity("research sources are being filtered")
    async def classify_references(self, goal: str, candidates: list[dict[str, str]]) -> set[str]:
        async with self._inference_lock:
            await self.ensure_loaded()
            response = await asyncio.to_thread(
                self.model.generate,
                [
                    {"role": "system", "content": (
                        "Select citation candidates that are substantively relevant to the research goal. Favor "
                        "primary studies and authoritative reviews; reject navigation, author profiles, unrelated "
                        "citations, and duplicate editions. Return only JSON: {\"relevant_ids\": [string]}."
                    )},
                    {"role": "user", "content": json.dumps({
                        "goal": goal,
                        "candidates": candidates,
                    }, ensure_ascii=False)},
                ],
                max_new_tokens=int(getattr(self.model.settings, "research_classifier_max_new_tokens", 512)),
                temperature=0,
                generation_class="research_relevance",
            )
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
            payload = json.loads(cleaned[cleaned.find("{"):cleaned.rfind("}") + 1])
            allowed = {candidate["id"] for candidate in candidates}
            return {str(value) for value in payload.get("relevant_ids", []) if str(value) in allowed}
        except (ValueError, TypeError, json.JSONDecodeError):
            raise ValueError("The relevance classifier did not return valid candidate IDs.")

    @tracked_activity("research sources are being summarized")
    async def task_analyze_sources(self, messages: list[dict[str, Any]]) -> str:
        async with self._inference_lock:
            await self.ensure_loaded()
            return await asyncio.to_thread(
                self.model.generate,
                messages,
                max_new_tokens=int(getattr(self.model.settings, "research_notes_max_new_tokens", 1536)),
                temperature=self.model.settings.tool_temperature,
                generation_class="research_notes",
            )

    async def task_capability_catalog(self, plugin_ids: list[str]) -> list[dict[str, Any]]:
        await self.mcp.ensure_started(plugin_ids)
        return [{
            "name": tool.exposed_name,
            "description": tool.description,
            "required": tool.input_schema.get("required", []),
        } for tool in self.mcp.tools_for_plugins(plugin_ids)]

    @staticmethod
    def _durable_request(content: str, has_project_tools: bool) -> bool:
        normalized = " ".join(content.lower().split())
        durable_markers = (
            "overnight", "while i am away", "while i'm away", "keep working", "long-running",
            "long running", "comprehensive report", "detailed report", "literature review",
            "deep research", "research this", "research the", "investigate the", "investigate this",
        )
        iterative_markers = (
            "optimize", "optimise", "improve the grid", "improve this grid", "iterate until",
            "until the metric", "until it meets", "run scenarios", "sensitivity analysis",
        )
        return any(marker in normalized for marker in durable_markers) or (
            has_project_tools and any(marker in normalized for marker in iterative_markers)
        )

    @staticmethod
    def _needs_web_research(content: str) -> bool:
        normalized = content.lower()
        return any(term in normalized for term in (
            "research", "literature", "sources", "citations", "web", "internet", "papers", "state of the art",
        ))

    async def _start_durable_chat_task(
        self,
        chat: dict[str, Any],
        content: str,
        plugin_ids: list[str],
    ) -> str:
        title = " ".join(content.strip().split())[:120] or "Durable task"
        is_research = "local.web-research" in plugin_ids
        settings = self.model.settings
        success_criteria = [
            "The requested work is executed with persisted evidence rather than only described.",
            "Every requested metric, conclusion, technique, comparison, or deliverable is addressed.",
            "The final report identifies evidence, assumptions, limitations, unresolved gaps, and next steps.",
            "Claims are traceable to source URLs or tool-produced artifact, case, session, and metric records.",
        ]
        deliverables = ["A complete standalone report posted back into the originating chat."]
        if is_research:
            success_criteria.extend([
                "The research report contains at least 6,000 substantive words organized into independently drafted sections.",
                "A downloadable Word document is produced and preserved in app data.",
            ])
            deliverables = [
                "A downloadable Word research document stored in app data.",
                "A concise completion message and document link posted into the originating chat.",
            ]
        definition = TaskDefinition.model_validate({
            "title": title,
            "goal": content,
            "success_criteria": success_criteria,
            "deliverables": deliverables,
            "constraints": [
                "Never invent tool results, measurements, sources, or completion evidence.",
                "Checkpoint each bounded work item and resume after application restart.",
                "Pause for explicit approval or missing user input when required by tool policy.",
            ],
            "execution": {
                "deadline": None,
                "max_steps_per_episode": 64,
                "maximum_attempts": None,
                "retry_initial_seconds": 30,
                "retry_maximum_seconds": 3600,
                "resume_after_restart": True,
            },
            "metadata": {
                "mode": "durable_tools",
                "chat_id": chat["id"],
                "project_id": chat.get("project_id"),
                "plugin_ids": plugin_ids,
                "report_min_words": 6000 if is_research else None,
                "report_min_sources": 12 if is_research else None,
                "report_target_words": 8000 if is_research else None,
                "report_format": "docx" if is_research else None,
                "research_seed_sources": settings.research_seed_sources if is_research else None,
                "research_depth_passes": settings.research_depth_passes if is_research else None,
                "research_max_sources": settings.research_max_sources if is_research else None,
                "research_references_per_source": settings.research_references_per_source if is_research else None,
                "research_notes_batch_size": settings.research_notes_batch_size if is_research else None,
            },
        })
        task = self.tasks.create_task(content, definition)
        self.tasks.start_task(task["id"])
        self.task_engine.wake()
        answer = (
            "I started this as a durable task in this chat. It will plan checkpointed tool work, keep evidence "
            "outside the model context, audit the requested outcome, and post the completed report here. Leave "
            "Local Model running for uninterrupted work; if it closes, the task resumes on the next launch."
        )
        self.store.append_exchange(chat["id"], content, answer)
        return answer

    async def deliver_task_status(self, task: dict[str, Any]) -> None:
        metadata = task.get("definition", {}).get("metadata") or {}
        chat_id = metadata.get("chat_id")
        if not chat_id or task.get("delivered_at"):
            return
        status = task.get("status")
        if status not in {TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value}:
            return
        if status == TaskStatus.COMPLETED.value:
            report = ""
            artifacts: list[dict[str, Any]] = []
            for item in reversed(self.tasks.work_items(task["id"])):
                outcome = item.get("result") or {}
                if item.get("kind") == "synthesis" and isinstance(outcome, dict):
                    report = str(outcome.get("result") or "").strip()
                    artifacts = list(outcome.get("artifacts") or [])
                    if report or artifacts:
                        break
            message = report or task.get("current_summary") or "The durable task completed."
            if artifacts:
                links = []
                for artifact in artifacts:
                    filename = Path(str(artifact.get("relative_path") or "")).name
                    if filename:
                        label = str(artifact.get("name") or filename)
                        links.append(f"[{label}](/api/tasks/{task['id']}/artifacts/{filename})")
                if links:
                    message = f"{message}\n\n" + "\n".join(links)
        else:
            message = (
                f"The durable task ended with status {status}: "
                f"{task.get('last_error') or task.get('current_summary') or 'No additional detail was recorded.'}"
            )
        try:
            self.store.append_message(str(chat_id), "assistant", message)
        except KeyError:
            pass
        self.tasks.mark_delivered(task["id"])

    async def deliver_pending_task_results(self) -> None:
        for task in self.tasks.undelivered_terminal_tasks():
            await self.deliver_task_status(task)

    @tracked_activity("a response or tool action is in progress")
    async def respond(self, chat_id: str, content: str) -> str:
        history = [{"role": item["role"], "content": item["content"]} for item in self.store.messages(chat_id)]
        chat = self.store.get_chat(chat_id)
        if chat.get("project_id"):
            for plugin_id in self.store.project_plugins(str(chat["project_id"])):
                self.store.set_plugin_selected(chat_id, plugin_id, selected=True)
        selected_plugins = self.store.selected_plugins(chat_id)
        requested_plugins = self.requested_plugins(content)
        for plugin_id in requested_plugins:
            self.store.set_plugin_selected(chat_id, plugin_id, selected=True)
        if requested_plugins:
            selected_plugins = self.store.selected_plugins(chat_id)
        if self._needs_web_research(content):
            try:
                self.mcp.registry.get_plugin("local.web-research")
            except KeyError:
                pass
            else:
                self.store.set_plugin_selected(chat_id, "local.web-research", selected=True)
                selected_plugins = self.store.selected_plugins(chat_id)
        if self._durable_request(content, bool(selected_plugins)):
            return await self._start_durable_chat_task(chat, content, selected_plugins)
        async with self._inference_lock:
            await self.ensure_loaded()
            if selected_plugins:
                coordinator = ToolCoordinator(
                    self.model,
                    self.mcp,
                    Scratchpad(self.data_directory / "scratchpads" / f"{chat_id}.jsonl"),
                    self.data_directory / "tool_activity" / f"{chat_id}.jsonl",
                    router=self.tool_router,
                    max_calls=self.model.settings.max_tool_calls_per_step,
                )
                answer = await coordinator.respond(content, history, selected_plugins)
            else:
                coordinator = Coordinator(
                    self.model,
                    Scratchpad(self.data_directory / "scratchpads" / f"{chat_id}.jsonl"),
                )
                coordinator.history = history
                answer = await asyncio.to_thread(coordinator.respond, content)
        self.store.append_exchange(chat_id, content, answer)
        return answer

    def requested_plugins(self, content: str) -> list[str]:
        def normalize(value: str) -> str:
            return " ".join(value.lower().replace("_", " ").replace("-", " ").replace(".", " ").split())

        normalized = normalize(content)
        requested: list[str] = []
        for plugin in self.mcp.registry.plugins:
            manifest = plugin.manifest
            short_id = normalize(manifest.id.rsplit(".", 1)[-1])
            names = {
                normalize(manifest.id),
                normalize(manifest.name),
                f"{normalize(manifest.tool_namespace)} server",
                f"{normalize(manifest.tool_namespace)} mcp",
                f"{short_id} server",
                f"{short_id} mcp",
            }
            if any(name and name in normalized for name in names):
                requested.append(manifest.id)
        return requested

    async def stop_unused_plugins(self, plugin_ids: list[str]) -> None:
        for plugin_id in plugin_ids:
            if self.store.plugin_selection_count(plugin_id) == 0:
                await self.mcp.stop(plugin_id)

    def shutdown_status(self) -> dict[str, Any]:
        running_tasks = [
            {
                "id": task["id"],
                "title": task["definition"].get("title") or task["request"],
            }
            for task in self.tasks.list_tasks()
            if task["status"] == "running"
        ]
        reasons: list[str] = []
        if self.loading:
            reasons.append("the model is loading")
        active_operations = getattr(self, "_active_operations", {})
        reasons.extend(dict.fromkeys(active_operations.values()))
        if self._inference_lock.locked() and not active_operations:
            reasons.append("a response or tool action is in progress")
        if running_tasks:
            count = len(running_tasks)
            reasons.append(f"{count} background task{' is' if count == 1 else 's are'} finishing a step")
        return {
            "busy": bool(reasons),
            "draining": self.draining,
            "reasons": reasons,
            "running_tasks": running_tasks,
        }

    def prepare_shutdown(self) -> dict[str, Any]:
        """Stop taking new background episodes and let current work checkpoint."""
        self.draining = True
        self.task_engine.request_stop()
        return self.shutdown_status()

    @tracked_activity("local tools are starting")
    async def mcp_content_catalog(self, chat_id: str) -> list[dict[str, Any]]:
        plugin_ids = self.store.selected_plugins(chat_id)
        await self.mcp.ensure_started(plugin_ids)
        items = [
            *self.mcp.resources_for_plugins(plugin_ids),
            *self.mcp.prompts_for_plugins(plugin_ids),
        ]
        return [{
            "name": item.exposed_name,
            "kind": item.kind,
            "title": item.title,
            "description": item.description,
            "arguments": item.input_schema,
        } for item in items]

    @tracked_activity("a tool action is in progress")
    async def invoke_mcp_content(self, chat_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        plugin_ids = self.store.selected_plugins(chat_id)
        await self.mcp.ensure_started(plugin_ids)
        allowed = {
            item.exposed_name
            for item in [
                *self.mcp.resources_for_plugins(plugin_ids),
                *self.mcp.prompts_for_plugins(plugin_ids),
            ]
        }
        if name not in allowed:
            raise KeyError(name)
        # This API is the explicit application/user selection boundary for resources and prompts.
        return _compact_result(await self.mcp.call_tool(name, arguments, approved=True))

    @staticmethod
    def _write_state(path: Path, state: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _log_activity(path: Path, event: str, **data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **data,
            }, ensure_ascii=False) + "\n")

    def approval_state(self, chat_id: str) -> dict[str, Any]:
        self.store.get_chat(chat_id)
        state_path = self.data_directory / "tool_activity" / f"{chat_id}.state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"status": "none"}
        status = str(state.get("status") or "none")
        if status != "waiting_for_approval":
            return {"status": status}
        return {
            "status": status,
            "tool": state.get("exposed_tool"),
            "arguments": state.get("arguments") or {},
        }

    @tracked_activity("an approved tool action is in progress")
    async def approve_tool_call(self, chat_id: str) -> str:
        self.store.get_chat(chat_id)
        activity_path = self.data_directory / "tool_activity" / f"{chat_id}.jsonl"
        state_path = activity_path.with_suffix(".state.json")
        async with self._inference_lock:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") != "waiting_for_approval":
                raise RuntimeError("This approval has already been consumed or is no longer pending.")
            plugin_id = str(state.get("plugin_id") or "")
            if plugin_id not in self.store.selected_plugins(chat_id):
                raise RuntimeError("The tool's plugin is no longer enabled for this chat.")
            tool_name = str(state["exposed_tool"])
            arguments = dict(state.get("arguments") or {})
            approved_at = datetime.now(timezone.utc).isoformat()
            consuming = {**state, "status": "executing_approved_call", "approved_at": approved_at}
            self._write_state(state_path, consuming)
            self._log_activity(activity_path, "approved_tool_execution_started", tool=tool_name, arguments=arguments)
            try:
                await self.mcp.ensure_started([plugin_id])
                durable_work_item_id = state.get("durable_work_item_id")

                async def checkpoint_approved_task(handle: dict[str, Any]) -> None:
                    if durable_work_item_id:
                        self.tasks.checkpoint_work_item(
                            str(durable_work_item_id), {"pending_mcp_task": handle}
                        )

                if durable_work_item_id:
                    result = await self.mcp.call_tool(
                        tool_name,
                        arguments,
                        approved=True,
                        task_checkpoint=checkpoint_approved_task,
                    )
                else:
                    result = await self.mcp.call_tool(tool_name, arguments, approved=True)
                self._log_activity(activity_path, "approved_tool_execution_finished", tool=tool_name, result=result)
                compact = _compact_result(result)
                call_id = f"approval_{uuid.uuid4().hex}"
                history = [
                    {"role": item["role"], "content": item["content"]}
                    for item in self.store.messages(chat_id)[-6:]
                ]
                settings = getattr(self.model, "settings", None)
                reply = await asyncio.to_thread(
                    self.model.chat,
                    [
                        {
                            "role": "system",
                            "content": (
                                "The user explicitly approved the pending tool call. Observe its result and answer "
                                "the user's request. Do not call another tool or claim anything not present in the result."
                            ),
                        },
                        *history,
                        {"role": "user", "content": "I approve this tool call."},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": call_id,
                                "type": "function",
                                "function": {"name": tool_name, "arguments": arguments},
                            }],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name,
                            "content": json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
                        },
                    ],
                    tools=[],
                    max_new_tokens=int(getattr(settings, "max_new_tokens", FINAL_TOKENS)),
                    temperature=float(getattr(settings, "tool_temperature", 0.0)),
                )
                answer = reply.content.strip() or "The approved tool call completed."
                terminal = "failed" if result.get("isError") else "complete"
                self._write_state(state_path, {
                    **consuming,
                    "status": terminal,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "result": compact,
                    "answer": answer,
                })
                self.store.append_message(chat_id, "assistant", answer)
                durable_task_id = state.get("durable_task_id")
                if terminal == "complete" and durable_task_id and durable_work_item_id:
                    self.tasks.complete_work_item(
                        str(durable_work_item_id),
                        WorkItemOutcome(
                            outcome="completed",
                            summary=f"Approved tool call completed: {tool_name}",
                            result=answer,
                            completion_evidence=[json.dumps(compact, ensure_ascii=False)[:2000]],
                        ).model_dump(mode="json"),
                    )
                    self.tasks.resume_task(str(durable_task_id))
                    self.task_engine.wake()
                return answer
            except Exception as exc:
                self._write_state(state_path, {
                    **consuming,
                    "status": "failed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                })
                self._log_activity(
                    activity_path,
                    "approved_tool_execution_failed",
                    tool=tool_name,
                    error=type(exc).__name__,
                    message=str(exc),
                )
                raise

    async def deny_tool_call(self, chat_id: str) -> str:
        self.store.get_chat(chat_id)
        activity_path = self.data_directory / "tool_activity" / f"{chat_id}.jsonl"
        state_path = activity_path.with_suffix(".state.json")
        async with self._inference_lock:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") != "waiting_for_approval":
                raise RuntimeError("This approval is no longer pending.")
            message = "I did not run the pending tool because you declined approval."
            self._write_state(state_path, {
                **state,
                "status": "denied",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            self._log_activity(activity_path, "tool_approval_denied", tool=state.get("exposed_tool"))
            self.store.append_message(chat_id, "assistant", message)
            durable_task_id = state.get("durable_task_id")
            if durable_task_id:
                self.tasks.cancel_task(str(durable_task_id))
                await self.deliver_task_status(self.tasks.get_task(str(durable_task_id)))
            return message


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = Runtime()
    app.state.runtime = runtime
    if os.getenv("LOCAL_MODEL_EAGER_LOAD", "true").lower() in {"1", "true", "yes"}:
        asyncio.create_task(runtime.ensure_loaded())
    await runtime.task_engine.start()
    await runtime.deliver_pending_task_results()
    try:
        yield
    finally:
        await runtime.task_engine.stop()
        await runtime.mcp.stop_all()
        runtime.tasks.close()
        runtime.store.close()


app = FastAPI(title="Local Model Server", version="0.1.0", lifespan=lifespan)
STATIC_ROOT = ROOT / "local_model_app" / "static"
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


@app.middleware("http")
async def prevent_desktop_asset_caching(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


def runtime(request: Request) -> Runtime:
    return request.app.state.runtime


@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/health")
async def health(request: Request):
    service = runtime(request)
    return {
        "status": "error" if service.load_error else ("ready" if service.model.loaded else "loading"),
        "loaded": service.model.loaded,
        "loading": service.loading,
        "error": service.load_error,
        "model": service.model.settings.model_id,
    }


@app.get("/api/lifecycle/status", include_in_schema=False)
async def lifecycle_status(request: Request):
    return runtime(request).shutdown_status()


@app.post("/api/lifecycle/prepare-shutdown", include_in_schema=False)
async def prepare_shutdown(request: Request):
    return runtime(request).prepare_shutdown()


@app.post("/api/lifecycle/exit", include_in_schema=False)
async def exit_application(request: Request):
    request_exit = getattr(request.app.state, "request_process_exit", None)
    if request_exit is None:
        raise HTTPException(status_code=409, detail="The desktop launcher does not own this process.")
    request_exit()
    return {"status": "closing"}


@app.get("/api/settings")
async def get_settings(request: Request):
    service = runtime(request)
    return {
        **service.app_settings.public_settings(),
        "installed_models": service.app_settings.installed_models(),
        "detected_context_window": service.model.native_context_window,
        "effective_context_window": service.model.effective_context_window,
    }


@app.put("/api/settings")
async def update_settings(body: SettingsUpdateRequest, request: Request):
    service = runtime(request)
    before = service.app_settings.public_settings()
    try:
        saved = service.app_settings.update(body.model_dump())
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        **saved,
        "installed_models": service.app_settings.installed_models(),
        "restart_required": any(saved.get(key) != before.get(key) for key in saved),
    }


@app.get("/api/developer/generations")
async def generation_telemetry(request: Request, limit: int = 50):
    service = runtime(request)
    selected_limit = max(1, min(int(limit), 500))
    path = service.data_directory / "generation_telemetry.jsonl"
    try:
        with path.open("r", encoding="utf-8") as stream:
            lines = list(deque(stream, maxlen=selected_limit))
    except OSError:
        lines = []
    records = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return {
        "path": str(path),
        "records": records,
    }


@app.get("/api/models/search")
async def search_models(request: Request, q: str = ""):
    try:
        return await runtime(request).huggingface.search(q.strip())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Hugging Face search failed: {exc}") from exc


@app.post("/api/models/download")
async def download_model(body: ModelDownloadRequest, request: Request):
    try:
        return runtime(request).huggingface.start_download(body.repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/models/download/{job_id}")
async def model_download_status(job_id: str, request: Request):
    try:
        return runtime(request).huggingface.status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Download not found") from exc


@app.get("/api/projects")
async def list_projects(request: Request):
    return runtime(request).store.list_projects()


@app.post("/api/projects")
async def create_project(body: NewProjectRequest, request: Request):
    service = runtime(request)
    known_plugins = {plugin.manifest.id for plugin in service.mcp.registry.plugins}
    unknown = sorted(set(body.plugin_ids) - known_plugins)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown tools: {', '.join(unknown)}")
    try:
        return service.store.create_project(body.path, name=body.name, plugin_ids=body.plugin_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/activate")
async def activate_project(project_id: str, request: Request):
    service = runtime(request)
    try:
        plugin_ids = service.store.project_plugins(project_id)
        await service.mcp.ensure_started(plugin_ids)
        return {"project_id": project_id, "plugin_ids": plugin_ids}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not start project tools: {exc}") from exc


@app.get("/api/plugins")
async def list_plugins(request: Request):
    return runtime(request).mcp.status()


@app.post("/api/plugins")
async def install_plugin(body: dict[str, Any], request: Request):
    try:
        return await runtime(request).mcp.install(body)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (PluginConfigurationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/plugins/{plugin_id}/start")
async def start_plugin(plugin_id: str, request: Request):
    try:
        return await runtime(request).mcp.start(plugin_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Plugin not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/plugins/{plugin_id}/stop")
async def stop_plugin(plugin_id: str, request: Request):
    try:
        return await runtime(request).mcp.stop(plugin_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Plugin not found") from exc


@app.post("/v1/chat/completions")
async def chat_completions(body: CompletionRequest, request: Request):
    try:
        reply = await runtime(request).chat_completion(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    message: dict[str, Any] = {"role": "assistant", "content": reply.content}
    if reply.reasoning_content:
        message["reasoning_content"] = reply.reasoning_content
    if reply.tool_calls:
        message["tool_calls"] = reply.tool_calls
    return {
        "object": "chat.completion",
        "model": body.model or "local-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if reply.tool_calls else "stop",
            }
        ],
    }


@app.post("/api/chats")
async def create_chat(body: NewChatRequest, request: Request):
    try:
        return runtime(request).store.create_chat(body.title, project_id=body.project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc


@app.get("/api/chats")
async def list_chats(request: Request, archived: bool = False):
    return runtime(request).store.list_chats(archived=archived)


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str, request: Request):
    try:
        chat = runtime(request).store.get_chat(chat_id)
        project = (
            runtime(request).store.get_project(str(chat["project_id"]))
            if chat.get("project_id") else None
        )
        return {**chat, "project": project, "messages": runtime(request).store.messages(chat_id)}
    except KeyError as exc:
        try:
            chat = runtime(request).store.get_archived_chat(chat_id)
            return {**chat, "messages": runtime(request).store.archived_messages(chat_id)}
        except KeyError:
            raise HTTPException(status_code=404, detail="Chat not found") from exc


@app.get("/api/chats/{chat_id}/plugins")
async def chat_plugins(chat_id: str, request: Request):
    service = runtime(request)
    try:
        selected = set(service.store.selected_plugins(chat_id))
        await service.mcp.ensure_started(list(selected))
    except KeyError as exc:
        try:
            selected = set(service.store.archived_plugins(chat_id))
        except KeyError:
            raise HTTPException(status_code=404, detail="Chat not found") from exc
    return [{**plugin, "selected": plugin["id"] in selected} for plugin in service.mcp.status()]


@app.patch("/api/chats/{chat_id}/plugins/{plugin_id}")
async def select_chat_plugin(chat_id: str, plugin_id: str, body: PluginSelectionRequest, request: Request):
    service = runtime(request)
    try:
        if body.enabled:
            await service.mcp.start(plugin_id)
            service.store.set_plugin_selected(chat_id, plugin_id, selected=True)
        else:
            service.store.set_plugin_selected(chat_id, plugin_id, selected=False)
            await service.stop_unused_plugins([plugin_id])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Chat or plugin not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    selected = set(service.store.selected_plugins(chat_id))
    return [{**plugin, "selected": plugin["id"] in selected} for plugin in service.mcp.status()]


@app.patch("/api/chats/{chat_id}/project")
async def move_chat_to_project(chat_id: str, body: MoveChatRequest, request: Request):
    service = runtime(request)
    try:
        chat = service.store.move_chat_to_project(chat_id, body.project_id)
        if body.project_id:
            await service.mcp.ensure_started(service.store.project_plugins(body.project_id))
        return chat
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Chat or project not found") from exc


@app.get("/api/chats/{chat_id}/mcp-content")
async def list_chat_mcp_content(chat_id: str, request: Request):
    try:
        return await runtime(request).mcp_content_catalog(chat_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc


@app.post("/api/chats/{chat_id}/mcp-content")
async def invoke_chat_mcp_content(chat_id: str, body: McpContentRequest, request: Request):
    try:
        return await runtime(request).invoke_mcp_content(chat_id, body.name, body.arguments)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Resource or prompt not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/chats/{chat_id}", status_code=204)
async def delete_chat(chat_id: str, request: Request):
    service = runtime(request)
    try:
        try:
            plugin_ids = service.store.selected_plugins(chat_id)
        except KeyError:
            plugin_ids = []
        service.store.delete_chat(chat_id)
        (service.paths.data_directory / "scratchpads" / f"{chat_id}.jsonl").unlink(missing_ok=True)
        (service.paths.data_directory / "tool_activity" / f"{chat_id}.jsonl").unlink(missing_ok=True)
        await service.stop_unused_plugins(plugin_ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc


@app.patch("/api/chats/{chat_id}/archive")
async def archive_chat(chat_id: str, body: ArchiveChatRequest, request: Request):
    service = runtime(request)
    try:
        plugin_ids = service.store.selected_plugins(chat_id) if body.archived else []
        result = service.store.archive_chat(chat_id, archived=body.archived)
        if body.archived:
            await service.stop_unused_plugins(plugin_ids)
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc


@app.get("/api/chats/{chat_id}/scratchpad")
async def get_scratchpad(chat_id: str, request: Request):
    try:
        runtime(request).store.get_chat(chat_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc
    pad = Scratchpad(runtime(request).paths.data_directory / "scratchpads" / f"{chat_id}.jsonl")
    return {"entries": pad.recent(limit=20)}


@app.get("/api/chats/{chat_id}/approval")
async def get_chat_approval(chat_id: str, request: Request):
    try:
        return runtime(request).approval_state(chat_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc


@app.post("/api/chats/{chat_id}/approval/approve")
async def approve_chat_tool(chat_id: str, request: Request):
    try:
        answer = await runtime(request).approve_tool_call(chat_id)
        return {"role": "assistant", "content": answer}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Chat or tool not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chats/{chat_id}/approval/deny")
async def deny_chat_tool(chat_id: str, request: Request):
    try:
        return {"role": "assistant", "content": await runtime(request).deny_tool_call(chat_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/chats/{chat_id}/messages")
async def send_chat_message(chat_id: str, body: ChatMessageRequest, request: Request):
    try:
        answer = await runtime(request).respond(chat_id, body.content)
        return {"role": "assistant", "content": answer}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Chat not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/tasks/propose")
async def propose_task(body: TaskProposalRequest, request: Request):
    try:
        proposal = await runtime(request).task_coordinator.propose(body.request)
        return proposal.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not assemble a task proposal: {exc}") from exc


@app.post("/api/tasks")
async def create_task(body: CreateTaskRequest, request: Request):
    return runtime(request).tasks.create_task(body.request, body.definition)


@app.get("/api/tasks")
async def list_tasks(request: Request):
    return runtime(request).tasks.list_tasks()


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    try:
        return runtime(request).tasks.get_task(task_id, include_details=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc


@app.get("/api/tasks/{task_id}/artifacts/{artifact_name}")
async def get_task_artifact(task_id: str, artifact_name: str, request: Request):
    service = runtime(request)
    try:
        service.tasks.get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
    artifact_root = (service.data_directory / "artifacts" / task_id).resolve()
    artifact_path = (artifact_root / artifact_name).resolve()
    if artifact_path.parent != artifact_root or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(artifact_path, filename=artifact_path.name)


@app.post("/api/tasks/{task_id}/start")
async def start_task(task_id: str, request: Request):
    service = runtime(request)
    try:
        task = service.tasks.start_task(task_id)
        service.task_engine.wake()
        return task
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/tasks/{task_id}/pause")
async def pause_task(task_id: str, request: Request):
    try:
        return runtime(request).tasks.pause_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/tasks/{task_id}/resume")
async def resume_task(task_id: str, request: Request):
    service = runtime(request)
    try:
        task = service.tasks.resume_task(task_id)
        service.task_engine.wake()
        return task
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, request: Request):
    try:
        return runtime(request).tasks.cancel_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
