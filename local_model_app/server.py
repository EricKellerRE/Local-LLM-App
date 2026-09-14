from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from local_model_app.chat_store import ChatStore
from local_model_app.config import Settings, _load_dotenv
from local_model_app.coordinator import Coordinator
from local_model_app.model import TransformersModel
from local_model_app.mcp_manager import McpPluginManager
from local_model_app.mcp_plugins import McpPluginRegistry, PluginConfigurationError
from local_model_app.scratchpad import Scratchpad
from local_model_app.task_coordinator import ModelTaskCoordinator
from local_model_app.task_engine import UniversalTaskEngine
from local_model_app.task_models import TaskDefinition
from local_model_app.task_store import TaskStore
from local_model_app.tool_coordinator import ToolCoordinator


ROOT = Path(__file__).resolve().parents[1]
_load_dotenv(ROOT / ".env")


class CompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)


class NewChatRequest(BaseModel):
    title: str = "New chat"


class ChatMessageRequest(BaseModel):
    content: str = Field(min_length=1)


class ArchiveChatRequest(BaseModel):
    archived: bool = True


class PluginSelectionRequest(BaseModel):
    enabled: bool


class TaskProposalRequest(BaseModel):
    request: str = Field(min_length=1)


class CreateTaskRequest(BaseModel):
    request: str = Field(min_length=1)
    definition: TaskDefinition


class Runtime:
    def __init__(self) -> None:
        self.model = TransformersModel(Settings.from_environment(ROOT))
        self.store = ChatStore(ROOT / "data" / "chats.sqlite3")
        self.tasks = TaskStore(ROOT / "data" / "tasks.sqlite3")
        self.mcp = McpPluginManager(McpPluginRegistry(ROOT / "config" / "mcp.d", project_root=ROOT))
        self.load_error: str | None = None
        self.loading = False
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self.task_coordinator = ModelTaskCoordinator(self.task_generate)
        self.task_engine = UniversalTaskEngine(
            self.tasks,
            self.task_coordinator,
            poll_seconds=float(os.getenv("LOCAL_TASK_POLL_SECONDS", "5")),
            lease_seconds=int(os.getenv("LOCAL_TASK_LEASE_SECONDS", "1800")),
        )

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

    async def chat_completion(self, body: CompletionRequest):
        async with self._inference_lock:
            await self.ensure_loaded()
            return await asyncio.to_thread(
                self.model.chat,
                body.messages,
                tools=body.tools,
                max_new_tokens=body.max_tokens,
                temperature=body.temperature,
            )

    async def task_generate(self, messages: list[dict[str, Any]]) -> str:
        async with self._inference_lock:
            await self.ensure_loaded()
            return await asyncio.to_thread(self.model.generate, messages)

    async def respond(self, chat_id: str, content: str) -> str:
        history = [{"role": item["role"], "content": item["content"]} for item in self.store.messages(chat_id)]
        selected_plugins = self.store.selected_plugins(chat_id)
        async with self._inference_lock:
            await self.ensure_loaded()
            if selected_plugins:
                coordinator = ToolCoordinator(
                    self.model,
                    self.mcp,
                    Scratchpad(ROOT / "data" / "scratchpads" / f"{chat_id}.jsonl"),
                    ROOT / "data" / "tool_activity" / f"{chat_id}.jsonl",
                )
                answer = await coordinator.respond(content, history, selected_plugins)
            else:
                coordinator = Coordinator(
                    self.model,
                    Scratchpad(ROOT / "data" / "scratchpads" / f"{chat_id}.jsonl"),
                )
                coordinator.history = history
                answer = await asyncio.to_thread(coordinator.respond, content)
        self.store.append_exchange(chat_id, content, answer)
        return answer

    async def stop_unused_plugins(self, plugin_ids: list[str]) -> None:
        for plugin_id in plugin_ids:
            if self.store.plugin_selection_count(plugin_id) == 0:
                await self.mcp.stop(plugin_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = Runtime()
    app.state.runtime = runtime
    if os.getenv("LOCAL_MODEL_EAGER_LOAD", "true").lower() in {"1", "true", "yes"}:
        asyncio.create_task(runtime.ensure_loaded())
    await runtime.task_engine.start()
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
    return runtime(request).store.create_chat(body.title)


@app.get("/api/chats")
async def list_chats(request: Request, archived: bool = False):
    return runtime(request).store.list_chats(archived=archived)


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str, request: Request):
    try:
        chat = runtime(request).store.get_chat(chat_id)
        return {**chat, "messages": runtime(request).store.messages(chat_id)}
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


@app.delete("/api/chats/{chat_id}", status_code=204)
async def delete_chat(chat_id: str, request: Request):
    service = runtime(request)
    try:
        try:
            plugin_ids = service.store.selected_plugins(chat_id)
        except KeyError:
            plugin_ids = []
        service.store.delete_chat(chat_id)
        (ROOT / "data" / "scratchpads" / f"{chat_id}.jsonl").unlink(missing_ok=True)
        (ROOT / "data" / "tool_activity" / f"{chat_id}.jsonl").unlink(missing_ok=True)
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
    pad = Scratchpad(ROOT / "data" / "scratchpads" / f"{chat_id}.jsonl")
    return {"entries": pad.recent(limit=20)}


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
