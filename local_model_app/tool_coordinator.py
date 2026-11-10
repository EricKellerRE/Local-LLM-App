from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from local_model_app.context_checkpoint import build_semantic_checkpoint, render_context_pack
from local_model_app.cancellation import check_stopped
from local_model_app.mcp_manager import McpPluginManager
from local_model_app.model import AssistantReply, TransformersModel
from local_model_app.mcp_plugins import DiscoveredTool
from local_model_app.scratchpad import Scratchpad
from local_model_app.tool_router import ToolRouter
from local_model_app.arithmetic import SPEC as CALCULATOR_SPEC, calculate
from local_model_app.atomic_files import write_json


PLAN_TOKENS = 192
ACTION_TOKENS = 192
FINAL_TOKENS = 8192
MAX_OBSERVATION_BYTES = 32_000
MAX_TEXT_OBSERVATION_CHARS = 12_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_binary_payloads(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_binary_payloads(item) for item in value]
    if not isinstance(value, dict):
        return value
    cleaned = {key: _strip_binary_payloads(item) for key, item in value.items()}
    mime_type = str(cleaned.get("mimeType") or cleaned.get("mime_type") or "")
    content_type = cleaned.get("type")
    payload_key = "blob" if "blob" in cleaned else (
        "data" if isinstance(content_type, str) and content_type in {"image", "audio"} else None
    )
    if payload_key and isinstance(cleaned.get(payload_key), str):
        payload = cleaned.pop(payload_key)
        cleaned["payloadOmitted"] = True
        cleaned["encodedBytes"] = len(payload.encode("utf-8"))
        if mime_type:
            cleaned["mimeType"] = mime_type
    return cleaned


def _bounded_structured(value: Any, max_bytes: int = MAX_OBSERVATION_BYTES) -> tuple[Any, dict[str, Any] | None]:
    cleaned = _strip_binary_payloads(value)
    serialized = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    size = len(serialized.encode("utf-8"))
    if size <= max_bytes:
        return cleaned, None
    preview = serialized.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
    return {
        "preview": preview,
        "truncated": True,
    }, {
        "originalBytes": size,
        "includedBytes": len(preview.encode("utf-8")),
        "estimatedOriginalTokens": (size + 3) // 4,
        "strategy": "deterministic_json_preview",
    }


def _compact_result(
    result: dict[str, Any],
    *,
    max_structured_bytes: int = MAX_OBSERVATION_BYTES,
    max_text_characters: int = MAX_TEXT_OBSERVATION_CHARS,
) -> dict[str, Any]:
    """Avoid sending duplicate MCP text and structured payloads back to the model."""
    compact: dict[str, Any] = {"isError": bool(result.get("isError"))}
    if result.get("error") or (result.get("isError") and result.get("message")):
        compact.update({key: result[key] for key in ("error", "message") if key in result})
        return compact
    structured = result.get("structuredContent")
    if structured is not None:
        compact["data"], truncation = _bounded_structured(structured, max_structured_bytes)
        if truncation:
            compact["truncation"] = truncation
        return compact
    text_parts = [
        str(item.get("text"))
        for item in result.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
    ]
    combined = "\n".join(text_parts)
    compact["content"] = combined[:max_text_characters]
    if len(combined) > max_text_characters:
        compact["truncation"] = {
            "originalCharacters": len(combined),
            "includedCharacters": max_text_characters,
            "estimatedOriginalTokens": (len(combined) + 3) // 4,
            "strategy": "text_prefix",
        }
    return compact


def _structured_data(compact: dict[str, Any]) -> dict[str, Any]:
    data = compact.get("data")
    return data if isinstance(data, dict) else {}


def _input_artifact_paths(text: str) -> set[str]:
    """Allow exact user-supplied artifacts as well as tool-produced artifacts."""
    return set(re.findall(r'(?:[A-Za-z]:[\\/]|/)[^\r\n<>"|?*]*?\.(?:csv|tsv|json|txt)\b', text, re.IGNORECASE))


def _input_case_paths(text: str) -> set[str]:
    return set(re.findall(r'(?:[A-Za-z]:[\\/]|/)[^\r\n<>"|?*]*?\.(?:pwb|raw|epc)\b', text, re.IGNORECASE))


class ToolCoordinator:
    def __init__(
        self,
        model: TransformersModel,
        manager: McpPluginManager,
        scratchpad: Scratchpad,
        activity_path: Path,
        router: ToolRouter | None = None,
        max_calls: int = 256,
        task_checkpoint: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        telemetry_context: dict[str, Any] | None = None,
        structured_plans: bool = True,
    ) -> None:
        self.model = model
        self.manager = manager
        self.scratchpad = scratchpad
        self.activity_path = activity_path
        self.router = router or ToolRouter()
        # This is a circuit breaker for one checkpoint, not a task-wide quota.
        # Durable work can run as many checkpoints/episodes as completion needs.
        self.max_calls = max(1, min(int(max_calls), 4096))
        self.task_checkpoint = task_checkpoint
        self.telemetry_context = dict(telemetry_context or {})
        self.structured_plans = structured_plans
        self._artifact_paths: set[str] = set()

    def _remember_artifacts(self, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith('artifact_path') and isinstance(item, str):
                    self._artifact_paths.add(item)
                elif isinstance(item, (dict, list)):
                    self._remember_artifacts(item)
        elif isinstance(value, list):
            for item in value:
                self._remember_artifacts(item)

    def _observation_limits(self) -> tuple[int, int]:
        context_window = getattr(self.model, "effective_context_window", None)
        if not isinstance(context_window, int):
            context_window = getattr(getattr(self.model, "settings", None), "context_window", None)
        if not isinstance(context_window, int):
            return MAX_OBSERVATION_BYTES, MAX_TEXT_OBSERVATION_CHARS
        observation_tokens = max(512, min(context_window // 8, 8_000))
        characters = observation_tokens * 4
        return characters, characters

    def _log(self, event: str, **data: Any) -> None:
        self.activity_path.parent.mkdir(parents=True, exist_ok=True)
        with self.activity_path.open("a", encoding="utf-8") as activity_file:
            activity_file.write(json.dumps({"timestamp": _now(), "event": event, **data}, ensure_ascii=False) + "\n")

    @property
    def _state_path(self) -> Path:
        return self.activity_path.with_suffix(".state.json")

    @property
    def _checkpoint_history_path(self) -> Path:
        return self.activity_path.with_suffix(".checkpoints.jsonl")

    def _load_state(self) -> dict[str, Any]:
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return state if isinstance(state, dict) else {}

    def _save_state(self, state: dict[str, Any]) -> None:
        write_json(self._state_path,state)

    def _record_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self._checkpoint_history_path.parent.mkdir(parents=True, exist_ok=True)
        with self._checkpoint_history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(checkpoint, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _clear_state(self) -> None:
        self._state_path.unlink(missing_ok=True)

    @staticmethod
    def _by_native_name(tools: list[DiscoveredTool], native_name: str) -> DiscoveredTool | None:
        return next((tool for tool in tools if tool.native_name == native_name), None)

    @staticmethod
    def _arguments_for(tool: DiscoveredTool, data: dict[str, Any], user_message: str) -> dict[str, Any] | None:
        if tool.native_name.endswith("route_request"):
            return {"request_text": user_message}
        if tool.native_name.endswith("list_tools") and data.get("capability"):
            return {"capability": data["capability"]}
        if tool.native_name.endswith("get_instructions") and data.get("capability"):
            return {"capability": data["capability"]}
        return None

    def _next_protocol_call(
        self,
        data: dict[str, Any],
        tools: list[DiscoveredTool],
        user_message: str,
    ) -> tuple[DiscoveredTool, dict[str, Any]] | None:
        # Chunked web reads advertise an exact cursor. Continuing this operation is
        # protocol mechanics, not a reasoning decision; preserve the original URL
        # byte-for-byte instead of asking the model to reproduce it.
        next_start = data.get("next_start")
        source_url = data.get("url")
        if data.get("complete") is False and isinstance(source_url, str) and source_url and isinstance(next_start, int):
            fetch_tool = next((tool for tool in tools if tool.native_name.endswith("fetch_url")), None)
            if fetch_tool:
                return fetch_tool, {"url": source_url, "start_index": next_start}

        next_name = data.get("next_tool")
        if isinstance(next_name, str):
            tool = self._by_native_name(tools, next_name)
            if tool:
                arguments = data.get("next_arguments")
                if isinstance(arguments, dict):
                    return tool, arguments
                inferred = self._arguments_for(tool, data, user_message)
                if inferred is not None:
                    return tool, inferred

        next_step = data.get("next_step")
        if isinstance(next_step, str):
            for tool in tools:
                if tool.native_name in next_step:
                    arguments = self._arguments_for(tool, data, user_message)
                    if arguments is not None:
                        return tool, arguments

        candidates = data.get("tools")
        if isinstance(candidates, list) and len(candidates) == 1 and isinstance(candidates[0], dict):
            candidate_name = candidates[0].get("name")
            schema_tool = next((tool for tool in tools if tool.native_name.endswith("get_tool_schema")), None)
            if schema_tool and isinstance(candidate_name, str):
                return schema_tool, {"tool_name": candidate_name}
        return None

    @staticmethod
    def _required_input_is_present(name: str, user_message: str) -> bool:
        lowered = user_message.lower()
        if name.lower() in lowered:
            return True
        if name in {"case_path", "pww_path"}:
            return bool(re.search(r"[a-z]:\\[^\r\n\"]+\.(?:pwb|pww)\b", user_message, re.IGNORECASE))
        if name.endswith("_csv"):
            return bool(re.search(r"[a-z]:\\[^\r\n\"]+\.csv\b", user_message, re.IGNORECASE))
        return False

    def _missing_required_inputs(self, data: dict[str, Any], user_message: str) -> tuple[str, list[str]] | None:
        tool = data.get("tool")
        if not isinstance(tool, dict) or not isinstance(tool.get("input_schema"), dict):
            return None
        schema = tool["input_schema"]
        required = [str(name) for name in schema.get("required", [])]
        missing = [name for name in required if not self._required_input_is_present(name, user_message)]
        return (str(tool.get("name") or "selected tool"), missing) if missing else None

    @staticmethod
    def _path_for_extension(text: str, extensions: str, *, last: bool = False) -> str | None:
        matches = re.findall(rf"[a-z]:\\[^\r\n\"]+?\.(?:{extensions})(?=\b|[\s.,;])", text, re.IGNORECASE)
        if not matches:
            return None
        candidate = matches[-1 if last else 0].rstrip(".,;")
        # Chat/Markdown users sometimes escape underscores in copied Windows paths.
        # Only repair that form when the repaired path is confirmed to exist.
        repaired = candidate.replace(r"\_", "_")
        if repaired != candidate and Path(repaired).exists():
            return repaired
        return candidate

    def _deterministic_gateway_call(
        self,
        data: dict[str, Any],
        tools: list[DiscoveredTool],
        conversation_text: str,
    ) -> tuple[DiscoveredTool, dict[str, Any]] | None:
        """Build the mechanical MCP gateway wrapper when schema inputs are explicit."""
        backend = data.get("tool")
        if not isinstance(backend, dict) or not isinstance(backend.get("input_schema"), dict):
            return None
        schema = backend["input_schema"]
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        arguments: dict[str, Any] = {}
        if "paths_csv" in properties:
            value = self._path_for_extension(conversation_text, "csv", last=True)
            if value:
                arguments["paths_csv"] = value
        if "case_path" in properties:
            value = self._path_for_extension(conversation_text, "pwb|pww")
            if value:
                arguments["case_path"] = value

        required = [str(name) for name in schema.get("required", [])]
        if any(name not in arguments for name in required):
            return None

        lowered = conversation_text.lower()
        if "separate" in lowered and ("working" in lowered or "output" in lowered):
            run_root = self.activity_path.parents[1] / "workflow_runs" / self.activity_path.stem
            if "out_dir" in properties:
                arguments["out_dir"] = str(run_root / "working")
            if "export_dir" in properties:
                arguments["export_dir"] = str(run_root / "output")

        gateway = next((tool for tool in tools if tool.native_name.endswith("call_tool")), None)
        backend_name = backend.get("name")
        if not gateway or not isinstance(backend_name, str):
            return None
        return gateway, {"tool_name": backend_name, "arguments": arguments}

    @staticmethod
    def _selection_spec(tools: list[DiscoveredTool]) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": "host__select_tools",
            "description": "Load schemas for the next step from the complete connected tool catalog. "
            "Use this when the current tools cannot advance the task. This does not execute the selected tools.\n"
            + "\n".join(f"{tool.exposed_name}: {tool.description[:100]}" for tool in tools),
            "parameters": {"type": "object", "required": ["names"], "properties": {
                "names": {"type": "array", "minItems": 1, "maxItems": 4,
                          "items": {"type": "string", "enum": [tool.exposed_name for tool in tools]}}
            }, "additionalProperties": False},
        }}

    @staticmethod
    def _narrowed_specs(
        tools: list[DiscoveredTool],
        data: dict[str, Any],
        default_tools: list[DiscoveredTool] | None = None,
        artifact_paths: set[str] | None = None,
        case_paths: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        candidate_rows = data.get("tools")
        if isinstance(candidate_rows, list) and len(candidate_rows) > 1:
            candidate_names = [row.get("name") for row in candidate_rows if isinstance(row, dict) and row.get("name")]
            schema_tool = next((tool for tool in tools if tool.native_name.endswith("get_tool_schema")), None)
            if schema_tool:
                spec = copy.deepcopy(schema_tool.openai_spec())
                properties = spec["function"]["parameters"].setdefault("properties", {})
                properties.setdefault("tool_name", {})["enum"] = candidate_names
                return [spec]

        tool_schema = data.get("tool")
        if isinstance(tool_schema, dict) and tool_schema.get("input_schema"):
            gateway = next((tool for tool in tools if tool.native_name.endswith("call_tool")), None)
            if gateway:
                return [gateway.openai_spec()]

        selected = list(default_tools if default_tools is not None else tools)
        # Input discovery is a prerequisite, even when analytical tools rank higher.
        for tool in tools:
            if tool.native_name == "workspace.resolve_inputs" and tool not in selected:
                selected.append(tool)
        specs = []
        for tool in selected:
            spec = copy.deepcopy(tool.openai_spec())
            properties = spec['function']['parameters'].get('properties', {})
            for key in ('case_path', 'case_path_hint'):
                if case_paths and key in properties:
                    properties[key]['enum'] = sorted(case_paths)
                    properties[key]['description'] = 'Use an exact case path supplied by the user. Do not retype, shorten, or reconstruct it.'
            if artifact_paths and tool.native_name.startswith('results.') and 'path' in properties:
                properties['path']['enum'] = sorted(artifact_paths)
                properties['path']['description'] = 'Choose an exact artifact path returned by a previous tool; do not reconstruct or invent a filename.'
            specs.append(spec)
        return specs

    async def _execute(
        self,
        tool: DiscoveredTool,
        arguments: dict[str, Any],
        messages: list[dict[str, Any]],
        *,
        call_id: str | None = None,
        append_assistant_call: bool = True,
    ) -> dict[str, Any]:
        check_stopped()
        call_id = call_id or f"call_{uuid.uuid4().hex}"
        call = {
            "id": call_id,
            "type": "function",
            "function": {"name": tool.exposed_name, "arguments": arguments},
        }
        if append_assistant_call:
            messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        self._log("tool_call", tool=tool.exposed_name, arguments=arguments, call_id=call_id)
        try:
            if self.task_checkpoint is None:
                full_result = await self.manager.call_tool(tool.exposed_name, arguments)
            else:
                full_result = await self.manager.call_tool(
                    tool.exposed_name,
                    arguments,
                    task_checkpoint=self.task_checkpoint,
                )
        except PermissionError as exc:
            full_result = {"isError": True, "error": "approval_required", "message": str(exc)}
        except Exception as exc:
            full_result = {"isError": True, "error": type(exc).__name__, "message": str(exc)}
            # MCP transport errors carry the backend response in ErrorData.data.
            # str(exc) alone discards it, leaving the agent with an opaque HTTP 500.
            detail = getattr(getattr(exc, "error", None), "data", None)
            if detail is not None:
                full_result["message"] += "\nBackend detail: " + (
                    detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
                )[:MAX_TEXT_OBSERVATION_CHARS]
        self._log("tool_result", tool=tool.exposed_name, result=full_result, call_id=call_id)
        if not full_result.get('isError'):
            self._remember_artifacts(full_result)
        max_bytes, max_characters = self._observation_limits()
        compact = _compact_result(
            full_result,
            max_structured_bytes=max_bytes,
            max_text_characters=max_characters,
        )
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool.exposed_name,
            "content": json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
        })
        return compact

    def _finish(self, answer: str, *, event: str = "answer") -> str:
        if event == "answer":
            self._clear_state()
        self.scratchpad.add("answer_summary", answer[:500])
        self._log(event, content=answer)
        return answer

    def _setting(self, name: str, default: int | float) -> int | float:
        return getattr(getattr(self.model, "settings", None), name, default)

    def _estimated_prompt_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> int:
        counter = getattr(self.model, "count_prompt_tokens", None)
        if callable(counter):
            try:
                return int(counter(messages, tools))
            except Exception:
                pass
        serialized = json.dumps(
            {"messages": messages, "tools": tools},
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return max(1, (len(serialized) + 3) // 4)

    def _context_pressure(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reserved_output_tokens: int,
    ) -> tuple[bool, int, int | None, int]:
        context_window = getattr(self.model, "effective_context_window", None)
        if not isinstance(context_window, int):
            context_window = getattr(getattr(self.model, "settings", None), "context_window", None)
        prompt_tokens = self._estimated_prompt_tokens(messages, tools)
        if not isinstance(context_window, int):
            return False, prompt_tokens, None, 0
        safety_margin = max(256, min(2048, context_window // 10))
        pressured = prompt_tokens + reserved_output_tokens + safety_margin >= context_window
        return pressured, prompt_tokens, context_window, safety_margin

    @staticmethod
    def _data_requires_action(data: dict[str, Any]) -> bool:
        return bool(data.get("next_step") or data.get("next_tool") or data.get("tools") or data.get("tool"))

    @staticmethod
    def _promises_execution(text: str) -> bool:
        return bool(re.search(r"\b(?:I\s+(?:will|shall)|I['’]ll)\s+(?:begin|proceed|scan|inspect|analy[sz]e|execute|use|run|search|query|calculate|start|discover)\b|\b(?:I am|I'm|I’m)\s+ready to (?:begin|start|proceed)\b", text, re.I))

    async def respond(
        self,
        user_message: str,
        history: list[dict[str, Any]],
        plugin_ids: list[str],
    ) -> str:
        await self.manager.ensure_started(plugin_ids)
        tools = self.manager.tools_for_plugins(plugin_ids)
        # AUX snapshots are exported text artifacts, not binary PowerWorld cases.
        # Generic "analyze" wording otherwise ranks this unrelated tool highly.
        user_intent = '\n'.join(str(m.get('content', '')) for m in history if m.get('role') == 'user') + '\n' + user_message
        binary_case_task = bool(re.search(r'\.pw[bw]\b|\bPowerWorld\s+(?:case|model)', user_intent, re.I))
        aux_requested = bool(re.search(r'\.aux\b|\baux(?:iliary)?\s+(?:file|snapshot|export)', user_intent, re.I))
        if binary_case_task and not aux_requested:
            tools = [tool for tool in tools if tool.native_name != 'results.analyze_aux_snapshots']
        if not tools:
            raise RuntimeError("The selected MCP plugins exposed no callable capabilities.")
        if self.structured_plans:
            from .plan_execution import PlanExecution
            return await PlanExecution(self, tools).run(user_message, history, plugin_ids)

        saved_state = self._load_state()
        self._artifact_paths = set(saved_state.get('artifact_paths') or [])
        self._artifact_paths.update(_input_artifact_paths(user_message))
        case_paths = _input_case_paths(user_message)
        self._remember_artifacts(saved_state.get('data'))
        for pointer in (saved_state.get('checkpoint') or {}).get('artifacts_and_identifiers', []):
            if isinstance(pointer, dict) and str(pointer.get('kind', '')).endswith('artifact_path') and isinstance(pointer.get('value'), str):
                self._artifact_paths.add(pointer['value'])
        resumed_checkpoint = (
            saved_state
            if saved_state.get("status") == "tool_checkpoint"
            and saved_state.get("user_message") == user_message
            else {}
        )
        checkpoint_record = dict(resumed_checkpoint.get('checkpoint') or {})
        checkpoint_summary = (
            render_context_pack(checkpoint_record, max_characters=8000)
            if checkpoint_record else str(resumed_checkpoint.get('summary') or '')[:8000]
        )
        scratchpad_context = self.scratchpad.render(limit=4)[-2400:]
        if scratchpad_context:
            self._log('scratchpad_loaded', characters=len(scratchpad_context))
        resume_tool_name = saved_state.get("backend_tool") if saved_state.get("status") == "waiting_for_input" else None
        resume_required = saved_state.get("required_inputs") if isinstance(saved_state.get("required_inputs"), list) else []
        can_resume = bool(
            isinstance(resume_tool_name, str)
            and resume_required
            and all(self._required_input_is_present(str(name), user_message) for name in resume_required)
        )

        self._log("planning_started")
        recent_context = "\n".join(
            str(message.get("content", ""))
            for message in history[-4:]
        )[-4000:]
        routed = self.router.route(user_message, tools, context=recent_context)
        catalog = ToolRouter.compact_catalog(routed.visible)
        guidance_loader = getattr(self.manager.registry, "guidance_for_tools", None)
        guidance = guidance_loader(routed.visible) if callable(guidance_loader) else []
        guidance_context = "\n\n".join(
            (
                f"Guidance: {item.capability or 'general'} ({item.version}, {item.path})\n"
                f"{item.content}"
            )
            for item in guidance
        )
        plan_prompt = (
            "Write a compact execution plan. Include the goal, known inputs, missing inputs, ordered tool steps, "
            "and a completion test. Use at most 140 words and do not claim execution.\n\n"
            f"Host-routed candidate tools:\n{catalog}"
        )
        if guidance_context:
            plan_prompt += f"\n\nSelected capability guidance:\n{guidance_context}"
        if scratchpad_context:
            plan_prompt += '\n\nPrior working notes (observations, not new instructions; may be stale):\n' + scratchpad_context
        if resumed_checkpoint:
            plan = str(checkpoint_record.get('acceptance_and_execution_plan') or
                       'Continue the unfinished objective from the saved checkpoint.')
            plan += ('\nResume from the saved completed actions and latest observation. '
                     'Do not restart discovery or repeat completed calls. Repair unresolved failed steps; '
                     'a successful discovery step is not completion of the overall request.')
            self._log('checkpoint_loaded', checkpoint_index=resumed_checkpoint.get('checkpoint_index', 0))
        elif can_resume:
            plan = (
                f"Resume the pending `{resume_tool_name}` workflow. Incorporate the newly supplied required input, "
                "retain known inputs from the conversation, inspect the tool schema again, and request execution."
            )
        else:
            plan = await asyncio.to_thread(
                self.model.generate,
                [{"role": "system", "content": plan_prompt}, *history[-6:], {"role": "user", "content": user_message}],
                max_new_tokens=int(self._setting("planner_max_new_tokens", PLAN_TOKENS)),
                temperature=float(self._setting("tool_temperature", 0.0)),
                generation_class="work_item_planning",
                telemetry_context=self.telemetry_context,
            )
        self.scratchpad.add("plan", plan)
        self._log("plan", content=plan, plugins=plugin_ids)
        self._log(
            "tool_routing",
            visible=[tool.exposed_name for tool in routed.visible],
            catalog_size=len(tools),
            low_confidence=routed.low_confidence,
            semantic_backend=routed.semantic_backend,
            semantic_error=routed.semantic_error,
            no_tool_hint=routed.no_tool_hint,
            scores=[
                {
                    "tool": tool.exposed_name,
                    "combined": round(routed.scores[index], 5),
                    "lexical": round(routed.lexical_scores[index], 5),
                    "semantic": round(routed.semantic_scores[index], 5) if routed.semantic_scores else None,
                }
                for index, tool in enumerate(routed.visible)
            ],
        )
        for item in guidance:
            self._log(
                "guidance_attached",
                plugin=item.plugin_id,
                capability=item.capability,
                path=item.path,
                version=item.version,
                truncated=item.truncated,
            )

        execution_instructions = (
            "Execute the planning note. Use only supplied tools and never invent results or missing arguments. "
            "Use host__calculate for arithmetic, including unit conversions and products; copy its results exactly. "
            "When a tool result gives an unambiguous next discovery step, the host may perform it automatically. "
            "Use host__select_tools to load other tools for each next step; the initial tools are only a shortlist. "
            "For binary PowerWorld cases (.pwb), discover exact case paths, then use case.query_objects and "
            "case field-discovery tools. results.analyze_aux_snapshots requires existing exported AUX text files; "
            "it cannot read .pwb cases. A directory of cases is not an AUX snapshot input. "
            "Call exactly one tool when action is required. Return a final answer only when the request is complete "
            "or when a required input, permission, or external condition prevents progress.\n\n"
        )
        system_prompt = execution_instructions + f"Planning note:\n{plan}"
        if guidance_context:
            system_prompt += f"\n\nSelected capability guidance:\n{guidance_context}"
        if scratchpad_context:
            system_prompt += ('\n\nPrior scratchpad notes (may be stale; checkpoint observations take precedence):\n'
                              + scratchpad_context)
        base_system_prompt = system_prompt
        checkpoint_index = int(resumed_checkpoint.get("checkpoint_index") or 0)
        if checkpoint_summary:
            system_prompt += (
                "\n\nPersisted continuation record from earlier tool checkpoints. Treat it as bounded "
                "observations from this same request and continue from it; do not repeat completed calls:\n"
                f"{checkpoint_summary}"
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *history[-6:],
            {"role": "user", "content": user_message},
        ]
        if checkpoint_summary:
            messages.append({
                "role": "user",
                "content": "Continue the same work from the persisted checkpoint. Call the next necessary tool.",
            })
        conversation_text = "\n".join(
            str(message.get("content", ""))
            for message in [*history, {"role": "user", "content": user_message}]
            if message.get("role") == "user"
        )
        max_calls = min(
            self.max_calls,
            *(self.manager.registry.get_plugin(plugin_id).manifest.policy.max_calls_per_turn for plugin_id in plugin_ids),
        )
        call_count = 0
        failed_calls: dict[str, int] = {}
        repeated_observations: dict[str, tuple[str, int]] = {}
        loop_recoveries: set[str] = set()
        selected_tools: list[DiscoveredTool] | None = [
            tool for tool in tools if tool.exposed_name in resumed_checkpoint.get('selected_tools', [])
        ] or None
        selection_attempts = 0
        malformed_attempts = 0
        retry_output_budget = 0
        data = (
            dict(resumed_checkpoint.get("data") or {})
            if isinstance(resumed_checkpoint.get("data"), dict)
            else {}
        )
        pending: tuple[DiscoveredTool, dict[str, Any], str | None, bool] | None = None

        schema_tool = next((tool for tool in tools if tool.native_name.endswith("get_tool_schema")), None)
        route_tool = next((tool for tool in tools if tool.native_name.endswith("route_request")), None)
        if can_resume and schema_tool:
            self._log("workflow_resumed", backend_tool=resume_tool_name, supplied_inputs=resume_required)
            pending = (schema_tool, {"tool_name": resume_tool_name}, None, True)
        elif not resumed_checkpoint and route_tool and route_tool in routed.visible:
            pending = (route_tool, {"request_text": user_message}, None, True)

        premature_retries = 0

        def apply_checkpoint(
            trigger: str,
            *,
            prompt_tokens: int | None = None,
            context_window: int | None = None,
            reserved_output_tokens: int | None = None,
            next_action: str | None = None,
        ) -> None:
            nonlocal checkpoint_index, checkpoint_record, checkpoint_summary
            nonlocal messages, call_count, premature_retries
            checkpoint_index += 1
            checkpoint_record = build_semantic_checkpoint(
                objective=user_message,
                plan=plan,
                messages=messages,
                structured_state=data,
                previous=checkpoint_record or checkpoint_summary,
                checkpoint_index=checkpoint_index,
                trigger=trigger,
                prompt_tokens=prompt_tokens,
                context_window=context_window,
                reserved_output_tokens=reserved_output_tokens,
                next_action=next_action,
            )
            checkpoint_summary = render_context_pack(checkpoint_record, max_characters=12000)
            self._save_state({
                "status": "tool_checkpoint",
                "user_message": user_message,
                "checkpoint_index": checkpoint_index,
                "checkpoint": checkpoint_record,
                "summary": checkpoint_summary,
                "data": data,
                "artifact_paths": sorted(self._artifact_paths),
                "selected_tools": [tool.exposed_name for tool in selected_tools or []],
            })
            self._record_checkpoint(checkpoint_record)
            self.scratchpad.add("tool_checkpoint", checkpoint_summary[:4000])
            self._log(
                "tool_checkpoint",
                checkpoint_index=checkpoint_index,
                checkpoint_id=checkpoint_record["checkpoint_id"],
                trigger=trigger,
                prompt_tokens=prompt_tokens,
                context_window=context_window,
                reserved_output_tokens=reserved_output_tokens,
                continuing=True,
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"{base_system_prompt}\n\nContext checkpoint {checkpoint_index}. The host compacted "
                        "working context without ending the task. Continue from this semantic record, preserve its "
                        "constraints and identifiers, and do not repeat completed calls:\n"
                        f"{checkpoint_summary}"
                    ),
                },
                {"role": "user", "content": user_message},
                {
                    "role": "user",
                    "content": "Continue the same work from the checkpoint. Call the next necessary tool.",
                },
            ]
            call_count = 0
            premature_retries = 0

        while True:
            check_stopped()
            if call_count >= max_calls:
                pending_action = None
                if pending:
                    pending_action = (
                        f"Call {pending[0].exposed_name} with "
                        f"{json.dumps(pending[1], ensure_ascii=False, separators=(',', ':'))}."
                    )
                apply_checkpoint("call_interval", next_action=pending_action)

            if pending:
                selected_tool, arguments, existing_call_id, append_assistant_call = pending
                pending = None
                fingerprint = selected_tool.exposed_name + json.dumps(arguments, sort_keys=True)
                if failed_calls.get(fingerprint, 0) >= 2:
                    return self._finish(
                        "Stopped because the agent repeated the same failed tool call without correcting it. "
                        "No case results were obtained from that call. Last error: "
                        + str(last_failure), event="answer",
                    )
                if repeated_observations.get(fingerprint, ("", 0))[1] >= 3:
                    next_action = (
                        "The following exact request is blocked because it already returned the same observation "
                        "three times: " + fingerprint + ". Use the saved observation/checkpoint. "
                        "Select tools for the next unfinished step, use genuinely different arguments if needed, "
                        "or report the specific blocker. Do not execute that exact request again."
                    )
                    apply_checkpoint("repeated_observation", next_action=next_action)
                    if fingerprint not in loop_recoveries and len(loop_recoveries) < 2:
                        loop_recoveries.add(fingerprint)
                        selected_tools = None
                        messages.append({"role": "user", "content": next_action})
                        self._log("loop_recovery", tool=selected_tool.exposed_name)
                        continue
                    self._log("loop_stopped", tool=selected_tool.exposed_name)
                    return self._finish(
                        "Stopped a tool loop: the agent requested the same tool and arguments after three "
                        "identical observations and did not recover. The analysis is incomplete; "
                        "no further identical call was executed. Progress was saved for resume.", event="incomplete",
                    )
                call_count += 1
                compact = await self._execute(
                    selected_tool,
                    arguments,
                    messages,
                    call_id=existing_call_id,
                    append_assistant_call=append_assistant_call,
                )
                if compact.get("isError"):
                    failed_calls[fingerprint] = failed_calls.get(fingerprint, 0) + 1
                    last_failure = compact.get("message") or compact.get("data") or compact.get("content")
                    # Keep the failed operation and its discovery tools visible.
                    # Otherwise the initial routing shortlist can strand the model
                    # on unrelated discovery tools after a recoverable error.
                    selected_tools = [selected_tool]
                    if selected_tool.native_name == "case.query_objects":
                        selected_tools += [tool for tool in tools if tool.native_name in {
                            "case.search_object_fields", "case.describe_object_type"
                        }]
                    messages.append({"role": "user", "content":
                        "Repair the failed tool call using the error below; its schema remains available. "
                        "Do not restart completed discovery or repeat unchanged failed arguments. "
                        "For unknown constraint columns: discover the canonical field for the same object type, "
                        "include that field in fields, then inspect its exported header with limit=1 and no constraints "
                        "if needed. Retry the original query with the corrected field/header and original constraints. "
                        "A voltage-level listing does not repair a missing query column. "
                        "For file/path failures, use an available input-discovery tool to resolve the supplied reference. "
                        "Never invent a filename. If recovery is impossible, report the actual error.\n"
                        + str(last_failure)})
                else:
                    failed_calls.pop(fingerprint, None)
                    digest = hashlib.sha256(json.dumps(compact, sort_keys=True).encode()).hexdigest()
                    previous_digest, previous_count = repeated_observations.get(fingerprint, ("", 0))
                    count = previous_count + 1 if previous_digest == digest else 1
                    repeated_observations[fingerprint] = (digest, count)
                    if count >= 2:
                        self._log("repeated_observation", tool=selected_tool.exposed_name, count=count)
                        messages.append({"role": "user", "content":
                            "This exact tool call returned the same observation again. It is not advancing the task. "
                            "Call host__select_tools to load tools for the next step, or explain the blocker. "
                            "Do not repeat this call."})
                if compact.get("error") == "approval_required":
                    message = str(compact.get("message") or "This tool requires approval.")
                    self._save_state({
                        "status": "waiting_for_approval",
                        "plugin_id": selected_tool.plugin_id,
                        "exposed_tool": selected_tool.exposed_name,
                        "arguments": arguments,
                    })
                    return self._finish(f"Approval is required before I can continue: {message}", event="waiting_for_approval")
                data = _structured_data(compact)
                # Persist each completed tool boundary without altering the live
                # prompt. Resume must not depend on reaching a context-limit checkpoint.
                boundary = build_semantic_checkpoint(
                    objective=user_message, plan=plan, messages=messages, structured_state=data,
                    previous=checkpoint_record or checkpoint_summary, checkpoint_index=checkpoint_index,
                    trigger='tool_boundary',
                )
                self._save_state({'status':'tool_checkpoint','user_message':user_message,
                                  'checkpoint_index':checkpoint_index,'checkpoint':boundary,
                                  'data':data,'artifact_paths':sorted(self._artifact_paths),
                                  'selected_tools':[tool.exposed_name for tool in selected_tools or []]})
                missing = self._missing_required_inputs(data, user_message)
                if missing and missing[1]:
                    tool_name, names = missing
                    self._save_state({
                        "status": "waiting_for_input",
                        "backend_tool": tool_name,
                        "required_inputs": names,
                    })
                    joined = ", ".join(f"`{name}`" for name in names)
                    return self._finish(
                        f"I reached `{tool_name}`, but cannot execute it without the required input {joined}. "
                        "Please provide that input and I can resume from this point.",
                        event="waiting_for_input",
                    )
                next_call = self._next_protocol_call(data, tools, user_message)
                if next_call:
                    pending = (*next_call, None, True)
                    continue
                gateway_call = self._deterministic_gateway_call(data, tools, conversation_text)
                if gateway_call:
                    self._log("gateway_call_prepared", backend_tool=data["tool"]["name"])
                    pending = (*gateway_call, None, True)
                    continue

            specs = self._narrowed_specs(tools, data, selected_tools or routed.visible, self._artifact_paths, case_paths)
            specs.append(self._selection_spec(tools))
            specs.append(copy.deepcopy(CALCULATOR_SPEC))
            requires_action = self._data_requires_action(data) or not data
            output_budget = int(
                self._setting("tool_action_max_new_tokens", ACTION_TOKENS)
                if requires_action
                else self._setting(
                    "post_tool_decision_max_new_tokens",
                    self._setting("max_new_tokens", FINAL_TOKENS),
                )
            )
            output_budget = max(output_budget, retry_output_budget)
            pressured, prompt_tokens, context_window, safety_margin = self._context_pressure(
                messages,
                specs,
                output_budget,
            )
            if pressured:
                apply_checkpoint(
                    "token_pressure",
                    prompt_tokens=prompt_tokens,
                    context_window=context_window,
                    reserved_output_tokens=output_budget,
                    next_action=(
                        str(data.get("next_step") or data.get("next_tool") or "")
                        or "Ask the model to choose the next necessary tool from the supplied schemas."
                    ),
                )
                self._log(
                    "context_pressure_relieved",
                    checkpoint_index=checkpoint_index,
                    safety_margin=safety_margin,
                )
                specs = self._narrowed_specs(tools, data, selected_tools or routed.visible, self._artifact_paths, case_paths)
                specs.append(self._selection_spec(tools))
                specs.append(copy.deepcopy(CALCULATOR_SPEC))
                # Compaction is not proof that the result fits. Recount the exact
                # messages plus schemas, retaining output and safety headroom.
                for pack_limit in (8000, 4000, 2000, 1024):
                    still_pressured, _, _, _ = self._context_pressure(messages, specs, output_budget)
                    if not still_pressured:
                        break
                    smaller = render_context_pack(checkpoint_record, max_characters=pack_limit)
                    messages = [
                        {"role": "system", "content": base_system_prompt + '\n\nContext checkpoint:\n' + smaller},
                        {"role": "user", "content": user_message},
                    ]
                if self._context_pressure(messages, specs, output_budget)[0]:
                    # The old base prompt retained the full planning text, guidance
                    # and scratchpad on every compaction. Shrinking only the new
                    # checkpoint could therefore never relieve this pressure.
                    base_system_prompt = execution_instructions
                    if guidance:
                        base_system_prompt += '\nCapability guidance:\n' + guidance[0].content[:4000]
                    messages = [
                        {'role':'system','content':base_system_prompt + '\nContext checkpoint:\n' + render_context_pack(checkpoint_record,max_characters=2000)},
                        {'role':'user','content':user_message},
                    ]
                    # Keep schemas intact, but disclose fewer at once. The selector
                    # remains available so omitted tools can be loaded next step.
                    selector = self._selection_spec(tools)
                    candidates = [spec for spec in specs if spec['function']['name'] != 'host__select_tools']
                    while self._context_pressure(messages, specs, output_budget)[0] and candidates:
                        candidates.pop()
                        specs = candidates + [selector]
                    if self._context_pressure(messages, specs, output_budget)[0] and output_budget > 512:
                        output_budget = 512
                    pressure, actual_tokens, window, margin = self._context_pressure(messages,specs,output_budget)
                    self._log('context_payload_reduced',prompt_tokens=actual_tokens,context_window=window,
                              reserved_output_tokens=output_budget,tool_schema_count=len(specs))
                    if pressure:
                        raise RuntimeError(f'Model context is too small even after compaction: {actual_tokens} prompt tokens + {output_budget} output tokens + {margin} safety tokens exceed {window}. Progress is saved.')
            reply: AssistantReply = await asyncio.to_thread(
                self.model.chat,
                messages,
                tools=specs,
                max_new_tokens=output_budget,
                temperature=float(self._setting("tool_temperature", 0.0)),
                generation_class="required_tool_action" if premature_retries else ("tool_action" if requires_action else "post_tool_decision"),
                telemetry_context=self.telemetry_context,
            )
            # Reject malformed/multiple calls before appending them to history:
            # chat templates may themselves fail on invalid JSON or orphan calls.
            try:
                if len(reply.tool_calls) > 1:
                    raise ValueError('Exactly one tool call is supported per step')
                for call in reply.tool_calls:
                    function = call['function']
                    raw = function.get('arguments', {})
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                    if not isinstance(parsed, dict):
                        raise ValueError('Arguments must be a JSON object')
                    if not isinstance(function.get('name'), str) or not call.get('id'):
                        raise ValueError('Tool call requires name and id')
                    function['arguments'] = parsed
            except (ValueError, TypeError, KeyError):
                malformed_attempts += 1
                self._log('tool_call_rejected', reason='invalid_call_format', attempt=malformed_attempts)
                if malformed_attempts >= 3:
                    return self._finish('The local model produced invalid tool-call arguments three times. '
                                        'Those calls were not executed. Analysis is incomplete.')
                if reply.stop_reason == 'length':
                    retry_output_budget = min(max(output_budget * 2, 1024), 4096)
                messages.append({'role': 'user', 'content':
                    'Your last tool call was not executed because its format was invalid. '
                    'Generate exactly one complete tool call with arguments as a valid JSON object. '
                    'Use double-quoted keys and escape backslashes in Windows paths. '
                    'Do not include commentary in the arguments.'})
                continue
            malformed_attempts = 0
            retry_output_budget = 0
            assistant_message: dict[str, Any] = {"role": "assistant", "content": reply.content}
            if reply.tool_calls:
                assistant_message["tool_calls"] = reply.tool_calls
            messages.append(assistant_message)

            if not reply.tool_calls:
                requires_action = self._data_requires_action(data) or self._promises_execution(reply.content)
                if requires_action and premature_retries < 3:
                    premature_retries += 1
                    self._log('premature_final_rejected', attempt=premature_retries)
                    messages.append({
                        "role": "user",
                        "content": "That response stopped before the required action. Call one supplied tool now; do not describe the call.",
                    })
                    continue
                if requires_action:
                    apply_checkpoint('premature_final', next_action='Execute the next required tool; the last response only promised action.')
                    return self._finish('Analysis is incomplete: the model repeatedly described its next action without executing it. Progress was saved for resume.', event='incomplete')
                if not data and routed.low_confidence and routed.has_more and premature_retries < 1:
                    premature_retries += 1
                    routed = routed.expanded()
                    self._log("tool_routing_expanded", visible=[tool.exposed_name for tool in routed.visible])
                    messages.append({
                        "role": "user",
                        "content": (
                            "Tool routing confidence was low, so the host expanded the candidate set. "
                            "Review the newly supplied tools. Call one if the request needs it; otherwise answer directly."
                        ),
                    })
                    continue
                return self._finish(reply.content.strip())

            premature_retries = 0
            for call in reply.tool_calls:
                function = call["function"]
                raw_arguments = function.get("arguments", {})
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
                if function['name'] == 'host__calculate':
                    try:
                        result = {'isError': False, 'results': calculate(arguments.get('expressions'))}
                    except (ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
                        result = {'isError': True, 'error': str(exc)}
                    self._log('tool_call', tool='host__calculate', arguments=arguments, call_id=call['id'])
                    self._log('tool_result', tool='host__calculate', result=result, call_id=call['id'])
                    messages.append({'role':'tool','tool_call_id':call['id'],'name':function['name'],
                                     'content':json.dumps(result)})
                    data = result
                    break
                if function["name"] == "host__select_tools":
                    selection_attempts += 1
                    if selection_attempts > 64:
                        return self._finish("Stopped repeated tool selection without completing the task. Analysis is incomplete.")
                    names = arguments.get("names")
                    valid = (isinstance(names, list) and 1 <= len(names) <= 4
                             and all(isinstance(name, str) and any(t.exposed_name == name for t in tools) for name in names))
                    if valid:
                        selected_tools = [t for t in tools if t.exposed_name in names]
                    messages.append({"role": "tool", "tool_call_id": call["id"], "name": function["name"],
                                     "content": json.dumps({"isError": not valid, "message":
                                         "Schemas loaded. Execute the next tool now." if valid else "Select 1 to 4 names from the catalog."})})
                    self._log("tool_selection", names=names if valid else [], valid=valid)
                    data = {}
                    break
                selected_tool = next((tool for tool in tools if tool.exposed_name == function["name"]), None)
                if selected_tool is None:
                    self._log("invalid_tool_call", tool=function["name"], arguments=arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": function["name"],
                        "content": json.dumps({"isError": True, "error": "unknown_tool"}),
                    })
                    continue
                pending = (selected_tool, arguments, call["id"], False)
                break
