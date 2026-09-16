from __future__ import annotations

import asyncio
import copy
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_model_app.mcp_manager import McpPluginManager
from local_model_app.model import AssistantReply, TransformersModel
from local_model_app.mcp_plugins import DiscoveredTool
from local_model_app.scratchpad import Scratchpad
from local_model_app.tool_router import ToolRouter


PLAN_TOKENS = 192
ACTION_TOKENS = 192
FINAL_TOKENS = 256
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


def _bounded_structured(value: Any) -> tuple[Any, dict[str, Any] | None]:
    cleaned = _strip_binary_payloads(value)
    serialized = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    size = len(serialized.encode("utf-8"))
    if size <= MAX_OBSERVATION_BYTES:
        return cleaned, None
    preview = serialized.encode("utf-8")[:MAX_OBSERVATION_BYTES].decode("utf-8", errors="ignore")
    return {
        "preview": preview,
        "truncated": True,
    }, {
        "originalBytes": size,
        "includedBytes": len(preview.encode("utf-8")),
        "estimatedOriginalTokens": (size + 3) // 4,
        "strategy": "deterministic_json_preview",
    }


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Avoid sending duplicate MCP text and structured payloads back to the model."""
    compact: dict[str, Any] = {"isError": bool(result.get("isError"))}
    if result.get("error"):
        compact.update({key: result[key] for key in ("error", "message") if key in result})
        return compact
    structured = result.get("structuredContent")
    if structured is not None:
        compact["data"], truncation = _bounded_structured(structured)
        if truncation:
            compact["truncation"] = truncation
        return compact
    text_parts = [
        str(item.get("text"))
        for item in result.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
    ]
    combined = "\n".join(text_parts)
    compact["content"] = combined[:MAX_TEXT_OBSERVATION_CHARS]
    if len(combined) > MAX_TEXT_OBSERVATION_CHARS:
        compact["truncation"] = {
            "originalCharacters": len(combined),
            "includedCharacters": MAX_TEXT_OBSERVATION_CHARS,
            "estimatedOriginalTokens": (len(combined) + 3) // 4,
            "strategy": "text_prefix",
        }
    return compact


def _structured_data(compact: dict[str, Any]) -> dict[str, Any]:
    data = compact.get("data")
    return data if isinstance(data, dict) else {}


class ToolCoordinator:
    def __init__(
        self,
        model: TransformersModel,
        manager: McpPluginManager,
        scratchpad: Scratchpad,
        activity_path: Path,
        router: ToolRouter | None = None,
    ) -> None:
        self.model = model
        self.manager = manager
        self.scratchpad = scratchpad
        self.activity_path = activity_path
        self.router = router or ToolRouter()

    def _log(self, event: str, **data: Any) -> None:
        self.activity_path.parent.mkdir(parents=True, exist_ok=True)
        with self.activity_path.open("a", encoding="utf-8") as activity_file:
            activity_file.write(json.dumps({"timestamp": _now(), "event": event, **data}, ensure_ascii=False) + "\n")

    @property
    def _state_path(self) -> Path:
        return self.activity_path.with_suffix(".state.json")

    def _load_state(self) -> dict[str, Any]:
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return state if isinstance(state, dict) else {}

    def _save_state(self, state: dict[str, Any]) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self._state_path)

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
    def _narrowed_specs(
        tools: list[DiscoveredTool],
        data: dict[str, Any],
        default_tools: list[DiscoveredTool] | None = None,
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

        return [tool.openai_spec() for tool in (default_tools if default_tools is not None else tools)]

    async def _execute(
        self,
        tool: DiscoveredTool,
        arguments: dict[str, Any],
        messages: list[dict[str, Any]],
        *,
        call_id: str | None = None,
        append_assistant_call: bool = True,
    ) -> dict[str, Any]:
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
            full_result = await self.manager.call_tool(tool.exposed_name, arguments)
        except PermissionError as exc:
            full_result = {"isError": True, "error": "approval_required", "message": str(exc)}
        except Exception as exc:
            full_result = {"isError": True, "error": type(exc).__name__, "message": str(exc)}
        self._log("tool_result", tool=tool.exposed_name, result=full_result, call_id=call_id)
        compact = _compact_result(full_result)
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

    @staticmethod
    def _data_requires_action(data: dict[str, Any]) -> bool:
        return bool(data.get("next_step") or data.get("next_tool") or data.get("tools") or data.get("tool"))

    async def respond(
        self,
        user_message: str,
        history: list[dict[str, Any]],
        plugin_ids: list[str],
    ) -> str:
        await self.manager.ensure_started(plugin_ids)
        tools = self.manager.tools_for_plugins(plugin_ids)
        if not tools:
            raise RuntimeError("The selected MCP plugins exposed no callable capabilities.")

        saved_state = self._load_state()
        resume_tool_name = saved_state.get("backend_tool") if saved_state.get("status") == "waiting_for_input" else None
        resume_required = saved_state.get("required_inputs") if isinstance(saved_state.get("required_inputs"), list) else []
        can_resume = bool(
            isinstance(resume_tool_name, str)
            and resume_required
            and all(self._required_input_is_present(str(name), user_message) for name in resume_required)
        )

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
        if can_resume:
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

        system_prompt = (
            "Execute the planning note. Use only supplied tools and never invent results or missing arguments. "
            "When a tool result gives an unambiguous next discovery step, the host may perform it automatically. "
            "Call exactly one tool when action is required. Return a final answer only when the request is complete "
            "or when a required input, permission, or external condition prevents progress.\n\n"
            f"Planning note:\n{plan}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *history[-6:],
            {"role": "user", "content": user_message},
        ]
        conversation_text = "\n".join(
            str(message.get("content", ""))
            for message in [*history, {"role": "user", "content": user_message}]
            if message.get("role") == "user"
        )
        max_calls = min(
            8,
            *(self.manager.registry.get_plugin(plugin_id).manifest.policy.max_calls_per_turn for plugin_id in plugin_ids),
        )
        call_count = 0
        data: dict[str, Any] = {}
        pending: tuple[DiscoveredTool, dict[str, Any], str | None, bool] | None = None

        schema_tool = next((tool for tool in tools if tool.native_name.endswith("get_tool_schema")), None)
        route_tool = next((tool for tool in tools if tool.native_name.endswith("route_request")), None)
        if can_resume and schema_tool:
            self._log("workflow_resumed", backend_tool=resume_tool_name, supplied_inputs=resume_required)
            pending = (schema_tool, {"tool_name": resume_tool_name}, None, True)
        elif route_tool and route_tool in routed.visible:
            pending = (route_tool, {"request_text": user_message}, None, True)

        premature_retries = 0
        while call_count < max_calls:
            if pending:
                selected_tool, arguments, existing_call_id, append_assistant_call = pending
                pending = None
                call_count += 1
                compact = await self._execute(
                    selected_tool,
                    arguments,
                    messages,
                    call_id=existing_call_id,
                    append_assistant_call=append_assistant_call,
                )
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

            specs = self._narrowed_specs(tools, data, routed.visible)
            reply: AssistantReply = await asyncio.to_thread(
                self.model.chat,
                messages,
                tools=specs,
                max_new_tokens=int(self._setting(
                    "tool_action_max_new_tokens" if self._data_requires_action(data) or not data else "tool_final_max_new_tokens",
                    ACTION_TOKENS if self._data_requires_action(data) or not data else FINAL_TOKENS,
                )),
                temperature=float(self._setting("tool_temperature", 0.0)),
            )
            assistant_message: dict[str, Any] = {"role": "assistant", "content": reply.content}
            if reply.reasoning_content:
                assistant_message["reasoning_content"] = reply.reasoning_content
            if reply.tool_calls:
                assistant_message["tool_calls"] = reply.tool_calls
            messages.append(assistant_message)

            if not reply.tool_calls:
                requires_action = self._data_requires_action(data)
                if requires_action and premature_retries < 1:
                    premature_retries += 1
                    messages.append({
                        "role": "user",
                        "content": "That response stopped before the required action. Call one supplied tool now; do not describe the call.",
                    })
                    continue
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
                if call_count >= max_calls:
                    break
                function = call["function"]
                raw_arguments = function.get("arguments", {})
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
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

        raise RuntimeError(f"Tool loop exceeded its {max_calls}-call budget without a final answer.")
