from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any


URL_PATTERN = re.compile(r"https?://[^\s<>\]\[\"']+", re.IGNORECASE)
WINDOWS_PATH_PATTERN = re.compile(r"[A-Za-z]:\\[^\r\n\"<>|]+")
POINTER_KEYS = (
    "artifact", "file", "path", "url", "uri", "handle", "task_id", "job_id",
    "case_id", "session_id", "source_id", "sha256",
)
MAX_ACTIONS = 128
MAX_FAILURES = 64
MAX_DECISIONS = 48
MAX_ARTIFACTS = 256


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded(value: Any, characters: int = 4000) -> Any:
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(serialized) <= characters:
        return value
    return {
        "preview": serialized[:characters],
        "truncated": True,
        "original_characters": len(serialized),
    }


def _dedupe(rows: list[Any], limit: int) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result[-limit:]


def _extract_pointers(value: Any, *, key: str = "") -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            if isinstance(child, (str, int)) and any(marker in child_key.lower() for marker in POINTER_KEYS):
                found.append({"kind": child_key, "value": str(child)})
            found.extend(_extract_pointers(child, key=child_key))
        return found
    if isinstance(value, list):
        for child in value:
            found.extend(_extract_pointers(child, key=key))
        return found
    if isinstance(value, str):
        for url in URL_PATTERN.findall(value):
            found.append({"kind": "url", "value": url.rstrip(".,;)")})
        for path in WINDOWS_PATH_PATTERN.findall(value):
            found.append({"kind": "path", "value": path.rstrip(".,;")})
    return found


def _previous_record(previous: dict[str, Any] | str | None) -> dict[str, Any]:
    if isinstance(previous, dict):
        return previous
    if not previous:
        return {}
    try:
        parsed = json.loads(previous)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_events(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calls: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            call_id = str(call.get("id") or "")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass
            calls[call_id] = {
                "call_id": call_id,
                "tool": str(function.get("name") or message.get("name") or "unknown"),
                "arguments": _bounded(arguments, 3000),
            }
        if message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "")
        row = dict(calls.get(call_id) or {
            "call_id": call_id,
            "tool": str(message.get("name") or "unknown"),
            "arguments": {},
        })
        content = message.get("content")
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except (ValueError, TypeError, json.JSONDecodeError):
            parsed = content
        row["result"] = _bounded(parsed, 5000)
        completed.append(row)
        if isinstance(parsed, dict) and (parsed.get("isError") or parsed.get("error")):
            failures.append(row)
    return completed, failures


def _decision_notes(messages: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    for message in messages:
        if message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        content = " ".join(str(message.get("content") or "").split())
        if content:
            notes.append(content[:4000])
    return notes


def _user_requirements(messages: list[dict[str, Any]]) -> list[str]:
    requirements: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = " ".join(str(message.get("content") or "").split())
        if content:
            requirements.append(content[:4000])
    return requirements


def build_semantic_checkpoint(
    *,
    objective: str,
    plan: str,
    messages: list[dict[str, Any]],
    structured_state: dict[str, Any],
    previous: dict[str, Any] | str | None,
    checkpoint_index: int,
    trigger: str,
    prompt_tokens: int | None = None,
    context_window: int | None = None,
    reserved_output_tokens: int | None = None,
    next_action: str | None = None,
) -> dict[str, Any]:
    """Create a non-nesting, host-built continuation record for a tool workflow."""
    prior = _previous_record(previous)
    completed, failures = _tool_events(messages)
    state = _bounded(structured_state, 16_000)
    decisions = _dedupe(
        [*list(prior.get("decisions") or []), *_decision_notes(messages)],
        MAX_DECISIONS,
    )
    requirements = _dedupe(
        [*list(prior.get("user_requirements") or []), *_user_requirements(messages)],
        MAX_DECISIONS,
    )
    completed_actions = _dedupe(
        [*list(prior.get("completed_actions") or []), *completed],
        MAX_ACTIONS,
    )
    failed_actions = _dedupe(
        [*list(prior.get("failed_actions") or []), *failures],
        MAX_FAILURES,
    )
    pointers = _dedupe(
        [
            *list(prior.get("artifacts_and_identifiers") or []),
            *_extract_pointers(structured_state),
            *_extract_pointers(completed_actions),
            *_extract_pointers(decisions),
        ],
        MAX_ARTIFACTS,
    )
    recent_messages = []
    for message in messages[-6:]:
        row = {
            key: value
            for key, value in message.items()
            if key in {"role", "name", "tool_call_id", "content", "tool_calls"}
        }
        row.pop("reasoning_content", None)
        recent_messages.append(_bounded(row, 4000))
    history = list(prior.get("checkpoint_history") or [])
    if prior.get("checkpoint_id"):
        history.append({
            "checkpoint_id": prior["checkpoint_id"],
            "index": prior.get("checkpoint_index"),
            "trigger": prior.get("trigger"),
            "created_at": prior.get("created_at"),
        })
    record: dict[str, Any] = {
        "schema_version": 2,
        "checkpoint_index": checkpoint_index,
        "created_at": _now(),
        "trigger": trigger,
        "objective": objective[:8000],
        "acceptance_and_execution_plan": plan[:6000],
        "invariants": [
            "Do not repeat completed tool calls unless their recorded result requires a retry.",
            "Do not claim completion without satisfying the recorded acceptance plan.",
            "Preserve exact artifact paths, URLs, handles, identifiers, measurements, assumptions, and errors.",
        ],
        "completed_actions": completed_actions,
        "failed_actions": failed_actions,
        "user_requirements": requirements,
        "decisions": decisions,
        "artifacts_and_identifiers": pointers,
        "latest_structured_state": state,
        "exact_next_action": next_action or (
            str(structured_state.get("next_step") or structured_state.get("next_tool") or "")
            or "Select the next necessary tool, or finish only when the acceptance plan is satisfied."
        ),
        "recent_messages": recent_messages,
        "context_budget": {
            "prompt_tokens_before_compaction": prompt_tokens,
            "context_window": context_window,
            "reserved_output_tokens": reserved_output_tokens,
        },
        "checkpoint_history": history[-64:],
    }
    digest_source = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    record["checkpoint_id"] = hashlib.sha256(digest_source).hexdigest()[:20]
    return record


def render_context_pack(record: dict[str, Any]) -> str:
    """Render a bounded model-facing pack while the full structured record stays on disk."""
    pack = {
        key: record.get(key)
        for key in (
            "schema_version", "checkpoint_id", "checkpoint_index", "trigger", "objective",
            "acceptance_and_execution_plan", "invariants", "exact_next_action", "context_budget",
        )
    }
    pack["completed_actions"] = [
        {
            "call_id": row.get("call_id"),
            "tool": row.get("tool"),
            "arguments": _bounded(row.get("arguments"), 1200),
            "result": _bounded(row.get("result"), 1800),
        }
        for row in list(record.get("completed_actions") or [])[-24:]
        if isinstance(row, dict)
    ]
    pack["failed_actions"] = list(record.get("failed_actions") or [])[-12:]
    pack["user_requirements"] = [
        str(value)[:1200] for value in list(record.get("user_requirements") or [])[-16:]
    ]
    pack["decisions"] = [str(value)[:1200] for value in list(record.get("decisions") or [])[-16:]]
    pack["artifacts_and_identifiers"] = list(record.get("artifacts_and_identifiers") or [])[-128:]
    pack["latest_structured_state"] = _bounded(record.get("latest_structured_state"), 8000)
    pack["recent_messages"] = list(record.get("recent_messages") or [])[-4:]
    serialized = json.dumps(pack, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(serialized) <= 32_000:
        return serialized
    pack["completed_actions"] = pack["completed_actions"][-12:]
    pack["decisions"] = pack["decisions"][-8:]
    pack["artifacts_and_identifiers"] = pack["artifacts_and_identifiers"][-64:]
    pack["recent_messages"] = pack["recent_messages"][-2:]
    serialized = json.dumps(pack, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(serialized) <= 32_000:
        return serialized
    pack["completed_actions"] = [
        {
            "call_id": row.get("call_id"),
            "tool": row.get("tool"),
            "arguments": _bounded(row.get("arguments"), 500),
            "result": _bounded(row.get("result"), 700),
        }
        for row in pack["completed_actions"][-6:]
    ]
    pack["failed_actions"] = pack["failed_actions"][-4:]
    pack["user_requirements"] = pack["user_requirements"][-8:]
    pack["decisions"] = pack["decisions"][-4:]
    pack["artifacts_and_identifiers"] = pack["artifacts_and_identifiers"][-32:]
    pack["latest_structured_state"] = _bounded(pack["latest_structured_state"], 4000)
    pack["recent_messages"] = []
    return json.dumps(pack, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
