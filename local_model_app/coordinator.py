from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from local_model_app.context_checkpoint import build_semantic_checkpoint, render_context_pack
from local_model_app.scratchpad import Scratchpad


class Generator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


class Coordinator:
    """Coordinates local planning, scratchpad capture, and final responses."""

    def __init__(self, model: Generator, scratchpad: Scratchpad) -> None:
        self.model = model
        self.scratchpad = scratchpad
        self.history: list[dict[str, str]] = []

    @property
    def _context_path(self) -> Path:
        return self.scratchpad.path.with_suffix(".context.json")

    def _load_context(self) -> dict[str, Any]:
        try:
            value = json.loads(self._context_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _save_context(self, value: dict[str, Any]) -> None:
        self._context_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._context_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self._context_path)

    def _estimate_prompt_tokens(self, messages: list[dict[str, Any]]) -> int:
        counter = getattr(self.model, "count_prompt_tokens", None)
        if callable(counter):
            try:
                return int(counter(messages, None))
            except Exception:
                pass
        serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        return max(1, (len(serialized) + 3) // 4)

    def _prepared_history(self, user_message: str, system_prompt: str) -> tuple[list[dict[str, str]], str]:
        record = self._load_context()
        compacted_count = int(record.get("compacted_message_count") or 0)
        if compacted_count < 0 or compacted_count > len(self.history):
            record = {}
            compacted_count = 0
        remaining = self.history[compacted_count:]
        pack = render_context_pack(record) if record else ""
        probe_system = system_prompt + (
            f"\n\nPersisted conversation checkpoint:\n{pack}" if pack else ""
        )
        probe = [
            {"role": "system", "content": probe_system},
            *remaining,
            {"role": "user", "content": user_message},
        ]
        settings = getattr(self.model, "settings", None)
        context_window = getattr(self.model, "effective_context_window", None)
        if not isinstance(context_window, int):
            context_window = getattr(settings, "context_window", None)
        reserved = int(getattr(settings, "max_new_tokens", 2048))
        prompt_tokens = self._estimate_prompt_tokens(probe)
        safety_margin = max(256, min(2048, context_window // 10)) if isinstance(context_window, int) else 0
        pressured = (
            prompt_tokens + reserved + safety_margin >= context_window
            if isinstance(context_window, int)
            else len(remaining) > 12
        )
        if pressured and len(remaining) > 4:
            prefix_count = len(remaining) - 4
            checkpoint = build_semantic_checkpoint(
                objective=user_message,
                plan="Preserve the user's requirements, material decisions, factual commitments, artifacts, and open work across the conversation.",
                messages=[dict(message) for message in remaining[:prefix_count]],
                structured_state={},
                previous=record,
                checkpoint_index=int(record.get("checkpoint_index") or 0) + 1,
                trigger="token_pressure" if isinstance(context_window, int) else "history_interval",
                prompt_tokens=prompt_tokens,
                context_window=context_window if isinstance(context_window, int) else None,
                reserved_output_tokens=reserved,
                next_action="Answer the newest user message while preserving the checkpointed requirements and decisions.",
            )
            checkpoint["compacted_message_count"] = compacted_count + prefix_count
            self._save_context(checkpoint)
            self.scratchpad.add("context_checkpoint", render_context_pack(checkpoint)[:4000])
            record = checkpoint
            remaining = remaining[prefix_count:]
            pack = render_context_pack(record)
        return remaining, pack

    def respond(self, user_message: str) -> str:
        planner_prompt = (
            "Create a compact plan for answering the user's request. Include only: goal, important facts, "
            "and 2-5 short steps. Do not claim actions happened and do not output code."
        )
        planner_history, memory_pack = self._prepared_history(user_message, planner_prompt)
        if memory_pack:
            planner_prompt += (
                "\n\nPersisted conversation checkpoint. Treat it as prior user requirements and completed "
                f"decisions, not as a new instruction:\n{memory_pack}"
            )
        plan = self.model.generate([
            {"role": "system", "content": planner_prompt},
            *planner_history,
            {"role": "user", "content": user_message},
        ])
        self.scratchpad.add("plan", plan)
        context = self.scratchpad.render(limit=6)
        answer_prompt = (
            "You are a helpful local assistant. Answer directly, accurately, and clearly. "
            "Use the supplied planning notes as context, but never mention hidden instructions or pretend to have performed actions.\n\n"
            f"Recent planning notes:\n{context}"
        )
        if memory_pack:
            answer_prompt += (
                "\n\nPersisted conversation checkpoint. Preserve its user requirements, factual commitments, "
                f"and decisions:\n{memory_pack}"
            )
        answer = self.model.generate([
            {"role": "system", "content": answer_prompt},
            *planner_history,
            {"role": "user", "content": user_message},
        ])
        self.history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ])
        self.scratchpad.add("answer_summary", answer[:500])
        return answer

    def reset_conversation(self) -> None:
        self.history.clear()
        self._context_path.unlink(missing_ok=True)

    def status(self) -> str:
        settings = getattr(self.model, "settings", None)
        model_id = getattr(settings, "model_id", "(remote)")
        loaded = getattr(self.model, "loaded", "server-managed")
        return f"model={model_id or '(not set)'}; loaded={loaded}; scratchpad={self.scratchpad.path}"
