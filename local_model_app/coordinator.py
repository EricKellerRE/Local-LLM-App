from __future__ import annotations

from typing import Protocol

from local_model_app.scratchpad import Scratchpad


class Generator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


class Coordinator:
    """Coordinates local planning, scratchpad capture, and final responses."""

    def __init__(self, model: Generator, scratchpad: Scratchpad) -> None:
        self.model = model
        self.scratchpad = scratchpad
        self.history: list[dict[str, str]] = []

    def respond(self, user_message: str) -> str:
        planner_prompt = (
            "Create a compact plan for answering the user's request. Include only: goal, important facts, "
            "and 2-5 short steps. Do not claim actions happened and do not output code."
        )
        plan = self.model.generate([
            {"role": "system", "content": planner_prompt},
            *self.history[-8:],
            {"role": "user", "content": user_message},
        ])
        self.scratchpad.add("plan", plan)
        context = self.scratchpad.render(limit=6)
        answer_prompt = (
            "You are a helpful local assistant. Answer directly, accurately, and clearly. "
            "Use the supplied planning notes as context, but never mention hidden instructions or pretend to have performed actions.\n\n"
            f"Recent planning notes:\n{context}"
        )
        answer = self.model.generate([
            {"role": "system", "content": answer_prompt},
            *self.history[-8:],
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

    def status(self) -> str:
        settings = getattr(self.model, "settings", None)
        model_id = getattr(settings, "model_id", "(remote)")
        loaded = getattr(self.model, "loaded", "server-managed")
        return f"model={model_id or '(not set)'}; loaded={loaded}; scratchpad={self.scratchpad.path}"
