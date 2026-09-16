from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_model_app.config import Settings


@dataclass(frozen=True)
class AssistantReply:
    content: str
    tool_calls: list[dict[str, Any]]
    reasoning_content: str | None = None


class TransformersModel:
    """Lazy Hugging Face Transformers loader and text generator."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tokenizer: Any | None = None
        self.processor: Any | None = None
        self.model: Any | None = None
        self.is_multimodal = False
        self.uses_device_map = False

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def native_context_window(self) -> int | None:
        """Return the first credible context limit reported by the loaded model."""
        config = getattr(self.model, "config", None)
        for source, names in (
            (config, ("max_position_embeddings", "max_sequence_length", "seq_length", "n_positions")),
            (getattr(config, "text_config", None), ("max_position_embeddings", "max_sequence_length", "seq_length", "n_positions")),
            (self.tokenizer, ("model_max_length",)),
            (getattr(self.processor, "tokenizer", None), ("model_max_length",)),
        ):
            for name in names:
                value = getattr(source, name, None)
                if isinstance(value, int) and 256 <= value <= 4_194_304:
                    return value
        return None

    @property
    def effective_context_window(self) -> int | None:
        requested = self.settings.context_window
        native = self.native_context_window
        if requested is None:
            return native
        return min(requested, native) if native is not None else requested

    def _load(self) -> None:
        if self.loaded:
            return
        if not self.settings.model_id:
            raise RuntimeError("Set LOCAL_MODEL_ID in .env before starting a conversation.")
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt into localmodel-env first.") from exc

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        kwargs: dict[str, Any] = {"trust_remote_code": self.settings.trust_remote_code}
        if self.settings.dtype in dtype_map:
            kwargs["dtype"] = dtype_map[self.settings.dtype]
        if self.settings.device == "auto":
            kwargs["device_map"] = "auto"
            self.uses_device_map = True
        if self.settings.cpu_memory_gb:
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {"cpu": f"{self.settings.cpu_memory_gb}GiB"}
            self.uses_device_map = True
            if self.settings.offload_dir:
                Path(self.settings.offload_dir).mkdir(parents=True, exist_ok=True)
                kwargs["offload_folder"] = self.settings.offload_dir
                kwargs["offload_state_dict"] = True

        try:
            config = AutoConfig.from_pretrained(self.settings.model_id, trust_remote_code=self.settings.trust_remote_code)
            self.is_multimodal = self.settings.model_kind == "multimodal" or (
                self.settings.model_kind == "auto" and config.model_type.startswith("gemma4")
            )
            if self.is_multimodal:
                self.processor = AutoProcessor.from_pretrained(self.settings.model_id, trust_remote_code=self.settings.trust_remote_code)
                self.model = AutoModelForMultimodalLM.from_pretrained(self.settings.model_id, **kwargs)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id, trust_remote_code=self.settings.trust_remote_code)
                self.model = AutoModelForCausalLM.from_pretrained(self.settings.model_id, **kwargs)
            if self.settings.device in {"cpu", "cuda"} and not self.uses_device_map:
                self.model.to(self.settings.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(f"Could not load '{self.settings.model_id}': {exc}") from exc

    def load(self) -> None:
        """Load the configured model once and retain it for future requests."""
        self._load()

    @staticmethod
    def _processor_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI messages to the structure used by HF chat templates."""
        converted: list[dict[str, Any]] = []
        for source in messages:
            message = copy.deepcopy(source)
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = [{"type": "text", "text": content}]
            elif content is None:
                message["content"] = []

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        function["arguments"] = json.loads(arguments)
                # Tool-call markup is represented by tool_calls, not message text.
                message["content"] = []
            converted.append(message)
        return converted

    def _template_reasoning_options(self, max_new_tokens: int) -> dict[str, Any]:
        """Supply the common thinking controls; templates that do not use them ignore them."""
        budget = self.settings.reasoning_budget
        if budget is None:
            return {}
        options: dict[str, Any] = {"enable_thinking": budget > 0}
        if budget > 0:
            effective_budget = min(budget, max_new_tokens)
            options.update({
                "thinking_budget": effective_budget,
                "reasoning_budget": effective_budget,
            })
        return options

    def _format_inputs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_new_tokens: int,
    ) -> Any:
        tokenizer = self.tokenizer
        template_options = {
            "tools": tools or None,
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            **self._template_reasoning_options(max_new_tokens),
        }
        if self.is_multimodal:
            assert self.processor is not None
            return self.processor.apply_chat_template(
                self._processor_messages(messages),
                **template_options,
            )
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                self._processor_messages(messages),
                **template_options,
            )
        assert tokenizer is not None
        text = "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages) + "\nASSISTANT:"
        return tokenizer(text, return_tensors="pt")

    @staticmethod
    def _without_oldest_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Drop one complete old turn while retaining leading system messages and the latest turn."""
        first_conversation_message = 0
        while (
            first_conversation_message < len(messages)
            and messages[first_conversation_message].get("role") == "system"
        ):
            first_conversation_message += 1
        next_user = next(
            (
                index
                for index in range(first_conversation_message + 1, len(messages))
                if messages[index].get("role") == "user"
            ),
            None,
        )
        if next_user is None:
            return None
        return messages[:first_conversation_message] + messages[next_user:]

    @staticmethod
    def _truncate_token_inputs(inputs: Any, max_input_tokens: int) -> Any:
        """Left-truncate sequence tensors when a single remaining turn is still too large."""
        original_length = int(inputs["input_ids"].shape[-1])
        if original_length <= max_input_tokens:
            return inputs
        for key in ("input_ids", "attention_mask", "token_type_ids", "position_ids", "cache_position"):
            value = inputs.get(key)
            if value is None or not hasattr(value, "shape") or not value.shape:
                continue
            if int(value.shape[-1]) != original_length:
                continue
            value = value[..., -max_input_tokens:]
            if key == "position_ids":
                value = value - value[..., :1]
            inputs[key] = value
        return inputs

    def _context_fitted_inputs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_new_tokens: int,
    ) -> Any:
        context_window = self.effective_context_window
        if context_window is None:
            return self._format_inputs(messages, tools, max_new_tokens)
        max_input_tokens = context_window - max_new_tokens
        if max_input_tokens < 1:
            raise RuntimeError(
                f"The response budget ({max_new_tokens}) must be smaller than the effective "
                f"context window ({context_window})."
            )

        retained = copy.deepcopy(messages)
        inputs = self._format_inputs(retained, tools, max_new_tokens)
        while int(inputs["input_ids"].shape[-1]) > max_input_tokens:
            shortened = self._without_oldest_turn(retained)
            if shortened is None:
                return self._truncate_token_inputs(inputs, max_input_tokens)
            retained = shortened
            inputs = self._format_inputs(retained, tools, max_new_tokens)
        return inputs

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AssistantReply:
        self._load()
        import torch

        assert self.model is not None
        tokenizer = self.tokenizer
        selected_max_new_tokens = max_new_tokens if max_new_tokens is not None else self.settings.max_new_tokens
        inputs = self._context_fitted_inputs(messages, tools, selected_max_new_tokens)
        device = torch.device("cuda:0") if torch.cuda.is_available() and self.settings.device != "cpu" else torch.device("cpu")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if self.is_multimodal:
            assert self.processor is not None
            pad_token_id = self.processor.tokenizer.eos_token_id
        else:
            assert tokenizer is not None
            pad_token_id = tokenizer.eos_token_id
        selected_temperature = self.settings.temperature if temperature is None else temperature
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": selected_max_new_tokens,
            "pad_token_id": pad_token_id,
        }
        if selected_temperature > 0:
            generate_kwargs.update({"do_sample": True, "temperature": selected_temperature, "top_p": self.settings.top_p})
        else:
            generate_kwargs["do_sample"] = False
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generate_kwargs)
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        decoder = self.processor if self.is_multimodal else tokenizer
        assert decoder is not None
        cleaned_content = decoder.decode(new_tokens, skip_special_tokens=True).strip()
        try:
            parsed = decoder.parse_response(new_tokens, prefix=inputs["input_ids"], tools=tools or None)
            if isinstance(parsed, list):
                parsed = parsed[0]
            content = str(parsed.get("content") or "")
            reasoning = parsed.get("reasoning") or parsed.get("reasoning_content")
            parsed_tool_calls = parsed.get("tool_calls") or []
        except (AttributeError, KeyError, TypeError, ValueError):
            try:
                from transformers.cli.serving.utils import parse_assistant_message

                content, reasoning, parsed_tool_calls = parse_assistant_message(
                    decoder,
                    self.model,
                    new_tokens,
                    inputs["input_ids"],
                    cleaned_content,
                )
            except (ImportError, AttributeError, KeyError, TypeError, ValueError):
                content, reasoning, parsed_tool_calls = cleaned_content, None, []

        tool_calls = []
        for call in parsed_tool_calls:
            if isinstance(call, dict):
                function = call.get("function", call)
                name = function.get("name")
                arguments = function.get("arguments", {})
                call_id = call.get("id") or f"call_{uuid.uuid4().hex}"
            else:
                name = call.name
                arguments = call.arguments
                call_id = f"call_{uuid.uuid4().hex}"
            if name:
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                })
        return AssistantReply(content=content.strip(), tool_calls=tool_calls, reasoning_content=reasoning)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        return self.chat(messages, max_new_tokens=max_new_tokens, temperature=temperature).content
