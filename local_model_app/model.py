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
        if self.is_multimodal:
            assert self.processor is not None
            multimodal_messages = self._processor_messages(messages)
            inputs = self.processor.apply_chat_template(
                multimodal_messages,
                tools=tools or None,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        elif getattr(tokenizer, "chat_template", None):
            template_messages = self._processor_messages(messages)
            inputs = tokenizer.apply_chat_template(
                template_messages,
                tools=tools or None,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            text = "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages) + "\nASSISTANT:"
            inputs = tokenizer(text, return_tensors="pt")
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
            "max_new_tokens": max_new_tokens or self.settings.max_new_tokens,
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
