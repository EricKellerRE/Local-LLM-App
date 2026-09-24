import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from local_model_app.model import TransformersModel


class FakeTokenizer:
    eos_token_id = 2

    def decode(self, tokens, skip_special_tokens=True):
        return "done"

    def parse_response(self, tokens, prefix=None, tools=None):
        return {"content": "done", "tool_calls": []}

    def encode(self, text, add_special_tokens=False):
        return [10] * len(text.split())


class FakeGenerationModel:
    config = SimpleNamespace(max_position_embeddings=4096)

    def generate(self, **inputs):
        prompt = inputs["input_ids"]
        suffix = torch.tensor([[10, 2]], dtype=prompt.dtype, device=prompt.device)
        return torch.cat([prompt, suffix], dim=-1)


class GenerationTelemetryTests(unittest.TestCase):
    def test_generation_records_counts_timing_class_and_stop_reason(self) -> None:
        settings = SimpleNamespace(
            model_id="fake/model",
            device="cpu",
            dtype="float32",
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            reasoning_budget=None,
            context_window=4096,
        )
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "generation.jsonl"
            model = TransformersModel(settings, telemetry_path=path)
            model.tokenizer = FakeTokenizer()
            model.model = FakeGenerationModel()
            model._context_fitted_inputs = lambda messages, tools, budget: {
                "input_ids": torch.tensor([[4, 5, 6]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

            reply = model.chat(
                [{"role": "user", "content": "hello"}],
                max_new_tokens=8,
                temperature=0.0,
                generation_class="post_tool_decision",
                telemetry_context={"task_id": "task-1"},
            )
            record = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(reply.generated_tokens, 2)
        self.assertEqual(reply.stop_reason, "eos")
        self.assertEqual(record["generation_class"], "post_tool_decision")
        self.assertEqual(record["task_id"], "task-1")
        self.assertEqual(record["prompt_tokens"], 3)
        self.assertEqual(record["generated_tokens"], 2)
        self.assertEqual(record["max_new_tokens"], 8)
        self.assertEqual(record["stop_reason"], "eos")
        self.assertGreaterEqual(record["elapsed_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
