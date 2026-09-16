import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from local_model_app.app_settings import AppSettingsStore


class AppSettingsStoreTests(unittest.TestCase):
    def test_defaults_use_application_directories(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            store = AppSettingsStore(root)
            paths = store.paths()
            self.assertEqual(paths.data_directory, (root / "data").resolve())
            self.assertEqual(paths.models_directory, (root / "models").resolve())
            self.assertEqual(store.public_settings()["max_new_tokens"], 8192)
            self.assertEqual(store.public_settings()["max_tool_calls_per_step"], 256)

    def test_update_persists_model_and_storage_settings(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            (root / ".env").write_text("UNRELATED=preserved\nLOCAL_MODEL_DEVICE=cpu\n", encoding="utf-8")
            store = AppSettingsStore(root)
            saved = store.update({
                "data_directory": str(root / "chosen-data"),
                "models_directory": str(root / "chosen-models"),
                "model_id": "owner/model",
                "device": "cuda",
                "context_window": 32768,
                "max_new_tokens": 900,
                "reasoning_budget": 400,
                "temperature": 0.2,
                "top_p": 0.8,
                "trust_remote_code": True,
            })

            self.assertEqual(saved["model_id"], "owner/model")
            self.assertEqual(saved["device"], "cuda")
            self.assertEqual(saved["context_window"], 32768)
            self.assertEqual(saved["reasoning_budget"], 400)
            self.assertTrue(saved["trust_remote_code"])
            self.assertTrue((root / "chosen-data").is_dir())
            self.assertTrue((root / "chosen-models").is_dir())
            self.assertIn("UNRELATED=preserved", (root / ".env").read_text(encoding="utf-8"))
            payload = json.loads((root / "config" / "app-settings.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(payload["models_directory"]), (root / "chosen-models").resolve())

    def test_blank_context_and_reasoning_budgets_use_model_defaults(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "LOCAL_MODEL_CONTEXT_WINDOW=\nLOCAL_MODEL_REASONING_BUDGET=\n",
                encoding="utf-8",
            )

            settings = AppSettingsStore(root).public_settings()

            self.assertIsNone(settings["context_window"])
            self.assertIsNone(settings["reasoning_budget"])


if __name__ == "__main__":
    unittest.main()
