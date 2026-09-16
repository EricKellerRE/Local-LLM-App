import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace

from local_model_app.app_settings import AppSettingsStore
from local_model_app.huggingface_service import HuggingFaceService


class HuggingFaceServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_is_limited_to_transformers_text_generation_models(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            service = HuggingFaceService(AppSettingsStore(Path(directory)))
            model = SimpleNamespace(
                id="owner/model", downloads=42, likes=3, gated=False, private=False,
                pipeline_tag="text-generation", library_name="transformers", safetensors=None,
            )
            with patch("local_model_app.huggingface_service.HfApi") as api_class:
                api_class.return_value.list_models.return_value = [model]
                results = await service.search("model")

            self.assertEqual(results[0]["id"], "owner/model")
            api_class.return_value.list_models.assert_called_once_with(
                search="model", pipeline_tag="text-generation", filter="transformers",
                sort="downloads", limit=20, full=True, fetch_config=True,
            )

    async def test_download_uses_selected_model_directory(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            store = AppSettingsStore(root)
            store.update({
                "data_directory": str(root / "data"),
                "models_directory": str(root / "model-library"),
            })

            def fake_download(*, repo_id, local_dir):
                self.assertEqual(repo_id, "owner/model")
                Path(local_dir).mkdir(parents=True, exist_ok=True)
                (Path(local_dir) / "config.json").write_text("{}", encoding="utf-8")

            service = HuggingFaceService(store)
            with patch("local_model_app.huggingface_service.snapshot_download", side_effect=fake_download):
                job = service.start_download("owner/model")
                for _ in range(50):
                    await asyncio.sleep(0.01)
                    status = service.status(job["id"])
                    if status["status"] not in {"queued", "downloading"}:
                        break

            self.assertEqual(status["status"], "complete")
            self.assertTrue(Path(status["destination"], "config.json").is_file())
            self.assertEqual(store.installed_models()[0]["name"], "owner/model")

    async def test_rejects_invalid_repository_ids(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            service = HuggingFaceService(AppSettingsStore(Path(directory)))
            with self.assertRaises(ValueError):
                service.start_download("not-a-qualified-id")


if __name__ == "__main__":
    unittest.main()
