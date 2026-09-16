from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import HfApi, snapshot_download

from local_model_app.app_settings import AppSettingsStore


class HuggingFaceService:
    """Search and download Transformers chat models from the Hugging Face Hub."""

    def __init__(self, settings: AppSettingsStore, activity: Callable[[str], Any] | None = None) -> None:
        self.settings = settings
        self.activity = activity
        self.downloads: dict[str, dict[str, Any]] = {}

    async def search(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        def run() -> list[dict[str, Any]]:
            api = HfApi()
            rows = api.list_models(
                search=query or None,
                pipeline_tag="text-generation",
                filter="transformers",
                sort="downloads",
                limit=limit,
                full=True,
                fetch_config=True,
            )
            results: list[dict[str, Any]] = []
            for model in rows:
                safetensors = getattr(model, "safetensors", None)
                parameters = getattr(safetensors, "total", None) if safetensors else None
                results.append({
                    "id": model.id,
                    "downloads": model.downloads or 0,
                    "likes": model.likes or 0,
                    "gated": bool(model.gated),
                    "private": bool(model.private),
                    "pipeline_tag": model.pipeline_tag,
                    "library_name": model.library_name,
                    "parameters": parameters,
                    "url": f"https://huggingface.co/{model.id}",
                })
            return results

        return await asyncio.to_thread(run)

    def start_download(self, repo_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
            raise ValueError("Use a Hugging Face model id such as organization/model-name.")
        job_id = uuid.uuid4().hex
        destination = self.settings.paths().models_directory / repo_id.replace("/", "--")
        job = {
            "id": job_id,
            "repo_id": repo_id,
            "destination": str(destination),
            "status": "queued",
            "error": None,
        }
        self.downloads[job_id] = job
        asyncio.create_task(self._download(job, destination), name=f"download-model-{job_id}")
        return dict(job)

    async def _download(self, job: dict[str, Any], destination: Path) -> None:
        job["status"] = "downloading"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            async def run() -> None:
                await asyncio.to_thread(
                    snapshot_download,
                    repo_id=job["repo_id"],
                    local_dir=destination,
                )

            if self.activity:
                async with self.activity("a model download is in progress"):
                    await run()
            else:
                await run()
            job["status"] = "complete"
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)

    def status(self, job_id: str) -> dict[str, Any]:
        try:
            return dict(self.downloads[job_id])
        except KeyError as exc:
            raise KeyError(job_id) from exc
