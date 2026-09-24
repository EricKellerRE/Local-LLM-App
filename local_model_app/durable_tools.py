from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from local_model_app.mcp_manager import McpPluginManager
from local_model_app.model import TransformersModel
from local_model_app.report_documents import (
    assemble_research_report,
    build_report_docx,
    report_word_count,
    unique_urls,
)
from local_model_app.scratchpad import Scratchpad
from local_model_app.task_models import TaskArtifact, WorkItemOutcome
from local_model_app.task_store import TaskStore
from local_model_app.tool_coordinator import ToolCoordinator
from local_model_app.tool_router import ToolRouter


AsyncGenerator = Callable[[list[dict[str, Any]]], Awaitable[str]]
MINIMUM_WORDS_PATTERN = re.compile(r"at least\s+([\d,]+)\s+(?:substantive\s+)?words", re.IGNORECASE)


class DurableToolExecutor:
    """Execute one checkpointed work item with the task's selected MCP capabilities."""

    def __init__(
        self,
        model: TransformersModel,
        manager: McpPluginManager,
        router: ToolRouter,
        data_directory: Path,
        store: TaskStore,
        inference_lock: asyncio.Lock,
    ) -> None:
        self.model = model
        self.manager = manager
        self.router = router
        self.data_directory = data_directory
        self.store = store
        self.inference_lock = inference_lock

    @staticmethod
    def _completed_context(items: list[dict[str, Any]]) -> str:
        records = []
        for item in items[-20:]:
            outcome = item.get("result") or {}
            records.append({
                "key": item.get("key"),
                "title": item.get("title"),
                "summary": outcome.get("summary") if isinstance(outcome, dict) else "",
                "result": outcome.get("result") if isinstance(outcome, dict) else outcome,
            })
        return json.dumps(records, ensure_ascii=False)

    @staticmethod
    def _attach_task_to_approval(path: Path, task_id: str, work_item_id: str) -> None:
        state_path = path.with_suffix(".state.json")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        state.update({"durable_task_id": task_id, "durable_work_item_id": work_item_id})
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(state_path)

    async def execute_work_item(
        self,
        task: dict[str, Any],
        item: dict[str, Any],
        completed_items: list[dict[str, Any]],
    ) -> WorkItemOutcome:
        metadata = task["definition"].get("metadata") or {}
        plugin_ids = [str(value) for value in metadata.get("plugin_ids") or []]
        if not plugin_ids:
            return WorkItemOutcome(
                outcome="waiting_for_tools",
                summary="This durable work item requires a selected MCP capability.",
            )
        chat_id = str(metadata.get("chat_id") or task["id"])
        activity_path = self.data_directory / "tool_activity" / f"{chat_id}.jsonl"
        checkpoint = item.get("result") or {}
        pending_task = checkpoint.get("pending_mcp_task") if isinstance(checkpoint, dict) else None
        if isinstance(pending_task, dict):
            try:
                result = await self.manager.resume_tool_task(pending_task)
            except Exception:
                self.store.checkpoint_work_item(item["id"], None)
                raise
            serialized = json.dumps(result, ensure_ascii=False)
            return WorkItemOutcome(
                outcome="completed",
                summary=f"Completed durable MCP task for checkpoint: {item['title']}",
                result=serialized,
                completion_evidence=[serialized[:2000]],
            )

        async def save_task_handle(handle: dict[str, Any]) -> None:
            self.store.checkpoint_work_item(item["id"], {"pending_mcp_task": handle})

        coordinator = ToolCoordinator(
            self.model,
            self.manager,
            Scratchpad(self.data_directory / "scratchpads" / f"{chat_id}.jsonl"),
            activity_path,
            router=self.router,
            max_calls=int(getattr(self.model.settings, "max_tool_calls_per_step", 256)),
            task_checkpoint=save_task_handle,
            telemetry_context={
                "task_id": str(task["id"]),
                "work_item_id": str(item["id"]),
                "chat_id": chat_id,
            },
        )
        prompt = (
            f"Durable task goal:\n{task['definition']['goal']}\n\n"
            f"Current checkpointed work item:\n{item['title']}\n{item['instructions']}\n\n"
            f"Completion check:\n{item['completion_check']}\n\n"
            "Use the available tools rather than merely describing what could be done. Work iteratively, inspect "
            "results, and preserve exact source URLs, artifact handles, case/session identifiers, metric values, "
            "assumptions, and errors in the answer. Do not claim the completion check passed without evidence.\n\n"
            f"Previously completed checkpoints:\n{self._completed_context(completed_items)}"
        )
        # Transformers generation is not safe to run concurrently. Durable work uses the
        # same lock as ordinary chat, planning, and synthesis so foreground requests cannot
        # race a background tool turn and shutdown can accurately report in-flight work.
        async with self.inference_lock:
            answer = await coordinator.respond(prompt, [], plugin_ids)
        lowered = answer.lower()
        if "approval is required" in lowered:
            self._attach_task_to_approval(activity_path, task["id"], item["id"])
            return WorkItemOutcome(outcome="waiting_for_input", summary=answer)
        if "cannot execute" in lowered or "please provide" in lowered:
            return WorkItemOutcome(outcome="waiting_for_input", summary=answer)
        urls = unique_urls(answer)
        return WorkItemOutcome(
            outcome="completed",
            summary=f"Completed checkpoint: {item['title']}",
            result=answer,
            completion_evidence=urls or [answer[:500]],
        )


class DurableSectionExecutor:
    """Draft one substantial report section from the relevant persisted evidence."""

    def __init__(self, generate: AsyncGenerator, data_directory: Path | None = None) -> None:
        self.generate = generate
        self.data_directory = data_directory

    def _checkpoint_path(self, task: dict[str, Any], item: dict[str, Any]) -> Path | None:
        if self.data_directory is None:
            return None
        safe_key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(item.get("key") or item.get("id") or "section"))
        return self.data_directory / "research" / str(task["id"]) / "sections" / f"{safe_key}.json"

    @staticmethod
    def _read_segments(path: Path | None) -> list[str]:
        if path is None:
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return []
        return [str(value) for value in payload.get("segments", []) if str(value).strip()]

    @staticmethod
    def _save_segments(path: Path | None, segments: list[str]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    async def execute_work_item(
        self,
        task: dict[str, Any],
        item: dict[str, Any],
        completed_items: list[dict[str, Any]],
    ) -> WorkItemOutcome:
        dependencies = set(item.get("depends_on") or [])
        checkpoint_path = self._checkpoint_path(task, item)
        segments = self._read_segments(checkpoint_path)
        evidence = []
        for completed in completed_items:
            if dependencies and completed.get("key") not in dependencies:
                continue
            outcome = completed.get("result") or {}
            evidence.append({
                "key": completed.get("key"),
                "title": completed.get("title"),
                "result": outcome.get("result") if isinstance(outcome, dict) else outcome,
                "completion_evidence": outcome.get("completion_evidence", []) if isinstance(outcome, dict) else [],
            })
            if completed.get("kind") == "research_discovery" and self.data_directory is not None:
                analysis_dir = self.data_directory / "research" / str(task["id"]) / "source-analysis"
                dossiers: list[dict[str, str]] = []
                for path in sorted(analysis_dir.glob("*.json")):
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError, TypeError):
                        continue
                    dossier = str(payload.get("dossier") or "").strip()
                    if payload.get("status") == "completed" and dossier:
                        dossiers.append({
                            "source_id": str(payload.get("source_id") or path.stem),
                            "title": str(payload.get("title") or "Source"),
                            "url": str(payload.get("url") or ""),
                            "dossier": dossier,
                        })
                if dossiers:
                    per_segment = 6
                    section_key = str(item.get("key") or item.get("title") or "section")
                    section_offset = sum(ord(character) for character in section_key)
                    offset = (section_offset + len(segments) * per_segment) % len(dossiers)
                    selected = (dossiers + dossiers)[offset:offset + min(per_segment, len(dossiers))]
                    evidence.append({
                        "key": "source-dossiers",
                        "title": "Checkpointed source-specific evidence dossiers",
                        "result": selected,
                        "completion_evidence": [row["url"] for row in selected if row["url"]],
                    })
        minimum_match = MINIMUM_WORDS_PATTERN.search(item.get("completion_check") or "")
        minimum_words = int(minimum_match.group(1).replace(",", "")) if minimum_match else 0
        existing = "\n\n".join(segments)
        remaining_words = max(0, minimum_words - report_word_count(existing))
        if segments:
            system = (
                "Continue one existing research-report section using only the recorded evidence. Write a new, "
                "non-redundant continuation only: do not repeat the title, opening, prior paragraphs, or source list. "
                "Develop underexplained mechanisms, boundary conditions, causal strength, disagreements, and "
                "limitations. Preserve exact source URLs beside new claims and do not invent evidence."
            )
        else:
            system = (
                "Write exactly one substantive section of a longer research report from the recorded evidence. Use "
                "connected explanatory prose, not a compressed checklist. Preserve mechanisms, conditions, timescales, "
                "study systems, disagreements, and evidential limitations. Put exact source URLs beside the claims they "
                "support. Do not invent evidence, do not write the entire report, and do not add a top-level document title."
            )
        segment = await self.generate([
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({
                "task_goal": task["definition"]["goal"],
                "section_title": item["title"],
                "section_instructions": item["instructions"],
                "completion_check": item["completion_check"],
                "minimum_additional_words": min(max(remaining_words, 300), 700),
                "existing_draft": existing,
                "recorded_evidence": evidence,
            }, ensure_ascii=False)},
        ])
        if segment.strip():
            segments.append(segment.strip())
            self._save_segments(checkpoint_path, segments)
        section = "\n\n".join(segments)
        words = report_word_count(section)
        urls = unique_urls(section)
        if words < minimum_words or not urls:
            reason = (
                f"Section checkpoint did not meet its acceptance check yet: {words} words across "
                f"{len(segments)} bounded segment(s) and {len(urls)} "
                f"source URLs; continuing toward at least {minimum_words} words with traceable URLs."
            )
            max_segments = int((task["definition"].get("metadata") or {}).get("section_max_segments") or 6)
            if len(segments) < max_segments:
                return WorkItemOutcome(outcome="retry", summary=reason, wait_seconds=1)
            return WorkItemOutcome(
                outcome="failed",
                summary=f"{reason} The configured {max_segments}-segment limit was reached.",
                result=section,
                completion_evidence=urls,
            )
        return WorkItemOutcome(
            outcome="completed",
            summary=f"Drafted {item['title']} ({words:,} words; {len(urls)} source URLs).",
            result=section,
            completion_evidence=urls,
        )


class DurableSynthesisExecutor:
    """Compose the user-facing report from durable, persisted checkpoint evidence."""

    def __init__(self, generate: AsyncGenerator, data_directory: Path) -> None:
        self.generate = generate
        self.data_directory = data_directory

    async def execute_work_item(
        self,
        task: dict[str, Any],
        item: dict[str, Any],
        completed_items: list[dict[str, Any]],
    ) -> WorkItemOutcome:
        section_items = [completed for completed in completed_items if completed.get("kind") == "section"]
        metadata = task["definition"].get("metadata") or {}
        if metadata.get("mode") == "durable_tools" and section_items:
            sections: list[tuple[str, str]] = []
            for completed in sorted(section_items, key=lambda value: int(value.get("sequence") or 0)):
                outcome = completed.get("result") or {}
                content = outcome.get("result") if isinstance(outcome, dict) else outcome
                if str(content or "").strip():
                    sections.append((str(completed.get("title") or "Report section"), str(content)))
            ledger_path = self.data_directory / "research" / str(task["id"]) / "source-ledger.json"
            try:
                ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
                corpus_sources = [
                    source for source in ledger.get("sources", [])
                    if source.get("status") in {"fetched", "analyzed"}
                ]
            except (OSError, json.JSONDecodeError, TypeError):
                corpus_sources = []
            if corpus_sources:
                sections.append((
                    "Research corpus",
                    "\n".join(
                        f"- {source.get('title') or source['url']} — {source['url']} "
                        f"(citation depth {source.get('depth', 0)})"
                        for source in corpus_sources
                    ),
                ))
            title = str(task["definition"].get("title") or "Research Report")
            report = assemble_research_report(title, sections)
            words = report_word_count(report)
            urls = unique_urls(report)
            minimum_words = int(metadata.get("report_min_words") or 6000)
            minimum_sources = int(metadata.get("report_min_sources") or 12)
            if words < minimum_words or len(urls) < minimum_sources:
                return WorkItemOutcome(
                    outcome="failed",
                    summary=(
                        f"The assembled report did not meet document acceptance criteria: {words:,} words and "
                        f"{len(urls)} unique source URLs; required at least {minimum_words:,} words and "
                        f"{minimum_sources} source URLs."
                    ),
                    result=report,
                    completion_evidence=[f"word_count={words}", f"source_url_count={len(urls)}"],
                )
            artifact_directory = self.data_directory / "artifacts" / str(task["id"])
            markdown_path = artifact_directory / "report.md"
            document_path = artifact_directory / "report.docx"
            artifact_directory.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(report, encoding="utf-8")
            await asyncio.to_thread(build_report_docx, report, document_path, title=title)
            relative_path = f"{task['id']}/report.docx"
            return WorkItemOutcome(
                outcome="completed",
                summary=(
                    f"Assembled a {words:,}-word research document from {len(sections)} independently drafted "
                    f"sections with {len(urls)} unique source URLs."
                ),
                result=(
                    f"The complete report is available as a Word document ({words:,} words, "
                    f"{len(urls)} source URLs)."
                ),
                completion_evidence=[relative_path, f"word_count={words}", f"source_url_count={len(urls)}"],
                artifacts=[TaskArtifact(
                    name="Research report",
                    relative_path=relative_path,
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    size_bytes=document_path.stat().st_size,
                )],
            )

        evidence = []
        for completed in completed_items:
            outcome = completed.get("result") or {}
            evidence.append({
                "key": completed.get("key"),
                "title": completed.get("title"),
                "summary": outcome.get("summary") if isinstance(outcome, dict) else "",
                "result": outcome.get("result") if isinstance(outcome, dict) else outcome,
                "completion_evidence": outcome.get("completion_evidence", []) if isinstance(outcome, dict) else [],
            })
        system = (
            "Write the final deliverable for a durable task from the recorded checkpoint evidence. Produce a "
            "complete standalone report, not a progress summary. Include an executive summary, scope and method, "
            "findings, quantitative results where available, techniques or interventions, conclusions, unresolved "
            "gaps, limitations, and recommended next steps. For research, cite exact source URLs beside supported "
            "claims and distinguish source evidence from inference. For engineering optimization, compare baseline "
            "and final metrics, enumerate changes, preserve artifact/case handles, and state whether each success "
            "criterion passed. Never invent evidence."
        )
        report = await self.generate([
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({
                "task": task["definition"],
                "synthesis_instructions": item["instructions"],
                "recorded_evidence": evidence,
            }, ensure_ascii=False)},
        ])
        return WorkItemOutcome(
            outcome="completed",
            summary="Final report synthesized from persisted checkpoint evidence.",
            result=report,
            completion_evidence=["Final report generated from completed work-item records."],
        )
