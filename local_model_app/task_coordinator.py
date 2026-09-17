from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Protocol

from local_model_app.task_models import (
    CompletionAudit,
    CriterionAudit,
    ProposedWorkItem,
    TaskDefinition,
    WorkItemOutcome,
)


AsyncGenerator = Callable[[list[dict[str, Any]]], Awaitable[str]]
CapabilityCatalog = Callable[[list[str]], Awaitable[list[dict[str, Any]]]]


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model did not return a JSON object.")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("The model response must be a JSON object.")
    return value


class TaskCoordinatorProtocol(Protocol):
    async def propose(self, request: str) -> TaskDefinition: ...

    async def plan_task(self, task: dict[str, Any]) -> list[ProposedWorkItem]: ...

    async def audit_completion(
        self,
        task: dict[str, Any],
        work_items: list[dict[str, Any]],
    ) -> CompletionAudit: ...


class WorkItemExecutorProtocol(Protocol):

    async def execute_work_item(
        self,
        task: dict[str, Any],
        item: dict[str, Any],
        completed_items: list[dict[str, Any]],
    ) -> WorkItemOutcome: ...

class ModelTaskCoordinator:
    """Uses the local model for planning while leaving lifecycle control to the host."""

    def __init__(
        self,
        generate: AsyncGenerator,
        capability_catalog: CapabilityCatalog | None = None,
    ) -> None:
        self.generate = generate
        self.capability_catalog = capability_catalog

    @staticmethod
    def _research_work_items() -> list[ProposedWorkItem]:
        discovery_items = [
            ProposedWorkItem(
                key="source-discovery",
                kind="research_discovery",
                title="Build and expand the source graph",
                instructions=(
                    "Collect the configured seed-source count, read those sources, extract their references, resolve "
                    "novel relevant references, and repeat breadth-first until the configured depth-pass limit, the "
                    "source cap, or a pass with no novel relevant references. Persist the source ledger and graph."
                ),
                completion_check=(
                    "The configured seed count was read and every citation-expansion pass has persisted counts, "
                    "exact URLs, parent links, and an objective stop reason."
                ),
                priority=20,
            ),
            ProposedWorkItem(
                key="source-notes",
                kind="research_notes",
                title="Extract reusable evidence from the source corpus",
                instructions=(
                    "Read every fetched source once in hardware-bounded batches. Preserve source-specific findings, "
                    "mechanisms, methods, study systems, evidential strength, limitations, disagreements, and exact URLs "
                    "as reusable notes for all report sections."
                ),
                completion_check="Every readable source has persisted, source-specific evidence notes with its exact URL.",
                priority=15,
                depends_on=["source-discovery"],
            ),
        ]
        evidence_dependency = ["source-notes"]
        section_items = [
            ProposedWorkItem(
                key="section-executive",
                kind="section",
                title="Executive summary and scope",
                instructions=(
                    "Write the opening section of the report. State the central conclusions, define the scope and "
                    "memory categories covered, explain the evidence-selection method, and preview the cross-scale "
                    "account. Use connected prose and cite exact source URLs beside material claims. Write 600-900 words."
                ),
                completion_check="At least 600 substantive words with conclusions, scope, method, and source URLs.",
                priority=8,
                depends_on=evidence_dependency,
            ),
            ProposedWorkItem(
                key="section-molecular-cellular",
                kind="section",
                title="Molecular and cellular mechanisms",
                instructions=(
                    "Write a detailed report section tracing induction, signaling, transcription, translation, "
                    "synaptic tagging and capture, structural plasticity, and candidate persistence mechanisms. "
                    "Distinguish necessity, sufficiency, correlation, timescale, species, and memory system. Explain "
                    "important disagreements rather than listing molecules. Cite exact source URLs. Write 1,200-1,800 words."
                ),
                completion_check="At least 1,200 substantive words with mechanistic explanation and exact source URLs.",
                priority=7,
                depends_on=evidence_dependency,
            ),
            ProposedWorkItem(
                key="section-circuit-systems",
                kind="section",
                title="Circuit and systems mechanisms",
                instructions=(
                    "Write a detailed report section on engram allocation and reactivation, hippocampal and cortical "
                    "interactions, replay, sleep and oscillations, systems consolidation theories, remote memory, and "
                    "precision or generalization. Connect circuit findings to molecular mechanisms and distinguish "
                    "causal from observational evidence. Cite exact source URLs. Write 1,000-1,600 words."
                ),
                completion_check="At least 1,000 substantive words covering circuit and systems mechanisms with URLs.",
                priority=7,
                depends_on=evidence_dependency,
            ),
            ProposedWorkItem(
                key="section-memory-phases",
                kind="section",
                title="Formation consolidation reconsolidation and maintenance",
                instructions=(
                    "Write a detailed chronological section that distinguishes acquisition, early and late "
                    "consolidation, systems consolidation, retrieval, destabilization and reconsolidation, persistence, "
                    "maintenance, updating, extinction, and forgetting. State boundary conditions and timescales and "
                    "explain where terminology or evidence remains contested. Cite exact URLs. Write 1,000-1,600 words."
                ),
                completion_check="At least 1,000 substantive words clearly distinguishing every requested phase with URLs.",
                priority=7,
                depends_on=evidence_dependency,
            ),
            ProposedWorkItem(
                key="section-methods",
                kind="section",
                title="Techniques and strength of evidence",
                instructions=(
                    "Write a comparative section covering behavioral paradigms, lesions and pharmacology, "
                    "electrophysiology, imaging, optogenetics and chemogenetics, activity tagging, molecular profiling, "
                    "and human translation. Explain what each technique can establish, its resolution, confounds, and "
                    "generalization limits. Include a compact comparison table where useful and exact URLs. Write 900-1,400 words."
                ),
                completion_check="At least 900 substantive words comparing methods, causal reach, limitations, and URLs.",
                priority=6,
                depends_on=evidence_dependency,
            ),
            ProposedWorkItem(
                key="section-disagreements-gaps",
                kind="section",
                title="Disagreements limitations and open research gaps",
                instructions=(
                    "Write a critical section on competing theories, replication and translation problems, measurement "
                    "limits, unresolved cross-scale links, and high-value open questions. Identify what evidence would "
                    "discriminate between alternatives. Separate sourced findings from synthesis and cite exact URLs. "
                    "Write 900-1,400 words."
                ),
                completion_check="At least 900 substantive words naming concrete disagreements, limitations, gaps, and URLs.",
                priority=6,
                depends_on=evidence_dependency,
            ),
            ProposedWorkItem(
                key="section-conclusions",
                kind="section",
                title="Integrated conclusions and research agenda",
                instructions=(
                    "Write the concluding section. Integrate molecular, cellular, circuit, and systems explanations; "
                    "state the strongest supported conclusions and their limits; identify useful experimental designs "
                    "for closing the major gaps; and avoid merely repeating the executive summary. Cite exact URLs for "
                    "claims that depend on particular studies. Write 700-1,100 words."
                ),
                completion_check="At least 700 substantive words with integrated conclusions, limitations, next steps, and URLs.",
                priority=5,
                depends_on=evidence_dependency,
            ),
        ]
        return [
            *discovery_items,
            *section_items,
            ProposedWorkItem(
                key="synthesis",
                kind="synthesis",
                title="Assemble and export the complete research document",
                instructions=(
                    "Assemble every completed report section without compressing it, preserve exact inline source "
                    "URLs, append a deduplicated source URL index, and export a readable Word document."
                ),
                completion_check="A traceable report of at least 6,000 words exists as a downloadable DOCX artifact.",
                depends_on=[item.key for item in section_items if item.key],
            ),
        ]

    async def propose(self, request: str) -> TaskDefinition:
        prompt = (
            "Convert the user's request into a durable long-running task definition. "
            "Do not claim that work has begun. Infer useful defaults, but preserve the user's intent. "
            "Return only JSON with: title, goal, success_criteria (non-empty string array), "
            "deliverables (string array), constraints (string array), execution, and metadata. "
            "execution must contain deadline (ISO-8601 or null), max_steps_per_episode (1-100), "
            "maximum_attempts (positive integer or null), retry_initial_seconds, retry_maximum_seconds, "
            "and resume_after_restart. Use null attempt/deadline values when the user requests indefinite work."
        )
        response = await self.generate(
            [{"role": "system", "content": prompt}, {"role": "user", "content": request}]
        )
        return TaskDefinition.model_validate(_parse_json_object(response))

    async def plan_task(self, task: dict[str, Any]) -> list[ProposedWorkItem]:
        definition = task["definition"]
        metadata = definition.get("metadata") or {}
        if metadata.get("mode") == "durable_tools":
            plugin_ids = [str(value) for value in metadata.get("plugin_ids") or []]
            if plugin_ids == ["local.web-research"]:
                return self._research_work_items()
            capabilities = (
                await self.capability_catalog(plugin_ids)
                if self.capability_catalog is not None and plugin_ids
                else []
            )
            prompt = (
                "Plan a durable, resumable tool-driven task. Return only JSON with a work_items array. "
                "Create 3-12 bounded evidence-producing items of kind 'tool', each covering a distinct research "
                "question, engineering stage, metric, scenario, or validation concern. Use dependencies where a "
                "later operation requires a case/session/artifact handle from earlier work. Then create exactly one "
                "kind 'synthesis' item that depends on every evidence item and produces the complete user-facing "
                "deliverable. For iterative optimization, include baseline measurement, candidate changes, repeated "
                "metric evaluation, regression/safety checks, and final comparison. For research, include diverse "
                "search angles, primary-source reading, disagreement/gap analysis, and citation capture. Do not put "
                "all work into one item. Each item has key, kind, title, instructions, completion_check, priority, "
                "and depends_on. Keys must start with a lowercase letter and contain only letters, digits, dot, "
                "underscore, or hyphen."
            )
            response = await self.generate([
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({
                    "definition": definition,
                    "available_capabilities": capabilities,
                }, ensure_ascii=False)},
            ])
            data = _parse_json_object(response)
            return [ProposedWorkItem.model_validate(item) for item in data.get("work_items", [])]
        prompt = (
            "Create the first bounded work items for this durable task. You are planning, not claiming execution. "
            "The host currently supports model reasoning and drafting but has no task-specific external tools. "
            "Create model work only when it can be honestly completed from supplied information. "
            "If external capabilities are necessary, create one item describing exactly what capability is required. "
            "Return only JSON: {\"work_items\": [{\"key\": str, \"kind\": \"model\", \"title\": str, "
            "\"instructions\": str, \"completion_check\": str, \"priority\": int, \"depends_on\": [key]}]}. "
            "Use stable unique keys and dependencies when ordering matters. "
            "Use 1-6 small items."
        )
        response = await self.generate(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(definition, ensure_ascii=False)},
            ]
        )
        data = _parse_json_object(response)
        return [ProposedWorkItem.model_validate(item) for item in data.get("work_items", [])]

    async def execute_work_item(
        self,
        task: dict[str, Any],
        item: dict[str, Any],
        completed_items: list[dict[str, Any]],
    ) -> WorkItemOutcome:
        history = [
            {
                "title": completed["title"],
                "result": completed.get("result"),
            }
            for completed in completed_items[-12:]
        ]
        prompt = (
            "Perform one bounded work item for a durable task. Never claim to search, read files, contact services, "
            "or perform other external actions: no task-specific tools are connected yet. You may reason, outline, "
            "draft, classify supplied material, or transform supplied text. If a required capability is unavailable, "
            "return waiting_for_tools and name it. If the user must decide something, return waiting_for_input. "
            "Return only JSON with: outcome (completed, retry, waiting_for_input, waiting_for_tools, or failed), "
            "summary, result, follow_up_items, completion_evidence, and wait_seconds. Follow-up items use the same "
            "shape as the current item and must be small and necessary. They may depend on existing item keys."
        )
        payload = {
            "task": task["definition"],
            "current_work_item": {
                "title": item["title"],
                "instructions": item["instructions"],
                "completion_check": item["completion_check"],
            },
            "completed_work": history,
        }
        response = await self.generate(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
        return WorkItemOutcome.model_validate(_parse_json_object(response))

    async def audit_completion(
        self,
        task: dict[str, Any],
        work_items: list[dict[str, Any]],
    ) -> CompletionAudit:
        metadata = task["definition"].get("metadata") or {}
        if metadata.get("plugin_ids") == ["local.web-research"]:
            by_kind: dict[str, list[dict[str, Any]]] = {}
            for work_item in work_items:
                by_kind.setdefault(str(work_item.get("kind")), []).append(work_item)
            discovery = by_kind.get("research_discovery", [])
            notes = by_kind.get("research_notes", [])
            sections = by_kind.get("section", [])
            synthesis = by_kind.get("synthesis", [])
            synthesis_outcome = synthesis[-1].get("result") or {} if synthesis else {}
            evidence = synthesis_outcome.get("completion_evidence", []) if isinstance(synthesis_outcome, dict) else []

            def metric(name: str) -> int:
                prefix = f"{name}="
                for value in evidence:
                    if str(value).startswith(prefix):
                        try:
                            return int(str(value)[len(prefix):])
                        except ValueError:
                            return 0
                return 0

            artifacts = synthesis_outcome.get("artifacts", []) if isinstance(synthesis_outcome, dict) else []
            required_words = int(metadata.get("report_min_words") or 6000)
            required_sources = int(metadata.get("report_min_sources") or 12)
            checks = {
                "citation graph": bool(discovery and discovery[-1].get("status") == "completed"),
                "source notes": bool(notes and notes[-1].get("status") == "completed"),
                "report sections": len(sections) == 7 and all(item.get("status") == "completed" for item in sections),
                "word count": metric("word_count") >= required_words,
                "source count": metric("source_url_count") >= required_sources,
                "Word document": any(
                    str(artifact.get("media_type")) ==
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    for artifact in artifacts
                ),
            }
            passed = all(checks.values())
            detail = ", ".join(f"{name}={'passed' if value else 'failed'}" for name, value in checks.items())
            return CompletionAudit(
                passed=passed,
                summary=("Research task passed host-enforced completion checks. " if passed else
                         "Research task did not pass host-enforced completion checks. ") + detail,
                criteria=[
                    CriterionAudit(
                        criterion=criterion,
                        satisfied=passed,
                        evidence=(
                            f"Host-enforced research checks: {detail}; words={metric('word_count')}; "
                            f"sources={metric('source_url_count')}."
                        ),
                    )
                    for criterion in task["definition"]["success_criteria"]
                ],
            )

        evidence = []
        for item in work_items[-40:]:
            outcome = item.get("result") or {}
            result = outcome.get("result") if isinstance(outcome, dict) else outcome
            result_text = str(result or "")
            evidence.append({
                "key": item.get("key"),
                "kind": item.get("kind"),
                "title": item["title"],
                "status": item["status"],
                "summary": outcome.get("summary") if isinstance(outcome, dict) else "",
                "word_count": len(re.findall(r"\b[\w'-]+\b", result_text)),
                "source_url_count": len(set(re.findall(r"https?://[^\s<>)\]}]+", result_text))),
                "completion_evidence": outcome.get("completion_evidence", []) if isinstance(outcome, dict) else [],
                "artifacts": outcome.get("artifacts", []) if isinstance(outcome, dict) else [],
                "result_preview": result_text[:1200],
            })
        durable_tools = metadata.get("mode") == "durable_tools"
        prompt = (
            "Audit a durable task against every success criterion using only recorded evidence. Be conservative. "
            "A plan or assertion is not evidence that external work occurred. Return only JSON with: passed, summary, "
            "criteria (each containing criterion, satisfied, evidence), and follow_up_items. passed may be true only "
            "when every criterion is satisfied. Create bounded follow-up items for remediable gaps."
            + (
                " For this tool-driven task, follow-up evidence or optimization items must use kind 'tool'. If any "
                "follow-up is created, also create a new kind 'synthesis' item depending on every new follow-up key. "
                "Do not pass an optimization without baseline/final metric evidence, and do not pass research without "
                "source URLs, conclusions, techniques, and explicit gaps or limitations."
                " For document-backed research, do not pass unless the recorded synthesis has a DOCX artifact and "
                "its completion evidence meets the configured minimum word count."
                if durable_tools else ""
            )
        )
        response = await self.generate(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"definition": task["definition"], "recorded_work": evidence}, ensure_ascii=False
                    ),
                },
            ]
        )
        return CompletionAudit.model_validate(_parse_json_object(response))
