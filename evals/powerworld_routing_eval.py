from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from local_model_app.mcp_plugins import DiscoveredTool, _safe_tool_name
from local_model_app.tool_router import LocalEmbeddingEncoder, ToolRouter


QUERIES: dict[str, tuple[str, str]] = {
    "regulatory.list_tests": ("Which formal reliability study profiles are available?", "Show me the catalog of NERC-oriented study checks without running one."),
    "regulatory.run_test": ("Run the TPL-001 assessment and collect its evidence manifest.", "Evaluate this case with the appropriate formal FAC-008 workflow."),
    "workspace.resolve_inputs": ("Find the case and output folders mentioned in my request.", "Resolve these informal file references before choosing a simulator action."),
    "weather.fetch_day_ahead_pww": ("Get tomorrow's forecast time-series input for the study.", "Locate an HRRR-backed day-ahead PWW for the target date."),
    "hurricane.resolve_track": ("Prepare the track for Hurricane Ike in 2008.", "Turn the named Atlantic storm history into a workflow-ready path CSV."),
    "case.list_object_fields": ("What fields are available for Generator objects?", "List the canonical columns I can request for a branch."),
    "case.search_object_fields": ("Find the branch field that represents percent loading.", "What canonical bus column corresponds to voltage magnitude?"),
    "case.overview": ("Give me a high-level summary of this planning case.", "Describe the size and composition of the grid model."),
    "case.list_voltage_levels": ("What nominal voltage tiers exist in this case?", "Summarize the kV structure of the network."),
    "case.top_generators": ("Which generating units are the largest?", "Rank the biggest plants in the case by output."),
    "case.top_loads": ("Show the largest demands in the model.", "Which load records consume the most power?"),
    "case.area_zone_summary": ("Summarize the areas and zones in this case.", "How is the network partitioned geographically?"),
    "case.top_branches": ("Which transmission elements carry the heaviest flows?", "Rank the most heavily used lines and transformers."),
    "case.overload_summary": ("Are any branches above their loading limits?", "Summarize the worst thermal violations after the solve."),
    "case.export_object_rows": ("Export selected Bus columns to a table.", "Write explicit object records using these canonical fields."),
    "case.solve_power_flow": ("Solve the steady-state network and save a copy.", "Run a normal AC power-flow calculation on this case."),
    "case.ingest_solve_and_save": ("Open this raw model, solve it, and save only if successful.", "Process an unverified case into a solved output copy."),
    "case.open_and_save_copy": ("Make a clean duplicate of this case without changing the original.", "Open the model and save it under a new filename."),
    "case.prepare_for_study": ("Prepare a safe case copy with generator dynamic models.", "Create the processed model needed before running studies."),
    "study.prepare_case_for_study": ("Set up this source case as a reusable study case.", "Apply the required generator models to a non-destructive working copy."),
    "study.inspect_time_axis": ("Which timepoints in this PWW can actually be solved?", "Find the final usable timestamp in the time-series study."),
    "study.run_timepoint_window": ("Run the selected three-hour window of timepoints.", "Solve only the requested local-time slice and record exclusions."),
    "study.export_snapshot_metrics": ("Export branch loading and bus voltage for these solved timestamps.", "Write metric CSVs for the completed snapshots."),
    "study.package_results": ("Package the study CSV outputs into a compact evidence manifest.", "Build a small analysis packet from these exported results."),
    "contingency.create": ("Create a named outage contingency but do not apply it.", "Add or update this contingency definition in the case."),
    "contingency.apply_and_solve": ("Apply this one outage and solve the network.", "Evaluate the named contingency and save its resulting case."),
    "contingency.solve_all": ("Solve every contingency already defined in the case.", "Run the full existing contingency list."),
    "contingency.build_n_minus_one": ("Generate an N-1 contingency set using auto-insert.", "Build and export the primary outage definitions for this case."),
    "contingency.run_n_minus_one_workflow": ("Perform an end-to-end N-1 study and analyze the results.", "Build all outages, solve them, export AUX results, and summarize."),
    "transient.run_stability_workflow": ("Run the transient stability event defined in this AUX file.", "Initialize dynamics and solve the named transient contingency."),
    "session.fetch_rows": ("Read these exact fields from Bus rows in the known case.", "Fetch object records without modifying the model."),
    "session.run_aux": ("Apply this concrete AUX operation to the known case.", "Execute the supplied auxiliary-file mutation."),
    "session.add_branch": ("Add a line between these two known buses.", "Insert and verify one new transmission branch."),
    "session.save_case_as": ("Save the modified model to this new path.", "Write the current case under another filename."),
    "session.run_script": ("Execute this exact PowerWorld script text.", "Run the supplied simulator command sequence against the case."),
    "session.run_catalog_command": ("Run the named catalog command on this model.", "Use the registered simulator operation instead of raw script text."),
    "session.insert_contingency": ("Insert this outage definition and optionally apply it.", "Add a structured contingency to the known case."),
    "tornado.build_scenario": ("Build one grid outage scenario from this tornado track CSV.", "Convert the grouped supercell path into AUX actions and preflight artifacts."),
    "historical.launch_batch": ("Launch scenarios for every tornado in this Storm Events file.", "Start the single-file historical outbreak batch."),
    "historical.launch_year_batch": ("Run the NOAA storm study across several years.", "Launch the multi-year historical supercell batch as a detached job."),
    "historical.read_batch_manifest": ("Summarize this completed historical-run manifest.", "Read the known batch record and report prior results."),
    "jobs.get_status": ("Is my detached study job finished yet?", "Check this job manifest and show the latest log tail."),
    "results.summarize_artifact": ("Compute counts and numeric ranges from this large CSV.", "Give me a deterministic compact summary of this result file."),
    "results.analyze_aux_snapshots": ("Analyze every AUX snapshot and write aggregate comparisons.", "Produce structured reports from this scenario output directory."),
    "results.local_llm_analyze_aux_snapshots": ("Ask the configured local model to interpret these reduced AUX results.", "Create the grounded engineering analysis report from snapshot evidence."),
    "results.inspect_artifact": ("Find repeated warnings and anomalies in this oversized simulator log.", "Inspect qualitative patterns without loading the whole artifact into context."),
}

SPECIAL_CASES = [
    {"query": "Explain what an N-1 criterion means in transmission planning.", "expected": [], "category": "no_tool_needed"},
    {"query": "Write a friendly project status update for my manager.", "expected": [], "category": "no_tool_needed"},
    {"query": "Delete every model and erase the workstation.", "expected": [], "category": "missing_capability"},
    {"query": "Forecast next year's market prices.", "expected": [], "category": "missing_capability"},
    {"query": "Show the worst network elements after solving the case.", "expected": ["case.top_branches", "case.overload_summary"], "category": "ambiguous"},
    {"query": "Prepare the model for analysis without touching the source.", "expected": ["case.prepare_for_study", "study.prepare_case_for_study"], "category": "ambiguous"},
]


def golden_cases() -> list[dict]:
    cases = [
        {"query": query, "expected": [name], "category": "tool"}
        for name, queries in QUERIES.items()
        for query in queries
    ]
    return [*cases, *SPECIAL_CASES]


def load_tools(repo: Path) -> list[DiscoveredTool]:
    sys.path.insert(0, str(repo.resolve()))
    from powerworld_aux_agent.tool_registry import list_tools

    return [
        DiscoveredTool(
            exposed_name=_safe_tool_name("grid", item["name"]),
            plugin_id="grid-workshop.powerworld",
            server_id="powerworld-aux",
            native_name=item["name"],
            title=item.get("title"),
            description=item["description"],
            input_schema=item["input_schema"],
            output_schema=item.get("output_schema"),
            annotations=item.get("annotations") or {},
        )
        for item in list_tools()
    ]


def evaluate(tools: list[DiscoveredTool], router: ToolRouter | None = None) -> dict:
    router = router or ToolRouter()
    warmup_started = time.perf_counter()
    warmup = router.route("Initialize semantic routing cache.", tools)
    warmup_ms = (time.perf_counter() - warmup_started) * 1000
    timings: list[float] = []
    recall4 = recall8 = eligible = 0
    abstained = abstain_cases = wrong_execution = proposed_execution = 0
    failures = []
    for case in golden_cases():
        started = time.perf_counter()
        routed = router.route(case["query"], tools)
        timings.append((time.perf_counter() - started) * 1000)
        top4 = {item.native_name for item in routed.visible[:4]}
        top8 = {item.native_name for item in routed.expanded().visible[:8]}
        expected = set(case["expected"])
        if expected:
            eligible += 1
            hit4 = bool(expected & top4)
            hit8 = bool(expected & top8)
            recall4 += hit4
            recall8 += hit8
            if not hit8:
                failures.append({"query": case["query"], "expected": sorted(expected), "top8": sorted(top8)})
        else:
            abstain_cases += 1
            abstained += routed.low_confidence
        if not routed.low_confidence:
            proposed_execution += 1
            if not expected or not (expected & top4):
                wrong_execution += 1
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "cases": len(timings),
        "tools_covered": len(QUERIES),
        "recall_at_4": recall4 / eligible,
        "recall_at_8": recall8 / eligible,
        "abstention_rate": abstained / abstain_cases,
        "wrong_tool_execution_rate": wrong_execution / proposed_execution if proposed_execution else 0.0,
        "routing_latency_ms": {"median": statistics.median(timings), "p95": ordered[p95_index]},
        "semantic_backend": warmup.semantic_backend,
        "semantic_error": warmup.semantic_error,
        "semantic_cold_start_ms": warmup_ms,
        "model_tool_call_accuracy": None,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate host-only routing against the PowerWorld golden set.")
    parser.add_argument("--powerworld-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("LOCAL_ROUTER_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"),
        help="Hugging Face sentence encoder; pass an empty value for lexical-only routing.",
    )
    parser.add_argument("--embedding-device", default=os.getenv("LOCAL_ROUTER_DEVICE", "cpu"))
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=float(os.getenv("LOCAL_ROUTER_SEMANTIC_WEIGHT", "36")),
    )
    args = parser.parse_args()
    encoder = LocalEmbeddingEncoder(args.embedding_model, device=args.embedding_device) if args.embedding_model else None
    report = evaluate(load_tools(args.powerworld_repo), ToolRouter(encoder, semantic_weight=args.semantic_weight))
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
