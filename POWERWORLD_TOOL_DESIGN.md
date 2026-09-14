# PowerWorld Tool Design

## Decision

The local model is the workflow planner. The PowerWorld hierarchy provides progressive discovery, operating instructions, schemas, validation, and execution boundaries. It should not encode a fixed workflow merely to compensate for a model that previously could not plan.

The coordinator uses a bounded plan-act-observe-replan loop:

1. Write a short plan with dependencies and a completion test.
2. Discover only the next relevant capability and tool schema.
3. Propose one typed tool call.
4. Let the adapter validate permissions, paths, and arguments.
5. Execute one call and record its observation in the scratchpad.
6. Continue, revise the remaining plan, ask for clarification, or stop.

The executor never blindly runs every item in a generated plan.

## Preserve workflow knowledge as study playbooks

Decomposing a combined script does not mean deleting the engineering process it encodes. Study knowledge belongs in versioned, literature-backed playbooks that the planner can retrieve before planning. A playbook describes what a valid study requires without forcing every case into one hard-coded call sequence.

Each playbook should carry:

- study type, objective, scope, and applicability limits;
- source literature, citations, publication dates, and provenance notes;
- required inputs and preconditions;
- required tool capabilities and simulator state;
- canonical phases and dependencies;
- decision points where observations may change the remaining plan;
- expected artifacts and naming conventions;
- engineering validation checks and completion criteria;
- safety constraints, failure handling, and restart/recovery guidance;
- playbook version and compatibility with tool-surface versions.

The literature-review workspace can remain the authoring system. This app should import approved playbook releases as read-only planner context, record the exact playbook version used in each study, and keep citations attached to the resulting plan and artifacts.

This produces three separate layers:

1. **Study playbook:** sourced knowledge about what process and evidence the study requires.
2. **Planner:** adapts that process to the request, available data, and observed tool results.
3. **Tools:** atomic operations or justified safety/transaction boundaries.

## Tool granularity rule

Prefer a small, composable operation when the model can safely sequence it. Keep a compound tool when the grouping supplies at least one of these guarantees:

- an atomic simulator transaction or required application lifecycle;
- copy-on-write protection and output-path enforcement;
- a deterministic reducer that keeps large case or AUX data out of context;
- a durable detached job with progress and recovery state;
- a domain algorithm whose intermediate steps are implementation details rather than planning choices.

## Initial surface audit

Likely good atomic or bounded operations include `workspace.resolve_inputs`, field discovery, case open/copy/solve/export operations, session mutations, artifact inspection, ledger operations, and job status.

Compound tools that appear justified include folder/object comparison reducers, formal Difference Case extraction, AUX snapshot reduction, historical batch launchers, scenario construction, and stability workflows. Their value is deterministic processing, simulator lifecycle control, or durable job handling.

These workflow-shaped tools should be reviewed first for decomposition or demotion to optional recipes:

- `case.ingest_solve_and_save`
- `case.prepare_for_study`
- `contingency.run_n_minus_one_workflow`
- `results.local_llm_analyze_aux_snapshots`
- `study.prepare_case_for_study`
- `study.run_timepoint_window`
- `study.package_results`

Some may remain compound after inspection, but each should document the safety or transactional boundary it provides. Otherwise the planner should call their component operations.

## Planner qualification

Before enabling PowerWorld execution in ordinary chats, use repeatable evaluations rather than one successful demonstration:

1. Discovery: route, list, select, and retrieve a schema without executing it.
2. Read-only case task: resolve a path, open a copy or read a case, query rows, and summarize evidence.
3. Corrective replanning: recover from a deliberately invalid field or missing path.
4. Safe mutation: make one edit in a copy, solve, verify, and save to a new path.
5. Detached workflow: launch a long job, retain its call ID, poll, and summarize artifacts.

Grade exact call sequence, schema validity, path safety, repeated-call behavior, completion recognition, and faithfulness of the final answer. Only expose mutation tools after the planner consistently passes the first three levels.

## Current evidence

Gemma 4 12B successfully completed the first two stages of a live discovery loop against the real PowerWorld bridge: it called `powerworld_route_request`, consumed the returned `comparison` capability, then called `powerworld_list_tools` with that capability and summarized the result. No simulator or backend execution tool was called. This proves basic multi-turn tool-state handling, but not yet reliable long-horizon planning or safe mutation.
