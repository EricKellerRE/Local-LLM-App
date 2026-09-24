# Token Budget Tuning Test Plan

## Objective

Reduce the time and tokens spent between research tool calls without reducing tool reliability, evidence quality, checkpoint completion, or final-report quality.

The first overnight report is the preserved baseline. It completed all reported stages but produced only about 1,080 words and a handful of unique sources. That is a quality failure even though the old categorical audit passed; future comparisons must treat it as the negative control, not a successful report.

Before another full run, replace model-directed source collection with a host-enforced research pipeline: apply a task-specific source policy, acquire the configured usable core-source count, retain bounded supplemental sources only for orientation or reference leads, fetch and analyze one selected core source at a time, refill unusable core slots, extract and expand cited references breadth-first, deduplicate canonical URLs and DOIs, and stop only at the depth/source limits or a pass with no novel relevant references. Analyze short papers whole and checkpoint long papers section by section, then consolidate one durable evidence dossier per core source. Persist the source graph, source roles, source text, section notes, dossiers, pass counts, and stop reason outside model context.

## Confirmed starting point

The coordinator currently uses materially different budgets for only two cases:

- Planning: 192 tokens inside `ToolCoordinator`, and up to 1,024 tokens for durable-task planning.
- A turn that the host knows still requires an action: 1,024 tokens.
- A turn after ordinary tool data is available: the global response limit, currently 8,192 tokens.
- Final report synthesis: the global response limit, currently 8,192 tokens.

The likely bloat is the third case. After a search or fetch succeeds, the model may need only to select another tool or produce a bounded checkpoint summary, but it receives the same budget as final synthesis. In the baseline run, web operations generally finish in seconds while local generation between calls takes tens of minutes.

## Hypotheses

1. Post-tool reasoning is the dominant avoidable token cost.
2. Tool selection can remain reliable with a substantially smaller budget than 1,024 tokens.
3. Checkpoint summaries need more space than tool selection but much less than final synthesis.
4. Final synthesis should retain an independently configurable large budget.
5. A token limit alone is insufficient: the system must record actual generated tokens and termination reasons to distinguish necessary output from failure to stop.

## Required instrumentation

Record one structured event for every model generation:

- Task, work-item, chat, and model identifiers.
- Turn class: task planning, work-item planning, tool selection, post-tool decision, checkpoint summary, ordinary chat, or final synthesis.
- Prompt tokens, configured output budget, actual generated tokens, and total context tokens.
- Start time, end time, and tokens per second.
- Termination reason: EOS, output limit, tool call, cancellation, or error.
- Whether the result parsed as a valid tool call.
- Selected tool and whether its result passed schema validation.
- Whether the work item subsequently completed, retried, or requested input.

Instrumentation must not store binary tool payloads or duplicate full fetched documents.

## Separate the budgets before tuning

Introduce explicit settings instead of overloading `max_new_tokens`:

- `task_planner_max_new_tokens`
- `work_item_planner_max_new_tokens`
- `tool_action_max_new_tokens`
- `post_tool_decision_max_new_tokens`
- `checkpoint_summary_max_new_tokens`
- `max_new_tokens` for ordinary responses
- `synthesis_max_new_tokens` for final deliverables

The global response limit must never silently control tool selection or post-tool decisions.

## Evaluation fixtures

Create deterministic replay fixtures from the completed baseline run, preserving prompts, compacted tool observations, expected tool families, exact URLs, and checkpoint criteria. Do not invoke the network during replay tests.

Cover at least these cases:

1. Initial search selection from a research checkpoint.
2. Selection of `fetch_url` from search results containing long URLs.
3. Choosing another search after incomplete evidence.
4. Producing a checkpoint summary after sufficient evidence.
5. Recovering from an output-schema validation failure.
6. Preserving exact source URLs in checkpoint evidence.
7. A Grid Workshop call with large structured observations.
8. A sequential Grid optimization requiring another iteration.
9. A no-tool final answer.
10. Full report synthesis from persisted checkpoint records.
11. Seed acquisition cannot complete below the configured readable-source count.
12. Linked, DOI-only, and plain-text bibliography entries enter the reference frontier.
13. Duplicate references reached through different parents collapse to one source while preserving both parent edges.
14. Citation expansion reaches the configured depth and stops early only when a pass adds no novel relevant sources.
15. Per-source analysis resumes without regenerating completed sections or completed source dossiers.

## Search procedure

Binary search is appropriate for finding the smallest passing budget within one fixed turn class, but model quality is not guaranteed to be perfectly monotonic. Use a bracketed binary search followed by boundary confirmation.

### Tool selection

Test the ordered candidates `128, 192, 256, 384, 512, 768, 1024` at temperature zero.

1. Establish one failing and one passing bound.
2. Binary-search the candidate indexes for the smallest passing value.
3. Test the selected value, the candidate immediately below it, and the candidate immediately above it across the complete fixture set.
4. Require repeated success at the selected boundary before adopting it.

### Post-tool decisions

Test `256, 384, 512, 768, 1024, 1536, 2048` using the same procedure. A pass means the model either makes the correct next tool call or explicitly finishes the checkpoint with adequate evidence.

### Checkpoint summaries

Test `512, 768, 1024, 1536, 2048, 3072, 4096`. Score factual coverage and URL retention, not prose length. The summary should be structured evidence for synthesis, not a miniature final report.

### Final synthesis

Do not optimize final synthesis solely for minimum tokens. Compare 4,096, 8,192, and 12,288 where the context window permits. Select based on report completeness and citation traceability, with latency treated as a secondary metric.

For any stochastic turn, run at least three fixed seeds per candidate. Tool-selection experiments remain deterministic at temperature zero.

## Automated scoring

Each replay produces machine-checkable scores:

- Valid tool-call parse rate.
- Correct tool selection rate.
- Required-argument completeness, including long URLs.
- Premature-final-answer rate.
- Checkpoint completion rate.
- Exact URL retention rate.
- Required-topic coverage.
- Unsupported-claim count.
- Generated tokens and wall time by turn class.
- Total model time per completed work item.

Use explicit expected fields and source URLs for deterministic checks. A separate rubric may assess prose quality, but it must not replace the mechanical checks.

## Test layers

1. Unit tests verify that every turn class receives its own configured budget and that synthesis never inherits an intermediate budget.
2. Coordinator tests verify correct budget transitions before and after tool results, including retries and validation failures.
3. Replay integration tests run stored observations through the real local model without network variability.
4. A short live research task verifies real search, fetch, checkpoint, and synthesis behavior.
5. One final overnight report compares end-to-end runtime and quality against this baseline.

## Acceptance criteria

- 100% valid tool-call parsing on the replay suite.
- No long-URL truncation failures.
- No increase in premature checkpoint completion.
- All required checkpoint topics and exact source URLs survive into synthesis.
- At least a 70% reduction in median non-synthesis generated tokens.
- At least a 50% reduction in median time between tool results and the next action.
- Final report coverage and citation traceability are no worse than the baseline.
- Shutdown, restart recovery, checkpoint persistence, and the no-overall-tool-limit behavior continue to pass.

## Execution order

1. Wait for the current report to finish and preserve it as the baseline.
2. Add generation telemetry and tests for telemetry correctness.
3. Add separate budget settings without changing existing defaults.
4. Build replay fixtures from the baseline activity and checkpoint records.
5. Verify the deterministic citation-frontier fixtures before spending local-model time.
6. Run bracketed binary searches independently for reference relevance, source-note extraction, tool selection, post-tool decisions, and checkpoint summaries.
7. Adopt conservative passing boundaries and rerun the full automated suite.
8. Run a short live research task that reaches at least one citation-expansion pass.
9. Run the comparable overnight report and evaluate it against the baseline.
10. Keep the tested budgets and traversal controls exposed in Developer Settings.

## Decision record

Keep the raw benchmark results and selected thresholds in a versioned JSON or CSV artifact. Record rejected budgets and their failure modes so later model or hardware changes can rerun the same procedure instead of relying on the thresholds indefinitely.

### 2026-09-17 Gemma 4 12B checkpoint

The first local boundary run used one deterministic repetition per tested value. It is sufficient to choose deliberately conservative defaults, but it is not a substitute for the three-repeat confirmation required before declaring a model/hardware profile final.

- Exact long-URL tool selection passed at 1,024, 384, 192, and 128 tokens. Adopt 192 rather than the observed 128-token floor.
- Numbered source relevance passed at 512, 256, 192, and 128 tokens. Adopt 256 because the production selector also supports a staged `needs_abstract_numbers` path and richer metadata.
- Source-specific evidence notes passed at 1,536, 768, and 512 tokens. Adopt 768 rather than the observed 512-token floor.
- A completed-tool evidence summary passed at 2,048, 768, 384, and 256 tokens. Adopt 256.
- A bounded 300–450 word report segment passed at 1,024, 768, and 512 tokens. Adopt 768 and checkpoint multiple segments durably to reach the section's total target.
- A monolithic report-section request failed at a 3,072-token ceiling by ending naturally after 461 words. More ceiling did not make the model use it, so section writing now accumulates bounded, nonredundant segments instead of relying on one long generation.
- Raw-model continuation of a chunked URL failed even with a 2,048-token ceiling because the model rewrote punctuation inside the URL. Chunk continuation is now host-controlled using the exact prior URL and `next_start`; token tuning is no longer expected to solve identifier integrity.

The measured wall times changed little when ceilings were reduced because the model usually stopped naturally below them. The main efficiency improvement therefore comes from smaller task shapes, host-owned navigation, numbered selection, durable checkpoints, and eliminating unnecessary model turns—not merely lowering token ceilings. The machine-readable record is in `evals/token-budget-decisions.json`; ignored raw run output remains under `data/benchmarks/` on the test machine.

## Developer tab

Add a clearly labeled **Developer** tab inside Settings for diagnostics, tuning, and model customization that ordinary users should not need during normal operation. Keep broadly useful controls such as model selection, context size, overall response length, and reasoning budget in the normal model settings; do not hide them behind Developer mode.

### Initial tuning surface

The first version should provide:

- Live generation telemetry: turn class, input tokens, generated tokens, configured ceiling, stop reason, elapsed time, and tokens per second.
- A task timeline showing checkpoint transitions, tool calls, validation failures, retries, and synthesis.
- Separate editable budgets for task planning, work-item planning, tool selection, post-tool decisions, checkpoint summaries, ordinary responses, and synthesis.
- Progressive-escalation tiers and the conditions that trigger a retry at the next tier.
- Read-only display of effective context window, model-native context window, device, dtype or quantization, memory use, and active model path.
- A replay runner that evaluates selected stored fixtures without network access.
- A benchmark runner for bracketed binary searches, with pass/fail criteria and comparison results visible in the app.
- Export and import of tuning profiles plus a one-click reset to tested defaults.

### Safety and lifecycle behavior

- Clearly mark settings that apply on the next generation versus those requiring a model or application restart.
- Never mutate the configuration of a generation already in progress.
- Validate that output budgets fit inside the effective context window.
- Preserve the last known-good profile and offer recovery when a model fails to load or a benchmark configuration is invalid.
- Keep raw reasoning content private by default; telemetry should expose counts and timings without automatically displaying hidden reasoning text.
- Require an explicit action to apply experimental values as application defaults.

### Future “make a model your own” surface

Reserve subsections for later additions:

- System-prompt and behavior profiles.
- LoRA or adapter selection and training.
- Dataset preparation and curation.
- Evaluation suites and regression history.
- Tool-use demonstrations and routing examples.
- Model conversion, quantization, and hardware profiles.
- Versioned customization bundles that can be exported, restored, or associated with a project.

The Developer tab should make experiments observable and reversible. It should not become a second task system or expose internal complexity in the main chat workflow.
