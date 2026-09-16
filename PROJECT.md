# Project: Local Model Coordinator

## Goal

Provide a desktop-friendly chat experience backed by a persistent local Python server whose model runs through Hugging Face Transformers. The app should feel like a hosted-model assistant while keeping model selection, inference, planning, history, and scratchpad data under local control.

## Non-goals

- No Ollama, LM Studio, hosted inference API, or proprietary model server.
- No unrestricted shell access, arbitrary code execution, or direct execution of unvalidated model output.
- No automatic model download without the user explicitly choosing a Hugging Face model identifier.

## Architecture

```
Browser or terminal UI
  -> Local HTTP API (FastAPI on 127.0.0.1)
      -> Coordinator
          -> Planner: goal, ordered steps, dependencies, completion test
          -> Executor: validate and perform only the next approved tool call
          -> Observer: record the result and replan when reality differs
          -> Response composer
          -> Persistent TransformersModel (Gemma loaded once)
          -> Per-chat scratchpad (JSONL)
      -> Universal task engine
          -> Model-generated, user-confirmed task contract
          -> Durable work queue, bounded episodes, and completion audit
          -> Lease recovery, scheduled retry, pause/resume/cancel
      -> ChatStore (SQLite conversations and messages)
      -> TaskStore (SQLite tasks, work items, events, and audits)
```

The server eagerly loads the configured model once and serializes inference requests. The coordinator first asks the model for a short structured plan, stores it in the chat's scratchpad, then asks for the user-facing answer using that plan and saved conversation history. Chats and messages are durable in SQLite; scratchpads remain visible through the API and `/scratchpad`.

The OpenAI-compatible chat endpoint also supports function definitions, assistant tool calls, and tool-result messages. Tool execution stays outside the model server in explicit adapters and agent loops. The model owns ordinary workflow orchestration: it creates a short plan, calls one small tool at a time, observes the result, and continues or revises the plan. Large tool collections use progressive disclosure to find those small tools without placing every schema in context.

## Conventions

- Keep runtime code in `local_model_app/`; keep the terminal entry point thin.
- All durable application state belongs in `data/`, which is ignored by Git.
- Bind the server to loopback by default; network exposure must be an explicit user choice.
- Treat chat IDs as stable UI routing keys and keep UI state out of the inference layer.
- Read model settings from `.env` or environment variables. Do not hard-code a model ID in Python.
- Keep planner output constrained and treat it as model-generated guidance, not executable instructions.
- Keep tool execution in allow-listed adapters. The model may propose a typed call; the adapter owns validation, authorization, execution, timeouts, and error handling.
- Prefer host-side hybrid lexical/semantic metadata routing and progressive schema disclosure for large tool sets. Do not place an entire backend registry or every schema into a local model's context; semantic failure must retain a deterministic lexical fallback.
- Prefer atomic, composable tools over prompt-specific workflow scripts. Routing and skills should help the planner discover constraints and schemas, not duplicate the planner's sequencing work.
- Keep a compound backend operation only when it provides a real boundary: transactionality, copy-on-write safety, deterministic reduction of large data, simulator lifecycle management, or durable detached-job handling.
- Plans must identify dependencies and a completion condition. Execute only the next valid step, append its observation to the scratchpad, then continue or replan; never blindly execute an entire generated plan.
- Preserve study methodology in versioned, literature-backed playbooks. Tool decomposition must not erase prerequisites, engineering decision points, validation checks, expected artifacts, or citations.
- Archive chats as self-contained gzip files outside the active SQLite database. Include messages and scratchpad data, write atomically before deleting active rows, and support lossless restoration.
- Persist enabled plugin ids per chat. Share one live MCP connection across chats using the same plugin, while exposing only each chat's selected tools to its planner.
- Record plans, tool calls, arguments, results, and final-answer summaries in an inspectable per-chat activity log; include that log in gzip archives.
- Treat server-provided staged gateways as optional optimizations, not requirements. Standard MCP catalogs must work through the generic host router.
- Normalize copied `\_` Windows-path escapes only when the repaired source exists; leave planning and ambiguous choices to the model.
- Prefer clear errors for unavailable CUDA, model-loading failures, or missing packages.
- Write tests without loading a model. Hardware-dependent inference remains a manual smoke test.

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Meaning |
| --- | --- |
| `LOCAL_MODEL_ID` | Hugging Face causal-LM repository or local model directory. |
| `LOCAL_MODEL_DEVICE` | `auto`, `cpu`, or `cuda`. |
| `LOCAL_MODEL_DTYPE` | `auto`, `float16`, `bfloat16`, or `float32`. |
| `LOCAL_MODEL_MAX_NEW_TOKENS` | Upper bound for a single model response. |
| `LOCAL_MODEL_TEMPERATURE` | Sampling temperature. Use `0` for greedy decoding. |
| `LOCAL_MODEL_TRUST_REMOTE_CODE` | Set only for models whose repository you trust. |
| `LOCAL_PLANNER_MAX_NEW_TOKENS` | Token budget for the bounded execution plan. |
| `LOCAL_TOOL_ACTION_MAX_NEW_TOKENS` | Token budget for a structured tool decision. |
| `LOCAL_TOOL_FINAL_MAX_NEW_TOKENS` | Token budget for the final answer after observations. |
| `LOCAL_TOOL_TEMPERATURE` | Sampling temperature for tool-enabled planning and action turns. |

## Plan

1. Done: build the local UI for starting, listing, and resuming saved chats.
2. Done: add OpenAI-compatible function calls and validate the standard PowerWorld MCP catalog loop.
3. Add streaming model output to the API and UI.
4. Add chat rename. Gzip archive/restore and permanent delete are complete.
5. Define and import a versioned schema for literature-backed study playbooks from the workflow-research workspace.
6. In progress: the composer plugin menu installs manifests and persists toggles per chat; selected chats share persistent MCP connections. The host stores complete catalogs internally, uses a measured hybrid semantic router over compact metadata, reveals schemas in bounded batches, keeps resources/prompts outside model-selected tools, validates and bounds observations, attaches capability guidance selectively, and supports one-shot approval in the UI. The official MCP Everything server is registered as an unrelated interoperability fixture. Planning and tool activity are logged and archived. Next, add running-call status, cancellation, activity-log UI, and protocol conformance automation.
7. In progress: the desktop window now owns API startup and shutdown, the installer creates the environment, tool plugins can declare owned companion processes, and closing the window safely drains or ends active work. Remaining packaging work is model-location selection and a distributable installer.
8. The separate long-running-task UI was removed. Longer work belongs in ordinary chats so it uses the same conversation, project context, and tools; durability remains an internal concern rather than a second user workflow.

## PowerWorld integration

`Grid-Workshop/powerworld-aux-agent` now advertises its ordinary MCP tool catalog. Local Model retains every schema internally, ranks compact metadata, and initially exposes only the top four schemas. Gemma chooses and replans around atomic backend calls. Existing combined workflows should be decomposed when they merely encode a fixed call sequence, while deterministic reducers, simulator transactions, copy-on-write operations, and detached jobs remain legitimate compound tools.

The existing PowerWorld agent loop can use this server through `http://127.0.0.1:8765/v1/chat/completions`; no LM Studio model server is involved. Conservative catalog, field-discovery, case-summary, artifact-summary, manifest, and job-status calls are explicitly allowed. Mutations and simulator execution remain approval-gated and subject to the PowerWorld adapter's path, copy-on-write, and detached-job conventions.

Generic MCP integration is configured through versioned drop-in manifests under `config/mcp.d/`; see `MCP_PLUGINS.md`. Local Model owns connection lifecycle and permissions, while Grid Workshop owns power-domain MCP implementations and literature-backed study playbooks.

`scripts/powerworld_tool_smoke.py` and `tests/mcp_state_machine_smoke.py` validate the standard catalog against the real Grid MCP server without loading generator weights. They require `regulatory.list_tests` to appear in the top four schemas, execute without approval, and be observed before the final answer. `evals/powerworld_routing_eval.py` measures recall@4, recall@8, abstention, wrong high-confidence proposals, semantic cold-start, and warm routing latency independently of model tool-call accuracy. `scripts/mcp_everything_smoke.py` validates the same generic host against the official, unrelated MCP Everything reference server.

## Environment note

`localmodel-env` at the repository root is a directory junction to the supplied Conda environment. The supplied Python 3.14 environment successfully installed the current CPU PyTorch and Transformers packages. The initial runtime is therefore CPU-only; install a CUDA-compatible PyTorch wheel later if this machine has a supported NVIDIA GPU and you want GPU inference. The configured model is Google's instruction-tuned Gemma 4 12B BF16 checkpoint, stored under `models/` on drive F and loaded directly into RAM.
