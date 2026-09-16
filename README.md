# Local Model Coordinator

Local Model Coordinator is a private, local-first chat and task application backed directly by Hugging Face `transformers`. It provides durable conversations, bounded planning, tool execution, and generic Model Context Protocol (MCP) hosting without requiring Ollama, LM Studio, or a hosted inference service.

For design rationale and current roadmap, see [PROJECT.md](PROJECT.md). For the MCP manifest format and integration contract, see [MCP_PLUGINS.md](MCP_PLUGINS.md).

## Feature list

### Local model runtime

- Loads one configured Hugging Face causal or multimodal model and retains it across requests.
- Supports CPU, CUDA, and Accelerate device-map/offload configurations.
- Serializes inference so concurrent requests do not corrupt model state.
- Exposes an OpenAI-compatible `/v1/chat/completions` endpoint with tool-call and tool-result messages.
- Runs on loopback by default; network exposure must be configured explicitly.

### Chat and planning

- Browser and terminal chat interfaces.
- Durable SQLite conversations and messages.
- Per-chat visible scratchpads containing plans and answer summaries.
- Archive, restore, and permanent-delete workflows.
- Bounded planner, tool-selection, observation, and final-answer phases.
- Explicit completion tests and replanning after unexpected observations.

### Generic MCP host

- Standard MCP discovery through `tools/list`, `resources/list`, resource templates, and `prompts/list`.
- Local stdio and Streamable HTTP transports.
- Persistent plugin selection per chat and shared connections between chats using the same plugin.
- Host-owned tool namespaces, permissions, validation, call budgets, and activity logs.
- Hybrid lexical/semantic routing over names, titles, and descriptions.
- Progressive disclosure: four schemas initially and a bounded expansion to eight when confidence is low.
- Local sentence embeddings with deterministic lexical fallback if embedding weights are unavailable.
- Standard MCP servers do not need custom routing or schema-gateway tools.

### Tool safety and context control

- JSON Schema validation for both tool arguments and structured outputs.
- Conservative `allow`, `ask`, and `deny` policies enforced outside the model.
- One-shot approval cards displaying the exact pending tool and arguments.
- Server annotations are retained as hints but never grant authority.
- Raw results remain in the audit log while malformed structured output is quarantined.
- Large observations are deterministically bounded before entering model context.
- Encoded image, audio, and blob payloads are omitted from model context with truncation metadata.
- Tools, resources, and prompts retain their distinct MCP control boundaries.

### Long-running tasks

- Model-generated task proposals reviewed before execution.
- Durable tasks, work items, dependencies, events, and completion audits.
- Bounded worker episodes with leases, retry/backoff, and restart recovery.
- Pause, resume, and cancel controls.
- Visible `waiting for tools` state instead of fabricated success when a capability is unavailable.

### Verification fixtures

- A 98-case PowerWorld routing evaluation covering all 46 Grid Workshop tools, ambiguity, unsupported requests, and no-tool requests.
- A live read-only PowerWorld catalog smoke test.
- The official MCP Everything reference server as an unrelated interoperability fixture.
- Unit tests that do not require loading the main generator model.

## Runtime lifecycle

The intended application lifecycle is:

```text
Start Local Model Coordinator
  -> start the local HTTP API and UI
  -> load the configured model
  -> start the durable task worker
  -> leave unneeded MCP servers stopped

Send a message in a chat with an MCP plugin selected
  -> ensure that plugin's MCP server is running
  -> initialize the MCP session
  -> discover and cache capabilities
  -> route the request and disclose a bounded schema set
  -> validate, authorize, and execute any selected call
  -> observe the result before composing the answer

Disable the plugin everywhere or shut down the application
  -> stop accepting new plugin work
  -> close MCP sessions
  -> terminate every MCP subprocess owned by the host
  -> stop the task worker and close local databases
```

The message path always calls `ensure_started()` for the chat's selected plugins. Selecting a plugin in the current browser UI also establishes its connection immediately so configuration errors are visible before the first message. Deselecting the final chat that uses a plugin stops that shared connection. Normal application shutdown calls `stop_all()` for every managed MCP connection.

### Current development-build boundary

The packaged desktop launcher is not complete yet. In the current development build, `server.ps1` is the application process. Opening or closing the browser does not start or stop that process. Use `Ctrl+C` in its PowerShell window for a graceful shutdown.

The Grid Workshop manifest currently launches a stdio MCP bridge that still expects a separately running HTTP tool backend on `127.0.0.1:8000`. That backend is not yet owned by Local Model Coordinator, so this is not the final plug-and-play lifecycle. The target is a self-contained Grid stdio MCP command that uses its registry in-process, or privately owns and terminates any child backend. Once that is implemented, no manual port-8000 process will be required.

## Initial configuration

1. Confirm that `localmodel-env\python.exe` exists. In this workspace it is a directory junction to the supplied Python environment.
2. Copy `.env.example` to `.env` if `.env` does not already exist.
3. Set `LOCAL_MODEL_ID` to a trusted Hugging Face repository ID or local model directory.
4. Set `GRID_WORKSHOP_ROOT` if using the Grid Workshop plugin.
5. Review device, memory, offload, and token settings before starting the application.

Important settings include:

| Variable | Purpose |
| --- | --- |
| `LOCAL_MODEL_ID` | Main Hugging Face model repository or local directory. |
| `LOCAL_MODEL_KIND` | `auto`, text, or multimodal model selection. |
| `LOCAL_MODEL_DEVICE` | `auto`, `cpu`, or `cuda`. |
| `LOCAL_MODEL_DTYPE` | `auto`, `float16`, `bfloat16`, or `float32`. |
| `LOCAL_MODEL_CPU_MEMORY_GB` | Optional Accelerate CPU memory ceiling. |
| `LOCAL_MODEL_OFFLOAD_DIR` | Optional model offload directory. |
| `LOCAL_ROUTER_MODEL_ID` | Local sentence encoder; blank selects lexical-only routing. |
| `LOCAL_ROUTER_DEVICE` | Router device; CPU is the default. |
| `GRID_WORKSHOP_ROOT` | Directory containing `powerworld-aux-agent`. |
| `LOCAL_MODEL_EAGER_LOAD` | Load the main model during server startup when true. |
| `LOCAL_TASK_POLL_SECONDS` | Durable-task worker polling interval. |

The default semantic router uses `sentence-transformers/all-MiniLM-L6-v2`. Its weights are downloaded and cached on first use if they are not already present. Set `LOCAL_ROUTER_MODEL_ID=` to disable semantic routing.

## Normal operating procedure

### Start the application

1. Open PowerShell in the repository root.
2. Start the application process:

   ```powershell
   .\server.ps1
   ```

3. Wait for model loading to complete. CPU loading can take substantial time.
4. Verify the API:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8765/health
   ```

5. Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

### Start a chat with MCP capabilities

1. Create or open a chat.
2. Open the composer `+` menu.
3. Enable the desired MCP plugin for that chat.
4. Confirm that the plugin reports `running` and shows discovered capability counts.
5. Send the request in ordinary language; do not needlessly use native tool names.
6. If an approval card appears, review the exact tool and arguments.
7. Approve once to execute and resume the answer, or deny to end the pending call.

Read-only tools explicitly listed in a manifest's `allow_tools` policy can execute without an approval card. Mutations and simulator execution should normally remain under `ask`.

### Use resources and prompts

- Tools may be selected by the model from the routed schema set.
- Resources are selected by the application and should usually be attached rather than treated as callable tools.
- Prompts are explicitly selected by the user.
- Large generated artifacts should remain resource links instead of being copied into model context.

### Shut down cleanly

1. Allow any important in-flight model or tool response to finish.
2. Press `Ctrl+C` in the PowerShell window running `server.ps1`.
3. Wait for Uvicorn to complete application shutdown.
4. Confirm that managed MCP subprocesses have exited if troubleshooting lifecycle behavior.

Avoid terminating the Python process directly during normal operation. Forced termination bypasses the FastAPI lifespan cleanup that stops the task worker, closes MCP sessions, and closes SQLite stores.

Closing only the browser tab does not stop the current development server.

## Automatic startup

Install the supplied Windows scheduled task to start the server after sign-in and restart it after failures:

```powershell
.\install-autostart.ps1
```

Remove the scheduled task with:

```powershell
.\remove-autostart.ps1
```

The computer must be powered on and the configured user signed in. The scheduled task runs `server.ps1`; it does not turn the current browser UI into a packaged desktop application.

## Chat and task operations

### Browser UI

- Create and reopen saved chats from the sidebar.
- Enable plugins per chat from the composer menu.
- Review and resolve pending tool approvals.
- Create long-running tasks, review the generated task contract, and then start execution.
- Pause, resume, or cancel durable tasks from their task page.

### Terminal client

With the server running, open another PowerShell window:

```powershell
.\run.ps1
```

Terminal commands:

- `/status` — model and scratchpad status.
- `/new` — create a new saved chat.
- `/chats` — list saved chats.
- `/resume ID` — resume a saved chat.
- `/scratchpad` — display the current scratchpad.
- `/reset` — clear only in-memory terminal history.
- `/quit` — exit the terminal client; it does not stop the server.

## MCP administration

MCP manifests live under `config/mcp.d/`. A manifest defines server transport, command or URL, namespace, optional guidance, permission policy, and call budget.

Standard addition procedure:

1. Verify the server independently.
2. Add a versioned manifest under `config/mcp.d/`.
3. Use a unique short namespace.
4. Keep secrets and machine paths in environment variables.
5. Begin with `default_access: "ask"`.
6. Explicitly allow only a conservative read-only subset.
7. Validate discovery and schemas.
8. Execute one harmless call.
9. Add routing and lifecycle evaluations before broadening access.

Inspect configured plugins:

```powershell
.\localmodel-env\python.exe .\scripts\mcp_plugins.py list
```

Validate enabled plugin discovery:

```powershell
.\localmodel-env\python.exe .\scripts\mcp_plugins.py validate
```

See [MCP_PLUGINS.md](MCP_PLUGINS.md) for the complete manifest and runtime contract.

## Verification procedures

### Unit and integration tests

```powershell
.\localmodel-env\python.exe -m unittest discover -s tests -v
```

### PowerWorld routing evaluation

```powershell
.\localmodel-env\python.exe -m evals.powerworld_routing_eval `
  --powerworld-repo 'C:\Users\your-name\Documents\Grid-Workshop\powerworld-aux-agent'
```

The evaluation records recall@4, recall@8, abstention, wrong high-confidence execution proposals, semantic cold-start time, and warm routing latency. Model tool-call accuracy is intentionally measured separately.

### Live read-only PowerWorld smoke

This test discovers the standard catalog, requires no more than four schemas in the initial planner call, executes `regulatory.list_tests`, and verifies that the result is observed before answer composition. It does not open a case or run the simulator.

```powershell
.\localmodel-env\python.exe .\scripts\powerworld_tool_smoke.py `
  --powerworld-repo 'C:\Users\your-name\Documents\Grid-Workshop\powerworld-aux-agent'
```

The transcript is written to `data/powerworld-tool-smoke.jsonl`.

Run the standalone state-machine smoke with:

```powershell
.\localmodel-env\python.exe .\tests\mcp_state_machine_smoke.py `
  --powerworld-repo 'C:\Users\your-name\Documents\Grid-Workshop\powerworld-aux-agent'
```

### Unrelated MCP interoperability smoke

The official MCP Everything package is downloaded and cached by `npx` the first time this command runs:

```powershell
.\localmodel-env\python.exe .\scripts\mcp_everything_smoke.py
```

It verifies ordinary tool, resource, and prompt discovery, semantic routing, explicit test approval, and result handling without server-specific coordinator logic.

## Data, logs, and recovery

Durable state is stored under `data/`:

- `chats.sqlite3` — conversations, messages, archives, and plugin selections.
- `tasks.sqlite3` — tasks, work items, leases, events, and audits.
- `scratchpads/` — per-chat plans and answer summaries.
- `tool_activity/` — tool routing, calls, raw results, approvals, and terminal state.
- `workflow_runs/` — host-owned working and output directories for applicable workflows.

The directory is intentionally ignored by Git. Back it up before moving machines or performing destructive maintenance.

After an unclean shutdown:

1. Start the application normally.
2. Check `/health` and the browser plugin status.
3. Durable tasks with expired leases will be recovered by the task engine.
4. Inspect per-chat activity and state files before retrying an uncertain external mutation.
5. Never approve the same uncertain mutation again until its external outcome has been verified.

## Known limitations

- The desktop packaging and true launch/quit wrapper are not complete; `server.ps1` owns the development server lifecycle.
- The Grid Workshop stdio MCP server still depends on a separately running HTTP backend. It must become self-contained before PowerWorld has the intended app-owned lifecycle.
- Streaming model output, running-call cancellation, and an activity-log browser remain planned work.
- The routing evaluation measures schema retrieval, not whether the generator always chooses the correct routed tool.
