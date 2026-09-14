# Local Model Coordinator

A private, local-first chat application that uses Hugging Face `transformers` directly. It does not depend on Ollama or LM Studio.

See [PROJECT.md](PROJECT.md) for the product goals, conventions, architecture, and next steps.

## Open the app

The server is currently available at [http://127.0.0.1:8765](http://127.0.0.1:8765). The browser UI can start a new conversation, reopen any saved chat, show whether Gemma is ready, and display a visible working state during CPU inference.

To start it again after a reboot:

1. Open PowerShell in this folder.
2. Start the persistent model server:

   ```powershell
   .\server.ps1
   ```

3. Open `http://127.0.0.1:8765` in a browser.

The supplied environment is linked at `localmodel-env`, and the configured Gemma checkpoint is already stored under `models/` on drive F. No model files are stored in the LM Studio directory or the default Hugging Face cache.

## Long-running tasks

Choose `Long-running task` in the sidebar and describe an outcome in ordinary language. The model assembles a reviewable goal, completion criteria, deliverables, constraints, and execution policy. Nothing runs until you confirm the proposal.

Confirmed tasks are stored in `data/tasks.sqlite3`. A background worker runs bounded episodes, checkpoints every work item and transition, retries transient failures with backoff, and recovers expired work after the server restarts. Tasks can be paused, resumed, or cancelled from their task page. Until task-specific MCP capabilities are connected, work that requires external access stops visibly in `waiting for tools`; the model is not allowed to claim that unavailable work happened.

To start the local server automatically after Windows sign-in and restart it after failures, run:

```powershell
.\install-autostart.ps1
```

Remove that scheduled task with `.\remove-autostart.ps1`. The computer must be powered on and signed in for local inference to run.

## Optional terminal chat

With the server running, open another PowerShell window and run:

```powershell
.\run.ps1
```

## PowerWorld tool smoke test

The local OpenAI-compatible endpoint supports function definitions, assistant tool calls, and tool-result messages. The repeatable discovery-only test uses the existing staged MCP hierarchy and never opens a case or invokes the simulator:

```powershell
.\localmodel-env\python.exe .\scripts\powerworld_tool_smoke.py `
  --powerworld-repo 'C:\Users\your-name\Documents\Grid-Workshop\powerworld-aux-agent'
```

Its inspectable transcript is written to `data/powerworld-tool-smoke.json`.

## MCP plugins

Generic MCP servers are registered with versioned manifests under `config/mcp.d/`. The registry supports stdio and Streamable HTTP, short model-facing namespaces, environment-variable substitution, JSON Schema argument validation, and host-owned permission policy. See [MCP_PLUGINS.md](MCP_PLUGINS.md) for the standard addition procedure.

```powershell
.\localmodel-env\python.exe .\scripts\mcp_plugins.py validate
```

## Useful commands in chat

- `/status` — display model and scratchpad information
- `/new` — start a new saved chat
- `/chats` — list saved chats and their IDs
- `/resume ID` — resume a saved chat
- `/scratchpad` — print the current scratchpad
- `/reset` — clear only the in-memory conversation history
- `/quit` — exit

## Run tests

```powershell
.\localmodel-env\python.exe -m unittest discover -s tests -v
```
