# Local Model

Local Model is a private desktop chat app. It runs a Hugging Face model on this computer, keeps conversations locally, and can use optional local tools.

You do not need to start or stop a server. The app owns everything it starts.

## Install

1. Double-click **`Install Local LLM.cmd`**.
2. Double-click **`Local LLM.cmd`**. Open **Settings** to choose an installed model, a local model folder, or a model from Hugging Face.

The installer creates (or updates) the Python environment and installs the app's dependencies. It preserves existing settings. If a sibling `Grid-Workshop` folder is present, it configures and installs that tool environment too.

## Open the app

Double-click **`Local LLM.cmd`**.

A Local Model window opens and the model begins loading. Loading a large model on CPU can take several minutes. The status in the lower-left corner changes to **Model ready** when it can answer.

Use the **+** button beside the message box to give a chat access to any installed tools. Local Model starts those tools only when they are needed and closes them with the app.

The app opens to a blank chat, ready for typing. A chat is saved when you first send a message or enable a tool.

## Close the app

Click the normal **X** in the window corner.

- If nothing is running, the window and everything it owns shut down immediately.
- If a response, tool action, model load, or background-task step is active, the app asks whether to let it finish, end it now, or cancel closing.
- **Let it finish** stops new background episodes, waits for the active work to checkpoint, and then closes automatically.
- **End now** closes immediately. Saved chat history remains available the next time the app opens.

There is no server window to manage and no second shutdown step.

## Everyday use

- **New chat** opens a blank conversation without creating clutter in the history.
- **Add project** associates a folder with the app. Project chats are grouped beneath that folder in the sidebar and inherit the tools chosen for the project.
- **Settings** controls the active model, context window, response and reasoning budgets, sampling, app-data folder, and model-library folder. Blank context and reasoning values use the model defaults.
- **Find models on Hugging Face** searches compatible Transformers text-generation models and downloads one into the selected model-library folder.
- Longer requests use the same chat and the same tools; there is no separate task workspace to manage.
- **Tools** enables optional capabilities for only the current chat.
- **Archive** removes a chat from the main list without deleting its saved history.

Chats, scratchpads, and activity records use `data/` by default. Models use `models/` by default. Both locations can be changed in **Settings**; changing the app-data location does not move existing data automatically.

## Research and long work

Ask for research, a literature review, an overnight report, or iterative project-tool work in an ordinary chat. The app turns that request into persisted evidence checkpoints, uses the selected tools, audits the requested outcome, and posts the final report back into the same chat. There is no eight-call or other fixed whole-task limit. The configurable tool-call value is a context-checkpoint interval: reaching it compacts the working context and execution continues automatically. The task stops only when it completes, genuinely needs user input or approval, is cancelled, or encounters an actual unrecoverable failure.

Versioned workflow skills under `config/skills/` define reliable long-task methods. A skill declares its trigger, required tools, structured inputs, checkpoint graph, completion gates, and deliverables. The coordinator extracts only the subject-specific inputs from a request and instantiates the selected skill; it does not turn the user's entire instruction into a search query or improvise a known multi-stage method on every run. Requests without a matching skill still use the generic planner.

Public-web research automatically uses the built-in search and chunked page reader. Long Grid Workshop operations use the MCP durable-task protocol, persist the Grid task ID before waiting, and resume polling that same operation after an app restart instead of launching a duplicate. Grid mutations and simulator runs still require explicit approval.

## Optional: open at sign-in

Run `install-autostart.ps1` once to open Local Model after Windows sign-in. Run `remove-autostart.ps1` to remove it.

## Troubleshooting

If the app cannot open, check `data/logs/application.log`. Common causes are:

- `LOCAL_MODEL_ID` is empty or points to a model that is not available.
- The selected model needs more memory than this computer has.
- Port 8765 is already being used by another program or another copy of Local Model.
- A CUDA device was requested but the installed PyTorch build or driver does not support it.

The optional router model is downloaded on first use. Set `LOCAL_ROUTER_MODEL_ID=` in `.env` to use local lexical routing without that download.

## Configuration

Normal configuration is available from **Settings** in the app. Changes that affect the model or storage location are applied the next time Local Model starts. The `.env` file remains available for advanced and unattended setups:

The usual settings are in `.env`:

| Setting | Purpose |
| --- | --- |
| `LOCAL_MODEL_ID` | Hugging Face repository ID or local model directory. |
| `LOCAL_MODEL_DEVICE` | `auto`, `cpu`, or `cuda`. |
| `LOCAL_MODEL_DTYPE` | `auto`, `float16`, `bfloat16`, or `float32`. |
| `LOCAL_MODEL_CPU_MEMORY_GB` | Optional CPU memory ceiling. |
| `LOCAL_MODEL_OFFLOAD_DIR` | Optional model offload directory. |
| `LOCAL_MODEL_CONTEXT_WINDOW` | Total token capacity for conversation history plus generation; blank uses the detected model limit. |
| `LOCAL_MODEL_MAX_NEW_TOKENS` | Total generation budget, including reasoning tokens; defaults to 8,192 and also governs final answers after tool use. |
| `LOCAL_MODEL_REASONING_BUDGET` | Blank uses the model default, `0` disables thinking, and a positive value is passed to compatible chat templates. |
| `LOCAL_MAX_TOOL_CALLS_PER_STEP` | Context-checkpoint interval; defaults to 256 (maximum 4,096). Reaching it compacts working context and continues rather than stopping the task. |
| `LOCAL_MODEL_EAGER_LOAD` | Load the model as soon as the app opens. |
| `LOCAL_ROUTER_MODEL_ID` | Optional local sentence encoder for choosing tools. |

Keep the default host on `127.0.0.1`; changing it exposes the internal API beyond this computer.

## Developer notes

Normal users should only need the two `.cmd` files. The scripts below are for development and diagnostics:

```powershell
# Run the unit tests without loading the main model
.\localmodel-env\python.exe -m unittest discover -s tests -v

# Run the API without the desktop wrapper
.\server.ps1

# Run the terminal client against an already-running API
.\localmodel-env\python.exe .\main.py

# Inspect configured tool plugins
.\localmodel-env\python.exe .\scripts\mcp_plugins.py list
```

The internal API remains OpenAI-compatible at `/v1/chat/completions`. It binds to loopback by default. `server.ps1` exists for development only; the desktop launcher is the production lifecycle owner.

For architecture and roadmap details, see [PROJECT.md](PROJECT.md). For plugin manifests and the MCP integration contract, see [MCP_PLUGINS.md](MCP_PLUGINS.md).

### Verification fixtures

- `tests/` contains model-free unit and integration coverage.
- `config/skills/` contains versioned durable-workflow skill manifests.
- `evals/powerworld_routing_eval.py` measures PowerWorld tool-routing quality.
- `scripts/powerworld_tool_smoke.py` runs a live read-only PowerWorld catalog smoke test.
- `scripts/mcp_everything_smoke.py` checks interoperability with the official MCP Everything reference server.

The Grid Workshop plugin's HTTP backend is declared as an owned companion process. Enabling the plugin starts the backend and MCP bridge together; disabling it or closing Local Model stops both. A detached Grid MCP task worker can finish independently; Local Model reconnects to its persisted task record on the next launch.
