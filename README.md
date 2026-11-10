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

Context checkpointing is also driven by the model's measured token headroom. Before a tool or chat prompt would consume the effective context window plus its reserved response budget and safety margin, the host writes a semantic continuation record containing the objective, acceptance plan, user requirements, decisions, completed and failed calls, exact artifact identifiers, current structured state, and next action. Repeated checkpoints merge this state rather than recursively nesting earlier summaries. The full checkpoint history is retained beside the tool activity log, while ordinary chats keep a reusable per-chat context record.

Every model generation is recorded as compressed JSON under `data/generation_records/<task-or-chat-id>/`. A record contains the visible messages and tool schemas supplied to the model, the visible response or tool call, generation settings, token counts, stop reason, hashes, and available task, work-item, source, and chat identifiers. Private reasoning text is not persisted. Telemetry links to the corresponding generation record and reports if the final model adapter still had to drop or left-truncate input. Research note and source-dossier generations automatically continue when they stop only because their output-token segment filled.

Versioned workflow skills under `config/skills/` define reliable long-task methods. A skill declares its trigger, required tools, structured inputs, checkpoint graph, completion gates, and deliverables. The coordinator extracts only the subject-specific inputs from a request and instantiates the selected skill; it does not turn the user's entire instruction into a search query or improvise a known multi-stage method on every run. Requests without a matching skill still use the generic planner.

Public-web research automatically uses the built-in search and chunked page reader. Each workflow carries an explicit source policy: core evidence satisfies the requested source target and supports report claims, supplemental material may supply orientation or reference leads without counting toward that target, and disallowed source types are excluded. The pipeline first fills the entire seed quota with reachable, validated core papers, replacing every inaccessible or rejected candidate before analysis begins. It then analyzes one paper at a time, checkpoints long papers section by section, writes a durable standalone report for every analyzed paper, and expands deduplicated references breadth-first. Synthesis sections consume those paper reports incrementally; disposable section and extraction notes are deleted only after the accepted final DOCX is safely exported. Original PDF, HTML, and other source bytes are retained under `data/research/<task-id>/originals/`; the source ledger records each archive path, final URL, content type, byte count, and SHA-256 checksum without placing binary data in model context. Short papers can use a whole-document pass; the thresholds and separate section-note/dossier budgets are exposed in Developer settings. Long Grid Workshop operations use the MCP durable-task protocol, persist the Grid task ID before waiting, and resume polling that same operation after an app restart instead of launching a duplicate. Grid mutations and simulator runs still require explicit approval.

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
| `LOCAL_RESEARCH_CLASSIFIER_MODEL_ID` | Optional lightweight Transformers model for research-source triage; blank uses the main model. |
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

### Structured tool execution

Tool-enabled chats use a persisted execution plan. Each step declares its dependencies, input/output variable names, and eligible tools. The host validates the plan and tool arguments, substitutes exact variable references, and stores full observations separately from the model context. The model receives the current step, bounded previews, and tool schemas; it can read stored results by JSON pointer instead of rediscovering them. Scratchpad prose is supplementary, not execution state.

Plans live in `data/tool_activity/<chat-id>.plan.json`; referenced observations live beside them in `<chat-id>.plan.results/`. Successful identical calls reuse stored evidence. Failed identical calls require repair rather than automatic reexecution. Outputs bind to successful result values, and a step cannot complete before its declared outputs exist. Approval resumes the same plan. An interrupted call with an unknown outcome pauses rather than risking a duplicate execution. Activity shows step IDs and completion events. A completed plan is execution evidence, not an independent guarantee that the model interpreted the data correctly.

Numeric variables include a unit report with source evidence. Supported tool unit metadata takes precedence over explicit unit-bearing column labels such as `InstalledMW`; a variable's chosen name never determines its unit. Unrecognized units remain unknown. MW quantities include a host-computed GW equivalent. `host__convert_units` records conversions as result evidence and requires an explicit power factor for MVA/GVA to MW/GW. Generic arithmetic results without unit metadata remain unknown rather than receiving guessed units.

Run `python -m unittest tests.test_plan_execution` for synthetic execution-state coverage and `python scripts/gguf_workflow_smoke.py` for the real GGUF plus synthetic CSV workflow. The legacy coordinator remains available explicitly with `structured_plans=False` for regression testing.

### Flexible case analysis

Read-only case questions now use a model-authored per-case analysis plan instead of the fixed 345 kV report. The model chooses fields, filters, measures, and output columns. The host validates the entire plan before querying a case, executes one case at a time, and saves each completed operation before advancing. The Activity panel shows the case number and current step.

The supported operations are selection, filtering, aggregation (`count`, `sum`, `mean`, `min`, `max`), unit conversion, and binary arithmetic on saved quantities. Initial object types are Branch, Gen, and Bus. Organizational tag inspection supports attribution decisions without putting raw object exports into model context. Selections use complete exports, verify row counts, and deduplicate identities; conflicting duplicates fail explicitly. Catalog units take precedence. Unknown units require a stated assumption; MVA-to-GW conversion requires a power factor. Unmatched organizational searches remain unknown rather than reporting zero assets.

Plans and typed intermediate values persist outside chat context. Resume reuses completed operations. Invalid fields, references, and unit combinations return a bounded correction opportunity with the rejected plan. Explicit transient read-only transport failures receive one retry; successful requests and unknown-outcome calls are never blindly repeated. Failed cases remain visible in the report while subsequent cases continue. This validates execution, not the model's interpretation of ownership or the user's intended question.

Examples include “Count Alpha branches whose endpoints are both 138 kV,” “Find Alpha's average installed generator capacity in GW,” and “Find the longest Alpha 345 kV line in miles.” Grouped reports, arbitrary Python execution, and simulator mutations are not part of this analysis interface; other plugin tasks retain the general planner.

Run `python -m unittest tests.test_analysis_plan` for synthetic operation and recovery checks. `python scripts/flexible_analysis_smoke.py` asks the configured local GGUF three different questions using two fresh synthetic PowerWorld cases and an isolated plugin backend. It never opens user cases. `--replay <synthetic-run-directory>` reexecutes saved model-authored plans for executor regression testing without regenerating them.
