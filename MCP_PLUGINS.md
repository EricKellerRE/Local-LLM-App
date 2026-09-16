# MCP Plugin System

## Ownership boundary

Local Model is the MCP host. It owns inference, planning, chat state, permissions, tool-call limits, and the user interface. Domain repositories own MCP servers and guidance. Grid Workshop can therefore contain power-system servers, study playbooks, examples, and tests without becoming part of the model runtime.

## Supported transports

- `stdio`: the host launches a local server subprocess and communicates over JSON-RPC.
- `streamable_http`: the host connects to an independent MCP HTTP endpoint.

Legacy SSE is excluded for new plugins.

## Standard addition procedure

1. Put the server and its tests in the domain repository.
2. Confirm that it initializes and returns valid, paginated `tools/list` results.
3. Add one versioned JSON manifest under `config/mcp.d/`.
4. Assign a short unique `tool_namespace`; exposed names become `<namespace>__<native_tool_name>`.
5. Reference machine paths and secrets with `${VARIABLE}`. Define values in `.env` or the process environment, never in the manifest.
6. Set the host-owned default permission, native-tool allow/deny lists, and call budget.
7. Run `scripts/mcp_plugins.py validate` with the linked environment.
8. Run one harmless, explicitly allowed call with `scripts/mcp_plugins.py call`.
9. Add planner evaluations for discovery, read-only use, error recovery, and mutations before broadening permissions.

## Manifest fields

- `manifest_version`: configuration format version; currently `1`.
- `id`, `name`, `version`, `description`: stable identity and display metadata.
- `tool_namespace`: a short collision-resistant prefix shown to the model.
- `enabled`: whether the registry connects to the plugin.
- `servers`: one or more stdio or Streamable HTTP servers.
- `servers[].companions`: optional non-MCP subprocesses owned by that server. Each companion declares a command, arguments, working directory, environment, optional readiness URL, and startup timeout.
- `guidance`: literature-backed playbooks or instructions with capability labels.
- `policy`: host-owned access defaults, explicit allow/deny lists, and call budget.

Tool annotations such as read-only, destructive, idempotent, and open-world are preserved as hints. They are not authorization; Local Model policy remains authoritative.

## Runtime flow

1. Validate enabled manifests and resolve `${VARIABLE}` references in memory.
2. Start any declared companion processes, wait for their readiness URLs, then connect with the official MCP client and negotiate the protocol.
3. Retrieve every page of `tools/list`, plus advertised resources, resource templates, and prompts.
4. Store the complete catalog internally. Rank compact names, titles, and descriptions with a hybrid lexical/semantic router, then place only the four highest-ranked entries and schemas in model context initially. The sentence encoder runs locally, caches catalog vectors, and falls back to lexical routing if its optional weights are unavailable.
5. Expand the candidate schema set in bounded batches when routing confidence is low.
6. Namespace every callable capability and validate proposed arguments against its JSON Schema.
7. Apply the host permission policy and per-turn call budget outside the model.
8. Validate advertised tool outputs with JSON Schema 2020-12, retain malformed raw results in the audit log, and quarantine them from model context.
9. Bound observations before model context, omit encoded binary payloads, and include deterministic truncation metadata.

Stateful servers should return explicit case, session, or job handles and require those handles on later calls. The planner records handles in its scratchpad rather than depending on hidden connection state.

## Chat interface

The composer `+` menu lists installed plugins, reports connection errors, and shows how many tools were discovered. Its toggles are per chat: enabling a plugin saves that choice with the chat and gives only that chat's planner the plugin's tool catalog. Chats that select the same plugin share one live MCP connection; the connection stops after the final selecting chat disables, archives, or deletes it.

Companion processes share that lifecycle. Local Model refuses to start a companion when its readiness address is already occupied, so it never silently adopts a server it does not own. Normal plugin or app shutdown closes the MCP connection and then terminates every companion it started.

`Add plugin` imports a user-selected JSON manifest but does not start its processes automatically. The Grid Workshop plugin is installed and stopped by default. Archived chats retain their plugin selections, scratchpad, and tool activity in the compressed record, and restoration recovers all three.

Each tool-enabled response writes a plan and an append-only activity log under `data/`. Arguments are schema-validated and checked against the manifest policy before execution. Tools covered by a plugin's `default_access: ask` policy stop at a visible approval card. Approval displays the exact tool and arguments, is consumed once, calls with `approved=True`, and returns the observation to the model before final composition. MCP annotations are retained with their protocol aliases but never grant authority.

Standard MCP tools, resources, and prompts remain distinct. Only tools enter model-selected function schemas. Resources are application-selected and prompts are user-selected through the explicit `/api/chats/{chat_id}/mcp-content` boundary. Static resources take no arguments, resource templates accept a validated matching `uri`, and prompts accept their advertised string arguments. The same manifest policy applies to their synthetic native names (`resource:<name>`, `resource_template:<name>`, and `prompt:<name>`).

After routing, the host reads at most two manifest guidance documents whose capability labels match the routed tools. Each document is content-bounded and its resolved path, SHA-256 version, capability, and truncation status are recorded in the activity log.

`LOCAL_ROUTER_MODEL_ID` selects the local Hugging Face sentence encoder and defaults to `sentence-transformers/all-MiniLM-L6-v2`; set it to an empty value for lexical-only routing. `LOCAL_ROUTER_DEVICE` defaults to CPU so routing does not consume the main generator's GPU allocation. The PowerWorld golden set records semantic cold-start and warm median/p95 latency alongside recall and abstention.

`config/mcp.d/mcp-everything.json` registers the official MCP Everything reference server with `default_access: ask`. `scripts/mcp_everything_smoke.py` provides an unrelated interoperability check covering ordinary tool, resource, and prompt discovery, semantic schema routing, permission override at an explicit test boundary, and result handling.

For staged discovery surfaces, the coordinator can treat explicit route/list/schema responses as an optional optimization. It automatically performs an unambiguous next discovery call, asks the model only when a semantic choice remains, and exposes the smallest relevant tool set for that choice. Servers do not need to implement this protocol: ordinary `tools/list` catalogs use the host's generic metadata router. When structured content is present, duplicate textual content is retained in the audit log but omitted from the next model prompt. Missing required schema inputs and approval requirements become explicit waiting responses instead of additional speculative model turns.
