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
- `guidance`: literature-backed playbooks or instructions with capability labels.
- `policy`: host-owned access defaults, explicit allow/deny lists, and call budget.

Tool annotations such as read-only, destructive, idempotent, and open-world are preserved as hints. They are not authorization; Local Model policy remains authoritative.

## Runtime flow

1. Validate enabled manifests and resolve `${VARIABLE}` references in memory.
2. Connect with the official MCP client and negotiate the protocol.
3. Retrieve every page of `tools/list`.
4. Namespace tools and expose only the planner-relevant subset.
5. Validate proposed arguments against the advertised JSON Schema.
6. Apply the host permission policy.
7. Call the native tool and return `content`, `structuredContent`, and `isError`.

Stateful servers should return explicit case, session, or job handles and require those handles on later calls. The planner records handles in its scratchpad rather than depending on hidden connection state.

## Chat interface

The composer `+` menu lists installed plugins, reports connection errors, and shows how many tools were discovered. Its toggles are per chat: enabling a plugin saves that choice with the chat and gives only that chat's planner the plugin's tool catalog. Chats that select the same plugin share one live MCP connection; the connection stops after the final selecting chat disables, archives, or deletes it.

`Add plugin` imports a user-selected JSON manifest but does not start its processes automatically. The Grid Workshop plugin is installed and stopped by default. Archived chats retain their plugin selections, scratchpad, and tool activity in the compressed record, and restoration recovers all three.

Each tool-enabled response writes a plan and an append-only activity log under `data/`. Arguments are schema-validated and checked against the manifest policy before execution. Tools covered by a plugin's `default_access: ask` policy are blocked until an approval interaction is added; the planner cannot bypass that boundary.

For staged discovery surfaces, the coordinator treats explicit route/list/schema responses as protocol. It automatically performs an unambiguous next discovery call, asks the model only when a semantic choice remains, and exposes the smallest relevant tool set for that choice. When structured content is present, duplicate textual content is retained in the audit log but omitted from the next model prompt. Missing required schema inputs and approval requirements become explicit waiting responses instead of additional speculative model turns.
