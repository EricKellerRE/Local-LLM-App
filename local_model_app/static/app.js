const state = {
  chatId: null, chats: [], projects: [], pendingProjectId: null, busy: false,
  showArchived: false, currentArchived: false, plugins: [], pluginBusy: null,
  detectedContextWindow: null,
  messageSignature: "",
  pendingPluginOverrides: new Map(),
};
const el = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.status === 204 ? null : response.json();
}

function escapeHtml(text) {
  const node = document.createElement("div"); node.textContent = text ?? ""; return node.innerHTML;
}
function renderMessageContent(text) {
  const escaped = escapeHtml(text);
  return escaped
    .replace(/\[([^\]]+)\]\((\/api\/tasks\/[a-zA-Z0-9-]+\/artifacts\/[a-zA-Z0-9._-]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/\n/g, "<br>");
}
function projectById(id) { return state.projects.find((project) => project.id === id) || null; }
function setProjectLabel(id) {
  const project = projectById(id);
  el("chat-project").hidden = false;
  el("chat-project").disabled = state.currentArchived;
  el("chat-project").textContent = project ? `Folder: ${project.name}` : "No project";
  el("chat-project").title = project?.path || "Choose a project for this chat";
}
function chatButton(chat) {
  const archived = state.showArchived;
  const archiveLabel = archived ? "Restore" : "Archive";
  const archiveIcon = archived
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8M8 12l4-4 4 4M12 8v8"/></svg>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M6 7v12h12V7M3 4h18v3H3zM9 11h6"/></svg>';
  const deleteIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13M10 11v5M14 11v5"/></svg>';
  return `<div class="chat-row ${chat.id === state.chatId ? "active" : ""}">
    <button class="chat-open" data-open-chat="${chat.id}" type="button">${escapeHtml(chat.title)}</button>
    <div class="chat-actions">
      <button class="chat-action" data-chat-action="archive" data-chat-id="${chat.id}" type="button" aria-label="${archiveLabel} ${escapeHtml(chat.title)}" title="${archiveLabel}">${archiveIcon}</button>
      <button class="chat-action danger" data-chat-action="delete" data-chat-id="${chat.id}" type="button" aria-label="Delete ${escapeHtml(chat.title)}" title="Delete">${deleteIcon}</button>
    </div>
  </div>`;
}

function renderSidebar() {
  el("project-list").innerHTML = state.projects.map((project) => {
    const chats = state.chats.filter((chat) => chat.project_id === project.id);
    return `<section class="project-group"><div class="project-heading"><button data-project-id="${project.id}" title="${escapeHtml(project.path)}">${escapeHtml(project.name)}</button><button class="project-new-chat" data-new-project-chat="${project.id}" title="New chat in ${escapeHtml(project.name)}">＋</button></div><div class="project-chat-list">${chats.map(chatButton).join("") || '<small class="empty-list">No chats yet</small>'}</div></section>`;
  }).join("") || '<small class="empty-list">No projects yet</small>';
  el("chat-list").innerHTML = state.chats.filter((chat) => !chat.project_id).map(chatButton).join("") || '<small class="empty-list">No chats yet</small>';
  document.querySelectorAll("[data-open-chat]").forEach((button) => button.addEventListener("click", () => openChat(button.dataset.openChat)));
  document.querySelectorAll("[data-chat-action]").forEach((button) => button.addEventListener("click", () => chatAction(button.dataset.chatId, button.dataset.chatAction)));
  document.querySelectorAll("[data-new-project-chat], [data-project-id]").forEach((button) => {
    button.addEventListener("click", () => startBlankChat(button.dataset.newProjectChat || button.dataset.projectId));
  });
}
async function loadProjects() { state.projects = await api("/api/projects"); renderSidebar(); }
async function loadChats() {
  state.chats = await api(`/api/chats?archived=${state.showArchived}`);
  el("archive-view").textContent = state.showArchived ? "← Active chats" : "Archived chats";
  renderSidebar();
}

function showBlankChat(projectId = null) {
  state.chatId = null; state.pendingProjectId = projectId || null; state.currentArchived = false; state.messageSignature = "";
  state.pendingPluginOverrides.clear();
  el("chat-title").textContent = "New chat";
  el("messages").hidden = false; el("composer").hidden = false; el("messages").innerHTML = "";
  setProjectLabel(state.pendingProjectId); renderApproval(null); renderSidebar(); loadPlugins().catch(() => {}); updateComposerState();
  el("message-input").focus(); document.querySelector(".shell").classList.remove("sidebar-open");
  updateHealth().catch(() => {});
}
async function startBlankChat(projectId = null) {
  if (state.showArchived) { state.showArchived = false; await loadChats(); }
  showBlankChat(projectId);
  if (projectId) {
    try { await api(`/api/projects/${projectId}/activate`, { method: "POST" }); await loadPlugins(); }
    catch (error) { window.alert(`The project opened, but its tools could not start: ${error.message}`); }
  }
}
async function createChat(projectId = state.pendingProjectId) {
  const chat = await api("/api/chats", { method: "POST", body: JSON.stringify({ title: "New chat", project_id: projectId || null }) });
  state.chatId = chat.id; state.pendingProjectId = chat.project_id || null;
  try {
    for (const [pluginId, enabled] of state.pendingPluginOverrides.entries()) {
      await api(`/api/chats/${chat.id}/plugins/${encodeURIComponent(pluginId)}`, { method: "PATCH", body: JSON.stringify({ enabled }) });
    }
  } catch (error) {
    await api(`/api/chats/${chat.id}`, { method: "DELETE" }).catch(() => {});
    state.chatId = null;
    throw error;
  }
  state.pendingPluginOverrides.clear(); await loadChats(); return chat;
}
async function openChat(chatId, { force = false } = {}) {
  if (state.busy && !force) return;
  const chat = await api(`/api/chats/${chatId}`);
  state.chatId = chat.id; state.pendingProjectId = chat.project_id || null; state.currentArchived = Boolean(chat.archived_at);
  state.pendingPluginOverrides.clear();
  el("chat-title").textContent = chat.title;
  state.messageSignature = JSON.stringify(chat.messages.map((message) => [message.role, message.created_at, message.content]));
  el("messages").hidden = false; el("composer").hidden = false;
  setProjectLabel(state.pendingProjectId); renderMessages(chat.messages); await Promise.all([loadApproval(), loadPlugins()]); renderSidebar(); updateComposerState();
  document.querySelector(".shell").classList.remove("sidebar-open");
}
async function refreshOpenChat() {
  if (!state.chatId || state.busy || document.hidden) return;
  try {
    const chat = await api(`/api/chats/${state.chatId}`);
    const signature = JSON.stringify(chat.messages.map((message) => [message.role, message.created_at, message.content]));
    if (signature === state.messageSignature) return;
    state.messageSignature = signature; renderMessages(chat.messages); await loadChats(); await loadApproval();
  } catch (error) { console.warn("Could not refresh durable-task progress", error); }
}
function renderMessages(messages) {
  el("messages").innerHTML = messages.map((message) => `<article class="message ${message.role}"><div class="bubble">${renderMessageContent(message.content)}</div></article>`).join("");
  el("messages").scrollTop = el("messages").scrollHeight;
}

function renderPlugins() {
  const list = el("plugin-list");
  if (!state.plugins.length) { list.innerHTML = '<div class="plugin-row"><div><strong>No plugins installed</strong><small>Add a JSON plugin manifest.</small></div></div>'; return; }
  list.innerHTML = state.plugins.map((plugin) => {
    const count = plugin.tool_count + (plugin.resource_count || 0) + (plugin.prompt_count || 0);
    const detail = plugin.selected && plugin.status === "running" ? `Enabled here · ${count} capabilities ready` : plugin.status === "running" ? `${count} capabilities ready for another chat` : plugin.error || "Starts automatically when enabled";
    return `<div class="plugin-row"><div><strong>${escapeHtml(plugin.name)}</strong><small class="${plugin.error ? "plugin-error" : ""}">${escapeHtml(detail)}</small></div><label class="switch"><input type="checkbox" data-plugin-id="${escapeHtml(plugin.id)}" ${plugin.selected ? "checked" : ""} ${state.pluginBusy === plugin.id || state.currentArchived ? "disabled" : ""}><span></span></label></div>`;
  }).join("");
  list.querySelectorAll("[data-plugin-id]").forEach((toggle) => toggle.addEventListener("change", () => togglePlugin(toggle.dataset.pluginId, toggle.checked)));
}
async function loadPlugins() {
  state.plugins = await api(state.chatId ? `/api/chats/${state.chatId}/plugins` : "/api/plugins");
  const projectPlugins = new Set(projectById(state.pendingProjectId)?.plugin_ids || []);
  state.plugins = state.plugins.map((plugin) => ({
    ...plugin,
    selected: state.chatId
      ? Boolean(plugin.selected)
      : state.pendingPluginOverrides.has(plugin.id)
        ? state.pendingPluginOverrides.get(plugin.id)
        : projectPlugins.has(plugin.id),
  }));
  renderPlugins();
}
async function togglePlugin(pluginId, enabled) {
  if (!state.chatId) {
    state.pendingPluginOverrides.set(pluginId, enabled);
    state.plugins = state.plugins.map((plugin) => plugin.id === pluginId ? { ...plugin, selected: enabled } : plugin);
    renderPlugins();
    return;
  }
  state.pluginBusy = pluginId; renderPlugins();
  try { state.plugins = await api(`/api/chats/${state.chatId}/plugins/${encodeURIComponent(pluginId)}`, { method: "PATCH", body: JSON.stringify({ enabled }) }); }
  catch (error) { window.alert(`Could not ${enabled ? "enable" : "disable"} plugin: ${error.message}`); }
  finally { state.pluginBusy = null; await loadPlugins(); }
}
async function installPluginFile(file) {
  if (!file) return;
  try {
    const manifest = JSON.parse(await file.text()); const name = manifest.name || manifest.id || file.name;
    if (!window.confirm(`Install ${name}? It will start only when a chat needs it.`)) return;
    await api("/api/plugins", { method: "POST", body: JSON.stringify(manifest) }); await loadPlugins();
  } catch (error) { window.alert(`Could not install plugin: ${error.message}`); }
  finally { el("plugin-file").value = ""; }
}

function renderApproval(approval) {
  const panel = el("approval-panel");
  if (!approval || approval.status !== "waiting_for_approval") { panel.hidden = true; panel.innerHTML = ""; return; }
  panel.innerHTML = `<div><strong>Tool approval required</strong><p>${escapeHtml(approval.tool)}</p><pre>${escapeHtml(JSON.stringify(approval.arguments || {}, null, 2))}</pre></div><div class="approval-actions"><button id="deny-tool" class="text-button" type="button">Deny</button><button id="approve-tool" class="primary-button" type="button">Approve once</button></div>`;
  panel.hidden = false; el("approve-tool").addEventListener("click", () => resolveApproval("approve")); el("deny-tool").addEventListener("click", () => resolveApproval("deny"));
}
async function loadApproval() { if (!state.chatId || state.currentArchived) return renderApproval(null); renderApproval(await api(`/api/chats/${state.chatId}/approval`)); }
async function resolveApproval(action) {
  if (!state.chatId || state.busy) return; setBusy(true);
  try { await api(`/api/chats/${state.chatId}/approval/${action}`, { method: "POST" }); await openChat(state.chatId, { force: true }); await loadChats(); }
  catch (error) { window.alert(`Could not ${action} this tool call: ${error.message}`); } finally { setBusy(false); }
}
function setBusy(busy) { state.busy = busy; updateComposerState(); }
function updateComposerState() {
  const disabled = state.busy || state.currentArchived; el("message-input").disabled = disabled; el("send").disabled = disabled;
  el("composer-note").textContent = state.currentArchived ? "Restore this chat before continuing it." : state.busy ? "Working in this chat until the current step is complete…" : "Press Enter to send · longer work stays in this chat";
}
async function sendMessage(event) {
  event.preventDefault(); const input = el("message-input"); const content = input.value.trim();
  if (!content || state.busy || state.currentArchived) return; if (!state.chatId) await createChat();
  const current = await api(`/api/chats/${state.chatId}`);
  renderMessages([...current.messages, { role: "user", content }, { role: "assistant", content: "Working locally…" }]);
  el("messages").lastElementChild.querySelector(".bubble").classList.add("working"); input.value = ""; input.style.height = "auto"; setBusy(true);
  try { await api(`/api/chats/${state.chatId}/messages`, { method: "POST", body: JSON.stringify({ content }) }); await openChat(state.chatId, { force: true }); await loadChats(); }
  catch (error) { renderMessages([...current.messages, { role: "user", content }, { role: "assistant", content: `Sorry, the local request failed: ${error.message}` }]); }
  finally { setBusy(false); input.focus(); }
}

function renderProjectChoices() {
  el("project-choices").innerHTML = [`<button type="button" data-choice="">No project<small>Keep this chat unassociated</small></button>`, ...state.projects.map((project) => `<button type="button" data-choice="${project.id}">${escapeHtml(project.name)}<small>${escapeHtml(project.path)}</small></button>`), `<button type="button" id="choice-new-project">＋ Add a project folder</button>`].join("");
  el("project-choices").querySelectorAll("[data-choice]").forEach((button) => button.addEventListener("click", async () => {
    const projectId = button.dataset.choice || null;
    if (state.chatId) await api(`/api/chats/${state.chatId}/project`, { method: "PATCH", body: JSON.stringify({ project_id: projectId }) });
    state.pendingProjectId = projectId; setProjectLabel(projectId); el("project-choice-dialog").close(); await loadChats();
    state.pendingPluginOverrides.clear();
    if (state.chatId) await loadPlugins();
    else if (projectId) {
      try { await api(`/api/projects/${projectId}/activate`, { method: "POST" }); await loadPlugins(); }
      catch (error) { window.alert(`The project was selected, but its tools could not start: ${error.message}`); }
    }
  }));
  el("choice-new-project").addEventListener("click", () => { el("project-choice-dialog").close(); openProjectDialog(); });
}
function openProjectChoice() { renderProjectChoices(); el("project-choice-dialog").showModal(); }
async function chooseDirectory(targetId) {
  try { if (window.pywebview?.api?.choose_directory) { const selected = await window.pywebview.api.choose_directory(); if (selected) el(targetId).value = selected; return; } }
  catch (error) { console.warn(error); }
  const selected = window.prompt("Enter the full folder path:", el(targetId).value); if (selected) el(targetId).value = selected;
}
async function openProjectDialog() {
  if (!state.plugins.length) await loadPlugins();
  el("project-path").value = ""; el("project-name").value = ""; el("project-status").textContent = "";
  el("project-plugins").innerHTML = state.plugins.map((plugin) => `<label class="checkbox-row"><input type="checkbox" value="${escapeHtml(plugin.id)}"> ${escapeHtml(plugin.name)}</label>`).join("") || "No tools installed.";
  el("project-dialog").showModal();
}
async function createProject() {
  const path = el("project-path").value.trim(); if (!path) { el("project-status").textContent = "Choose a project folder first."; return; }
  const pluginIds = [...el("project-plugins").querySelectorAll("input:checked")].map((input) => input.value); el("create-project").disabled = true;
  try {
    const project = await api("/api/projects", { method: "POST", body: JSON.stringify({ path, name: el("project-name").value.trim() || null, plugin_ids: pluginIds }) });
    el("project-dialog").close(); await loadProjects(); await startBlankChat(project.id);
  } catch (error) { el("project-status").textContent = error.message; } finally { el("create-project").disabled = false; }
}

function formatNumber(value) { return value == null ? "?" : Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value); }
function renderModelResults(models) {
  el("hf-results").innerHTML = models.map((model) => `<article class="model-card"><div><strong>${escapeHtml(model.id)}</strong><small>${formatNumber(model.parameters)} params · ${formatNumber(model.downloads)} downloads${model.gated ? " · gated" : ""}</small></div><button class="text-button" type="button" data-download-model="${escapeHtml(model.id)}">Download</button></article>`).join("") || '<small class="empty-list">No compatible models found.</small>';
  el("hf-results").querySelectorAll("[data-download-model]").forEach((button) => button.addEventListener("click", () => downloadModel(button.dataset.downloadModel, button)));
}
async function searchModels() {
  el("search-models").disabled = true; el("hf-status").textContent = "Searching Hugging Face…";
  try { const models = await api(`/api/models/search?q=${encodeURIComponent(el("hf-search").value.trim())}`); renderModelResults(models); el("hf-status").textContent = `${models.length} compatible model${models.length === 1 ? "" : "s"}`; }
  catch (error) { el("hf-status").textContent = error.message; } finally { el("search-models").disabled = false; }
}
async function downloadModel(repoId, button) {
  button.disabled = true; button.textContent = "Starting…";
  try {
    let job = await api("/api/models/download", { method: "POST", body: JSON.stringify({ repo_id: repoId }) });
    while (["queued", "downloading"].includes(job.status)) { button.textContent = "Downloading…"; await new Promise((resolve) => setTimeout(resolve, 1000)); job = await api(`/api/models/download/${job.id}`); }
    if (job.status === "failed") throw new Error(job.error || "Download failed");
    button.textContent = "Downloaded"; el("model-id").value = job.destination; el("hf-status").textContent = `${repoId} is ready. Save settings to make it the active model.`;
  } catch (error) { button.disabled = false; button.textContent = "Retry"; el("hf-status").textContent = error.message; }
}
function updatePromptBudgetNote() {
  const requested = el("context-window").value === "" ? null : Number(el("context-window").value);
  const response = Number(el("max-tokens").value);
  const effective = requested && state.detectedContextWindow ? Math.min(requested, state.detectedContextWindow) : requested || state.detectedContextWindow;
  if (!effective || !response) { el("prompt-budget-note").textContent = "Usable prompt capacity will be shown once a context and response budget are known."; return; }
  const prompt = effective - response;
  el("prompt-budget-note").textContent = prompt > 0 ? `Usable prompt capacity: ${prompt.toLocaleString()} tokens (${effective.toLocaleString()} context − ${response.toLocaleString()} reserved for generation).` : "Response budget must be smaller than the effective context window.";
}
function selectSettingsTab(name) {
  document.querySelectorAll("[data-settings-tab]").forEach((button) => {
    const selected = button.dataset.settingsTab === name;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", selected ? "true" : "false");
  });
  document.querySelectorAll("[data-settings-panel]").forEach((panel) => { panel.hidden = panel.dataset.settingsPanel !== name; });
  if (name === "developer") loadGenerationTelemetry();
}
function resetTuningFields() {
  el("task-planner-tokens").value = 1024;
  el("work-item-planner-tokens").value = 192;
  el("tool-action-tokens").value = 1024;
  el("post-tool-tokens").value = el("max-tokens").value || 8192;
  el("section-tokens").value = 3072;
  el("synthesis-tokens").value = el("max-tokens").value || 8192;
  el("research-classifier-tokens").value = 512;
  el("research-notes-tokens").value = 1536;
  el("research-seed-sources").value = 12;
  el("research-depth-passes").value = 3;
  el("research-max-sources").value = 80;
  el("research-references-per-source").value = 12;
  el("research-notes-batch-size").value = 6;
  el("settings-status").textContent = "Tuning fields reset. Save settings to persist them.";
}
function renderGenerationTelemetry(payload) {
  const records = payload.records || [];
  el("telemetry-summary").textContent = records.length
    ? `${records.length} recent generation${records.length === 1 ? "" : "s"} · ${payload.path}`
    : `No generation telemetry recorded yet. Data will be written to ${payload.path}.`;
  el("telemetry-rows").innerHTML = [...records].reverse().map((record) => {
    const elapsed = record.elapsed_seconds == null ? "—" : `${Number(record.elapsed_seconds).toFixed(1)}s`;
    const rate = record.tokens_per_second == null ? "—" : Number(record.tokens_per_second).toFixed(2);
    return `<tr><td>${escapeHtml(record.generation_class || "unknown")}</td><td>${formatNumber(record.prompt_tokens)}</td><td>${formatNumber(record.generated_tokens)} / ${formatNumber(record.max_new_tokens)}</td><td>${escapeHtml(record.stop_reason || "—")}</td><td>${elapsed}</td><td>${rate}</td></tr>`;
  }).join("") || '<tr><td colspan="6">No records</td></tr>';
}
async function loadGenerationTelemetry() {
  el("refresh-telemetry").disabled = true;
  try { renderGenerationTelemetry(await api("/api/developer/generations?limit=50")); }
  catch (error) { el("telemetry-summary").textContent = error.message; }
  finally { el("refresh-telemetry").disabled = false; }
}
async function openSettings() {
  el("settings-status").textContent = "Loading…"; selectSettingsTab("general"); el("settings-dialog").showModal();
  try {
    const settings = await api("/api/settings");
    const values = { "data-directory": settings.data_directory, "models-directory": settings.models_directory, "model-id": settings.model_id, "offload-directory": settings.offload_dir, "model-kind": settings.model_kind, "model-device": settings.device, "model-dtype": settings.dtype, "cpu-memory": settings.cpu_memory_gb ?? "", "context-window": settings.context_window ?? "", "max-tokens": settings.max_new_tokens, "reasoning-budget": settings.reasoning_budget ?? "", "max-tool-calls": settings.max_tool_calls_per_step, temperature: settings.temperature, "top-p": settings.top_p, "task-planner-tokens": settings.task_planner_max_new_tokens, "work-item-planner-tokens": settings.work_item_planner_max_new_tokens, "tool-action-tokens": settings.tool_action_max_new_tokens, "post-tool-tokens": settings.post_tool_decision_max_new_tokens, "section-tokens": settings.section_max_new_tokens, "synthesis-tokens": settings.synthesis_max_new_tokens, "research-classifier-tokens": settings.research_classifier_max_new_tokens, "research-notes-tokens": settings.research_notes_max_new_tokens, "research-seed-sources": settings.research_seed_sources, "research-depth-passes": settings.research_depth_passes, "research-max-sources": settings.research_max_sources, "research-references-per-source": settings.research_references_per_source, "research-notes-batch-size": settings.research_notes_batch_size };
    Object.entries(values).forEach(([id, value]) => { el(id).value = value; }); el("trust-remote-code").checked = settings.trust_remote_code;
    const nativeContext = settings.detected_context_window;
    const effectiveContext = settings.effective_context_window;
    state.detectedContextWindow = nativeContext;
    el("developer-model-summary").textContent = `${settings.model_id || "No model selected"} · ${settings.device} · ${settings.dtype} · effective context ${effectiveContext?.toLocaleString() || "unknown"}`;
    el("context-window-note").textContent = nativeContext ? `Detected model limit: ${nativeContext.toLocaleString()} tokens${effectiveContext && effectiveContext !== nativeContext ? ` · effective limit: ${effectiveContext.toLocaleString()}` : ""}.` : "The model's native context limit is detected after it loads.";
    updatePromptBudgetNote();
    el("installed-models").innerHTML = settings.installed_models.map((model) => `<option value="${escapeHtml(model.path)}">${escapeHtml(model.name)}</option>`).join(""); el("settings-status").textContent = "";
  } catch (error) { el("settings-status").textContent = error.message; }
}
async function saveSettings() {
  const numeric = (id) => el(id).value === "" ? null : Number(el(id).value);
  const body = { data_directory: el("data-directory").value, models_directory: el("models-directory").value, model_id: el("model-id").value.trim(), offload_dir: el("offload-directory").value.trim(), model_kind: el("model-kind").value, device: el("model-device").value, dtype: el("model-dtype").value, cpu_memory_gb: numeric("cpu-memory"), context_window: numeric("context-window"), max_new_tokens: numeric("max-tokens"), reasoning_budget: numeric("reasoning-budget"), task_planner_max_new_tokens: numeric("task-planner-tokens"), work_item_planner_max_new_tokens: numeric("work-item-planner-tokens"), tool_action_max_new_tokens: numeric("tool-action-tokens"), post_tool_decision_max_new_tokens: numeric("post-tool-tokens"), section_max_new_tokens: numeric("section-tokens"), synthesis_max_new_tokens: numeric("synthesis-tokens"), research_classifier_max_new_tokens: numeric("research-classifier-tokens"), research_notes_max_new_tokens: numeric("research-notes-tokens"), research_seed_sources: numeric("research-seed-sources"), research_depth_passes: numeric("research-depth-passes"), research_max_sources: numeric("research-max-sources"), research_references_per_source: numeric("research-references-per-source"), research_notes_batch_size: numeric("research-notes-batch-size"), max_tool_calls_per_step: numeric("max-tool-calls"), temperature: numeric("temperature"), top_p: numeric("top-p"), trust_remote_code: el("trust-remote-code").checked };
  el("save-settings").disabled = true;
  try { const result = await api("/api/settings", { method: "PUT", body: JSON.stringify(body) }); el("settings-status").textContent = result.restart_required ? "Saved. Restart Local Model to apply these changes." : "Settings are up to date."; }
  catch (error) { el("settings-status").textContent = error.message; } finally { el("save-settings").disabled = false; }
}

async function toggleArchiveView() { if (state.busy) return; state.showArchived = !state.showArchived; await loadChats(); showBlankChat(); }
async function chatAction(chatId, action) {
  if (state.busy) return;
  if (action === "archive") {
    await api(`/api/chats/${chatId}/archive`, { method: "PATCH", body: JSON.stringify({ archived: !state.showArchived }) });
  } else if (action === "delete") {
    const chat = state.chats.find((item) => item.id === chatId);
    if (!window.confirm(`Permanently delete “${chat?.title || "this chat"}”? This cannot be undone.`)) return;
    await api(`/api/chats/${chatId}`, { method: "DELETE" });
  }
  await loadChats();
  if (state.chatId === chatId) showBlankChat();
}
async function updateHealth() {
  try { const health = await api("/health"); el("status-dot").className = health.status === "ready" ? "ready" : health.status === "error" ? "error" : ""; el("status-text").textContent = health.status === "ready" ? "Model ready" : health.status === "error" ? "Model error" : "Loading model…"; el("model-name").textContent = health.model ? health.model.split(/[\\/]/).pop() : ""; }
  catch { el("status-dot").className = "error"; el("status-text").textContent = "Local Model unavailable"; }
}

el("new-chat").addEventListener("click", () => startBlankChat()); el("new-project").addEventListener("click", openProjectDialog); el("chat-project").addEventListener("click", openProjectChoice);
el("close-project-dialog").addEventListener("click", () => el("project-dialog").close()); el("browse-project").addEventListener("click", () => chooseDirectory("project-path")); el("create-project").addEventListener("click", createProject); el("close-project-choice").addEventListener("click", () => el("project-choice-dialog").close());
el("open-settings").addEventListener("click", openSettings); el("close-settings").addEventListener("click", () => el("settings-dialog").close()); el("save-settings").addEventListener("click", saveSettings); el("search-models").addEventListener("click", searchModels); el("hf-search").addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); searchModels(); } });
document.querySelectorAll("[data-settings-tab]").forEach((button) => button.addEventListener("click", () => selectSettingsTab(button.dataset.settingsTab))); el("refresh-telemetry").addEventListener("click", loadGenerationTelemetry); el("reset-tuning-defaults").addEventListener("click", resetTuningFields);
el("context-window").addEventListener("input", updatePromptBudgetNote); el("max-tokens").addEventListener("input", updatePromptBudgetNote);
document.querySelectorAll(".browse-directory").forEach((button) => button.addEventListener("click", () => chooseDirectory(button.dataset.target)));
el("plugin-menu-button").addEventListener("click", async () => { const menu = el("plugin-menu"); menu.hidden = !menu.hidden; el("plugin-menu-button").classList.toggle("active", !menu.hidden); if (!menu.hidden) await loadPlugins(); });
el("add-plugin").addEventListener("click", () => el("plugin-file").click()); el("plugin-file").addEventListener("change", (event) => installPluginFile(event.target.files[0]));
document.addEventListener("click", (event) => { if (!event.target.closest(".plugin-control")) { el("plugin-menu").hidden = true; el("plugin-menu-button").classList.remove("active"); } });
el("archive-view").addEventListener("click", toggleArchiveView); el("composer").addEventListener("submit", sendMessage); el("sidebar-toggle").addEventListener("click", () => document.querySelector(".shell").classList.toggle("sidebar-open"));
el("message-input").addEventListener("input", (event) => { event.target.style.height = "auto"; event.target.style.height = `${Math.min(event.target.scrollHeight, 180)}px`; }); el("message-input").addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); el("composer").requestSubmit(); } });
el("messages").addEventListener("click", () => { if (!state.currentArchived) el("message-input").focus(); });
document.addEventListener("keydown", (event) => {
  if (state.busy || state.currentArchived || event.ctrlKey || event.metaKey || event.altKey || event.key.length !== 1) return;
  if (event.target.closest("input, textarea, select, button, dialog")) return;
  const input = el("message-input");
  event.preventDefault(); input.focus();
  input.setRangeText(event.key, input.selectionStart, input.selectionEnd, "end");
  input.dispatchEvent(new Event("input", { bubbles: true }));
});

Promise.all([loadChats(), loadProjects(), updateHealth(), loadPlugins()]).then(() => showBlankChat()).catch((error) => { el("status-dot").className = "error"; el("status-text").textContent = error.message; });
setInterval(updateHealth, 15000);
setInterval(refreshOpenChat, 5000);
