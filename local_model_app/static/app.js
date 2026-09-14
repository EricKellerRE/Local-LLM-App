const state = { chatId: null, chats: [], busy: false, showArchived: false, currentArchived: false, plugins: [], pluginBusy: null, mode: "chat", taskId: null, tasks: [], taskProposal: null };
const el = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  if (response.status === 204) return null;
  return response.json();
}

function escapeHtml(text) {
  const node = document.createElement("div");
  node.textContent = text ?? "";
  return node.innerHTML;
}

function renderPlugins() {
  const list = el("plugin-list");
  if (!state.plugins.length) {
    list.innerHTML = '<div class="plugin-row"><div><strong>No plugins installed</strong><small>Add a JSON plugin manifest.</small></div></div>';
    return;
  }
  list.innerHTML = state.plugins.map((plugin) => {
    const detail = plugin.selected && plugin.status === "running"
      ? `Enabled here · ${plugin.tool_count} tools ready`
      : plugin.status === "running"
        ? `${plugin.tool_count} tools ready for another chat`
        : plugin.error || `${plugin.server_count} server${plugin.server_count === 1 ? "" : "s"} · stopped`;
    return `<div class="plugin-row"><div><strong>${escapeHtml(plugin.name)}</strong><small class="${plugin.error ? "plugin-error" : ""}">${escapeHtml(detail)}</small></div><label class="switch" title="${plugin.selected ? "Disable" : "Enable"} ${escapeHtml(plugin.name)} for this chat"><input type="checkbox" data-plugin-id="${escapeHtml(plugin.id)}" ${plugin.selected ? "checked" : ""} ${state.pluginBusy === plugin.id || state.currentArchived ? "disabled" : ""}><span></span></label></div>`;
  }).join("");
  list.querySelectorAll("input[data-plugin-id]").forEach((toggle) => {
    toggle.addEventListener("change", () => togglePlugin(toggle.dataset.pluginId, toggle.checked));
  });
}

async function loadPlugins() {
  state.plugins = await api(state.chatId ? `/api/chats/${state.chatId}/plugins` : "/api/plugins");
  state.plugins = state.plugins.map((plugin) => ({ ...plugin, selected: Boolean(plugin.selected) }));
  renderPlugins();
}

async function togglePlugin(pluginId, enabled) {
  if (!state.chatId) await newChat();
  state.pluginBusy = pluginId;
  renderPlugins();
  try {
    state.plugins = await api(`/api/chats/${state.chatId}/plugins/${encodeURIComponent(pluginId)}`, {
      method: "PATCH",
      body: JSON.stringify({ enabled }),
    });
  } catch (error) {
    window.alert(`Could not ${enabled ? "enable" : "disable"} plugin: ${error.message}`);
  } finally {
    state.pluginBusy = null;
    await loadPlugins();
  }
}

async function installPluginFile(file) {
  if (!file) return;
  try {
    const manifest = JSON.parse(await file.text());
    const name = manifest.name || manifest.id || file.name;
    if (!window.confirm(`Install ${name}? Installing registers its commands and permissions. It will not start until you turn it on.`)) return;
    await api("/api/plugins", { method: "POST", body: JSON.stringify(manifest) });
    await loadPlugins();
  } catch (error) {
    window.alert(`Could not install plugin: ${error.message}`);
  } finally {
    el("plugin-file").value = "";
  }
}

function renderChats() {
  el("chat-list").innerHTML = state.chats.map((chat) =>
    `<button class="chat-item ${chat.id === state.chatId ? "active" : ""}" data-id="${chat.id}">${escapeHtml(chat.title)}</button>`
  ).join("");
  document.querySelectorAll(".chat-item").forEach((button) => {
    button.addEventListener("click", () => openChat(button.dataset.id));
  });
}

function statusLabel(status) {
  return String(status || "").replaceAll("_", " ");
}

function renderTasks() {
  el("task-list").innerHTML = state.tasks.map((task) =>
    `<button class="task-item ${state.mode === "task" && task.id === state.taskId ? "active" : ""}" data-task-id="${task.id}"><span>${escapeHtml(task.definition.title)}</span><small>${escapeHtml(statusLabel(task.status))}</small></button>`
  ).join("");
  document.querySelectorAll(".task-item").forEach((button) => {
    button.addEventListener("click", () => openTask(button.dataset.taskId));
  });
}

async function loadTasks() {
  state.tasks = await api("/api/tasks");
  renderTasks();
}

function renderMessages(messages) {
  const welcome = messages.length === 0 ? el("welcome").outerHTML : "";
  el("messages").innerHTML = welcome + messages.map((message) =>
    `<article class="message ${message.role}"><div class="bubble">${escapeHtml(message.content)}</div></article>`
  ).join("");
  el("messages").scrollTop = el("messages").scrollHeight;
}

async function loadChats() {
  state.chats = await api(`/api/chats?archived=${state.showArchived}`);
  el("archive-view").textContent = state.showArchived ? "← Active chats" : "Archived chats";
  renderChats();
}

async function newChat() {
  if (state.busy) return;
  state.showArchived = false;
  state.mode = "chat";
  const chat = await api("/api/chats", { method: "POST", body: JSON.stringify({ title: "New chat" }) });
  state.chatId = chat.id;
  await loadChats();
  await openChat(chat.id);
  el("message-input").focus();
}

async function openChat(chatId, { force = false } = {}) {
  if (state.busy && !force) return;
  const chat = await api(`/api/chats/${chatId}`);
  state.mode = "chat";
  state.chatId = chat.id;
  state.currentArchived = Boolean(chat.archived_at);
  el("chat-title").textContent = chat.title;
  el("archive-chat").hidden = false;
  el("archive-chat").textContent = state.currentArchived ? "Restore" : "Archive";
  el("delete-chat").hidden = false;
  el("start-task").hidden = true;
  el("pause-task").hidden = true;
  el("resume-task").hidden = true;
  el("cancel-task").hidden = true;
  el("messages").hidden = false;
  el("composer").hidden = false;
  el("task-detail").hidden = true;
  renderMessages(chat.messages);
  renderChats();
  await loadPlugins();
  updateComposerState();
  document.querySelector(".shell").classList.remove("sidebar-open");
}

function taskControls(task) {
  const terminal = ["completed", "failed", "cancelled"].includes(task.status);
  el("start-task").hidden = task.status !== "draft";
  el("pause-task").hidden = !["runnable", "running", "waiting"].includes(task.status);
  el("resume-task").hidden = !["paused", "waiting", "waiting_for_input", "waiting_for_tools", "failed"].includes(task.status);
  el("cancel-task").hidden = terminal;
}

function renderTaskDetail(task) {
  const criteria = task.definition.success_criteria.map((criterion) => `<li>${escapeHtml(criterion)}</li>`).join("");
  const deliverables = task.definition.deliverables.length
    ? `<div class="task-section"><h3>Deliverables</h3><ul>${task.definition.deliverables.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>`
    : "";
  const items = task.work_items.length
    ? task.work_items.map((item) => `<div class="work-item"><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(statusLabel(item.status))} · ${escapeHtml(item.kind)}</small>${item.result?.result ? `<p>${escapeHtml(item.result.result)}</p>` : item.error ? `<p class="task-error">${escapeHtml(item.error)}</p>` : ""}</div>`).join("")
    : '<p class="working">Work items will be assembled after the task starts.</p>';
  const summary = task.current_summary ? `<div class="task-summary">${escapeHtml(task.current_summary)}</div>` : "";
  el("task-detail").innerHTML = `<div class="task-hero"><span class="status-chip">${escapeHtml(statusLabel(task.status))}</span><h2>${escapeHtml(task.definition.title)}</h2><p>${escapeHtml(task.definition.goal)}</p>${summary}${task.last_error ? `<p class="task-error">${escapeHtml(task.last_error)}</p>` : ""}</div><div class="task-section"><h3>Completion criteria</h3><ul>${criteria}</ul></div>${deliverables}<div class="task-section"><h3>Work items</h3>${items}</div>`;
}

async function openTask(taskId) {
  if (state.busy) return;
  const task = await api(`/api/tasks/${taskId}`);
  state.mode = "task";
  state.taskId = task.id;
  el("chat-title").textContent = task.definition.title;
  el("model-name").textContent = `Long-running task · ${statusLabel(task.status)}`;
  el("archive-chat").hidden = true;
  el("delete-chat").hidden = true;
  el("messages").hidden = true;
  el("composer").hidden = true;
  el("task-detail").hidden = false;
  taskControls(task);
  renderTaskDetail(task);
  renderTasks();
  renderChats();
  document.querySelector(".shell").classList.remove("sidebar-open");
}

function renderTaskProposal(proposal) {
  const criteria = proposal.success_criteria.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const deliverables = proposal.deliverables.length ? `<h4>Deliverables</h4><ul>${proposal.deliverables.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : "";
  el("task-proposal").innerHTML = `<h3>${escapeHtml(proposal.title)}</h3><p>${escapeHtml(proposal.goal)}</p><h4>Complete when</h4><ul>${criteria}</ul>${deliverables}<p>${proposal.execution.deadline ? `Deadline: ${escapeHtml(proposal.execution.deadline)}` : "No deadline"} · ${proposal.execution.maximum_attempts ? `${proposal.execution.maximum_attempts} attempts maximum` : "Continue retrying until resolved"}</p>`;
  el("task-proposal").hidden = false;
}

function openTaskDialog() {
  state.taskProposal = null;
  el("task-request").value = "";
  el("task-proposal").hidden = true;
  el("task-dialog-status").textContent = "";
  el("propose-task").hidden = false;
  el("create-task").hidden = true;
  el("task-dialog").showModal();
  el("task-request").focus();
}

async function proposeTask() {
  const request = el("task-request").value.trim();
  if (!request) return;
  el("propose-task").disabled = true;
  el("task-dialog-status").textContent = "Assembling the task contract locally…";
  try {
    state.taskProposal = await api("/api/tasks/propose", { method: "POST", body: JSON.stringify({ request }) });
    renderTaskProposal(state.taskProposal);
    el("propose-task").hidden = true;
    el("create-task").hidden = false;
    el("task-dialog-status").textContent = "Review the goal and completion criteria before starting.";
  } catch (error) {
    el("task-dialog-status").textContent = error.message;
  } finally {
    el("propose-task").disabled = false;
  }
}

async function createAndStartTask() {
  if (!state.taskProposal) return;
  el("create-task").disabled = true;
  try {
    const request = el("task-request").value.trim();
    const task = await api("/api/tasks", { method: "POST", body: JSON.stringify({ request, definition: state.taskProposal }) });
    await api(`/api/tasks/${task.id}/start`, { method: "POST" });
    el("task-dialog").close();
    await loadTasks();
    await openTask(task.id);
  } catch (error) {
    el("task-dialog-status").textContent = error.message;
  } finally {
    el("create-task").disabled = false;
  }
}

async function taskAction(action) {
  if (!state.taskId) return;
  await api(`/api/tasks/${state.taskId}/${action}`, { method: "POST" });
  await loadTasks();
  await openTask(state.taskId);
}

function showEmptyState() {
  state.chatId = null;
  state.currentArchived = false;
  el("chat-title").textContent = state.showArchived ? "Archived chats" : "New chat";
  el("archive-chat").hidden = true;
  el("delete-chat").hidden = true;
  renderMessages([]);
  loadPlugins().catch(() => {});
  updateComposerState();
}

async function toggleArchiveView() {
  if (state.busy) return;
  state.showArchived = !state.showArchived;
  await loadChats();
  if (state.chats.length) await openChat(state.chats[0].id);
  else showEmptyState();
}

async function archiveCurrentChat() {
  if (!state.chatId || state.busy) return;
  await api(`/api/chats/${state.chatId}/archive`, {
    method: "PATCH",
    body: JSON.stringify({ archived: !state.currentArchived }),
  });
  await loadChats();
  if (state.chats.length) await openChat(state.chats[0].id);
  else showEmptyState();
}

async function deleteCurrentChat() {
  if (!state.chatId || state.busy) return;
  if (!window.confirm("Permanently delete this chat? This cannot be undone.")) return;
  await api(`/api/chats/${state.chatId}`, { method: "DELETE" });
  await loadChats();
  if (state.chats.length) await openChat(state.chats[0].id);
  else showEmptyState();
}

function setBusy(busy) {
  state.busy = busy;
  updateComposerState();
}

function updateComposerState() {
  const disabled = state.busy || state.currentArchived;
  el("message-input").disabled = disabled;
  el("send").disabled = disabled;
  el("composer-note").textContent = state.currentArchived
    ? "Restore this chat before continuing it."
    : state.busy
      ? "Planning and generating locally…"
      : "Responses may take a few minutes on CPU.";
}

async function sendMessage(event) {
  event.preventDefault();
  const input = el("message-input");
  const content = input.value.trim();
  if (!content || state.busy || state.currentArchived) return;
  if (!state.chatId) await newChat();

  const current = await api(`/api/chats/${state.chatId}`);
  renderMessages([...current.messages, { role: "user", content }, { role: "assistant", content: "Working locally…" }]);
  el("messages").lastElementChild.querySelector(".bubble").classList.add("working");
  input.value = "";
  input.style.height = "auto";
  setBusy(true);
  try {
    await api(`/api/chats/${state.chatId}/messages`, { method: "POST", body: JSON.stringify({ content }) });
    await openChat(state.chatId, { force: true });
    await loadChats();
  } catch (error) {
    renderMessages([...current.messages, { role: "user", content }, { role: "assistant", content: `Sorry, the local request failed: ${error.message}` }]);
  } finally {
    setBusy(false);
    input.focus();
  }
}

async function updateHealth() {
  try {
    const health = await api("/health");
    el("status-dot").className = health.status === "ready" ? "ready" : health.status === "error" ? "error" : "";
    el("status-text").textContent = health.status === "ready" ? "Model ready" : health.status === "error" ? "Model error" : "Loading model…";
    el("model-name").textContent = health.model ? health.model.split(/[\\/]/).pop() : "";
  } catch {
    el("status-dot").className = "error";
    el("status-text").textContent = "Server unavailable";
  }
}

el("new-chat").addEventListener("click", newChat);
el("new-task").addEventListener("click", openTaskDialog);
el("close-task-dialog").addEventListener("click", () => el("task-dialog").close());
el("propose-task").addEventListener("click", proposeTask);
el("create-task").addEventListener("click", createAndStartTask);
el("start-task").addEventListener("click", () => taskAction("start"));
el("pause-task").addEventListener("click", () => taskAction("pause"));
el("resume-task").addEventListener("click", () => taskAction("resume"));
el("cancel-task").addEventListener("click", () => {
  if (window.confirm("Cancel this long-running task? Its history will be retained.")) taskAction("cancel");
});
el("plugin-menu-button").addEventListener("click", async () => {
  const menu = el("plugin-menu");
  menu.hidden = !menu.hidden;
  el("plugin-menu-button").classList.toggle("active", !menu.hidden);
  el("plugin-menu-button").setAttribute("aria-expanded", String(!menu.hidden));
  if (!menu.hidden) await loadPlugins();
});
el("add-plugin").addEventListener("click", () => el("plugin-file").click());
el("plugin-file").addEventListener("change", (event) => installPluginFile(event.target.files[0]));
document.addEventListener("click", (event) => {
  if (!event.target.closest(".plugin-control")) {
    el("plugin-menu").hidden = true;
    el("plugin-menu-button").classList.remove("active");
    el("plugin-menu-button").setAttribute("aria-expanded", "false");
  }
});
el("archive-view").addEventListener("click", toggleArchiveView);
el("archive-chat").addEventListener("click", archiveCurrentChat);
el("delete-chat").addEventListener("click", deleteCurrentChat);
el("composer").addEventListener("submit", sendMessage);
el("sidebar-toggle").addEventListener("click", () => document.querySelector(".shell").classList.toggle("sidebar-open"));
el("message-input").addEventListener("input", (event) => {
  event.target.style.height = "auto";
  event.target.style.height = `${Math.min(event.target.scrollHeight, 180)}px`;
});
el("message-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el("composer").requestSubmit();
  }
});

Promise.all([loadChats(), loadTasks(), updateHealth(), loadPlugins()]).then(async () => {
  if (state.chats.length) await openChat(state.chats[0].id);
  else showEmptyState();
}).catch((error) => {
  el("status-dot").className = "error";
  el("status-text").textContent = error.message;
});
setInterval(updateHealth, 15000);
setInterval(async () => {
  await loadTasks().catch(() => {});
  if (state.mode === "task" && state.taskId) await openTask(state.taskId).catch(() => {});
}, 10000);
