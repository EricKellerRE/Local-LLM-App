const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync("local_model_app/static/app.js", "utf8");
const html = fs.readFileSync("local_model_app/static/index.html", "utf8");
const controls = new Map();
for (const match of html.matchAll(/<(input|select)\b[^>]*\bid="([^"]+)"[^>]*>/g)) {
  const listeners = {};
  controls.set(match[2], { id: match[2], value: "1", checked: false, tagName: match[1].toUpperCase(),
    type: /type="([^"]+)"/.exec(match[0])?.[1] || "text",
    addEventListener(event, fn) { listeners[event] = fn; }, listeners });
}
const elements = new Map(controls);
elements.set("settings-status", {});
elements.set("settings-dialog", {addEventListener() {}});
const writes = [];
const sandbox = { el: id => elements.get(id), setTimeout, clearTimeout,
  document: {querySelectorAll: () => [...controls.values()]},
  api: async (path, options) => { await new Promise(r => setTimeout(r, 5)); writes.push(JSON.parse(options.body)); return {restart_required: true}; } };
vm.createContext(sandbox);
vm.runInContext(source.slice(source.indexOf("let settingsSaveQueue"), source.indexOf("async function openSettings")), sandbox);
vm.runInContext(source.slice(source.indexOf("async function saveSettings"), source.indexOf("async function toggleArchiveView")), sandbox);
vm.runInContext(source.slice(source.indexOf("// Persist configuration controls")), sandbox);
vm.runInContext("settingsReady = true", sandbox);
(async () => {
  for (const control of controls.values()) {
    if (control.id === "hf-search") { assert.equal(control.listeners.change, undefined); continue; }
    assert.equal(typeof control.listeners.change, "function", control.id);
  }
  controls.get("model-id").value = "synthetic-first.gguf";
  controls.get("model-id").listeners.change();
  controls.get("model-id").value = "synthetic-second.gguf";
  controls.get("model-id").listeners.change();
  await vm.runInContext("settingsSaveQueue", sandbox);
  assert.deepEqual(writes.map(w => w.model_id), ["synthetic-first.gguf", "synthetic-second.gguf"]);
  controls.get("temperature").value = "0.3";
  controls.get("temperature").listeners.input();
  await new Promise(r => setTimeout(r, 450));
  await vm.runInContext("settingsSaveQueue", sandbox);
  assert.equal(writes.at(-1).temperature, 0.3);
  controls.get("trust-remote-code").checked = true;
  controls.get("trust-remote-code").listeners.change();
  await vm.runInContext("settingsSaveQueue", sandbox);
  assert.equal(writes.at(-1).trust_remote_code, true);
  assert.match(elements.get("settings-status").textContent, /^Saved/);
  console.log("PASS: all settings wired, rapid edits ordered, typing autosaved, checkbox persisted.");
})().catch(error => { console.error(error); process.exitCode = 1; });
