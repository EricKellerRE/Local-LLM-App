const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('local_model_app/static/app.js', 'utf8');
async function scenario(fails) {
  const events = [];
  const sandbox = {
    state: {chatId: 'synthetic', busy: false},
    setBusy(value) { sandbox.state.busy = value; events.push(['busy', value]); },
    renderApproval(value) { events.push(['panel', value]); },
    async api() { events.push(['request']); if (fails) throw Error('synthetic HTTP 500'); },
    async openChat() {}, async loadChats() {},
    async loadApproval() { events.push(['refresh']); },
    window: {alert(message) { events.push(['alert', message]); }},
  };
  vm.createContext(sandbox);
  vm.runInContext(source.slice(source.indexOf('async function resolveApproval('), source.indexOf('function setBusy(')), sandbox);
  await sandbox.resolveApproval('approve');
  assert.equal(sandbox.state.busy, false);
  assert.deepEqual(events[1], ['panel', null]);
  assert.ok(events.findIndex(e => e[0] === 'refresh') > events.findIndex(e => e[0] === 'request'));
  if (fails) assert.equal(events.at(-1)[0], 'alert');
}
(async () => { await scenario(false); await scenario(true); console.log('Approval UI success/failure cleanup passed'); })().catch(e => {console.error(e); process.exitCode=1;});
