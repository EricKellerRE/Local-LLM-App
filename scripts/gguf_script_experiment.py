"""Ask the actual local GGUF to author code; synthetic-only execution is separate."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_model_app.config import Settings
from local_model_app.gguf_model import GgufModel
from local_model_app.mcp_manager import McpPluginManager
from local_model_app.mcp_plugins import McpPluginRegistry

OUT = ROOT / 'synthetic-script-experiment'

async def main():
    OUT.mkdir(exist_ok=True)
    settings = replace(Settings.from_environment(ROOT), context_window=16384,
                       max_new_tokens=4096, reasoning_budget=0)
    if not settings.model_id.lower().endswith('.gguf'):
        raise ValueError('Experiment requires the configured direct GGUF model.')
    manager = McpPluginManager(McpPluginRegistry(ROOT/'config/mcp.d', project_root=ROOT))
    try:
        await manager.start('grid-workshop.powerworld')
        query = next(t for t in manager.tools_for_plugins(['grid-workshop.powerworld'])
                     if t.native_name == 'case.query_objects')
        schema = query.input_schema
    finally:
        await manager.stop_all()
    prompt = '''Write Python code only, no Markdown, implementing:
async def analyze(cases, company, call, read_rows):
It returns a list of result dicts, one per case, processing cases sequentially.
cases is an already recursively discovered list of PowerWorld .pwb paths.
call is an async callback: await call("case.query_objects", arguments).
It calls the installed Grid Workshop plugin and returns its unwrapped dict.
read_rows(result) is a supplied synchronous callback which reads ALL rows from
the returned complete CSV artifact, validates its row count, and normalizes
CSV aliases to the canonical field names requested in the tool call.
Never use result preview rows; use read_rows. No imports or filesystem access:
all input/output must go through these two callbacks. Do not call other tools.

For each case, query Branch and Gen, select rows where ANY supplied organizational
name field equals company case-insensitively after stripping surrounding whitespace.
This synthetic fixture's declared attribution basis is organizational membership,
not proof of ownership. Search OwnerName, AreaName, AreaName:1 for branches,
and OwnerName, AreaName for generators. Use constraints=[] to retrieve the complete
source export and perform OR attribution in Python. limit=1 is only preview size.
Branch fields: BusNum, BusNum:1, LineCircuit, BusNomVolt, BusNomVolt:1,
LineLength (fixture unit miles), LineAMVA (MVA), OwnerName, AreaName, AreaName:1.
Gen fields: BusNum, GenID, GenMWMax (MW), OwnerName, AreaName.
Keep branches only when BOTH endpoint nominal voltages equal 345.
Deduplicate branch identities (BusNum, BusNum:1, LineCircuit) and generators
(BusNum, GenID); reject conflicting duplicates rather than silently using one.
Reject missing or invalid numeric measurements, negative values, NaN or infinity.
For no entity matches, report unknown (None), not a proven zero.
For matching branches all at other voltages, transmission metrics are zero.
Sum all matching 345 kV lengths as miles, sum rating MVA /1000 at PF=1 as
transmission_gw, sum generator maximum MW /1000 as generation_gw, and compute
gw_miles = miles * transmission_gw. Never use dispatch.
Result keys: case, miles, transmission_gw, generation_gw, gw_miles, error.
A failing case must remain in the output with error and None metrics; continue
to the next case. No hardcoded answers or entity names. No top-level execution.
Actual case.query_objects JSON input schema:
''' + json.dumps(schema)
    messages=[{'role':'user','content':prompt}]
    if '--repair' in sys.argv:
        previous=(OUT/'generated.py').read_text(encoding='utf-8')
        (OUT/'generated-first.py').write_text(previous,encoding='utf-8')
        (OUT/'result-first.json').write_text((OUT/'result.json').read_text(encoding='utf-8'),encoding='utf-8')
        feedback='''Your script failed the actual plugin test: [] should be non-empty for fields.
You invented a query response key called fields. There is no such field-discovery
interface. The canonical field lists were already provided in the prompt. Use
those exact literal lists in one Branch query and one Gen query per case, with
constraints=[] and limit=1. Then use read_rows(result) for all exported rows.
Identical duplicate rows must count once, not raise an error; only conflicting
duplicates should fail. Your unused numeric helper includes float('elif'), which
is invalid. Validate all measurements with f >= 0 and f < float('inf'); this also
rejects NaN. Missing numeric fields must fail, never default to zero.
Return None independently for a missing branch attribution or generator attribution.
Keep sequential per-case processing and recovery after a failed case.
Rewrite the full function, concise Python only, no imports, no top-level execution.
Do not add a discovery query or hardcode test results.'''
        messages += [{'role':'assistant','content':previous},{'role':'user','content':feedback}]
        (OUT/'repair-feedback.txt').write_text(feedback,encoding='utf-8')
    (OUT/'prompt.txt').write_text(prompt, encoding='utf-8')
    model = GgufModel(settings, telemetry_path=OUT/'telemetry.jsonl')
    started = time.monotonic()
    try:
        print(json.dumps({'stage':'loading_model','model':Path(settings.model_id).name}), flush=True)
        await asyncio.to_thread(model.load)
        print(json.dumps({'stage':'generating_code','load_seconds':round(time.monotonic()-started,2)}), flush=True)
        reply = await asyncio.to_thread(model.chat, messages,
                                       temperature=0, max_new_tokens=4096)
        (OUT/'raw-response.txt').write_text(reply.content, encoding='utf-8')
        code = reply.content.strip()
        if code.startswith('```'):
            code = code.split('\n',1)[1].rsplit('```',1)[0].strip()
        (OUT/'generated.py').write_text(code+'\n', encoding='utf-8')
        print(json.dumps({'stage':'code_saved','seconds':round(time.monotonic()-started,2),
                          'stop_reason':reply.stop_reason,'characters':len(code)}), flush=True)
    finally:
        model.close()

if __name__ == '__main__':
    asyncio.run(main())
