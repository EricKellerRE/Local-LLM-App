"""Real local GGUF + real Grid Workshop; fresh synthetic cases only."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import socket
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from local_model_app.config import Settings
from local_model_app.gguf_model import GgufModel
from local_model_app.mcp_manager import McpPluginManager
from local_model_app.mcp_plugins import McpPluginRegistry
from local_model_app.scratchpad import Scratchpad
from local_model_app.tool_coordinator import ToolCoordinator
from synthetic_grid_fixture import create_fixture

async def main():
    replay=Path(sys.argv[sys.argv.index('--replay')+1]).resolve() if '--replay' in sys.argv else None
    if replay and not replay.is_relative_to((ROOT/'synthetic-workflow-check').resolve()): raise ValueError('Replay must be a synthetic experiment')
    out=ROOT/'synthetic-workflow-check'/('flexible-'+uuid.uuid4().hex);out.mkdir(parents=True)
    settings=replace(Settings.from_environment(ROOT),context_window=16384,reasoning_budget=0,max_new_tokens=3072)
    if not settings.model_id.lower().endswith('.gguf'): raise ValueError('Direct GGUF required')
    model=GgufModel(settings,telemetry_path=out/'telemetry.jsonl')
    cases=[str(create_fixture(1)),str(create_fixture(2))]
    (out/'cases.json').write_text(json.dumps(cases),encoding='utf-8')
    # Each experiment owns its backend port, independent of the desktop app.
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1',0)); port=reservation.getsockname()[1]
    manifests=out/'mcp.d'; manifests.mkdir()
    manifest=(ROOT/'config/mcp.d/grid-workshop-powerworld.json').read_text(encoding='utf-8')
    (manifests/'grid.json').write_text(manifest.replace('8766',str(port)),encoding='utf-8')
    manager=McpPluginManager(McpPluginRegistry(manifests,project_root=ROOT))
    results=[]
    try:
        await manager.start('grid-workshop.powerworld')
        original=manager.call_tool
        async def safe_call(name,args,**kwargs):
            if 'case_path' in args and args['case_path'] not in cases: raise ValueError('Synthetic cases only')
            if name.endswith('workspace_resolve_inputs'): raise ValueError('Explicit synthetic case paths already supplied')
            if not name.endswith(('case_query_objects','case_search_object_fields')): raise ValueError('Read-only synthetic analysis tools only')
            print(json.dumps({'tool':name,'case_index':cases.index(args['case_path']) if 'case_path' in args else None}),flush=True)
            return await original(name,args,**kwargs)
        manager.call_tool=safe_call
        original_chat=model.chat
        def chat(*args,**kwargs):
            result=original_chat(*args,**kwargs)
            print(json.dumps({'model_actions':result.tool_calls}),flush=True)
            return result
        model.chat=chat
        tests=[
            ('How many Alpha branches have BOTH endpoints at 138 kV? Report only their count.',[1,1]),
            ('What is the average installed maximum generator capacity in GW for Alpha? Do not use dispatch.',[.75,1.5]),
            ('What is the maximum line length in miles among Alpha branches with BOTH endpoints at 345 kV? The synthetic fixture defines line length in miles.',[20,40]),
        ]
        for index,(question,expected) in enumerate(tests):
            started=time.monotonic(); path=out/str(index);path.mkdir()
            if replay:
                from tests.test_tool_coordinator import FakeModel
                from tests.test_plan_execution import action
                previous=json.loads((replay/str(index)/'activity.plan.json').read_text())
                selected_model=FakeModel([action('host__run_analysis',**previous['analysis_program'])])
            else: selected_model=model
            host=ToolCoordinator(selected_model,manager,Scratchpad(path/'notes.jsonl'),path/'activity.jsonl')
            prompt=question+' Analyze each of these synthetic PowerWorld cases separately: '+', '.join('"'+p+'"' for p in cases)+'. In this fixture Alpha membership is AreaName = Alpha; for branches use both endpoint AreaName fields. Do not infer ownership. Use installed tools.'
            print(json.dumps({'question':index+1,'stage':'started'}),flush=True)
            answer=await host.respond(prompt,[],['grid-workshop.powerworld'])
            (path/'answer.txt').write_text(answer,encoding='utf-8')
            state=json.loads((path/'activity.plan.json').read_text())
            program=state.get('analysis_program',{}); execution=state.get('analysis_execution',{}).get('cases',{})
            actual=[]
            for case in cases:
                row=execution.get(case,{})
                actual.append([row.get('values',{}).get(o['source'],{}).get('value') for o in program.get('outputs',[])])
            passed=actual==[[value] for value in expected] and all(r.get('status')=='completed' for r in execution.values())
            results.append({'question':question,'passed':passed,'expected':expected,'actual':actual,'seconds':round(time.monotonic()-started,2)})
            print(json.dumps(results[-1]),flush=True)
            (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(json.dumps({'passed':all(r['passed'] for r in results),'report':str(out/'results.json'),'real_cases_opened':0}),flush=True)
        if not all(r['passed'] for r in results): raise AssertionError('Synthetic GGUF analysis check failed')
    finally:
        model.close();await manager.stop_all()

if __name__=='__main__': asyncio.run(main())
