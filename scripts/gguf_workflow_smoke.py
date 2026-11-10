"""Real GGUF + installed MCP summarizer; entirely synthetic CSV, including recovery."""
import asyncio, json, sys, re, uuid
from pathlib import Path
from contextlib import nullcontext
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from local_model_app.config import Settings, _load_dotenv
from local_model_app.gguf_model import GgufModel
from local_model_app.mcp_plugins import McpPluginRegistry
from local_model_app.mcp_manager import McpPluginManager
from local_model_app.tool_coordinator import ToolCoordinator
from local_model_app.scratchpad import Scratchpad

async def main():
    print('Synthetic agent check: loading configuration.',flush=True)
    _load_dotenv(ROOT/'.env')
    settings=Settings.from_environment(ROOT)
    model=GgufModel(replace(settings,model_kind='gguf',context_window=16384,max_new_tokens=2048))
    if '--plan-debug' in sys.argv:
        original_chat=model.chat
        def debug_chat(*args,**kwargs):
            reply=original_chat(*args,**kwargs)
            print(json.dumps({'synthetic_decision':reply.tool_calls,'stop_reason':reply.stop_reason}),flush=True)
            return reply
        model.chat=debug_chat
    manager=McpPluginManager(McpPluginRegistry(ROOT/'config/mcp.d',project_root=ROOT))
    try:
        print('Synthetic agent check: starting Grid Workshop plugin.',flush=True)
        await manager.start('grid-workshop.powerworld')
        print('Synthetic agent check: plugin connected.',flush=True)
        original_tools=manager.tools_for_plugins
        available=original_tools(['grid-workshop.powerworld'])
        case_mode='--case' in sys.argv
        allowed={'results.summarize_artifact'}
        if case_mode: allowed.update({'case.query_objects','case.search_object_fields','case.describe_object_type','workspace.resolve_inputs'})
        manager.tools_for_plugins=lambda ids:[t for t in available if t.native_name in allowed]
        original_call=manager.call_tool
        attempts=[]
        async def call(name,args,**kwargs):
            # Inject a real rejected column into the first call, then let the agent recover.
            sent=dict(args)
            if case_mode:
                if name.endswith('workspace_resolve_inputs'):
                    sent['case_search_roots']=[str(Path(case).parent)]
                    sent['case_path_hint']=str(case)
                if 'case_path' in sent and Path(sent['case_path']).resolve()!=Path(case).resolve():
                    return {'isError':True,'message':'Synthetic test permits only its explicitly supplied fixture case.'}
            inject=not attempts and not case_mode and '--resume-fixture' not in sys.argv
            if inject: sent['metric']='MissingRating'
            attempts.append({'name':name,'arguments':sent})
            print(json.dumps({'step':len(attempts),'tool':name,'arguments':sent,'injected_error':inject}),flush=True)
            if len(attempts)>24: raise RuntimeError('Synthetic test exceeded 24 calls')
            result = await original_call(name,sent,**kwargs)
            print(json.dumps({'synthetic_result':result.get('structuredContent',result)})[:2500],flush=True)
            return result
        manager.call_tool=call
        fixture = ROOT/"synthetic-workflow-check"/uuid.uuid4().hex
        if '--resume-fixture' in sys.argv:
            fixture=Path(sys.argv[sys.argv.index('--resume-fixture')+1]).resolve()
            if not fixture.is_relative_to((ROOT/'synthetic-workflow-check').resolve()):
                raise ValueError('Resume is restricted to this test script\'s synthetic fixture directory.')
        fixture.mkdir(parents=True,exist_ok=True)
        with nullcontext(str(fixture)) as directory:
            root=Path(directory)
            branches=root/'branches.csv'
            generation=root/'generation.csv'
            branch_text='Entity,kV,Miles,RatingMVA\nAlpha,345,10,1000\nAlpha,345,20,2000\nAlpha,138,99,9000\nBeta,345,70,4000\n'
            generation_text='Entity,InstalledMW\nAlpha,700\nAlpha,800\nBeta,9000\n'
            if '--resume-fixture' in sys.argv:
                assert branches.read_text()==branch_text and generation.read_text()==generation_text
            else:
                branches.write_text(branch_text)
                generation.write_text(generation_text)
            coordinator=ToolCoordinator(model,manager,Scratchpad(root/'notes.jsonl'),root/'activity.jsonl')
            prompt=f"""Analyze synthetic entity Alpha using installed tools and these CSVs: branches {branches}, generation {generation}.
Report total miles of Alpha 345 kV branches, total Alpha 345 kV RatingMVA converted to GW assuming power factor 1, total Alpha installed generation GW, and product of total miles times transmission GW. Ignore other entities and other voltages. Use all matching rows. If a tool rejects a column, inspect the headers and retry with correct columns. Reply finally with only a JSON object whose numeric keys are miles, transmission_gw, generation_gw, gw_miles, plus assumption."""
            if case_mode:
                from synthetic_grid_fixture import create_fixture
                case=create_fixture()
                prompt=f'''Analyze the synthetic entity Alpha in this synthetic PowerWorld case: {case}.
For this fixture, entity attribution is explicitly AreaName Alpha: require both endpoint buses in Alpha for branches, and generator bus in Alpha for generation.
Use only 345 kV branches with both endpoints 345 kV. Report total line miles, sum of MVA rating A converted to GW assuming power factor 1, installed generation maximum capability in GW, and total miles multiplied by total transmission GW.
Discover the actual fields; use case.query_objects with explicit constraints and aggregate complete exports. Do not solve or modify the case. Do not report dispatch as capacity.
Final answer must be a JSON object with numeric keys miles, transmission_gw, generation_gw, gw_miles, plus assumption.'''
            print('Synthetic agent check: loading GGUF and running agent (900 second limit).',flush=True)
            answer=await asyncio.wait_for(coordinator.respond(prompt,[],['grid-workshop.powerworld']),timeout=900)
            match=re.search(r'\{.*\}',answer,re.S)
            if not match: raise AssertionError('Final answer is not JSON: '+answer)
            result=json.loads(match[0])
            expected={'miles':30,'transmission_gw':3,'generation_gw':1.5,'gw_miles':90}
            assert all(result.get(k)==v for k,v in expected.items()),result
            assert len(attempts)>1 or '--resume-fixture' in sys.argv
            print(json.dumps({'passed':True,'expected':expected,'actual':result,'calls':len(attempts),'real_cases_opened':0}),flush=True)
    finally:
        print('Synthetic agent check: shutting down model and plugin.',flush=True)
        model.close()
        await manager.stop_all()
        print('Synthetic agent check: shutdown complete.',flush=True)
if __name__=='__main__': asyncio.run(main())
