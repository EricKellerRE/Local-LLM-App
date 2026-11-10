"""Execute reviewed model code with synthetic-case-only plugin capabilities."""
import ast
import asyncio
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from local_model_app.config import _load_dotenv
from local_model_app.mcp_manager import McpPluginManager
from local_model_app.mcp_plugins import McpPluginRegistry
from local_model_app.grid_metrics import unpack
from synthetic_grid_fixture import create_fixture
OUT=ROOT/'synthetic-script-experiment'

async def main():
    _load_dotenv(ROOT/'.env')
    if '--prepare' in sys.argv:
        cases=[str(create_fixture(scale=1)),str(create_fixture(scale=2))]
        (OUT/'synthetic-cases.json').write_text(json.dumps(cases),encoding='utf-8')
        print('Two fresh synthetic cases created.',flush=True)
        return
    code=(OUT/'generated.py').read_text(encoding='utf-8')
    tree=ast.parse(code)
    # Defense in depth, not a general-purpose Python sandbox. Code is manually
    # reviewed before this script is launched; no arbitrary model code is trusted.
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            raise ValueError('Generated code must not import modules')
        if isinstance(node,ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('Private attribute access is forbidden')
        if isinstance(node,ast.Name) and node.id.startswith('__'):
            raise ValueError('Dunder access is forbidden')
    if any(not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in tree.body):
        raise ValueError('Only function definitions may appear at top level')
    allowed=('str','float','int','len','sum','min','max','abs','all','any','set','dict',
             'list','tuple','enumerate','zip','range','isinstance','sorted','bool',
             'ValueError','Exception','TypeError','KeyError')
    import builtins
    env={'__builtins__':{name:getattr(builtins,name) for name in allowed}}
    exec(compile(tree,'model-generated.py','exec'),env)
    cases=json.loads((OUT/'synthetic-cases.json').read_text(encoding='utf-8'))
    if len(cases)!=2 or any(not Path(p).resolve().is_relative_to((ROOT/'synthetic-workflow-check').resolve()) for p in cases):
        raise ValueError('Only this experiment\'s synthetic fixture paths are allowed')
    manager=McpPluginManager(McpPluginRegistry(ROOT/'config/mcp.d',project_root=ROOT))
    calls=[]; artifacts={}; catalogs={}
    try:
        await manager.start('grid-workshop.powerworld')
        tools={t.native_name:t for t in manager.tools_for_plugins(['grid-workshop.powerworld'])}
        for kind in ('Branch','Gen'):
            result=unpack(await manager.call_tool(tools['case.search_object_fields'].exposed_name,
                         {'object_type':kind,'query':'','limit':100000}))
            catalogs[kind]={f['variable_name']:f.get('concise_name') for f in result['matches']}
        async def call(name,args):
            if name!='case.query_objects' or args.get('case_path') not in cases:
                raise ValueError('Only the two fresh synthetic cases may be queried')
            if args.get('object_type') not in catalogs:
                raise ValueError('Only Branch and Gen queries allowed')
            if set(args)-{'case_path','object_type','fields','constraints','limit','offset'}:
                raise ValueError('Unexpected arguments')
            if len(calls)>=12: raise ValueError('Exceeded synthetic test call budget')
            calls.append({'case_index':cases.index(args['case_path']), 'kind':args['object_type']})
            print(json.dumps({'query':calls[-1]}),flush=True)
            result=unpack(await manager.call_tool(tools[name].exposed_name,args))
            artifacts[id(result)]=(args['object_type'],args['fields'])
            return result
        def read_rows(result):
            if id(result) not in artifacts: raise ValueError('Unknown artifact')
            if result.get('artifact_complete') is not True: raise ValueError('Incomplete artifact')
            kind,fields=artifacts[id(result)]
            with Path(result['artifact_path']).open(encoding='utf-8-sig',newline='') as stream:
                for _ in range(int(result.get('artifact_skip_rows',0))): next(stream)
                reader=csv.DictReader(stream)
                headers={f:next((h for h in (f,catalogs[kind].get(f)) if h in reader.fieldnames),None) for f in fields}
                if not all(headers.values()): raise ValueError('Missing synthetic export columns: '+str(headers))
                rows=[{f:r[h] for f,h in headers.items()} for r in reader]
            if len(rows)!=result['artifact_row_count']: raise ValueError('Export row count mismatch')
            return rows
        started=time.monotonic()
        actual=await env['analyze'](cases,'Alpha',call,read_rows)
        expected=[(30,3,1.5,90),(60,6,3,360)]
        keys=('miles','transmission_gw','generation_gw','gw_miles')
        passed=(len(actual)==2 and all(row.get('case')==case and not row.get('error')
                  and tuple(row.get(k) for k in keys)==values
                  for row,case,values in zip(actual,cases,expected)))
        # Additional in-memory synthetic probes distinguish arithmetic success
        # from safe handling of absent entities, duplicate IDs and failed cases.
        branch={'BusNum':'1','BusNum:1':'2','LineCircuit':'1','BusNomVolt':'345','BusNomVolt:1':'345','LineLength':'10','LineAMVA':'1000','OwnerName':'Other','AreaName':'Alpha','AreaName:1':'Alpha'}
        gen={'BusNum':'1','GenID':'1','GenMWMax':'1500','OwnerName':'Other','AreaName':'Alpha'}
        probes=[]
        for label,branch_rows,gen_rows,expected_values,expect_error in [
            ('duplicate_ids',[branch,dict(branch)],[gen,dict(gen)],(10,1,1.5,10),False),
            ('no_entity',[dict(branch,AreaName='Other',**{'AreaName:1':'Other'})],[dict(gen,AreaName='Other')],(None,None,None,None),False),
            ('other_voltage',[dict(branch,BusNomVolt='138')],[gen],(0,0,1.5,0),False),
            ('nan',[dict(branch,LineLength='NaN')],[gen],None,True),
            ('infinity',[dict(branch,LineAMVA='inf')],[gen],None,True),
            ('missing_measurement',[{k:v for k,v in branch.items() if k!='LineLength'}],[gen],None,True),
            ('missing_generation_attribution',[branch],[dict(gen,AreaName='Other')],(10,1,None,10),False),
            ('missing_branch_attribution',[dict(branch,AreaName='Other',**{'AreaName:1':'Other'})],[gen],(None,None,1.5,None),False),
            ('conflicting_duplicate',[branch,dict(branch,LineLength='20')],[gen],None,True),
        ]:
            async def fake_call(name,args):
                return {'rows':branch_rows if args['object_type']=='Branch' else gen_rows}
            try:
                rows=await env['analyze'](['synthetic-memory'],'Alpha',fake_call,lambda r:r['rows'])
                row=rows[0]
                ok=bool(row.get('error')) if expect_error else tuple(row.get(k) for k in keys)==expected_values
                probes.append({'name':label,'passed':ok,'actual':row})
            except Exception as exc:
                probes.append({'name':label,'passed':False,'exception':str(exc)})
        async def recovery_call(name,args):
            if args['case_path']=='failed': raise ValueError('Injected synthetic backend failure')
            return {'rows':[branch] if args['object_type']=='Branch' else [gen]}
        try:
            recovered=await env['analyze'](['failed','healthy'],'Alpha',recovery_call,lambda r:r['rows'])
            ok=len(recovered)==2 and bool(recovered[0].get('error')) and tuple(recovered[1].get(k) for k in keys)==(10,1,1.5,10)
            probes.append({'name':'continue_after_failure','passed':ok})
        except Exception as exc:
            probes.append({'name':'continue_after_failure','passed':False,'exception':str(exc)})
        overall=passed and all(p['passed'] for p in probes)
        summary={'passed':overall,'plugin_case_test_passed':passed,'actual':actual,'expected':expected,'calls':calls,'synthetic_edge_probes':probes,
                 'execution_seconds':round(time.monotonic()-started,2),'real_cases_opened':0}
        (OUT/'result.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2),flush=True)
        if not overall: raise AssertionError('Model-authored script failed one or more synthetic checks')
    finally:
        await manager.stop_all()

if __name__=='__main__': asyncio.run(main())
