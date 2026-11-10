"""Exercise the real Grid query schema with synthetic arguments; execute no tools."""
import sys
import json
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parent/'Grid-Workshop'/'powerworld-aux-agent'))
from local_model_app.config import Settings
from local_model_app.gguf_model import GgufModel
from powerworld_aux_agent.object_query import QueryObjectsRequest
from jsonschema import validate

model=GgufModel(replace(Settings.from_environment(ROOT),model_id=sys.argv[1],model_kind='gguf',device='cpu',context_window=4096,reasoning_budget=0,max_new_tokens=512))
schema=QueryObjectsRequest.model_json_schema()
tools=[{'type':'function','function':{'name':'query_objects','description':'Query objects using explicit constraints.','parameters':schema}}]
try:
    for i in range(2):
        reply=model.chat([{'role':'user','content':
            r'Call query_objects exactly once for C:\synthetic\example.pwb, object_type Branch, fields ["ID","kV"], constraints [{"column":"kV","op":"eq","value":345}], limit 10. This is a synthetic formatting test; do not execute anything.'}],tools=tools,temperature=0)
        assert len(reply.tool_calls)==1, 'Expected one call'
        call=reply.tool_calls[0]['function']
        assert call['name']=='query_objects', 'Wrong tool'
        validate(call['arguments'],schema)
        assert call['arguments']['case_path']==r'C:\synthetic\example.pwb'
        assert call['arguments']['constraints']==[{'column':'kV','op':'eq','value':345}],call['arguments']
        print(json.dumps({'trial':i+1,'valid_schema':True,'correct_arguments':True,'seconds':reply.elapsed_seconds,'case_files_opened':0}),flush=True)
finally:
    model.close()
