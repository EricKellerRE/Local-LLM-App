"""Small direct-GGUF tool-call benchmark; never opens a power-system case."""
import argparse
import json
import sys
import time
from dataclasses import replace, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_model_app.config import Settings
from local_model_app.gguf_model import GgufModel

parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()
settings = replace(Settings.from_environment(ROOT), model_id=args.model, model_kind='gguf',
                   device='cpu', context_window=4096, reasoning_budget=0, max_new_tokens=128)
model = GgufModel(settings, telemetry_path=ROOT/'data/gguf-bench/telemetry.jsonl')
try:
    start = time.monotonic()
    model.load()
    print(json.dumps({'loaded':True,'seconds':time.monotonic()-start,'model':Path(args.model).name}), flush=True)
    tools = [{'type':'function','function':{'name':'get_inventory','description':'Return the requested entity inventory.',
             'parameters':{'type':'object','properties':{'entity':{'type':'string'}},'required':['entity']}}}]
    messages = [{'role':'system','content':(ROOT/'config/guidance/grid-analysis.md').read_text(encoding='utf-8')},
                {'role':'user','content':'Call get_inventory with entity Alpha. Do not guess the inventory.'}]
    print(json.dumps({'prompt_tokens':model.count_prompt_tokens(messages, tools)}), flush=True)
    reply = model.chat(messages, tools=tools, temperature=0, generation_class='gguf_tool_smoke')
    print(json.dumps(asdict(reply)), flush=True)
    assert len(reply.tool_calls)==1 and reply.tool_calls[0]['function']['name']=='get_inventory'
    assert reply.tool_calls[0]['function']['arguments']=={'entity':'Alpha'}
    messages += [{'role':'assistant','content':reply.content,'tool_calls':reply.tool_calls},
                 {'role':'tool','tool_call_id':reply.tool_calls[0]['id'], 'content':'{"miles":30,"rating_mva":3000}'},
                 {'role':'user','content':'At unity power factor, convert rating_mva to GW by dividing by 1000, then multiply total miles by that GW rating. Return JSON only with numeric keys rating_gw and gw_miles.'}]
    reply = model.chat(messages, tools=tools, temperature=0, generation_class='gguf_result_smoke')
    print(json.dumps(asdict(reply)), flush=True)
    answer = json.loads(reply.content.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
    assert answer == {'rating_gw':3, 'gw_miles':90}, answer
finally:
    model.close()
