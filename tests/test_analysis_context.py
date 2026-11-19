import asyncio
import copy
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from jsonschema import validate, ValidationError
from local_model_app.analysis_context import (contract_for,check_contract,step_schema,append_step,
    inspect_selection,remember,observation_context,partial_answer)
from local_model_app.analysis_plan import render
from tests.test_analysis_plan import CATALOGS,BRANCHES,branch_plan,gen_plan
from tests import test_analysis_plan as fixtures
from tests.test_tool_coordinator import FakeManager,FakeModel,make_tool
from tests.test_plan_execution import action
from local_model_app.tool_coordinator import ToolCoordinator
from local_model_app.scratchpad import Scratchpad


class ContextTests(unittest.TestCase):
    def test_contract_requires_objects_and_complete_case_coverage(self):
        state={'objective':'Save matching line objects in each case.'}
        contract=contract_for(state,['one.pwb','two.pwb'])
        program=branch_plan()
        with self.assertRaisesRegex(ValueError,'requires matching objects'): check_contract(contract,program,CATALOGS)
        program['outputs'].append({'label':'Lines','source':'voltage'})
        check_contract(contract,program,CATALOGS)
        with self.assertRaisesRegex(ValueError,'every case'): check_contract(contract,program,CATALOGS,{'cases':{}})
        with self.assertRaisesRegex(ValueError,'scope changed'): contract_for(state,['one.pwb'])

    def test_explicit_directory_rejects_fallback_repository(self):
        state={'objective':r'Find branches in "C:\requested cases"'}
        with self.assertRaisesRegex(ValueError,'outside'): contract_for(state,[r'C:\examples\synthetic.pwb'])
        self.assertEqual(contract_for(state,[r'C:\requested cases\nested\one.pwb'])['case_paths'],[r'C:\requested cases\nested\one.pwb'])

    def test_step_schema_only_accepts_compatible_existing_handles(self):
        program=gen_plan(); steps=program['steps'][:2]
        schema=step_schema(steps,CATALOGS)
        wrong_field=copy.deepcopy(program['steps'][0]); wrong_field['fields']=['InventedField']
        with self.assertRaises(ValidationError): validate({'step':wrong_field},schema)
        validate({'step':program['steps'][2]},schema)
        bad=copy.deepcopy(program['steps'][2]);bad['source']='units'
        with self.assertRaises(ValidationError): validate({'step':bad},schema)
        bad={'id':'bad','op':'aggregate','source':'average','method':'count'}
        with self.assertRaises(ValidationError): validate({'step':bad},schema)
        bad={'id':'bad','op':'aggregate','source':'units','method':'mean','field':'GenMWMax','assumed_unit':'GW'}
        with self.assertRaises(ValidationError): validate({'step':bad},schema)
        bad={'id':'bad','op':'filter','source':'units','filters':[{'field':'Bogus','op':'eq','value':1}]}
        with self.assertRaises(ValidationError): validate({'step':bad},schema)
        state={}; append_step(state,program['steps'][0],CATALOGS)
        with self.assertRaisesRegex(ValueError,'Duplicate'): append_step(state,program['steps'][0],CATALOGS)
        self.assertEqual(len(state['analysis_draft']),1)

    def test_object_outputs_have_complete_artifact_and_bounded_preview(self):
        with TemporaryDirectory() as directory:
            program=branch_plan(); program['outputs']=[{'label':'Matching lines','source':'voltage'}]
            state,_,_=fixtures.AnalysisTests().run_program(Path(directory),program)
            selection=state['cases']['one.pwb']['values']['voltage']
            page=inspect_selection(selection,fields=['BusNum'],limit=1)
            self.assertEqual(page,{'count':2,'rows':[{'BusNum':'1'}],'next_offset':1})
            self.assertEqual(len(json.loads(Path(selection['path']).read_text())),2)
            answer=render(program,state)
            self.assertIn('2 objects',answer); self.assertIn('BusNum',answer); self.assertIn(selection['path'],answer)
            with self.assertRaises(ValueError): inspect_selection(selection,limit=10000)

    def test_observations_survive_feedback_replacement_and_are_bounded(self):
        state={}
        for i in range(30): remember(state,'inspect',{'i':i},{'evidence':'x'*500})
        state['analysis_feedback']={'validation_error':'new error'}
        self.assertEqual(len(state['analysis_observations']),30)
        self.assertLess(len(json.dumps(observation_context(state))),6200)

    def test_pause_reports_available_evidence_without_claiming_absence(self):
        state={'objective':'Find managed branches.'}
        contract_for(state,['synthetic.pwb','unprocessed.pwb'])
        remember(state,'host__inspect_tags',{'case_index':1,'object_type':'Branch','text':'Synthetic Utility'},
                 {'matches':[{'field':'AreaName','value':'Synthetic Utility','count':7}]})
        answer=partial_answer(state,'The next action was invalid.')
        self.assertIn('AreaName = Synthetic Utility (7 objects)',answer)
        self.assertIn('does not establish ownership',answer)
        self.assertIn('Cases with completed analysis: 0',answer)
        self.assertIn('cannot be generalized to unprocessed cases',answer)
        remember(state,'host__inspect_tags',{'case_index':1,'object_type':'Branch','text':'Absent'}, {'matches':[]})
        self.assertIn('does not establish that the entity has no managed assets',partial_answer(state,'Paused.'))

    def test_download_only_serves_registered_selection_for_this_chat(self):
        from local_model_app.server import analysis_selection_artifact
        from fastapi import HTTPException
        with TemporaryDirectory() as directory:
            root=Path(directory); artifacts=root/'tool_activity'/'test.plan.results'/'selections'
            artifacts.mkdir(parents=True)
            name='a'*20+'.json'; target=artifacts/name; target.write_text('[]')
            (root/'tool_activity'/'test.plan.json').write_text(json.dumps({'analysis_execution':{'cases':{
                'synthetic.pwb':{'values':{'rows':{'kind':'selection','path':str(target)}}}}}}))
            service=SimpleNamespace(data_directory=root,store=SimpleNamespace(get_chat=lambda _:{}))
            with patch('local_model_app.server.runtime',return_value=service):
                response=asyncio.run(analysis_selection_artifact('test',name,None))
                self.assertEqual(Path(response.path),target.resolve())
                for bad in ('../secret.json','b'*20+'.json'):
                    with self.assertRaises(HTTPException): asyncio.run(analysis_selection_artifact('test',bad,None))

    def test_repeated_tag_search_reuses_evidence_then_recovers_without_duplicate_backend_call(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); case=str(root/'synthetic.pwb')
            search=make_tool('case.search_object_fields'); query=make_tool('case.query_objects')
            inspect=action('host__inspect_tags',case_index=1,object_type='Branch',text='Alpha')
            program=branch_plan();program['outputs'].append({'label':'Matching lines','source':'voltage'})
            model=FakeModel([inspect,inspect,inspect,*[action('host__add_step',step=s) for s in program['steps']],
                             action('host__finish_analysis',outputs=program['outputs'])])
            manager=FakeManager([search,query],{})
            async def call(name,args):
                manager.calls.append((name,args))
                if name==search.exposed_name: return {'structuredContent':{'matches':CATALOGS[args['object_type']]}}
                path=root/('rows'+str(len(manager.calls))+'.csv')
                with path.open('w',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=args['fields'],extrasaction='ignore');writer.writeheader();writer.writerows(BRANCHES)
                return {'structuredContent':{'artifact_complete':True,'artifact_path':str(path),'artifact_row_count':len(BRANCHES)}}
            manager.call_tool=call
            host=ToolCoordinator(model,manager,Scratchpad(root/'notes.jsonl'),root/'activity.jsonl')
            answer=asyncio.run(host.respond('Save matching Alpha branch objects at 345 kV in "'+case+'"',[],['grid.plugin']))
            self.assertIn('2 objects',answer)
            self.assertEqual(sum(name==query.exposed_name for name,args in manager.calls),2) # inspection + selection
            context=json.loads(model.chat_calls[3]['messages'][1]['content'])
            self.assertTrue(context['saved_observations'][0]['feedback']['matches'])
            offered=[t['function']['name'] for t in model.chat_calls[3]['tools']]
            self.assertNotIn('host__inspect_tags',offered)
            saved=json.loads((root/'activity.plan.json').read_text())
            self.assertEqual(saved['status'],'completed')
            self.assertEqual(saved['analysis_contract']['case_paths'],[case])

    def test_rejected_final_step_keeps_valid_prefix_for_direct_finish(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); case=str(root/'synthetic.pwb')
            search=make_tool('case.search_object_fields'); query=make_tool('case.query_objects')
            good=branch_plan(); bad=copy.deepcopy(good)
            bad['steps'].append({'id':'invalid_report','op':'aggregate','source':'answer','method':'count'})
            bad['outputs']=[{'label':'Wrong','source':'invalid_report'}]
            model=FakeModel([action('host__run_analysis',**bad),action('host__finish_analysis',outputs=good['outputs'])])
            manager=FakeManager([search,query],{})
            async def call(name,args):
                manager.calls.append((name,args))
                if name==search.exposed_name: return {'structuredContent':{'matches':CATALOGS[args['object_type']]}}
                path=root/'rows.csv'
                with path.open('w',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=args['fields'],extrasaction='ignore');writer.writeheader();writer.writerows(BRANCHES)
                return {'structuredContent':{'artifact_complete':True,'artifact_path':str(path),'artifact_row_count':len(BRANCHES)}}
            manager.call_tool=call
            host=ToolCoordinator(model,manager,Scratchpad(root/'notes.jsonl'),root/'activity.jsonl')
            answer=asyncio.run(host.respond('Sum Alpha branch miles at 345 kV in "'+case+'"',[],['grid.plugin']))
            self.assertIn('30 miles',answer)
            context=json.loads(model.chat_calls[1]['messages'][1]['content'])
            self.assertEqual(context['typed_handles']['answer']['unit'],'miles')
            self.assertEqual(len(context['saved_steps']),3)
            self.assertEqual(sum(name==query.exposed_name for name,args in manager.calls),1)


if __name__=='__main__': unittest.main()
