import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from local_model_app.grid_selection import CORE, select_objects,calculate_selection,render_selection,matches_company
from tests.test_grid_metrics import export
from tests.test_plan_execution import action
from tests.test_tool_coordinator import FakeManager,FakeModel,make_tool
from local_model_app.tool_coordinator import ToolCoordinator
from local_model_app.scratchpad import Scratchpad


def catalog(kind):
    if kind=='Bus': return [{'variable_name':'BusNum','concise_name':'BusNum'}]
    return [{'variable_name':name,'concise_name':name} for name in CORE[kind]+['OwnerName','AreaName']]


class SelectionTests(unittest.TestCase):
    def test_directory_selection_then_calculation_never_requeries(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            b=export(root/'b.csv',','.join(CORE['Branch']+['OwnerName','AreaName'])+'\n1,2,1,345,345,10,1000,Other,Alpha Utility\n2,3,1,138,138,99,9000,Alpha Utility,Other\n')
            g=export(root/'g.csv',','.join(CORE['Gen']+['OwnerName','AreaName'])+'\n1,1,1500,Alpha Utility,Other\n')
            calls=[]
            async def call(args): calls.append(args); return b if args['object_type']=='Branch' else g
            selection=asyncio.run(select_objects(['one.pwb','two.pwb'],'Alpha Utility',{k:catalog(k) for k in CORE},call,root/'selected',lambda *x:None))
            self.assertEqual(len(calls),4)
            reports=calculate_selection(selection)
            self.assertEqual(len(calls),4)
            self.assertEqual(len(reports),2)
            for row in reports:
                self.assertEqual([row[k] for k in ('miles','transmission','generation','gw_miles')],[10,1,1.5,10])
            self.assertIn('Cases discovered: 2. Completed: 2.',render_selection(selection,reports))
            self.assertFalse(matches_company('ALPHABET','Alpha'))

    def test_empty_and_failed_cases_stay_in_report(self):
        manifest={'company':'Alpha','cases':[{'case':'a.pwb','objects':{'Branch':{'count':0},'Gen':{'error':'Synthetic failure'}}}]}
        reports=calculate_selection(manifest)
        self.assertIsNone(reports[0]['miles'])
        self.assertIn('Unknown',render_selection(manifest,reports))
        self.assertIn('Unresolved or failed: 1',render_selection(manifest,reports))

    def test_coordinator_context_has_no_object_or_tag_payloads(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); case=str(root/'synthetic.pwb'); second=str(root/'synthetic2.pwb')
            b=export(root/'b.csv',','.join(CORE['Branch']+['OwnerName','AreaName'])+'\n1,2,1,345,345,10,1000,SECRET_UNRELATED_TAG,Alpha Utility\n')
            g=export(root/'g.csv',','.join(CORE['Gen']+['OwnerName','AreaName'])+'\n1,1,1500,Alpha Utility,SECRET_UNRELATED_TAG\n')
            query=make_tool('case.query_objects'); search=make_tool('case.search_object_fields')
            manager=FakeManager([query,search],{})
            async def call(name,args):
                manager.calls.append((name,args))
                if name==search.exposed_name: return {'structuredContent':{'matches':catalog(args['object_type'])}}
                if args['case_path']==second:
                    saved=json.loads((root/'activity.plan.json').read_text())
                    self.assertEqual(saved['analysis_execution']['cases'][case]['status'],'completed')
                return b if args['object_type']=='Branch' else g
            manager.call_tool=call
            from tests.test_analysis_plan import branch_plan
            program=branch_plan(); program['steps'][0]['match']['text']='Alpha Utility'
            model=FakeModel([action('host__run_analysis',**program)])
            host=ToolCoordinator(model,manager,Scratchpad(root/'notes.jsonl'),root/'activity.jsonl')
            answer=asyncio.run(host.respond('Analyze Alpha Utility 345 kV GW metrics for "'+case+'" and "'+second+'"',[],['grid.plugin']))
            self.assertIn('10 miles | completed',answer)
            self.assertEqual(len(model.chat_calls),1)
            self.assertIn('synthetic2.pwb',answer)
            context=json.dumps(model.chat_calls)
            self.assertNotIn('SECRET_UNRELATED_TAG',context)
            self.assertLess(len(context),18000)
