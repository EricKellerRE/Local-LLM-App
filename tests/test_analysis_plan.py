import asyncio
import copy
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from local_model_app.analysis_plan import Executor, validate_plan, render, compile_plan
from local_model_app.cancellation import ResponseStopped, stop_event
from tests.test_plan_execution import action
from tests.test_tool_coordinator import FakeManager, FakeModel, make_tool
from local_model_app.tool_coordinator import ToolCoordinator
from local_model_app.scratchpad import Scratchpad

FIELDS={'Branch':['BusNum','BusNum:1','LineCircuit','BusNomVolt','BusNomVolt:1','LineLength','LineAMVA','AreaName','OwnerName'],
        'Gen':['BusNum','GenID','GenMWMax','GenMW','AreaName','OwnerName'], 'Bus':['BusNum','BusNomVolt','AreaName']}
CATALOGS={kind:[{'variable_name':f,'concise_name':f,'description':{'GenMWMax':'Maximum MW','GenMW':'Dispatch MW','LineAMVA':'Rating MVA','LineLength':'Length'}.get(f,'')} for f in fields] for kind,fields in FIELDS.items()}
BRANCHES=[dict(zip(FIELDS['Branch'],r)) for r in [
    ['1','2','1','345','345','10','1000','Alpha','Other'],
    ['2','3','1','345','345','20','2000','Alpha','Other'],
    ['4','5','1','138','138','99','9000','Alpha','Other'],
    ['6','7','1','345','345','40','4000','Beta','Other']]]
GENERATORS=[dict(zip(FIELDS['Gen'],r)) for r in [['1','1','700','100','Alpha','Other'],['2','1','800','200','Alpha','Other'],['6','1','9000','400','Beta','Other']]]

def branch_plan(voltage=345,method='sum'):
    return {'steps':[
        {'id':'lines','op':'select','object_type':'Branch','fields':['LineLength','BusNomVolt','BusNomVolt:1'],'match':{'text':'Alpha'}},
        {'id':'voltage','op':'filter','source':'lines','filters':[{'field':f,'op':'eq','value':voltage} for f in ('BusNomVolt','BusNomVolt:1')]},
        {'id':'answer','op':'aggregate','source':'voltage','method':method,**({'field':'LineLength','assumed_unit':'miles','assumption':'Synthetic fixture length unit is miles.'} if method!='count' else {})}],
        'outputs':[{'label':'Requested metric','source':'answer'}]}

def gen_plan():
    return {'steps':[{'id':'units','op':'select','object_type':'Gen','fields':['GenMWMax'],'match':{'text':'Alpha'}},
                     {'id':'average','op':'aggregate','source':'units','method':'mean','field':'GenMWMax'},
                     {'id':'gw','op':'convert','source':'average','unit':'GW'}],
            'outputs':[{'label':'Average installed capacity','source':'gw'}]}

class AnalysisTests(unittest.TestCase):
    def test_compiler_omits_unneeded_work_but_preserves_needed_validation(self):
        program=gen_plan()
        program['steps'].append({'id':'unused','op':'aggregate','source':'gw','method':'sum','field':'bad'})
        self.assertEqual(len(compile_plan(program,CATALOGS)['steps']),3)
        program['outputs'].append({'label':'Needed','source':'unused'})
        with self.assertRaisesRegex(ValueError,'selection'): compile_plan(program,CATALOGS)
    def run_program(self,root,program,state=None,fail_case=None,duplicates=False):
        calls=[]; state={} if state is None else state; saves=[]
        async def call(args):
            calls.append(args)
            if args['case_path']==fail_case: raise ValueError('Synthetic backend error')
            if args['case_path']=='two.pwb': self.assertIn(state['cases']['one.pwb']['status'],('completed','failed'))
            rows=copy.deepcopy(BRANCHES if args['object_type']=='Branch' else GENERATORS)
            if duplicates: rows.append(copy.deepcopy(rows[0]))
            if args['case_path']=='two.pwb':
                for row in rows:
                    for f in ('LineLength','GenMWMax'):
                        if f in row: row[f]=str(float(row[f])*2)
            path=root/('export'+str(len(calls))+'.csv')
            with path.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=args['fields'],extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
            return {'artifact_complete':True,'artifact_path':str(path),'artifact_row_count':len(rows)}
        executor=Executor(program,CATALOGS,call,root/'selections',lambda:saves.append(copy.deepcopy(state)),lambda *a:None)
        asyncio.run(executor.run(['one.pwb','two.pwb'],state))
        return state,calls,saves

    def test_different_questions_and_case_sequential_execution(self):
        for program,expected in [(branch_plan(),[30,60]),(branch_plan(138,'count'),[1,1]),(branch_plan(345,'max'),[20,40]),(gen_plan(),[.75,1.5])]:
            with self.subTest(program=program),TemporaryDirectory() as directory:
                state,calls,saves=self.run_program(Path(directory),program)
                source=program['outputs'][0]['source']
                self.assertEqual([r['values'][source]['value'] for r in state['cases'].values()],expected)
                self.assertEqual(len(calls),2)
                self.assertTrue(saves)

    def test_deduplication_and_resume_do_not_repeat_queries(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); state,calls,_=self.run_program(root,branch_plan(),duplicates=True)
            self.assertEqual(state['cases']['one.pwb']['values']['answer']['value'],30)
            state,calls,_=self.run_program(root,branch_plan(),state)
            self.assertEqual(calls,[])

    def test_validation_precedes_execution(self):
        bad=branch_plan(); bad['steps'][0]['fields'].append('ImaginaryField')
        with self.assertRaisesRegex(ValueError,'Unknown Branch'): validate_plan(bad,CATALOGS)
        bad=gen_plan(); bad['steps'][-1]['source']='not_a_variable'
        with self.assertRaisesRegex(ValueError,'earlier quantities'): validate_plan(bad,CATALOGS)
        bad=gen_plan(); bad['steps'][-1]['unit']='miles'
        with self.assertRaisesRegex(ValueError,'incompatible'): validate_plan(bad,CATALOGS)
        bad=branch_plan(); bad['steps'][2]['assumed_unit']='GW'; bad['steps'].append({'id':'converted','op':'convert','source':'answer','unit':'miles'})
        with self.assertRaises(ValueError): validate_plan(bad,CATALOGS)

    def test_unmatched_entity_unknown_but_voltage_absence_zero(self):
        with TemporaryDirectory() as directory:
            program=branch_plan(); program['steps'][0]['match']['text']='Absent'
            state,_,_=self.run_program(Path(directory),program)
            self.assertIsNone(state['cases']['one.pwb']['values']['answer']['value'])
            self.assertIn('unresolved',render(program,state))
        with TemporaryDirectory() as directory:
            state,_,_=self.run_program(Path(directory),branch_plan(500))
            self.assertEqual(state['cases']['one.pwb']['values']['answer']['value'],0)

    def test_failed_case_is_reported_and_next_case_runs(self):
        with TemporaryDirectory() as directory:
            state,_,_=self.run_program(Path(directory),branch_plan(),fail_case='one.pwb')
            self.assertEqual(state['cases']['one.pwb']['status'],'failed')
            self.assertEqual(state['cases']['two.pwb']['status'],'completed')
            self.assertIn('Synthetic backend error',render(branch_plan(),state))

    def test_stop_and_resume_preserve_completed_selection(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); state={}; event=Event(); calls=[]
            async def call(args):
                calls.append(args)
                path=root/'rows.csv'
                with path.open('w',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=args['fields'],extrasaction='ignore');writer.writeheader();writer.writerows(BRANCHES)
                return {'artifact_path':str(path),'artifact_complete':True,'artifact_row_count':len(BRANCHES)}
            def save():
                if state.get('cases',{}).get('one.pwb',{}).get('values',{}).get('lines'): event.set()
            executor=Executor(branch_plan(),CATALOGS,call,root/'selections',save,lambda *a:None)
            token=stop_event.set(event)
            try:
                with self.assertRaises(ResponseStopped): asyncio.run(executor.run(['one.pwb'],state))
                event.clear(); executor.save=lambda:None
                asyncio.run(executor.run(['one.pwb'],state))
            finally: stop_event.reset(token)
            self.assertEqual(len(calls),1)
            self.assertEqual(state['cases']['one.pwb']['values']['answer']['value'],30)

    def test_conversion_and_composed_quantity(self):
        program=branch_plan()
        program['steps'][0]['fields'].append('LineAMVA')
        program['steps'] += [
            {'id':'rating','op':'aggregate','source':'voltage','method':'sum','field':'LineAMVA'},
            {'id':'gw','op':'convert','source':'rating','unit':'GW','power_factor':1},
            {'id':'product','op':'calculate','left':'answer','right':'gw','method':'multiply'}]
        program['outputs']=[{'label':'GW miles','source':'product'}]
        with TemporaryDirectory() as directory:
            state,_,_=self.run_program(Path(directory),program)
            self.assertEqual(state['cases']['one.pwb']['values']['product']['value'],90)
            self.assertEqual(state['cases']['one.pwb']['values']['product']['unit'],'miles × GW')
        del program['steps'][-2]['power_factor']
        with self.assertRaisesRegex(ValueError,'power_factor'): validate_plan(program,CATALOGS)

    def test_malformed_filter_rejected_before_execution(self):
        program=branch_plan(); program['steps'][1]['filters'][0]['value']=[345]
        with self.assertRaisesRegex(ValueError,'array'): validate_plan(program,CATALOGS)

    def test_known_unit_confirmation_does_not_require_assumption(self):
        program=gen_plan(); program['steps'][1]['assumed_unit']='MW'; program['steps'][2]['power_factor']=1
        self.assertEqual(validate_plan(program,CATALOGS)['gw']['unit'],'GW')
        program['steps'][2]['power_factor']=.5
        with self.assertRaises(ValueError): validate_plan(program,CATALOGS)

    def test_mutating_requests_keep_general_planner(self):
        from types import SimpleNamespace
        from local_model_app.analysis_runner import applies
        tool=make_tool('case.query_objects')
        plan=SimpleNamespace(tools={'query':tool},state={'objective':'Solve synthetic.pwb'})
        self.assertFalse(applies(plan))
        plan.state['objective']='Count branches in synthetic.pwb. Do not solve or modify.'
        self.assertTrue(applies(plan))

    def test_model_invalid_field_is_corrected_without_querying_it(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); case=str(root/'synthetic.pwb')
            search=make_tool('case.search_object_fields'); query=make_tool('case.query_objects')
            bad=gen_plan(); bad['steps'][0]['fields'].append('BadField')
            model=FakeModel([action('host__run_analysis',**bad),action('host__run_analysis',**gen_plan())])
            manager=FakeManager([search,query],{})
            async def call(name,args):
                manager.calls.append((name,args))
                if name==search.exposed_name: return {'structuredContent':{'matches':CATALOGS[args['object_type']]}}
                self.assertNotIn('BadField',args['fields'])
                if sum(n==query.exposed_name for n,a in manager.calls)==1:
                    return {'isError':True,'message':'503 temporarily unavailable (synthetic)'}
                path=root/'g.csv'
                with path.open('w',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=args['fields'],extrasaction='ignore');writer.writeheader();writer.writerows(GENERATORS)
                return {'structuredContent':{'artifact_path':str(path),'artifact_complete':True,'artifact_row_count':len(GENERATORS)}}
            manager.call_tool=call
            host=ToolCoordinator(model,manager,Scratchpad(root/'notes.jsonl'),root/'activity.jsonl')
            answer=asyncio.run(host.respond('Average installed generation capacity in GW for Alpha in "'+case+'"',[],['grid.plugin']))
            self.assertIn('0.75 GW',answer)
            self.assertEqual(sum(name==query.exposed_name for name,args in manager.calls),2)
            self.assertIn('Unknown Gen fields',model.chat_calls[1]['messages'][1]['content'])
            self.assertNotIn('grid_selection_runner',answer)

if __name__=='__main__': unittest.main()
