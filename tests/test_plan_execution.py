import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from local_model_app.plan_execution import PlanExecution
from local_model_app.tool_coordinator import ToolCoordinator
from local_model_app.scratchpad import Scratchpad
from local_model_app.model import AssistantReply
from tests.test_tool_coordinator import FakeManager, FakeModel, make_tool


def action(tool_name, **args):
    return AssistantReply(content='',tool_calls=[{'id':'synthetic','type':'function',
        'function':{'name':tool_name,'arguments':args}}])


def step(name, tool, outputs, depends=(), inputs=()):
    return {'id':name,'goal':'Synthetic '+name,'tools':[tool],'depends_on':list(depends),
            'inputs':list(inputs),'outputs':outputs}


def bind(name, result, pointer):
    return action('host__bind_variable',name=name,result=result,pointer=pointer)


class PlanTests(unittest.TestCase):
    def test_rejected_unchanged_repair_is_withheld_next_turn(self):
        tool=make_tool('query')
        model=FakeModel([
            action('host__set_plan',steps=[step('query',tool.exposed_name,['value'])]),
            action(tool.exposed_name),
            action('host__repair_step',tools=[tool.exposed_name],reason='Synthetic retry'),
            action('host__complete_step',bindings=[{'name':'value','result':'r1','pointer':'/structuredContent/value'}]),
            AssistantReply(content='Synthetic done.',tool_calls=[])])
        manager=FakeManager([tool],{tool.exposed_name:{'structuredContent':{'value':42}}})
        with TemporaryDirectory() as directory:
            host=self.setup_run(Path(directory),model,manager)
            self.assertEqual(asyncio.run(host.respond('Synthetic',[],['grid.plugin'])),'Synthetic done.')
        self.assertNotIn('host__repair_step',[s['function']['name'] for s in model.chat_calls[3]['tools']])
        self.assertIn('already-enabled',json.loads(model.chat_calls[3]['messages'][1]['content'])['feedback']['next'])
        self.assertEqual(len(manager.calls),1)

    def test_repeated_query_recovers_using_saved_observation(self):
        tool=make_tool('query')
        model=FakeModel([
            action('host__set_plan',steps=[step('query',tool.exposed_name,['value'])]),
            action(tool.exposed_name),action(tool.exposed_name),
            action('host__complete_step',bindings=[{'name':'value','result':'r1','pointer':'/structuredContent/value'}]),
            AssistantReply(content='Synthetic done.',tool_calls=[])])
        manager=FakeManager([tool],{tool.exposed_name:{'structuredContent':{'value':42}}})
        with TemporaryDirectory() as directory:
            host=self.setup_run(Path(directory),model,manager)
            self.assertEqual(asyncio.run(host.respond('Synthetic',[],['grid.plugin'])),'Synthetic done.')
        recovery=model.chat_calls[3]
        self.assertNotIn(tool.exposed_name,[s['function']['name'] for s in recovery['tools']])
        context=json.loads(recovery['messages'][1]['content'])
        self.assertEqual(context['feedback']['cached_result'],'r1')
        self.assertIn('42',str(context['feedback']['preview']))
        self.assertEqual(context['action_recovery']['tool'],tool.exposed_name)
        self.assertEqual(len(manager.calls),1)

    def test_repeated_read_retains_observation_and_requires_another_action(self):
        tool=make_tool('query')
        read=action('host__read_variable',name='r1',pointer='/structuredContent/value',offset=0,limit=1)
        model=FakeModel([
            action('host__set_plan',steps=[step('query',tool.exposed_name,['value'])]),
            action(tool.exposed_name),read,read,
            bind('value','r1','/structuredContent/value'),action('host__complete_step'),
            AssistantReply(content='Synthetic done.',tool_calls=[])])
        manager=FakeManager([tool],{tool.exposed_name:{'structuredContent':{'value':42}}})
        with TemporaryDirectory() as directory:
            host=self.setup_run(Path(directory),model,manager)
            self.assertEqual(asyncio.run(host.respond('Synthetic',[],['grid.plugin'])),'Synthetic done.')
        self.assertNotIn('host__read_variable',[s['function']['name'] for s in model.chat_calls[4]['tools']])
        self.assertEqual(json.loads(model.chat_calls[4]['messages'][1]['content'])['feedback']['value'],'42')
        self.assertNotIn('host__read_variable',[s['function']['name'] for s in model.chat_calls[5]['tools']])
        self.assertEqual(len(manager.calls),1)

    def test_reference_paths_are_paired_with_their_values(self):
        from jsonschema import validate, ValidationError
        tool=make_tool('query',required=['case_path'])
        with TemporaryDirectory() as directory:
            host=self.setup_run(Path(directory),FakeModel([]),FakeManager([tool],{}))
            executor=PlanExecution(host,[tool])
            executor.state={'variables':{'artifact':{'result':'r1','pointer':'/structuredContent/path'}},
                'observations':{'r1':{'ok':True,'tool':tool.exposed_name},'r2':{'ok':True,'tool':tool.exposed_name}}}
            values={'artifact':'synthetic.csv','r1':{'structuredContent':{'path':'synthetic.csv'}},'r2':{'structuredContent':{'count':2}}}
            executor.read=lambda name: values[name]
            current=step('query',tool.exposed_name,['out'],inputs=['artifact'])
            current['results']=['r1','r2']
            schemas={s['function']['name']:s['function']['parameters'] for s in executor.specs(current)}
            validate({'case_path':{'$var':'artifact','pointer':''}},schemas[tool.exposed_name])
            with self.assertRaises(ValidationError):
                validate({'case_path':{'$var':'artifact','pointer':'/structuredContent/path'}},schemas[tool.exposed_name])
            with self.assertRaises(ValidationError):
                validate({'name':'out','result':'r1','pointer':'/structuredContent/count'},schemas['host__bind_variable'])
            validate({'name':'out','result':'r2','pointer':'/structuredContent/count'},schemas['host__bind_variable'])
            with self.assertRaises(ValidationError):
                validate({'name':'artifact','pointer':'/structuredContent','offset':0,'limit':1},schemas['host__read_variable'])

    def test_explicit_json_keys_are_enforced(self):
        from local_model_app.plan_execution import requested_json_report
        from jsonschema import validate, ValidationError
        objective='Return a JSON object whose numeric keys are miles, transmission_gw, generation_gw, gw_miles, plus assumption.'
        schema=requested_json_report(objective)['function']['parameters']
        expected={'miles':30,'transmission_gw':3,'generation_gw':1.5,'gw_miles':90,'assumption':'PF=1'}
        validate(expected,schema)
        self.assertEqual(set(schema['properties']),set(expected))
        with self.assertRaises(ValidationError): validate({'30':30,'3':3,'1.5':1.5,'90':90},schema)
        tool=make_tool('query')
        model=FakeModel([action('host__set_plan',steps=[step('query',tool.exposed_name,['value'])]),
            action(tool.exposed_name),action('host__complete_step',bindings=[{'name':'value','result':'r1','pointer':'/structuredContent'}]),
            action('host__report_json',**expected)])
        with TemporaryDirectory() as directory:
            host=self.setup_run(Path(directory),model,FakeManager([tool],{tool.exposed_name:{'structuredContent':expected}}))
            self.assertEqual(json.loads(asyncio.run(host.respond(objective,[],['grid.plugin']))),expected)
        self.assertEqual(model.chat_calls[-1]['generation_class'],'required_tool_action')

    def test_units_survive_misleading_variable_name_and_conversion(self):
        tool=make_tool('query')
        model=FakeModel([
            action('host__set_plan',steps=[step('query',tool.exposed_name,['generation_gw']),
                step('convert','host__convert_units',['converted_gw'],['query'],['generation_gw'])]),
            action(tool.exposed_name),
            action('host__complete_step',bindings=[{'name':'generation_gw','result':'r1',
                'pointer':'/structuredContent/numeric_columns/InstalledMW/sum'}]),
            action('host__convert_units',name='generation_gw',pointer='',target_unit='GW'),
            action('host__complete_step',bindings=[{'name':'converted_gw','result':'r2','pointer':'/results/value'}]),
            AssistantReply(content='Synthetic done.',tool_calls=[])])
        manager=FakeManager([tool],{tool.exposed_name:{'structuredContent':{'numeric_columns':{'InstalledMW':{'sum':1500}}}}})
        with TemporaryDirectory() as directory:
            host=self.setup_run(Path(directory),model,manager)
            self.assertEqual(asyncio.run(host.respond('Synthetic',[],['grid.plugin'])),'Synthetic done.')
            saved=json.loads(host.activity_path.with_suffix('.plan.json').read_text())
            self.assertEqual(saved['variable_units']['generation_gw']['unit'],'MW')
            self.assertEqual(saved['variable_units']['generation_gw']['equivalent']['value'],1.5)
            self.assertEqual(saved['variable_units']['converted_gw']['unit'],'GW')
            self.assertEqual(saved['variable_units']['converted_gw']['value'],1.5)
        context=json.loads(model.chat_calls[3]['messages'][1]['content'])
        self.assertEqual(context['unit_report']['generation_gw']['unit'],'MW')

    def test_rejected_csv_columns_constrain_retry_schema(self):
        from jsonschema import validate, ValidationError
        tool=make_tool('summarize')
        with TemporaryDirectory() as directory:
            host=self.setup_run(Path(directory),FakeModel([]),FakeManager([tool],{}))
            executor=PlanExecution(host,[tool])
            current=step('query',tool.exposed_name,['value'])
            current.update(results=['r1'])
            executor.state={'variables':{},'observations':{'r1':{'tool':tool.exposed_name,'ok':False,
                'arguments':{'path':'synthetic.csv'}}}}
            executor.read=lambda name:{'detail':"Unknown CSV columns: ['Voltage']; available: ['Entity', 'kV']"}
            original=tool.openai_spec()
            original['function']['parameters']={'type':'object','properties':{
                'path':{'type':'string'},'metric':{'type':'string'},'filters':{'type':'array','items':{
                    'type':'object','properties':{'column':{'type':'string'}}}}}}
            with patch.object(type(tool),'openai_spec',return_value=original):
                schema=executor.specs(current)[0]['function']['parameters']
            validate({'path':'synthetic.csv','filters':[{'column':'kV'}]},schema)
            with self.assertRaises(ValidationError):
                validate({'path':'synthetic.csv','filters':[{'column':'Voltage'}]},schema)
            with self.assertRaises(ValidationError):
                validate({'path':'other.csv','metric':'kV'},schema)

    def test_paginated_read_exposes_bindable_paths_beyond_preview(self):
        tool=make_tool('query')
        rows=[{'value':index} for index in range(30)]
        model=FakeModel([
            action('host__set_plan',steps=[step('query',tool.exposed_name,['last'])]),
            action(tool.exposed_name),
            action('host__read_variable',name='r1',pointer='/structuredContent/rows',offset=29,limit=1),
            bind('last','r1','/structuredContent/rows/29/value'),action('host__complete_step'),
            AssistantReply(content='Synthetic done.',tool_calls=[])])
        manager=FakeManager([tool],{tool.exposed_name:{'structuredContent':{'rows':rows}}})
        with TemporaryDirectory() as directory:
            host=self.setup_run(Path(directory),model,manager)
            self.assertEqual(asyncio.run(host.respond('Synthetic',[],['grid.plugin'])),'Synthetic done.')
            saved=json.loads(host.activity_path.with_suffix('.plan.json').read_text())
            self.assertEqual(saved['variables']['last']['pointer'],'/structuredContent/rows/29/value')
        self.assertEqual(len(manager.calls),1)

    def test_pointer_math_and_duplicate_bindings(self):
        tool=make_tool('query')
        model=FakeModel([
            action('host__set_plan',steps=[step('query',tool.exposed_name,['summary']),
                step('math','host__calculate',['metrics'],['query'],['summary'])]),
            action(tool.exposed_name),
            action('host__complete_step',bindings=[{'name':'summary','result':'r1','pointer':'/structuredContent'}]),
            action('host__calculate',expressions={'gw':'999 / 1000'}),
            action('host__calculate',expressions={'gw':'${summary#/mva} / 1000','product':'${summary#/miles} * ${summary#/mva} / 1000'}),
            action('host__complete_step',bindings=[{'name':'metrics','result':'r2','pointer':'/results/gw'},
                {'name':'metrics','result':'r2','pointer':'/results/product'}]),
            action('host__complete_step',bindings=[{'name':'metrics','result':'r2','pointer':'/results'}]),
            AssistantReply(content='Synthetic done.',tool_calls=[])])
        manager=FakeManager([tool],{tool.exposed_name:{'structuredContent':{'mva':3000,'miles':30}}})
        with TemporaryDirectory() as directory:
            host=self.setup_run(Path(directory),model,manager)
            self.assertEqual(asyncio.run(host.respond('Synthetic',[],['grid.plugin'])),'Synthetic done.')
            executor=PlanExecution(host,[tool])
            executor.state=json.loads(executor.path.read_text())
            executor.results=executor.results/executor.state['result_namespace']
            self.assertEqual(executor.read('metrics'),{'gw':3,'product':90})
            self.assertEqual(len(executor.state['observations']),2)

    def setup_run(self, root, model, manager):
        return ToolCoordinator(model,manager,Scratchpad(root/'notes.jsonl'),root/'activity.jsonl')

    def test_complete_step_binds_outputs_atomically(self):
        tool=make_tool('query')
        model=FakeModel([action('host__set_plan',steps=[step('query',tool.exposed_name,['a','b'])]),
            action(tool.exposed_name),action('host__complete_step',bindings=[
                {'name':'a','result':'r1','pointer':'/structuredContent/a'},
                {'name':'b','result':'r1','pointer':'/structuredContent/missing'}]),
            action('host__complete_step',bindings=[
                {'name':'a','result':'r1','pointer':'/structuredContent/a'},
                {'name':'b','result':'r1','pointer':'/structuredContent/b'}]),
            AssistantReply(content='Synthetic completed.',tool_calls=[])])
        manager=FakeManager([tool],{tool.exposed_name:{'structuredContent':{'a':1,'b':2}}})
        with TemporaryDirectory() as directory:
            host=self.setup_run(Path(directory),model,manager)
            self.assertEqual(asyncio.run(host.respond('Synthetic',[],['grid.plugin'])),'Synthetic completed.')
            saved=json.loads(host.activity_path.with_suffix('.plan.json').read_text())
            self.assertEqual(set(saved['variables']),{'a','b'})
        self.assertEqual(len(manager.calls),1)

    def test_current_step_cannot_execute_unlisted_tool(self):
        first=make_tool('first');other=make_tool('other')
        model=FakeModel([action('host__set_plan',steps=[step('first',first.exposed_name,['value'])]),
            *[action(other.exposed_name) for _ in range(3)]])
        manager=FakeManager([first,other],{})
        with TemporaryDirectory() as directory:
            host=self.setup_run(Path(directory),model,manager)
            self.assertIn('paused',asyncio.run(host.respond('Synthetic',[],['grid.plugin'])))
        self.assertFalse(manager.calls)

    def test_unknown_inflight_call_is_not_replayed_after_restart(self):
        from local_model_app.cancellation import ResponseStopped
        tool=make_tool('query')
        class Interrupted(FakeManager):
            async def call_tool(self,name,args):
                self.calls.append((name,args))
                raise ResponseStopped('Synthetic interruption')
        manager=Interrupted([tool],{})
        with TemporaryDirectory() as directory:
            root=Path(directory)
            model=FakeModel([action('host__set_plan',steps=[step('query',tool.exposed_name,['value'])]),action(tool.exposed_name)])
            host=self.setup_run(root,model,manager)
            with self.assertRaises(ResponseStopped): asyncio.run(host.respond('Synthetic',[],['grid.plugin']))
            resumed=self.setup_run(root,FakeModel([]),manager)
            self.assertIn('outcome still unknown',asyncio.run(resumed.respond('Resume',[],['grid.plugin'])))
            self.assertEqual(len(manager.calls),1)

    def test_atomic_checkpoint_retries_transient_sharing_lock(self):
        from local_model_app.atomic_files import write_json
        with TemporaryDirectory() as directory:
            path=Path(directory)/'state.json'; original=Path.replace; attempts=[]
            def replace(source,target):
                attempts.append(1)
                if len(attempts)==1: raise PermissionError('Synthetic sharing lock')
                return original(source,target)
            with patch.object(Path,'replace',replace): write_json(path,{'value':42})
            self.assertEqual(json.loads(path.read_text()),{'value':42})
            self.assertEqual(len(attempts),2)

    def test_evidence_variables_dependencies_math_and_small_context(self):
        discovery=make_tool('discover')
        query=make_tool('query',required=['case_path'])
        plan=[step('discover',discovery.exposed_name,['case_path']),
              step('query',query.exposed_name,['miles','gw'],['discover'],['case_path']),
              step('math','host__calculate',['gw_miles'],['query'],['miles','gw'])]
        model=FakeModel([
            action('host__set_plan',steps=plan),action(discovery.exposed_name),
            action(discovery.exposed_name), # Cached, not re-executed.
            bind('case_path','r1','/structuredContent/path'),action('host__complete_step'),
            action(query.exposed_name,case_path={'$var':'case_path'}),
            bind('miles','r2','/structuredContent/miles'),bind('gw','r2','/structuredContent/gw'),
            action('host__complete_step'),action('host__calculate',expressions={'product':'${miles} * ${gw}'}),
            bind('gw_miles','r3','/results/product'),action('host__complete_step'),
            AssistantReply(content='Synthetic product: 90.',tool_calls=[]),
        ])
        manager=FakeManager([discovery,query],{
            discovery.exposed_name:{'structuredContent':{'path':r'C:\synthetic\fixture.pwb','bulk':'Z'*100000}},
            query.exposed_name:{'structuredContent':{'miles':30,'gw':3}},
        })
        with TemporaryDirectory() as directory:
            root=Path(directory)
            host=self.setup_run(root,model,manager)
            answer=asyncio.run(host.respond('Analyze synthetic fixture',[],['grid.plugin']))
            saved=json.loads(host.activity_path.with_suffix('.plan.json').read_text())
            executor=PlanExecution(host,[discovery,query]);executor.state=saved
            executor.results=executor.results/saved['result_namespace']
            self.assertEqual(executor.read('gw_miles'),90)
            self.assertEqual(saved['status'],'completed')
            self.assertFalse(host._state_path.exists())
        self.assertEqual(answer,'Synthetic product: 90.')
        self.assertEqual(manager.calls,[(discovery.exposed_name,{}),(query.exposed_name,{'case_path':r'C:\synthetic\fixture.pwb'})])
        self.assertTrue(all(len(json.dumps(c['messages']))<12000 for c in model.chat_calls))
        after_binding=json.loads(model.chat_calls[4]['messages'][1]['content'])
        self.assertIn('case_path',after_binding['bound_outputs'])
        self.assertIn('case_path',after_binding['variables'])
        self.assertNotIn(discovery.exposed_name,[s['function']['name'] for s in model.chat_calls[4]['tools']])
        query_call=model.chat_calls[5]
        self.assertNotIn(discovery.exposed_name,[s['function']['name'] for s in query_call['tools']])

    def test_resume_keeps_variables_and_does_not_repeat_discovery(self):
        tool=make_tool('discover')
        model=FakeModel([action('host__set_plan',steps=[step('discover',tool.exposed_name,['count'])]),
            action(tool.exposed_name),bind('count','r1','/structuredContent/count'),
            action('host__block_plan',reason='Synthetic external dependency unavailable')])
        manager=FakeManager([tool],{tool.exposed_name:{'structuredContent':{'count':4}}})
        with TemporaryDirectory() as directory:
            root=Path(directory)
            host=self.setup_run(root,model,manager)
            asyncio.run(host.respond('Synthetic objective',[],['grid.plugin']))
            resumed=FakeModel([action('host__complete_step'),AssistantReply(content='Four.',tool_calls=[])])
            host=self.setup_run(root,resumed,manager)
            self.assertEqual(asyncio.run(host.respond('Resume',[],['grid.plugin'])),'Four.')
            self.assertEqual(len(manager.calls),1)
            self.assertIn('count',resumed.chat_calls[0]['messages'][1]['content'])

    def test_plan_and_output_validation_prevent_false_completion(self):
        tool=make_tool('query')
        model=FakeModel([
            action('host__set_plan',steps=[step('query',tool.exposed_name,['count'],['missing'])]),
            action('host__set_plan',steps=[step('query',tool.exposed_name,['count'])]),
            action('host__complete_step'),bind('count','invented',''),action('host__complete_step')])
        manager=FakeManager([tool],{})
        with TemporaryDirectory() as directory:
            root=Path(directory); host=self.setup_run(root,model,manager)
            answer=asyncio.run(host.respond('Synthetic',[],['grid.plugin']))
            saved=json.loads(host.activity_path.with_suffix('.plan.json').read_text())
        self.assertIn('paused',answer)
        self.assertEqual(saved['steps'][0]['status'],'pending')
        self.assertEqual(saved['variables'],{})
        self.assertFalse(manager.calls)

    def test_failed_calls_are_not_reexecuted_and_arguments_validate_after_binding(self):
        tool=make_tool('query',required=['case_path'])
        model=FakeModel([action('host__set_plan',steps=[step('query',tool.exposed_name,['value'])]),
            action(tool.exposed_name,case_path='synthetic'),action(tool.exposed_name,case_path='synthetic'),
            bind('value','r1',''),action('host__block_plan',reason='Synthetic failure')])
        manager=FakeManager([tool],{tool.exposed_name:{'isError':True,'message':'Synthetic missing file'}})
        with TemporaryDirectory() as directory:
            host=self.setup_run(Path(directory),model,manager)
            answer=asyncio.run(host.respond('Synthetic',[],['grid.plugin']))
        self.assertEqual(len(manager.calls),1)
        self.assertIn('incomplete',answer)

    def test_approval_returns_to_plan_instead_of_finishing_analysis(self):
        from local_model_app.server import Runtime
        from local_model_app.chat_store import ChatStore
        tool=make_tool('query')
        class Manager(FakeManager):
            async def call_tool(self,name,args,approved=False):
                self.calls.append((name,args,approved))
                if not approved: raise PermissionError('Synthetic approval')
                return {'structuredContent':{'count':7}}
        manager=Manager([tool],{})
        model=FakeModel([action('host__set_plan',steps=[step('query',tool.exposed_name,['count'])]),
            action(tool.exposed_name),bind('count','r1','/structuredContent/count'),
            action('host__complete_step'),AssistantReply(content='Seven.',tool_calls=[])])
        with TemporaryDirectory() as directory:
            root=Path(directory);store=ChatStore(root/'data'/'chats.sqlite3');chat=store.create_chat()
            store.set_plugin_selected(chat['id'],'grid.plugin',selected=True)
            activity=root/'data'/'tool_activity'/f"{chat['id']}.jsonl"
            host=ToolCoordinator(model,manager,Scratchpad(root/'notes.jsonl'),activity)
            self.assertIn('Approval',asyncio.run(host.respond('Synthetic',[],['grid.plugin'])))
            runtime=Runtime.__new__(Runtime);runtime.store=store;runtime.mcp=manager;runtime.model=model
            runtime._inference_lock=asyncio.Lock()
            with patch('local_model_app.server.ROOT',root):
                answer=asyncio.run(runtime.approve_tool_call(chat['id']))
            self.assertEqual(answer,'Seven.')
            self.assertEqual([c[2] for c in manager.calls],[False,True])
            saved=json.loads(activity.with_suffix('.plan.json').read_text())
            self.assertEqual(saved['status'],'completed')
            store.close()


if __name__=='__main__': unittest.main()
