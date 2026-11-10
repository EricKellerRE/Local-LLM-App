import asyncio
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from local_model_app.mcp_plugins import DiscoveredTool
from local_model_app.model import AssistantReply
from local_model_app.scratchpad import Scratchpad
from local_model_app.tool_coordinator import ToolCoordinator as ProductionToolCoordinator, _compact_result, _input_artifact_paths, _input_case_paths


class ToolCoordinator(ProductionToolCoordinator):
    """Regression coverage for the retained legacy execution mode."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, structured_plans=False, **kwargs)


def make_tool(native_name: str, *, required: list[str] | None = None) -> DiscoveredTool:
    return DiscoveredTool(
        exposed_name=f"grid__{native_name}", plugin_id="grid.plugin", server_id="grid",
        native_name=native_name, title=None, description=f"Test tool {native_name}",
        input_schema={"type": "object", "properties": {name: {"type": "string"} for name in (required or [])}, "required": required or []},
        output_schema=None, annotations={},
    )


class FakeModel:
    def __init__(self, replies: list[AssistantReply] | None = None) -> None:
        self.replies = list(replies or [])
        self.chat_calls = []

    def generate(self, messages, **kwargs):
        return "Route, discover, inspect the schema, then execute only when required inputs are known."

    def chat(self, messages, *, tools, **kwargs):
        self.chat_calls.append({"messages": messages, "tools": tools, **kwargs})
        return self.replies.pop(0)


class FakeManager:
    def __init__(self, tools: list[DiscoveredTool], results: dict[str, dict]) -> None:
        policy = SimpleNamespace(max_calls_per_turn=8)
        manifest = SimpleNamespace(policy=policy)
        self.registry = SimpleNamespace(get_plugin=lambda plugin_id: SimpleNamespace(manifest=manifest))
        self.tools = tools
        self.results = results
        self.calls = []

    async def ensure_started(self, plugin_ids):
        self.started = list(plugin_ids)

    def tools_for_plugins(self, plugin_ids):
        return self.tools

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.results[name]


class ToolCoordinatorTests(unittest.TestCase):
    def test_binary_case_task_does_not_offer_aux_snapshot_analyzer(self):
        aux=make_tool('results.analyze_aux_snapshots')
        query=make_tool('case.query_objects')
        for prompt,history,expected in [
            ('Analyze PowerWorld cases in the synthetic directory',[],False),
            ('Continue',[{'role':'user','content':'Analyze fixture.pwb'}],False),
            ('Analyze AUX snapshots exported from fixture.pwb',[],True),
        ]:
            with self.subTest(prompt=prompt), TemporaryDirectory(dir=Path.cwd()) as directory:
                root=Path(directory)
                model=FakeModel([AssistantReply(content='Synthetic complete',tool_calls=[])]*2)
                coordinator=ToolCoordinator(model,FakeManager([aux,query],{}),Scratchpad(root/'notes.jsonl'),root/'activity.jsonl')
                asyncio.run(coordinator.respond(prompt,history,['grid.plugin']))
                for call in model.chat_calls:
                    self.assertEqual(aux.exposed_name in json.dumps(call['tools']),expected)

    def test_repeated_observation_recovers_without_fourth_execution(self):
        lookup = make_tool('lookup')
        summarize = make_tool('summarize')
        def call(tool, index):
            return AssistantReply(content='', tool_calls=[{'id':str(index),'type':'function',
                'function':{'name':tool.exposed_name,'arguments':{}}}])
        model = FakeModel([call(lookup,i) for i in range(4)] + [call(summarize,4)] +
                          [AssistantReply(content='Synthetic task complete.',tool_calls=[])] * 2)
        manager = FakeManager([lookup,summarize], {t.exposed_name:{'isError':False,
            'structuredContent':{'synthetic_count':2}} for t in [lookup,summarize]})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            coordinator=ToolCoordinator(model,manager,Scratchpad(root/'notes.jsonl'),root/'activity.jsonl')
            answer=asyncio.run(coordinator.respond('Summarize synthetic objects',[],['grid.plugin']))
            events=[json.loads(line)['event'] for line in (root/'activity.jsonl').read_text().splitlines()]
        self.assertEqual(answer,'Synthetic task complete.')
        self.assertEqual([name for name,args in manager.calls].count(lookup.exposed_name),3)
        self.assertIn((summarize.exposed_name,{}),manager.calls)
        self.assertIn('loop_recovery',events)
        self.assertNotIn('loop_stopped',events)

    def test_unrecovered_loop_preserves_resume_checkpoint(self):
        tool=make_tool('lookup')
        replies=[AssistantReply(content='',tool_calls=[{'id':str(i),'type':'function',
            'function':{'name':tool.exposed_name,'arguments':{}}}]) for i in range(5)]
        manager=FakeManager([tool],{tool.exposed_name:{'isError':False,'structuredContent':{'synthetic_count':2}}})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            activity=root/'activity.jsonl'
            coordinator=ToolCoordinator(FakeModel(replies),manager,Scratchpad(root/'notes.jsonl'),activity)
            answer=asyncio.run(coordinator.respond('Inspect synthetic objects',[],['grid.plugin']))
            saved=json.loads(activity.with_suffix('.state.json').read_text())
            events=[json.loads(line)['event'] for line in activity.read_text().splitlines()]
        self.assertEqual(len(manager.calls),3)
        self.assertIn('incomplete',answer)
        self.assertEqual(saved['status'],'tool_checkpoint')
        self.assertEqual(events[-1],'incomplete')

    def test_pressure_reduces_oversized_base_prompt_and_schema_batch(self):
        tools=[make_tool('lookup_'+str(i)) for i in range(6)]
        for tool in tools:
            tool.input_schema['description']='synthetic schema reference ' * 350
        model=FakeModel([AssistantReply(content='Synthetic check complete.',tool_calls=[])] * 2)
        model.generate=lambda *args,**kwargs: 'synthetic planning reference ' * 1500
        model.effective_context_window=4096
        model.settings=SimpleNamespace(tool_action_max_new_tokens=512)
        model.count_prompt_tokens=lambda messages,tools: len(json.dumps([messages,tools]))//4
        manager=FakeManager(tools,{})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            coordinator=ToolCoordinator(model,manager,Scratchpad(root/'notes.jsonl'),root/'activity.jsonl')
            answer=asyncio.run(coordinator.respond('Check synthetic context sizing',[],['grid.plugin']))
            events=[json.loads(line)['event'] for line in (root/'activity.jsonl').read_text().splitlines()]
        self.assertEqual(answer,'Synthetic check complete.')
        self.assertIn('context_payload_reduced',events)
        call=model.chat_calls[0]
        self.assertLess(model.count_prompt_tokens(call['messages'],call['tools'])+call['max_new_tokens']+409,4096)
        self.assertIn('host__select_tools',[spec['function']['name'] for spec in call['tools']])

    def test_plan_only_reply_is_replaced_by_execution(self):
        tool=make_tool('workspace.resolve_inputs')
        model=FakeModel([
            AssistantReply(content='I will begin by recursively scanning the directory. I am ready to begin.',tool_calls=[]),
            AssistantReply(content='',tool_calls=[{'id':'discover','type':'function','function':{'name':tool.exposed_name,'arguments':{}}}]),
            AssistantReply(content='Synthetic discovery complete.',tool_calls=[]),
        ])
        manager=FakeManager([tool],{tool.exposed_name:{'isError':False,'structuredContent':{'count':0}}})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            coordinator=ToolCoordinator(model,manager,Scratchpad(root/'notes.jsonl'),root/'activity.jsonl')
            answer=asyncio.run(coordinator.respond('Discover synthetic files',[],['grid.plugin']))
        self.assertEqual(answer,'Synthetic discovery complete.')
        self.assertEqual(len(manager.calls),1)
        self.assertEqual(model.chat_calls[1]['generation_class'],'required_tool_action')

    def test_explicit_case_path_is_bound_to_query_and_resolver_schemas(self):
        path=r'C:\synthetic folder\long-0123456789abcdef\fixture.pwb'
        paths=_input_case_paths(f'Analyze this case: {path}.')
        self.assertEqual(paths,{path})
        query=make_tool('case.query_objects',required=['case_path'])
        resolve=make_tool('workspace.resolve_inputs',required=['case_path_hint'])
        specs=ToolCoordinator._narrowed_specs([query,resolve],{},[query],case_paths=paths)
        self.assertEqual(specs[0]['function']['parameters']['properties']['case_path']['enum'],[path])
        self.assertEqual(specs[1]['function']['parameters']['properties']['case_path_hint']['enum'],[path])
        self.assertNotIn('enum',query.input_schema['properties']['case_path'])

    def test_calculator_executes_locally_and_returns_exact_result_to_model(self):
        model = FakeModel([
            AssistantReply(content='',tool_calls=[{'id':'math1','type':'function','function':{
                'name':'host__calculate','arguments':{'expressions':{'gw_miles':'30 * (3000 / 1000)'}}}}]),
            AssistantReply(content='90 GW-miles',tool_calls=[]),
        ])
        manager=FakeManager([make_tool('results.summarize_artifact')],{})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            coordinator=ToolCoordinator(model,manager,Scratchpad(root/'notes.jsonl'),root/'activity.jsonl')
            answer=asyncio.run(coordinator.respond('Calculate a synthetic product',[],['grid.plugin']))
        self.assertEqual(answer,'90 GW-miles')
        self.assertFalse(manager.calls)
        result=next(message for message in model.chat_calls[1]['messages'] if message.get('name')=='host__calculate')
        self.assertEqual(json.loads(result['content'])['results']['gw_miles'],90)

    def test_user_supplied_artifacts_remain_available_after_first_summary(self):
        first = r'C:\synthetic folder\branches.csv'
        second = r'C:\synthetic folder\generation.csv'
        paths = _input_artifact_paths(f'branches {first}, generation {second}. Case C:\\synthetic\\case.pwb')
        self.assertEqual(paths,{first,second})
        tool = make_tool('results.summarize_artifact',required=['path'])
        specs = ToolCoordinator._narrowed_specs([tool],{},[tool],paths)
        self.assertEqual(set(specs[0]['function']['parameters']['properties']['path']['enum']),paths)

    def test_resume_reads_notes_and_checkpoint_without_fresh_planning(self):
        from unittest.mock import Mock
        tool=make_tool('summarize')
        model=FakeModel([AssistantReply(content='Done',tool_calls=[])])
        model.generate=Mock(side_effect=AssertionError('Resume must not replan from scratch'))
        manager=FakeManager([tool],{})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            pad=Scratchpad(root/'scratch.jsonl')
            pad.add('progress','Synthetic discovery completed; filtered summary pending.')
            activity=root/'activity.jsonl'
            activity.with_suffix('.state.json').write_text(json.dumps({
                'status':'tool_checkpoint','user_message':'Synthetic task','checkpoint_index':7,
                'selected_tools':[tool.exposed_name],
                'checkpoint':{'checkpoint_id':'synthetic','acceptance_and_execution_plan':'Filter then summarize',
                              'latest_structured_state':{'next_action':'summarize existing artifact'}},
                'data':{'ok':True},
            }))
            coordinator=ToolCoordinator(model,manager,pad,activity)
            asyncio.run(coordinator.respond('Synthetic task',[],['grid.plugin']))
            prompt=model.chat_calls[0]['messages'][0]['content']
            self.assertIn('filtered summary pending',prompt)
            self.assertIn('summarize existing artifact',prompt)
            self.assertIn('Filter then summarize',prompt)
            events=[json.loads(line)['event'] for line in activity.read_text().splitlines()]
            self.assertIn('scratchpad_loaded',events)
            self.assertIn('checkpoint_loaded',events)
        model.generate.assert_not_called()

    def test_artifact_paths_constrain_consumer_without_mutating_catalog(self):
        tool=make_tool('results.summarize_artifact',required=['path'])
        paths={r'C:\synthetic\exact-export.csv'}
        specs=ToolCoordinator._narrowed_specs([tool],{},[tool],paths)
        self.assertEqual(specs[0]['function']['parameters']['properties']['path']['enum'],sorted(paths))
        self.assertNotIn('enum',tool.input_schema['properties']['path'])

    def test_artifact_reference_survives_large_observation_compaction(self):
        tool=make_tool('query')
        path=r'C:\synthetic\complete.csv'
        manager=FakeManager([tool],{tool.exposed_name:{'isError':False,'structuredContent':{
            'rows':['x'*1000]*40,'artifact_path':path}}})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            coordinator=ToolCoordinator(FakeModel(),manager,Scratchpad(root/'scratch.jsonl'),root/'activity.jsonl')
            result=asyncio.run(coordinator._execute(tool,{},[]))
            self.assertIn(path,coordinator._artifact_paths)
            self.assertTrue(result['data']['truncated'])

    def test_malformed_call_is_regenerated_without_executing_or_replaying_it(self):
        tool=make_tool('lookup')
        def reply(arguments):
            return AssistantReply(content='',tool_calls=[{'id':'call','type':'function','function':{'name':tool.exposed_name,'arguments':arguments}}])
        model=FakeModel([reply('{"broken":'), reply({}), AssistantReply(content='Done',tool_calls=[])])
        manager=FakeManager([tool],{tool.exposed_name:{'isError':False,'structuredContent':{'ok':True}}})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            coordinator=ToolCoordinator(model,manager,Scratchpad(root/'scratch.jsonl'),root/'activity.jsonl')
            answer=asyncio.run(coordinator.respond('Lookup',[],['grid.plugin']))
        self.assertEqual(answer,'Done')
        self.assertEqual(manager.calls,[(tool.exposed_name,{})])
        self.assertNotIn('broken',json.dumps(model.chat_calls[1]['messages']))

    def test_repeated_malformed_calls_end_without_tool_execution(self):
        tool=make_tool('lookup')
        invalid=AssistantReply(content='',tool_calls=[{'id':'call','type':'function','function':{'name':tool.exposed_name,'arguments':'[]'}}])
        model=FakeModel([invalid]*3)
        manager=FakeManager([tool],{})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            coordinator=ToolCoordinator(model,manager,Scratchpad(root/'scratch.jsonl'),root/'activity.jsonl')
            answer=asyncio.run(coordinator.respond('Lookup',[],['grid.plugin']))
        self.assertIn('three times',answer)
        self.assertEqual(manager.calls,[])

    def test_unfit_compacted_prompt_never_reaches_generation(self):
        tool = make_tool('lookup')
        model = FakeModel()
        model.effective_context_window = 1024
        model.settings = SimpleNamespace(tool_action_max_new_tokens=192)
        model.count_prompt_tokens = lambda messages, tools: 1100
        manager = FakeManager([tool], {})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            coordinator = ToolCoordinator(model, manager, Scratchpad(root/'scratch.jsonl'), root/'activity.jsonl')
            with self.assertRaisesRegex(RuntimeError, 'even after compaction'):
                asyncio.run(coordinator.respond('Find facts', [], ['grid.plugin']))
        self.assertEqual(model.chat_calls, [])

    def test_success_loop_stops_across_checkpoints(self):
        tool = make_tool("case.overview")
        reply = AssistantReply(content="", tool_calls=[{"id":"loop", "type":"function", "function":{"name":tool.exposed_name,"arguments":{}}}])
        model = FakeModel([reply] * 5)
        manager = FakeManager([tool], {tool.exposed_name:{"isError":False,"structuredContent":{"value":123}}})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            coordinator = ToolCoordinator(model, manager, Scratchpad(root/'scratch.jsonl'), root/'activity.jsonl', max_calls=1)
            answer = asyncio.run(coordinator.respond('Analyze inputs', [], ['grid.plugin']))
        self.assertEqual(len(manager.calls), 3)
        self.assertIn('Stopped a tool loop', answer)

    def test_agent_can_select_tool_outside_initial_shortlist(self):
        tools = [make_tool(f'tool_{i}') for i in range(10)]
        target = tools[-1]
        model = FakeModel([
            AssistantReply(content='',tool_calls=[{'id':'select','type':'function','function':{'name':'host__select_tools','arguments':{'names':[target.exposed_name]}}}]),
            AssistantReply(content='',tool_calls=[{'id':'execute','type':'function','function':{'name':target.exposed_name,'arguments':{}}}]),
            AssistantReply(content='Done',tool_calls=[]),
        ])
        manager = FakeManager(tools, {target.exposed_name:{'isError':False,'structuredContent':{'ok':True}}})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            coordinator = ToolCoordinator(model, manager, Scratchpad(root/'scratch.jsonl'), root/'activity.jsonl')
            answer = asyncio.run(coordinator.respond('Use tool_0 then the needed next tool', [], ['grid.plugin']))
        self.assertEqual(answer,'Done')
        self.assertEqual(manager.calls, [(target.exposed_name,{})])
        self.assertIn(target.exposed_name,[s['function']['name'] for s in model.chat_calls[1]['tools']])

    def test_input_discovery_survives_narrow_catalog(self):
        analyze = make_tool("case.overview")
        resolve = make_tool("workspace.resolve_inputs")
        specs = ToolCoordinator._narrowed_specs([analyze, resolve], {}, [analyze])
        self.assertEqual([s["function"]["name"] for s in specs], [analyze.exposed_name, resolve.exposed_name])

    def test_backend_error_detail_reaches_agent(self):
        from mcp.shared.exceptions import MCPError
        tool = make_tool("case.overview")
        manager = FakeManager([tool], {})
        async def fail(*args):
            raise MCPError(code=-32000, message="Tool server HTTP error: 500", data='{"error":"File does not exist"}')
        manager.call_tool = fail
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            coordinator = ToolCoordinator(FakeModel(), manager, Scratchpad(root / "scratch.jsonl"), root / "activity.jsonl")
            result = asyncio.run(coordinator._execute(tool, {}, []))
        self.assertIn("File does not exist", result["message"])

    def test_identical_failed_calls_stop_after_two_attempts(self):
        tool = make_tool("case.overview")
        reply = AssistantReply(content="", tool_calls=[{"id":"call_fail", "type":"function", "function":{"name":tool.exposed_name, "arguments":{}}}])
        model = FakeModel([reply, reply, reply])
        manager = FakeManager([tool], {tool.exposed_name: {"isError":True, "error":"missing_file", "message":"File does not exist"}})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            coordinator = ToolCoordinator(model, manager, Scratchpad(root / "scratch.jsonl"), root / "activity.jsonl")
            answer = asyncio.run(coordinator.respond("Inspect case", [], ["grid.plugin"]))
        self.assertEqual(len(manager.calls), 2)
        self.assertIn("File does not exist", answer)
        self.assertIn("Stopped", answer)

    def test_query_failure_keeps_repair_schemas_and_exact_error_visible(self):
        query = make_tool('case.query_objects')
        search = make_tool('case.search_object_fields')
        voltage = make_tool('case.list_voltage_levels')
        model = FakeModel([
            AssistantReply(content='', tool_calls=[{'id':'query1','type':'function','function':{
                'name':query.exposed_name,'arguments':{}}}]),
            AssistantReply(content='Cannot complete synthetic query', tool_calls=[]),
        ])
        manager = FakeManager([query, search, voltage], {query.exposed_name:{
            'isError':True,'message':'Unknown constraint columns: SyntheticVoltage'}})
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            coordinator = ToolCoordinator(model,manager,Scratchpad(root/'notes.jsonl'),root/'activity.jsonl')
            asyncio.run(coordinator.respond('Query synthetic objects',[],['grid.plugin']))
        retry = model.chat_calls[1]
        names = [spec['function']['name'] for spec in retry['tools']]
        self.assertIn(query.exposed_name,names)
        self.assertIn(search.exposed_name,names)
        self.assertNotIn(voltage.exposed_name,names)
        self.assertIn('Unknown constraint columns: SyntheticVoltage', str(retry['messages']))

    def test_compact_result_prefers_structured_content(self) -> None:
        result = {
            "isError": False,
            "content": [{"type": "text", "text": "duplicate and verbose"}],
            "structuredContent": {"ok": True, "capability": "scenario"},
        }
        self.assertEqual(_compact_result(result), {"isError": False, "data": {"ok": True, "capability": "scenario"}})

    def test_compact_result_omits_binary_payloads_and_bounds_large_data(self) -> None:
        binary = _compact_result({
            "isError": False,
            "structuredContent": {
                "contents": [{"uri": "file://plot.png", "mimeType": "image/png", "blob": "A" * 1000}],
            },
        })
        large = _compact_result({
            "isError": False,
            "structuredContent": {"rows": ["x" * 1000 for _ in range(100)]},
        })

        content = binary["data"]["contents"][0]
        self.assertNotIn("blob", content)
        self.assertTrue(content["payloadOmitted"])
        self.assertTrue(large["data"]["truncated"])
        self.assertEqual(large["truncation"]["strategy"], "deterministic_json_preview")

    def test_copied_windows_path_underscore_escape_is_repaired_only_if_it_exists(self) -> None:
        copied = r"E:\Data\Tornado\_Tracks\_1.csv"
        with patch("local_model_app.tool_coordinator.Path.exists", return_value=True):
            self.assertEqual(ToolCoordinator._path_for_extension(copied, "csv"), r"E:\Data\Tornado_Tracks_1.csv")
        with patch("local_model_app.tool_coordinator.Path.exists", return_value=False):
            self.assertEqual(ToolCoordinator._path_for_extension(copied, "csv"), copied)

    def test_staged_protocol_reaches_missing_input_without_extra_model_turns(self) -> None:
        route = make_tool("powerworld_route_request", required=["request_text"])
        list_tools = make_tool("powerworld_list_tools", required=["capability"])
        get_schema = make_tool("powerworld_get_tool_schema", required=["tool_name"])
        model = FakeModel()
        manager = FakeManager(
            [route, list_tools, get_schema],
            {
                route.exposed_name: {"isError": False, "structuredContent": {"ok": True, "capability": "tornado", "next_step": "Call powerworld_list_tools for this capability."}},
                list_tools.exposed_name: {"isError": False, "structuredContent": {"ok": True, "capability": "tornado", "tools": [{"name": "tornado.build_scenario"}]}},
                get_schema.exposed_name: {"isError": False, "structuredContent": {"ok": True, "tool": {"name": "tornado.build_scenario", "input_schema": {"type": "object", "required": ["paths_csv"], "properties": {"paths_csv": {"type": "string"}}}}}},
            },
        )

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            activity = root / "activity.jsonl"
            coordinator = ToolCoordinator(model, manager, Scratchpad(root / "scratchpad.jsonl"), activity)
            answer = asyncio.run(coordinator.respond(r"Run one tornado on C:\cases\study.pwb", [], ["grid.plugin"]))

            self.assertIn("`paths_csv`", answer)
            self.assertEqual([name for name, _ in manager.calls], [route.exposed_name, list_tools.exposed_name, get_schema.exposed_name])
            self.assertEqual(model.chat_calls, [])
            events = [json.loads(line)["event"] for line in activity.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events, ["planning_started", "plan", "tool_routing", "tool_call", "tool_result", "tool_call", "tool_result", "tool_call", "tool_result", "waiting_for_input"])

    def test_generic_tool_call_is_observed_before_answer(self) -> None:
        tool = make_tool("lookup", required=["query"])
        model = FakeModel([
            AssistantReply(content="", tool_calls=[{"id": "call_1", "type": "function", "function": {"name": tool.exposed_name, "arguments": '{"query":"cases"}'}}], reasoning_content="Use lookup."),
            AssistantReply(content="The lookup completed.", tool_calls=[]),
        ])
        model.settings = SimpleNamespace(max_new_tokens=12_000)
        manager = FakeManager([tool], {tool.exposed_name: {"isError": False, "structuredContent": {"ok": True}}})

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            coordinator = ToolCoordinator(model, manager, Scratchpad(root / "scratchpad.jsonl"), root / "activity.jsonl")
            answer = asyncio.run(coordinator.respond("look up cases", [], ["grid.plugin"]))

        self.assertEqual(answer, "The lookup completed.")
        self.assertEqual(manager.calls, [(tool.exposed_name, {"query": "cases"})])
        self.assertEqual(model.chat_calls[0]["max_new_tokens"], 192)
        self.assertEqual(model.chat_calls[1]["max_new_tokens"], 12_000)
        tool_message = next(message for message in model.chat_calls[1]["messages"] if message["role"] == "tool")
        self.assertNotIn("duplicate", tool_message["content"])

    def test_post_tool_decision_has_an_independent_budget_and_turn_class(self) -> None:
        tool = make_tool("lookup", required=["query"])
        model = FakeModel([
            AssistantReply(content="", tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": tool.exposed_name, "arguments": '{"query":"memory"}'},
            }]),
            AssistantReply(content="Evidence recorded.", tool_calls=[]),
        ])
        model.settings = SimpleNamespace(
            max_new_tokens=8192,
            tool_action_max_new_tokens=384,
            post_tool_decision_max_new_tokens=1536,
        )
        manager = FakeManager([tool], {tool.exposed_name: {
            "isError": False, "structuredContent": {"results": ["source"]},
        }})

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            coordinator = ToolCoordinator(
                model,
                manager,
                Scratchpad(root / "scratchpad.jsonl"),
                root / "activity.jsonl",
                telemetry_context={"task_id": "task-1", "work_item_id": "item-1"},
            )
            answer = asyncio.run(coordinator.respond("Research memory", [], ["grid.plugin"]))

        self.assertEqual(answer, "Evidence recorded.")
        self.assertEqual(model.chat_calls[0]["max_new_tokens"], 384)
        self.assertEqual(model.chat_calls[0]["generation_class"], "tool_action")
        self.assertEqual(model.chat_calls[1]["max_new_tokens"], 1536)
        self.assertEqual(model.chat_calls[1]["generation_class"], "post_tool_decision")
        self.assertEqual(model.chat_calls[1]["telemetry_context"]["work_item_id"], "item-1")

    def test_chunked_fetch_continues_deterministically_without_model_url_rewrite(self) -> None:
        url = (
            "https://example.org/paper?comparison=early-versus-late&"
            "source=exact-long-url"
        )
        fetch = make_tool("fetch_url", required=["url"])
        model = FakeModel([
            AssistantReply(content="", tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": fetch.exposed_name, "arguments": {"url": url}},
            }]),
            AssistantReply(content="Source fully read.", tool_calls=[]),
        ])
        manager = FakeManager([fetch], {fetch.exposed_name: {
            "isError": False,
            "structuredContent": {"url": url, "content": "chunk", "next_start": 30000, "complete": False},
        }})
        # The deterministic continuation sees a terminal result on the second call.
        calls = 0

        async def call_tool(exposed_name, arguments, **kwargs):
            nonlocal calls
            calls += 1
            manager.calls.append((exposed_name, arguments))
            if calls == 1:
                return {"isError": False, "structuredContent": {
                    "url": url, "content": "first", "next_start": 30000, "complete": False,
                }}
            return {"isError": False, "structuredContent": {
                "url": url, "content": "second", "next_start": None, "complete": True,
            }}

        manager.call_tool = call_tool
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            coordinator = ToolCoordinator(
                model, manager, Scratchpad(root / "scratchpad.jsonl"), root / "activity.jsonl",
            )
            answer = asyncio.run(coordinator.respond("Read the complete source", [], ["grid.plugin"]))

        self.assertEqual(answer, "Source fully read.")
        self.assertEqual(manager.calls[1], (fetch.exposed_name, {"url": url, "start_index": 30000}))
        self.assertEqual(len(model.chat_calls), 2)

    def test_tool_checkpoint_interval_compacts_and_continues(self) -> None:
        tool = make_tool("lookup", required=["query"])
        model = FakeModel([
            AssistantReply(content="", tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": tool.exposed_name, "arguments": '{"query":"first"}'},
            }]),
            AssistantReply(content="", tool_calls=[{
                "id": "call_2",
                "type": "function",
                "function": {"name": tool.exposed_name, "arguments": '{"query":"second"}'},
            }]),
            AssistantReply(content="Both lookups completed.", tool_calls=[]),
        ])
        model.settings = SimpleNamespace(max_new_tokens=12_000)
        manager = FakeManager([tool], {tool.exposed_name: {
            "isError": False, "structuredContent": {"ok": True},
        }})

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            activity = root / "activity.jsonl"
            coordinator = ToolCoordinator(
                model,
                manager,
                Scratchpad(root / "scratchpad.jsonl"),
                activity,
                max_calls=1,
            )
            answer = asyncio.run(coordinator.respond("Perform both lookups", [], ["grid.plugin"]))
            events = [
                json.loads(line)["event"]
                for line in activity.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(answer, "Both lookups completed.")
        self.assertEqual(
            manager.calls,
            [
                (tool.exposed_name, {"query": "first"}),
                (tool.exposed_name, {"query": "second"}),
            ],
        )
        self.assertGreaterEqual(events.count("tool_checkpoint"), 1)

    def test_token_pressure_compacts_before_the_model_silently_drops_history(self) -> None:
        tool = make_tool("lookup", required=["query"])

        class PressureModel(FakeModel):
            effective_context_window = 1024
            settings = SimpleNamespace(
                tool_action_max_new_tokens=192,
                post_tool_decision_max_new_tokens=192,
                max_new_tokens=192,
                tool_temperature=0.0,
            )

            def count_prompt_tokens(self, messages, tools):
                if any("Context checkpoint" in str(message.get("content") or "") for message in messages):
                    return 180
                if any(message.get("role") == "tool" for message in messages):
                    return 900
                return 200

        model = PressureModel([
            AssistantReply(content="", tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": tool.exposed_name, "arguments": '{"query":"memory"}'},
            }]),
            AssistantReply(content="Lookup complete.", tool_calls=[]),
        ])
        manager = FakeManager([tool], {tool.exposed_name: {
            "isError": False,
            "structuredContent": {
                "ok": True,
                "source_id": "source-1",
                "url": "https://example.test/source-1",
            },
        }})

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            activity = root / "activity.jsonl"
            coordinator = ToolCoordinator(
                model,
                manager,
                Scratchpad(root / "scratchpad.jsonl"),
                activity,
                max_calls=8,
            )
            answer = asyncio.run(coordinator.respond("Research memory", [], ["grid.plugin"]))
            events = [json.loads(line) for line in activity.read_text(encoding="utf-8").splitlines()]
            persisted_checkpoints = [
                json.loads(line)
                for line in activity.with_suffix(".checkpoints.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        checkpoints = [row for row in events if row["event"] == "tool_checkpoint"]
        self.assertEqual(answer, "Lookup complete.")
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]["trigger"], "token_pressure")
        self.assertEqual(checkpoints[0]["prompt_tokens"], 900)
        self.assertEqual(persisted_checkpoints[0]["checkpoint_id"], checkpoints[0]["checkpoint_id"])
        checkpoint_prompt = model.chat_calls[1]["messages"][0]["content"]
        self.assertIn('"objective":"Research memory"', checkpoint_prompt)
        self.assertIn('"source_id":"source-1"', checkpoint_prompt)

    def test_generic_router_exposes_only_top_schema_batch(self) -> None:
        weather = make_tool("weather_forecast", required=["city"])
        tools = [weather] + [make_tool(name) for name in (
            "calendar_events", "invoice_lookup", "repository_search", "send_email", "customer_record",
        )]
        model = FakeModel([
            AssistantReply(content="", tool_calls=[{
                "id": "call_weather", "type": "function",
                "function": {"name": weather.exposed_name, "arguments": '{"city":"Chicago"}'},
            }]),
            AssistantReply(content="Rain is expected.", tool_calls=[]),
        ])
        manager = FakeManager(tools, {weather.exposed_name: {
            "isError": False, "structuredContent": {"forecast": "rain"},
        }})

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            coordinator = ToolCoordinator(
                model, manager, Scratchpad(root / "scratchpad.jsonl"), root / "activity.jsonl",
            )
            answer = asyncio.run(coordinator.respond("Show the Chicago weather forecast", [], ["grid.plugin"]))

        first_specs = model.chat_calls[0]["tools"]
        self.assertEqual(answer, "Rain is expected.")
        self.assertEqual(len(first_specs), 6)
        self.assertEqual(first_specs[-2]['function']['name'], 'host__select_tools')
        self.assertEqual(first_specs[-1]['function']['name'], 'host__calculate')
        self.assertIn(weather.exposed_name, [item["function"]["name"] for item in first_specs])

    def test_supplied_missing_input_resumes_pending_backend_without_rerouting(self) -> None:
        route = make_tool("powerworld_route_request", required=["request_text"])
        get_schema = make_tool("powerworld_get_tool_schema", required=["tool_name"])
        gateway = make_tool("powerworld_call_tool", required=["tool_name", "arguments"])
        model = FakeModel([
            AssistantReply(content="", tool_calls=[{
                "id": "call_resume", "type": "function",
                "function": {
                    "name": gateway.exposed_name,
                    "arguments": '{"tool_name":"tornado.build_scenario","arguments":{"paths_csv":"E:\\\\tracks.csv","case_path":"C:\\\\study.pwb"}}',
                },
            }]),
        ])
        manager = FakeManager(
            [route, get_schema, gateway],
            {
                get_schema.exposed_name: {"isError": False, "structuredContent": {"ok": True, "tool": {
                    "name": "tornado.build_scenario",
                    "input_schema": {"type": "object", "required": ["paths_csv"], "properties": {
                        "paths_csv": {"type": "string"}, "case_path": {"type": ["string", "null"]},
                    }},
                }}},
                gateway.exposed_name: {"isError": True, "error": "approval_required", "message": "Approval required."},
            },
        )

        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            activity = root / "activity.jsonl"
            state = activity.with_suffix(".state.json")
            state.write_text(json.dumps({
                "status": "waiting_for_input",
                "backend_tool": "tornado.build_scenario",
                "required_inputs": ["paths_csv"],
            }), encoding="utf-8")
            coordinator = ToolCoordinator(model, manager, Scratchpad(root / "scratchpad.jsonl"), activity)
            answer = asyncio.run(coordinator.respond(r"paths_csv is E:\tracks.csv", [
                {"role": "user", "content": r"Use C:\study.pwb"},
                {"role": "assistant", "content": "Please provide paths_csv."},
            ], ["grid.plugin"]))

            self.assertIn("Approval is required", answer)
            self.assertEqual([name for name, _ in manager.calls], [get_schema.exposed_name, gateway.exposed_name])
            self.assertEqual(model.chat_calls, [])
            gateway_arguments = manager.calls[1][1]
            self.assertEqual(gateway_arguments["tool_name"], "tornado.build_scenario")
            self.assertEqual(gateway_arguments["arguments"]["paths_csv"], r"E:\tracks.csv")
            self.assertEqual(gateway_arguments["arguments"]["case_path"], r"C:\study.pwb")
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "waiting_for_approval")


if __name__ == "__main__":
    unittest.main()
