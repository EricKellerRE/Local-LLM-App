import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from local_model_app.gguf_model import GgufModel, create_model


class GgufTests(unittest.TestCase):
    def test_required_action_schema_cannot_emit_final_plan(self):
        from local_model_app.structured_tools import prepare, decode
        tools=[{'function':{'name':'synthetic_lookup','parameters':{'type':'object','properties':{}}}}]
        _,schema=prepare([{'role':'user','content':'synthetic'}],tools,require_tool=True)
        self.assertEqual(len(schema['oneOf']),1)
        self.assertEqual(schema['oneOf'][0]['properties']['kind']['const'],'tool')
        content,calls=decode('{"kind":"final","content":"I am ready to begin"}',schema)
        self.assertEqual(content,'')
        self.assertEqual(calls[0]['function']['name'],'invalid_structured_decision')

    def test_tool_request_uses_nested_schema_and_same_prompt_for_counting(self):
        model=GgufModel(self.settings())
        model._load=Mock()
        model.base_url='http://127.0.0.1:1'
        model._write_telemetry=Mock()
        model._write_generation_record=Mock()
        model._post=Mock(side_effect=[{'prompt':'synthetic'},{'tokens':[1,2]}])
        tools=[{'type':'function','function':{'name':'lookup','parameters':{'type':'object','properties':{},'additionalProperties':False}}}]
        messages=[{'role':'user','content':'synthetic'}]
        model.count_prompt_tokens(messages,tools)
        counted_messages=model._post.call_args_list[0].args[1]['messages']
        chunk={'choices':[{'delta':{'content':'{"kind":"tool","name":"lookup","arguments":{}}'},'finish_reason':'stop'}]}
        with patch('local_model_app.gguf_model.httpx.Client') as client:
            stream=client.return_value.__enter__.return_value.stream
            stream.return_value.__enter__.return_value.iter_lines.return_value=['data: '+json.dumps(chunk),'data: [DONE]']
            reply=model.chat(messages,tools=tools)
        payload=stream.call_args.kwargs['json']
        self.assertIn('schema',payload['response_format']['json_schema'])
        self.assertNotIn('tools',payload)
        self.assertEqual(payload['messages'],counted_messages)
        self.assertEqual(reply.tool_calls[0]['function']['name'],'lookup')

    def settings(self):
        return SimpleNamespace(model_id='synthetic.gguf', model_kind='auto', context_window=4096,
                               max_new_tokens=128, temperature=0, top_p=.9, reasoning_budget=None)

    def test_auto_selects_gguf(self):
        model = create_model(self.settings())
        self.assertIsInstance(model, GgufModel)
        self.assertEqual(model.effective_context_window, 4096)

    def test_wire_arguments_are_serialized_without_mutating_history(self):
        history = [{'role':'assistant','tool_calls':[{'function':{'name':'x','arguments':{'a':1}}}]}]
        wire = GgufModel._wire_messages(history)
        self.assertEqual(json.loads(wire[0]['tool_calls'][0]['function']['arguments']), {'a':1})
        self.assertIsInstance(history[0]['tool_calls'][0]['function']['arguments'], dict)

    def test_fragmented_tool_stream(self):
        model = GgufModel(self.settings())
        model._load = Mock()
        model.base_url = 'http://127.0.0.1:1'
        model._write_telemetry = Mock()
        model._write_generation_record = Mock()
        chunks = [
            {'choices':[{'delta':{'tool_calls':[{'index':0,'id':'call1','function':{'name':'lookup','arguments':'{"a":'}}]}}]},
            {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'1}'}}]},'finish_reason':'tool_calls'}],
             'usage':{'prompt_tokens':10,'completion_tokens':5}},
        ]
        with patch('local_model_app.gguf_model.httpx.Client') as client:
            response = client.return_value.__enter__.return_value.stream.return_value.__enter__.return_value
            response.iter_lines.return_value = ['data: '+json.dumps(c) for c in chunks] + ['data: [DONE]']
            reply = model.chat([{'role':'user','content':'test'}])
        self.assertEqual(reply.tool_calls[0]['function']['arguments'], {'a':1})
        self.assertEqual(reply.stop_reason, 'tool_call')
        self.assertEqual(reply.generated_tokens, 5)

    def test_rejects_projector_without_launching(self):
        settings = self.settings()
        settings.model_id = 'mmproj-example.gguf'
        model = GgufModel(settings)
        with patch.object(Path, 'is_file', return_value=True), self.assertRaisesRegex(ValueError, 'projector'):
            model.load()

    def test_malformed_tool_json_is_returned_for_coordinator_recovery(self):
        model = GgufModel(self.settings())
        model._load = Mock()
        model.base_url = 'http://127.0.0.1:1'
        model._write_telemetry = Mock()
        model._write_generation_record = Mock()
        chunk = {'choices':[{'delta':{'tool_calls':[{'index':0,'id':'bad','function':{'name':'lookup','arguments':'{"path":'}}]},'finish_reason':'length'}]}
        with patch('local_model_app.gguf_model.httpx.Client') as client:
            response=client.return_value.__enter__.return_value.stream.return_value.__enter__.return_value
            response.iter_lines.return_value=['data: '+json.dumps(chunk),'data: [DONE]']
            reply=model.chat([{'role':'user','content':'synthetic'}])
        self.assertEqual(reply.tool_calls[0]['function']['arguments'],'{"path":')
        self.assertEqual(reply.stop_reason,'length')
