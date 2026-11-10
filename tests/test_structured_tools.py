import json
import unittest
from local_model_app.structured_tools import prepare, decode


class StructuredToolTests(unittest.TestCase):
    def setUp(self):
        self.tools=[{'type':'function','function':{'name':'query','description':'Synthetic query',
            'parameters':{'type':'object','required':['path','voltage'], 'properties':{
                'path':{'type':'string'},'voltage':{'type':'number'}},'additionalProperties':False}}}]

    def test_valid_windows_path_and_numeric_argument_round_trip(self):
        _,schema=prepare([],self.tools)
        args={'path':r'C:\synthetic\case.pwb','voltage':345}
        content,calls=decode(json.dumps({'kind':'tool','name':'query','arguments':args}),schema)
        self.assertEqual(calls[0]['function']['arguments'],args)
        self.assertEqual(content,'')

    def test_wrong_tool_missing_argument_and_wrong_type_are_not_executable(self):
        _,schema=prepare([],self.tools)
        for decision in [
            {'kind':'tool','name':'unknown','arguments':{'path':'x','voltage':345}},
            {'kind':'tool','name':'query','arguments':{'path':'x'}},
            {'kind':'tool','name':'query','arguments':{'path':'x','voltage':'bad'}},
        ]:
            _,calls=decode(json.dumps(decision),schema)
            self.assertEqual(calls[0]['function']['name'],'invalid_structured_decision')

    def test_history_is_translated_without_native_call_syntax_or_mutation(self):
        history=[{'role':'assistant','tool_calls':[{'id':'one','function':{'name':'query','arguments':{'path':'x','voltage':345}}}]},
                 {'role':'tool','tool_call_id':'one','content':'synthetic observation'}]
        original=json.dumps(history)
        wire,schema=prepare(history,self.tools)
        self.assertEqual(json.dumps(history),original)
        self.assertFalse(any('tool_calls' in message or message['role']=='tool' for message in wire))
        self.assertEqual(decode('{"kind":"final","content":"Done"}',schema),('Done',[]))
