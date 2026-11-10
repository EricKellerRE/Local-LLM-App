import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from local_model_app.activity_view import read_events, track_chat


class ActivityTests(unittest.TestCase):
    def test_analysis_case_and_step_are_visible_without_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl'
            path.write_text(json.dumps({'event':'analysis_step_active','case_index':2,'case_total':5,'step_id':'capacity','rows':'private'}))
            events=read_events(path)
            self.assertEqual(events[0]['step_id'],'Case 2/5: capacity')
            self.assertNotIn('private',json.dumps(events))
    def test_reader_ignores_partial_lines_and_private_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_text('\n'.join([
                json.dumps({'event':'plan','content':'private planning text'}),
                json.dumps({'event':'tool_result','result':{'isError':True,'content':'large secret'}}),
                '{"incomplete":',
            ]), encoding='utf-8')
            events = read_events(path)
            self.assertEqual(len(events), 2)
            self.assertTrue(events[1]['failed'])
            self.assertNotIn('private', json.dumps(events))
            self.assertNotIn('secret', json.dumps(events))

    def test_tracking_cleans_up_after_error(self):
        class Runtime:
            data_directory = Path('.')
            events = []
            def _log_activity(self, path, event, **metadata):
                self.events.append(event)
            @track_chat
            async def respond(self, chat_id):
                self.assert_active = chat_id in self._active_chat_requests
                raise ValueError('failed')
        runtime = Runtime()
        with self.assertRaises(ValueError):
            asyncio.run(runtime.respond('chat'))
        self.assertTrue(runtime.assert_active)
        self.assertEqual(runtime._active_chat_requests, {})
        self.assertEqual(runtime.events, ['request_started','request_failed'])

    def test_large_results_do_not_make_reader_unbounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_text(json.dumps({'event':'tool_result','result':'x'*300000}) + '\n' +
                            json.dumps({'event':'answer'}), encoding='utf-8')
            self.assertEqual([e['event'] for e in read_events(path)], ['answer'])
