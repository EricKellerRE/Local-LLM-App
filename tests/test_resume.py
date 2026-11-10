import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from fastapi import HTTPException
from local_model_app.activity_view import save_resume, resume_record, track_chat
from local_model_app.server import resume_chat, chat_activity


class ResumeTests(unittest.TestCase):
    def test_restart_detects_interrupted_request_and_resubmits_exact_prompt(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            save_resume(root,'chat','Synthetic original request','running')
            service=SimpleNamespace(data_directory=root,store=SimpleNamespace(get_chat=Mock()),
                                    loading=False,respond=AsyncMock(return_value='Done'))
            with patch('local_model_app.server.runtime',return_value=service):
                state=asyncio.run(chat_activity('chat',None))
                self.assertFalse(state['active'])
                self.assertTrue(state['resumable'])
                asyncio.run(resume_chat('chat',None))
            service.respond.assert_awaited_once_with('chat','Synthetic original request')

    def test_completed_and_running_chat_cannot_be_resumed(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            service=SimpleNamespace(data_directory=root,store=SimpleNamespace(get_chat=Mock()),respond=AsyncMock())
            for status,active in [('completed',{}),('running',{'chat':'now'})]:
                save_resume(root,'chat','Synthetic',status)
                service._active_chat_requests=active
                with patch('local_model_app.server.runtime',return_value=service),self.assertRaises(HTTPException) as error:
                    asyncio.run(resume_chat('chat',None))
                self.assertEqual(error.exception.status_code,409)
            service.respond.assert_not_awaited()

    def test_failure_retains_prompt_and_success_clears_resume(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            class Service:
                data_directory=Path(directory)
                _log_activity=Mock()
                fail=True
                @track_chat
                async def respond(self,chat_id,content):
                    if self.fail: raise ValueError('synthetic failure')
                    return 'Done'
            service=Service()
            with self.assertRaises(ValueError):asyncio.run(service.respond('chat','Synthetic'))
            self.assertEqual(resume_record(service.data_directory,'chat')['status'],'failed')
            service.fail=False
            asyncio.run(service.respond('chat','Synthetic'))
            self.assertEqual(resume_record(service.data_directory,'chat')['status'],'completed')

    def test_old_interrupted_checkpoint_is_resumable(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            folder=root/'tool_activity';folder.mkdir()
            (folder/'chat.state.json').write_text(json.dumps({'user_message':'Synthetic','status':'tool_checkpoint'}))
            (folder/'chat.jsonl').write_text(json.dumps({'event':'request_stopped'}))
            self.assertEqual(resume_record(root,'chat')['content'],'Synthetic')
