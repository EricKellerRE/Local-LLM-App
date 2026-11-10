import asyncio
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from tempfile import TemporaryDirectory
from local_model_app.cancellation import stop_event, check_stopped, generation_stop_requested, ResponseStopped
from local_model_app.activity_view import track_chat


class CancellationTests(unittest.TestCase):
    def test_signal_propagates_to_inference_thread_and_is_reset(self):
        async def run():
            event = threading.Event()
            token = stop_event.set(event)
            try:
                self.assertFalse(await asyncio.to_thread(generation_stop_requested))
                event.set()
                self.assertTrue(await asyncio.to_thread(generation_stop_requested))
                with self.assertRaises(ResponseStopped):
                    await asyncio.to_thread(check_stopped)
            finally:
                stop_event.reset(token)
            self.assertFalse(generation_stop_requested())
        asyncio.run(run())

    def test_stopped_response_preserves_exchange_and_cleans_up(self):
        class Runtime:
            data_directory = Path('.')
            store = SimpleNamespace(append_exchange=Mock())
            _log_activity = Mock()
            @track_chat
            async def respond(self, chat_id, content):
                self._chat_stop_events[chat_id].set()
                check_stopped()
        runtime = Runtime()
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime.data_directory = Path(directory)
            answer = asyncio.run(runtime.respond('chat', 'question'))
        self.assertIn('Stopped by user', answer)
        runtime.store.append_exchange.assert_called_once_with('chat', 'question', answer)
        self.assertEqual(runtime._chat_stop_events, {})
        self.assertEqual(runtime._active_chat_requests, {})
