from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from launcher import DesktopApi, WindowLifecycle
from local_model_app.task_engine import UniversalTaskEngine
from local_model_app.task_store import TaskStore


class WindowLifecycleTests(unittest.TestCase):
    def test_browser_api_does_not_expose_the_recursive_native_window(self) -> None:
        self.assertEqual(vars(DesktopApi()), {})

    def test_idle_window_closes_without_prompt(self) -> None:
        request = Mock(return_value={"busy": False})
        ask = Mock()
        lifecycle = WindowLifecycle("http://local", Mock(), request_json=request, ask=ask)

        self.assertIsNone(lifecycle.closing())
        self.assertTrue(lifecycle.allow_close)
        ask.assert_not_called()
        self.assertEqual(request.call_count, 2)

    def test_end_now_marks_forced_shutdown(self) -> None:
        request = Mock(return_value={"busy": True, "reasons": ["a response is in progress"]})
        lifecycle = WindowLifecycle(
            "http://local",
            Mock(),
            request_json=request,
            ask=Mock(return_value=WindowLifecycle.IDNO),
        )

        self.assertIsNone(lifecycle.closing())
        self.assertTrue(lifecycle.force_shutdown)
        self.assertTrue(lifecycle.allow_close)

    def test_cancel_keeps_window_open(self) -> None:
        request = Mock(return_value={"busy": True, "reasons": ["the model is loading"]})
        lifecycle = WindowLifecycle(
            "http://local",
            Mock(),
            request_json=request,
            ask=Mock(return_value=WindowLifecycle.IDCANCEL),
        )

        self.assertFalse(lifecycle.closing())
        self.assertFalse(lifecycle.allow_close)


class TaskEngineShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_stop_wakes_idle_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.sqlite3")
            engine = UniversalTaskEngine(store, Mock(), poll_seconds=60)
            await engine.start()
            engine.request_stop()
            await asyncio.wait_for(engine.stop(), timeout=1)
            store.close()


if __name__ == "__main__":
    unittest.main()
