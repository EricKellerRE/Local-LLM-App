from __future__ import annotations

import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from local_model_app.app_settings import AppSettingsStore
from local_model_app.config import _load_dotenv


ROOT = Path(__file__).resolve().parent
LOG_PATH = AppSettingsStore(ROOT).paths().data_directory / "logs" / "application.log"
_desktop_window: Any | None = None


class DesktopApi:
    def choose_directory(self) -> str | None:
        if _desktop_window is None:
            return None
        import webview

        selected = _desktop_window.create_file_dialog(webview.FOLDER_DIALOG, allow_multiple=False)
        return str(selected[0]) if selected else None


def _request_json(url: str, *, method: str = "GET", timeout: float = 1.0) -> dict[str, Any]:
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _message_box(message: str, title: str, flags: int = 0x10) -> int:
    if sys.platform == "win32":
        return int(ctypes.windll.user32.MessageBoxW(None, message, title, flags | 0x40000))
    print(f"{title}: {message}", file=sys.stderr)
    return 1


def _start_server(host: str, port: int) -> subprocess.Popen[bytes]:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = LOG_PATH.open("ab", buffering=0)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    command = [
        sys.executable,
        "-m",
        "local_model_app.server_process",
        "--host",
        host,
        "--port",
        str(port),
    ]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    try:
        return subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    finally:
        log.close()


def _wait_for_server(process: subprocess.Popen[bytes], base_url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = "The local service did not answer."
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"The local service stopped during startup. See {LOG_PATH}.")
        try:
            _request_json(f"{base_url}/health", timeout=0.5)
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(0.1)
    raise RuntimeError(f"Local Model could not start: {last_error}. See {LOG_PATH}.")


def _stop_server(
    process: subprocess.Popen[bytes],
    *,
    force: bool,
    base_url: str | None = None,
    grace_seconds: float = 15.0,
) -> None:
    if process.poll() is not None:
        return
    if force:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        return
    if base_url:
        try:
            _request_json(f"{base_url}/api/lifecycle/exit", method="POST", timeout=1.0)
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except OSError:
            pass
    else:
        try:
            process.send_signal(signal.SIGINT)
        except OSError:
            pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


class WindowLifecycle:
    """Translate the window's close button into an application shutdown."""

    IDYES = 6
    IDNO = 7
    IDCANCEL = 2
    YES_NO_CANCEL_WARNING = 0x00000003 | 0x00000030

    def __init__(
        self,
        base_url: str,
        window: Any,
        *,
        request_json: Callable[..., dict[str, Any]] = _request_json,
        ask: Callable[[str, str, int], int] = _message_box,
    ) -> None:
        self.base_url = base_url
        self.window = window
        self.request_json = request_json
        self.ask = ask
        self.allow_close = False
        self.force_shutdown = False
        self.waiting = False

    def closing(self) -> bool | None:
        if self.allow_close:
            return None
        try:
            status = self.request_json(f"{self.base_url}/api/lifecycle/status", timeout=1.0)
        except Exception:
            return None
        if not status.get("busy"):
            try:
                status = self.request_json(
                    f"{self.base_url}/api/lifecycle/prepare-shutdown",
                    method="POST",
                    timeout=1.0,
                )
            except Exception:
                self.allow_close = True
                return None
            if not status.get("busy"):
                self.allow_close = True
                return None

        reason = ", ".join(status.get("reasons") or ["work is still in progress"])
        choice = self.ask(
            f"Local Model cannot close yet because {reason}.\n\n"
            "Choose Yes to let it finish and close automatically.\n"
            "Choose No to end it now.\n"
            "Choose Cancel to keep the app open.",
            "Close Local Model?",
            self.YES_NO_CANCEL_WARNING,
        )
        if choice == self.IDNO:
            self.force_shutdown = True
            self.allow_close = True
            return None
        if choice == self.IDYES and not self.waiting:
            self.waiting = True
            try:
                self.window.set_title("Local Model — finishing current work…")
            except Exception:
                pass
            try:
                self.request_json(
                    f"{self.base_url}/api/lifecycle/prepare-shutdown",
                    method="POST",
                    timeout=1.0,
                )
            except Exception:
                self.waiting = False
                return False
            threading.Thread(target=self._close_when_idle, daemon=True).start()
        return False

    def _close_when_idle(self) -> None:
        while True:
            try:
                status = self.request_json(f"{self.base_url}/api/lifecycle/status", timeout=1.0)
                if not status.get("busy"):
                    self.allow_close = True
                    self.window.destroy()
                    return
            except Exception:
                self.allow_close = True
                self.window.destroy()
                return
            time.sleep(0.5)


def main() -> int:
    global _desktop_window
    _load_dotenv(ROOT / ".env")
    host = os.getenv("LOCAL_MODEL_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("LOCAL_MODEL_SERVER_PORT", "8765"))
    base_url = f"http://{host}:{port}"

    if _port_is_open(host, port):
        _message_box(
            f"Local Model is already running, or port {port} is in use.",
            "Local Model",
        )
        return 1

    try:
        import webview
    except ImportError:
        _message_box(
            "The desktop component is not installed. Run 'Install Local LLM.cmd' and try again.",
            "Local Model",
        )
        return 1

    process = _start_server(host, port)
    lifecycle: WindowLifecycle | None = None
    try:
        _wait_for_server(process, base_url)
        desktop_api = DesktopApi()
        frontend_build = (ROOT / "local_model_app" / "static" / "app.js").stat().st_mtime_ns
        window = webview.create_window(
            "Local Model",
            f"{base_url}/?build={frontend_build}",
            js_api=desktop_api,
            width=1280,
            height=820,
            min_size=(760, 560),
            background_color="#f7f7f5",
            confirm_close=False,
        )
        _desktop_window = window
        lifecycle = WindowLifecycle(base_url, window)
        window.events.closing += lifecycle.closing
        webview.start(debug=False)
        return 0
    except Exception as exc:
        _message_box(str(exc), "Local Model could not start")
        return 1
    finally:
        _stop_server(
            process,
            force=bool(lifecycle and lifecycle.force_shutdown),
            base_url=base_url,
        )


if __name__ == "__main__":
    raise SystemExit(main())
