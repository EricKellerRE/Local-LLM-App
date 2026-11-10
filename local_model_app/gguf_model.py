"""Direct GGUF inference through an app-owned standalone llama.cpp process."""
from __future__ import annotations

import atexit
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
import uuid

import httpx

from local_model_app.model import TransformersModel, AssistantReply
from local_model_app.cancellation import check_stopped, stop_event, ResponseStopped
from local_model_app.generation_context import current_generation_context
from local_model_app.structured_tools import prepare as prepare_structured_tools, decode as decode_structured_tools


class GgufModel(TransformersModel):
    def __init__(self, settings, **kwargs):
        super().__init__(settings, **kwargs)
        self.process = None
        self.base_url = None
        self._context = settings.context_window or 16384
        self._load_lock = threading.Lock()
        self.log_path = (self.telemetry_path.parent if self.telemetry_path else Path('data')) / 'logs' / f'llama-{uuid.uuid4().hex[:8]}.log'
        atexit.register(self.close)

    @property
    def loaded(self):
        return self.process is not None and self.process.poll() is None and self.model is not None

    @property
    def native_context_window(self):
        return self._context

    def close(self):
        process, self.process = self.process, None
        self.model = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    def _load(self):
        with self._load_lock:
            if self.loaded:
                return
            check_stopped()
            root = Path(__file__).resolve().parents[1]
            model = (root / Path(self.settings.model_id).expanduser()).resolve()
            if not model.is_file() or model.suffix.lower() != '.gguf':
                raise ValueError('GGUF model ID must be a local .gguf file (first shard for split models).')
            if model.name.lower().startswith('mmproj'):
                raise ValueError('Select the language model GGUF, not its vision projector.')
            executable = root / Path(os.getenv('LOCAL_LLAMA_SERVER', 'runtimes/llama-b12144/llama-server.exe'))
            if not executable.is_file():
                raise RuntimeError('Standalone llama-server is missing. Set LOCAL_LLAMA_SERVER to its executable.')
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                port = reservation.getsockname()[1]
            self.base_url = f'http://127.0.0.1:{port}'
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [str(executable), '-m', str(model), '--host', '127.0.0.1', '--port', str(port),
                       '-c', str(self._context), '-np', '1', '-t', os.getenv('LOCAL_GGUF_THREADS', '8'),
                       '-ngl', '0' if self.settings.device == 'cpu' else 'auto', '--jinja', '--no-webui']
            with self.log_path.open('ab') as log:
                runtime_env = os.environ.copy()
                if runtime_env.get('LOCAL_CUDA_BIN'):
                    runtime_env['PATH'] = runtime_env['LOCAL_CUDA_BIN'] + os.pathsep + runtime_env.get('PATH', '')
                self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                                env=runtime_env,
                                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            try:
                deadline = time.monotonic() + 300
                with httpx.Client(trust_env=False, timeout=2) as client:
                    while time.monotonic() < deadline:
                        check_stopped()
                        if self.process.poll() is not None:
                            raise RuntimeError(f'llama.cpp exited during load; see {self.log_path}')
                        try:
                            response = client.get(self.base_url + '/health')
                            if response.status_code == 200:
                                self.model = True
                                return
                        except httpx.HTTPError:
                            pass
                        time.sleep(.2)
                raise TimeoutError(f'GGUF load exceeded five minutes; see {self.log_path}')
            except BaseException:
                self.close()
                raise

    def _post(self, path, payload):
        with httpx.Client(trust_env=False, timeout=120) as client:
            result = client.post(self.base_url + path, json=payload)
            result.raise_for_status()
            return result.json()

    def count_prompt_tokens(self, messages, tools=None):
        self._load()
        wire = prepare_structured_tools(messages, tools)[0] if tools else self._wire_messages(messages)
        payload = {'messages': wire, 'add_generation_prompt': True}
        formatted = self._post('/apply-template', payload)['prompt']
        return len(self._post('/tokenize', {'content': formatted, 'add_special': True})['tokens'])

    @staticmethod
    def _wire_messages(messages):
        messages = copy.deepcopy(messages)
        for message in messages:
            for call in message.get('tool_calls') or []:
                function = call.get('function', {})
                if isinstance(function.get('arguments'), dict):
                    function['arguments'] = json.dumps(function['arguments'])
        return messages

    def chat(self, messages, *, tools=None, max_new_tokens=None, temperature=None,
             generation_class='ordinary_response', telemetry_context=None):
        self._load()
        check_stopped()
        budget = max_new_tokens or self.settings.max_new_tokens
        wire, decision_schema = prepare_structured_tools(messages, tools, require_tool=generation_class == 'required_tool_action') if tools else (self._wire_messages(messages), None)
        payload = {'model': self.settings.model_id, 'messages': wire,
                   'max_tokens': budget, 'temperature': self.settings.temperature if temperature is None else temperature,
                   'top_p': self.settings.top_p, 'stream': True, 'stream_options': {'include_usage': True},
                   'cache_prompt': True}
        if decision_schema:
            payload['response_format'] = {'type':'json_schema', 'json_schema':{'name':'tool_decision','strict':True,'schema':decision_schema}}
        if self.settings.reasoning_budget is not None:
            payload['chat_template_kwargs'] = {'enable_thinking': self.settings.reasoning_budget > 0}
            payload['reasoning_budget'] = self.settings.reasoning_budget
        started = time.monotonic()
        content, calls, usage, finish = '', {}, {}, None
        event = stop_event.get()
        done = threading.Event()
        # Terminate only this adapter's child to interrupt blocked prefill as well as decoding.
        def cancel_watch():
            while not done.wait(.1):
                if event is not None and event.is_set():
                    self.close()
                    return
        watcher = threading.Thread(target=cancel_watch, daemon=True)
        watcher.start()
        try:
            with httpx.Client(trust_env=False, timeout=httpx.Timeout(600, connect=5)) as client:
                with client.stream('POST', self.base_url + '/v1/chat/completions', json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        check_stopped()
                        if not line.startswith('data:'):
                            continue
                        raw = line[5:].strip()
                        if raw == '[DONE]':
                            break
                        chunk = json.loads(raw)
                        usage = chunk.get('usage') or usage
                        for choice in chunk.get('choices', []):
                            finish = choice.get('finish_reason') or finish
                            delta = choice.get('delta') or {}
                            content += delta.get('content') or ''
                            for part in delta.get('tool_calls') or []:
                                call = calls.setdefault(part['index'], {'id':'', 'type':'function', 'function':{'name':'','arguments':''}})
                                if part.get('id'):
                                    call['id'] = part['id']
                                for key in ['name', 'arguments']:
                                    call['function'][key] += (part.get('function') or {}).get(key) or ''
            check_stopped()
        except Exception:
            check_stopped()
            raise
        finally:
            done.set()
            watcher.join(timeout=12)
        if finish is None:
            raise RuntimeError('GGUF stream ended without a completion marker.')
        tool_calls = [calls[index] for index in sorted(calls)]
        for call in tool_calls:
            call['id'] = call['id'] or f'call_{uuid.uuid4().hex}'
            raw = call['function']['arguments'] or '{}'
            try:
                call['function']['arguments'] = json.loads(raw)
            except json.JSONDecodeError:
                # Preserve invalid output for the coordinator's bounded retry.
                # Never guess or repair an argument before executing a tool.
                call['function']['arguments'] = raw
        if decision_schema:
            content, tool_calls = decode_structured_tools(content, decision_schema)
        elapsed = time.monotonic() - started
        generated = usage.get('completion_tokens')
        record = {'timestamp':datetime.now(timezone.utc).isoformat(), 'generation_class':generation_class,
                  'model_id':self.settings.model_id, 'backend':'llama.cpp', 'prompt_tokens':usage.get('prompt_tokens'),
                  'generated_tokens':generated, 'max_new_tokens':budget, 'elapsed_seconds':elapsed,
                  'tokens_per_second':generated / elapsed if generated is not None and elapsed else None,
                  'stop_reason':'length' if finish == 'length' else 'tool_call' if tool_calls else 'completed',
                  **current_generation_context(), **(telemetry_context or {})}
        self._write_telemetry(record)
        self._write_generation_record({'schema_version':1, 'generation_id':uuid.uuid4().hex,
            'context':current_generation_context(), 'request':{'messages':self._public_messages(messages), 'tools':tools or []},
            'response':{'content':content, 'tool_calls':tool_calls}, 'telemetry':record}, current_generation_context())
        return AssistantReply(content=content.strip(), tool_calls=tool_calls,
                              prompt_tokens=usage.get('prompt_tokens'), generated_tokens=generated,
                              stop_reason=record['stop_reason'], elapsed_seconds=elapsed)


def create_model(settings, **kwargs):
    cls = GgufModel if settings.model_kind == 'gguf' or settings.model_id.lower().endswith('.gguf') else TransformersModel
    return cls(settings, **kwargs)
