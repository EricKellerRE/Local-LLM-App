"""Bounded, public execution events for the chat activity panel."""
import json
import threading
import traceback
from local_model_app.atomic_files import write_json
from local_model_app.cancellation import stop_event, ResponseStopped
from functools import wraps
from datetime import datetime, timezone


def resume_record(directory, chat_id):
    path = directory / 'tool_activity' / f'{chat_id}.resume.json'
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        # Existing runs from before the Resume feature have checkpoint + activity
        # files, but no dedicated resume record yet.
        try:
            checkpoint = json.loads(path.with_name(f'{chat_id}.state.json').read_text(encoding='utf-8'))
            events = read_events(path.with_name(f'{chat_id}.jsonl'))
            lifecycle = [e['event'] for e in events if e['event'] in {'request_started','request_finished','request_stopped','request_failed'}]
            if lifecycle and lifecycle[-1] != 'request_finished' and checkpoint.get('user_message'):
                return {'content':checkpoint['user_message'], 'status':'stopped'}
        except (OSError, ValueError):
            pass
        return {}


def save_resume(directory, chat_id, content, status):
    if not content:
        return
    path = directory / 'tool_activity' / f'{chat_id}.resume.json'
    write_json(path,{'content':content, 'status':status})


def track_chat(function):
    @wraps(function)
    async def wrapped(self, chat_id, *args, **kwargs):
        active = getattr(self, '_active_chat_requests', None)
        if active is None:
            active = self._active_chat_requests = {}
        if chat_id in active:
            raise RuntimeError('A response is already running for this chat.')
        content = args[0] if args else kwargs.get('content', '')
        save_resume(self.data_directory, chat_id, content, 'running')
        active[chat_id] = datetime.now(timezone.utc).isoformat()
        stops = getattr(self, '_chat_stop_events', None)
        if stops is None:
            stops = self._chat_stop_events = {}
        event = threading.Event()
        stops[chat_id] = event
        token = stop_event.set(event)
        path = self.data_directory / 'tool_activity' / f'{chat_id}.jsonl'
        self._log_activity(path, 'request_started')
        try:
            result = await function(self, chat_id, *args, **kwargs)
            try:
                checkpoint=json.loads(path.with_suffix('.state.json').read_text(encoding='utf-8'))
            except (OSError,ValueError):
                checkpoint={}
            save_resume(self.data_directory, chat_id, content,
                        'stopped' if checkpoint.get('status')=='tool_checkpoint' else 'completed')
            self._log_activity(path, 'request_finished')
            return result
        except ResponseStopped:
            save_resume(self.data_directory, chat_id, content, 'stopped')
            self._log_activity(path, 'request_stopped')
            answer = 'Stopped by user. Completed tool activity has been preserved.'
            content = args[0] if args else kwargs.get('content', '')
            if content:
                self.store.append_exchange(chat_id, content, answer)
            else:
                self.store.append_message(chat_id, 'assistant', answer)
            return answer
        except BaseException as exc:
            save_resume(self.data_directory, chat_id, content, 'failed')
            self._log_activity(path, 'request_failed', error_type=type(exc).__name__,
                               stack=[{'function': frame.name, 'line': frame.lineno,
                                       'file': frame.filename.replace('\\', '/').rsplit('/', 1)[-1]}
                                      for frame in traceback.extract_tb(exc.__traceback__)])
            raise
        finally:
            active.pop(chat_id, None)
            stops.pop(chat_id, None)
            stop_event.reset(token)
    return wrapped


def read_events(path):
    if not path.exists():
        return []
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 262144))
        if size > 262144:
            stream.readline()
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    events = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        name = item.get('event')
        if name not in {'analysis_plan_saved','analysis_step_active','analysis_retry','plan_step_active','plan_step_completed','plan_step_result','plan_action_rejected','incomplete','scratchpad_loaded','checkpoint_loaded','tool_call_rejected','tool_selection','repeated_observation','loop_stopped','request_stopped','stop_requested','request_started','request_finished','request_failed','model_acquired',
                        'planning_started','plan','tool_call','tool_result','answer',
                        'waiting_for_input','context_checkpoint','approved_tool_execution_started',
                        'approved_tool_execution_finished'}:
            continue
        row = {key: item.get(key) for key in ['timestamp','event','tool','call_id','step_id']}
        if name == 'analysis_step_active':
            row['step_id'] = f"Case {item.get('case_index')}/{item.get('case_total')}: {item.get('step_id')}"
        if name == 'request_failed':
            row['error_type'] = item.get('error_type')
        if 'arguments' in item:
            row['arguments'] = json.dumps(item['arguments'], ensure_ascii=False)[:4000]
        if name in {'tool_result','approved_tool_execution_finished'}:
            result = item.get('result') or {}
            row['failed'] = bool(result.get('isError') or result.get('error'))
        events.append(row)
    return events[-40:]
