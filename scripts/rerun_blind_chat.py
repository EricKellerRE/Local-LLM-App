"""Resubmit an existing user prompt without exposing case answers to the operator."""
import hashlib
import json
import sys
import httpx

chat_id = sys.argv[1]
base = 'http://127.0.0.1:8765'
with httpx.Client(timeout=30) as client:
    status = client.get(base + '/api/lifecycle/status')
    status.raise_for_status()
    if status.json()['busy']:
        raise SystemExit('App busy; no duplicate request submitted.')
    chat = client.get(f'{base}/api/chats/{chat_id}')
    chat.raise_for_status()
    # Only user-role content is selected. Never print or inspect assistant answers.
    prompts = [m['content'] for m in chat.json()['messages'] if m['role'] == 'user']
    if not prompts:
        raise SystemExit('No saved user prompt; nothing submitted.')
    prompt = prompts[-1]
    print(json.dumps({'event': 'resubmitting', 'chat_id': chat_id,
                      'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}), flush=True)
    with client.stream('POST', f'{base}/api/chats/{chat_id}/messages',
                       json={'content': prompt}, timeout=None) as response:
        # Discard the body, including all case answers. The app stores it for its UI.
        print(json.dumps({'event': 'request_returned', 'http_status': response.status_code}), flush=True)
        for _ in response.iter_bytes():
            pass
