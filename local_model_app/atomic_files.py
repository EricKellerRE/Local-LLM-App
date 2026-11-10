"""Atomic JSON checkpoints with a bounded retry for transient Windows sharing locks."""
import json
import time
import uuid


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
        for attempt in range(6):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt==5:
                    raise
                time.sleep(0.02*(attempt+1))
    finally:
        temporary.unlink(missing_ok=True)
