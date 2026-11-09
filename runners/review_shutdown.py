"""Conservative completion proof for an owned, noninteractive review process."""
from datetime import datetime, timedelta
import json
import os
from pathlib import Path


def completed_shutdown(config, entry, current):
    """Return proof only after an exact-token exec session ends and grace expires.

    Elapsed runtime, inactivity and errors are never sufficient. Ambiguous or
    malformed transcripts leave the owner alive. This is local operating data.
    """
    grace = config.get('completed_review_shutdown_grace_seconds', 0)
    if grace < 120 or not entry.get('review_token'):
        return None
    began = datetime.fromisoformat(entry['at'])
    if current - began < timedelta(seconds=grace):
        return None
    directory = Path(config['review_user_profile']) / '.codex/sessions'
    candidates = []
    # Session directories use local calendar dates, while records use UTC.
    for day in (began - timedelta(days=1), began, began + timedelta(days=1)):
        folder = directory / day.strftime('%Y/%m/%d')
        for path in folder.glob('rollout-*.jsonl'):
            try:
                with path.open('rb') as stream:
                    head = stream.read(1024 * 1024)
                meta = json.loads(head.split(b'\n', 1)[0])
                body = meta['payload']
                started = datetime.fromisoformat(body['timestamp'].replace('Z', '+00:00'))
                if (meta['type'] != 'session_meta' or body.get('source') != 'exec'
                        or not 0 <= (started - began).total_seconds() <= 60
                        or os.path.normcase(os.path.normpath(body['cwd'])) !=
                           os.path.normcase(os.path.normpath(config['cwd']))
                        or entry['review_token'].encode() not in head):
                    continue
                candidates.append(path)
            except (OSError, ValueError, KeyError, TypeError):
                continue
    if len(candidates) != 1:
        return None
    path = candidates[0]
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            tail = stream.read()
        # A partial write is not completion evidence.
        if not tail.endswith(b'\n'):
            return None
        final = json.loads(tail.splitlines()[-1])
        finished = datetime.fromisoformat(final['timestamp'].replace('Z', '+00:00'))
        if (final['type'] != 'event_msg' or final['payload']['type'] != 'task_complete'
                or finished < began or (current - finished).total_seconds() < grace):
            return None
        return dict(state='MODEL_COMPLETED_PROCESS_SHUTDOWN_RECOVERED',
                    review_pid=entry['pid'], review_token=entry['review_token'],
                    transcript=str(path), model_completed_at=finished.isoformat(),
                    recovered_at=current.isoformat(), grace_seconds=grace,
                    active_model_timeout=False)
    except (OSError, ValueError, KeyError, TypeError):
        return None
