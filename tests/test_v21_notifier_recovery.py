import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from runners import watch_v21_events as watcher


def test_failed_cli_is_retained_without_broken_pipe(tmp_path):
    state = tmp_path / 'notifications'
    prompt = tmp_path / 'prompt.txt'
    prompt.write_text('A' * 200000, encoding='utf-8')
    write(tmp_path / 'ACCEPTANCE.json', {})
    write(tmp_path / 'events' / 'failure.json', dict(id='failure', kind='supervisor_failed'))
    config = dict(command=[sys.executable, '-c', 'import sys; sys.exit(7)'],
                  cwd=str(tmp_path), executable_sha256=file_digest(Path(sys.executable)),
                  prompt=str(prompt), prompt_sha256=file_digest(prompt))
    write(state / 'CONFIG.json', config)
    (state / 'ARMED').touch()
    watcher.supervise(tmp_path, state, state / 'CONFIG.json', once=True)
    delivery = read(state / 'DELIVERY.json')
    assert delivery['attempts'][0]['exit_code'] == 7
    assert delivery['attempts'][0]['acknowledged'] == []
    assert delivery['retry_not_before']
    assert not (state / 'ARMED').exists()


def test_profile_survives_missing_inherited_home(tmp_path, monkeypatch):
    (tmp_path / '.codex').mkdir()
    monkeypatch.delenv('USERPROFILE', raising=False)
    environment = watcher.review_environment(dict(review_user_profile=str(tmp_path)))
    assert environment['USERPROFILE'] == str(tmp_path)
    assert 'USERPROFILE' not in watcher.os.environ


def test_completions_wait_but_health_and_hold_release(tmp_path):
    config = dict(batch_completions_until_health=True)
    completed = dict(id='result', kind='batch_complete')
    assert watcher.delivery_events(tmp_path, config, [completed]) == []
    health = dict(id='health', kind='twelve_hour_native_health')
    assert watcher.delivery_events(tmp_path, config, [completed, health]) == [completed, health]
    (tmp_path / 'HOLD').touch()
    assert watcher.delivery_events(tmp_path, config, [completed]) == [completed]


def test_fixed_twelve_hour_checkpoints_are_once_only(tmp_path):
    config = dict(checkpoints=[dict(id=f'health-{hour}', kind='twelve_hour_native_health',
                  at=f'2026-10-12T{hour:02d}:00:00+00:00') for hour in (9, 21)])
    watcher.checkpoint_events(tmp_path, {}, config, datetime(2026, 10, 12, 9, tzinfo=timezone.utc))
    first = file_digest(tmp_path / 'events' / 'health-9.json')
    watcher.checkpoint_events(tmp_path, {}, config, datetime(2026, 10, 12, 21, tzinfo=timezone.utc))
    assert file_digest(tmp_path / 'events' / 'health-9.json') == first
    assert len(list((tmp_path / 'events').glob('*.json'))) == 2


def test_closed_campaign_does_not_emit_or_deliver(tmp_path, monkeypatch):
    state = tmp_path / 'notifications'
    report = tmp_path / 'FINAL_REPORT.md'
    report.write_text('Complete with explicit dispositions.', encoding='utf-8')
    write(tmp_path / 'CLOSEOUT.json', dict(state='closed', owned_scientific_services_stopped=True,
          final_report=str(report), final_report_sha256=file_digest(report)))
    write(tmp_path / 'QUEUE-STATUS.json', dict(pid=123))
    monkeypatch.setattr(watcher, 'alive', lambda pid: False)
    write(state / 'CONFIG.json', {})
    (state / 'ARMED').touch()
    watcher.supervise(tmp_path, state, state / 'CONFIG.json', once=True)
    assert read(state / 'STATUS.json')['state'] == 'closed; no further wakes'
    assert not (state / 'ARMED').exists()
    assert not (tmp_path / 'events').exists()
    assert not (state / 'DELIVERY.json').exists()
