"""Operational recovery controls; no scientific fits or discovery worlds."""
from datetime import datetime, timedelta, timezone
import pytest
from runners import release_v21_resource_hold as R


def fixture(tmp_path, monkeypatch):
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    (campaign / 'HOLD').write_text('fixture hold', encoding='utf-8')
    owner = dict(pid=12, creation=34, alive=True)
    reviewer = dict(pid=13, creation=35, alive=True)
    contract = campaign / 'helper' / 'CONTRACT.json'
    R.write(contract, dict(hold_sha256=R.file_digest(campaign / 'HOLD'),
                          supervisor=owner, reviewer=reviewer))
    R.write(campaign / 'ACCEPTANCE.json', dict(free_memory_reserve_bytes=16 * 2**30,
        worker_memory_ceiling_bytes=8 * 2**30, workers=3,
        report_start=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()))
    R.write(campaign / 'notifications/STATUS.json', dict(active_review_pid=None))
    R.write(campaign / 'QUEUE-STATUS.json', dict(active=[]))
    monkeypatch.setattr(R, 'native_identity', lambda pid: owner if pid == 12 else None)
    monkeypatch.setattr(R, 'available_memory', lambda: 25 * 2**30)
    return campaign, contract


def test_commit_dip_resets_two_observation_recovery(tmp_path, monkeypatch):
    campaign, contract = fixture(tmp_path, monkeypatch)
    capacities = iter([41, 7, 41, 41])
    monkeypatch.setattr(R, 'commit_headroom', lambda: dict(commit_free_bytes=next(capacities) * 2**30))
    observations = []
    def step(_):
        assert (campaign / 'HOLD').exists()
        observations.append(R.read(contract.parent / 'STATUS.json')['stable_observations'])
    monkeypatch.setattr(R.time, 'sleep', step)
    R.supervise(campaign, contract)
    assert observations == [1, 0, 1]
    assert not (campaign / 'HOLD').exists()
    assert R.read(contract.parent / 'STATUS.json')['required_commit_free_bytes'] == 40.5 * 2**30


@pytest.mark.parametrize('boundary', ['STOP', 'SUPERVISOR_STOP', 'changed_hold', 'changed_owner'])
def test_recovery_preserves_hold_at_boundary(tmp_path, monkeypatch, boundary):
    campaign, contract = fixture(tmp_path, monkeypatch)
    if boundary == 'changed_hold':
        (campaign / 'HOLD').write_text('replacement', encoding='utf-8')
    elif boundary == 'changed_owner':
        monkeypatch.setattr(R, 'native_identity', lambda pid: dict(pid=pid, creation=999, alive=True))
    else:
        (campaign / boundary).touch()
    monkeypatch.setattr(R, 'commit_headroom', lambda: pytest.fail('boundary must precede sampling'))
    R.supervise(campaign, contract)
    assert (campaign / 'HOLD').exists()


def test_failed_commit_probe_cannot_release(tmp_path, monkeypatch):
    campaign, contract = fixture(tmp_path, monkeypatch)
    def unavailable():
        raise OSError('native counter unavailable')
    monkeypatch.setattr(R, 'commit_headroom', unavailable)
    with pytest.raises(OSError):
        R.supervise(campaign, contract)
    assert (campaign / 'HOLD').exists()
