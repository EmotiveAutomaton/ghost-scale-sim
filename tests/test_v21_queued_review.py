import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_3.io import write, file_digest
from ghostscale.validation.soundingline.v21.metrics import scores, mean_scores
from runners.queue_review_v21 import checked_packet, regroup_rows


def fixture(tmp_path):
    p = tmp_path / 'packet'
    p.mkdir()
    write(p / 'PLAN.json', dict(design={'handler': 'learned'}, environment={'fixture': True},
                               sources={}, acceptance_sha256='accepted', archive_sha256=''))
    (p / 'SOURCE.zip').write_bytes(b'fixture')
    plan = __import__('json').loads((p / 'PLAN.json').read_text())
    plan['archive_sha256'] = file_digest(p / 'SOURCE.zip')
    write(p / 'PLAN.json', plan, immutable=False)
    rows = []
    for fit in (0, 1):
        for world in (1, 2):
            for reader in ('converged-direct', 'shared-maker'):
                prob = np.full((2, 128), 1 / 128)
                if reader == 'shared-maker':
                    prob[:, 0] += .1 * world
                    prob /= prob.sum(1, keepdims=True)
                lp = np.log(prob)
                y = np.zeros(2, int)
                a = scores(lp, y)
                raw = p / 'raw' / f'{fit}-{world}-{reader}_points.npz'
                raw.parent.mkdir(exist_ok=True)
                np.savez_compressed(raw, logp=lp, truth=y, **a)
                rows.append(dict(fit=fit, lineage=world, reader=reader, metrics=mean_scores(a)))
    write(p / 'SUMMARY.json', dict(population=rows))
    files = {x.relative_to(p).as_posix(): file_digest(x) for x in p.rglob('*')
             if x.is_file() and x.name not in ('PLAN.json', 'SOURCE.zip')}
    write(p / 'COMPLETE.json', dict(plan_sha256=file_digest(p / 'PLAN.json'), files=files))
    return p, dict(root=str(p), source=str(p), complete_sha256=file_digest(p / 'COMPLETE.json'),
                   plan_sha256=file_digest(p / 'PLAN.json'), acceptance_sha256='accepted')


def test_full_scores_and_nested_world_regroup(tmp_path):
    p, bound = fixture(tmp_path)
    r = checked_packet(bound, {'fixture': True}, {}, lambda **kw: None)
    assert r['forecast_rows'] == 16
    report = regroup_rows(p)
    assert len(report['world_rows']) == 4
    assert {r['fits_nested'] for r in report['world_rows']} == {2}
    assert next(r for r in report['contrasts'] if r['metric'] == 'log_loss')['upper'] < 0


def test_reused_score_receipt_still_checks_retained_bytes(tmp_path):
    p, bound = fixture(tmp_path)
    prior = tmp_path / 'receipt.json'
    write(prior, checked_packet(bound, {'fixture': True}, {}, lambda **kw: None))
    bound.update(score_receipt=str(prior), score_receipt_sha256=file_digest(prior))
    assert 'reused_score_receipt_sha256' in checked_packet(bound, {'fixture': True}, {}, lambda **kw: None)
    next((p / 'raw').glob('*.npz')).write_bytes(b'changed')
    with pytest.raises(AssertionError):
        checked_packet(bound, {'fixture': True}, {}, lambda **kw: None)


def test_changed_summary_is_not_a_valid_reaggregation(tmp_path):
    p, _ = fixture(tmp_path)
    from ghostscale.validation.soundingline.v18_3.io import read
    s = read(p / 'SUMMARY.json')
    s['population'][0]['metrics']['log_loss'] += 1
    write(p / 'SUMMARY.json', s, immutable=False)
    with pytest.raises(AssertionError):
        regroup_rows(p)


def test_resource_release_requires_owner_exit_and_headroom():
    from functools import partial
    from runners.release_v21_resource_hold import release_ready as check, matches
    release_ready = partial(check, commit_free=41, commit_required=41)
    assert not release_ready(24, 25, True, False, [])
    assert not release_ready(25, 25, True, True, [])
    assert not release_ready(25, 25, False, False, [])
    assert not release_ready(25, 25, True, False, ['active'])
    assert not release_ready(25, 25, True, False, [], other_review_alive=True)
    assert release_ready(25, 25, True, False, [])
    expected = dict(pid=12, creation=34)
    assert matches(dict(expected, alive=True), expected)
    assert not matches(dict(pid=12, creation=35, alive=True), expected)


def test_resource_release_fails_closed_without_commit_headroom():
    from runners.release_v21_resource_hold import release_ready
    assert not release_ready(100, 25, True, False, [])
    assert not release_ready(100, 25, True, False, [], commit_free=7, commit_required=41)
    assert release_ready(25, 25, True, False, [], commit_free=41, commit_required=41)


def test_complete_queue_consumer_dispatch(tmp_path, monkeypatch):
    from runners import queue_review_v21
    from ghostscale.validation.soundingline.v21.studies import run_study
    from ghostscale.validation.soundingline.v18_3.io import read
    packet, bound = fixture(tmp_path)
    monkeypatch.setattr(queue_review_v21, 'fingerprint', lambda: {'fixture': True})
    output = tmp_path / 'review'
    result = run_study(output, dict(handler='review-complete', packets=[bound],
                                   replay_receipts=[], family_replays=[]), lambda **kw: None)
    assert all(result['controls'].values()) and result['forecasts'] == 16
    assert result['scientific_acceptance'] is False
    assert read(output / 'scores/packet.json')['passed']
    assert len(read(output / 'regroup/packet.json')['world_rows']) == 4
