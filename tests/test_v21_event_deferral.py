from datetime import datetime, timezone
import pytest
from ghostscale.validation.soundingline.v18_3.io import write
from runners.watch_v21_events import actionable


def fixture(tmp_path):
    state = tmp_path / 'notifications'
    event = dict(id='complete', kind='batch_complete', event_sha256='frozen')
    rule = dict(event_sha256='frozen', requires=['replay'], until='2026-10-10T22:00:00+00:00')
    write(state / 'DEFERRED.json', dict(events={'complete': rule}))
    return state, event


def test_waiting_has_no_ack_and_health_bypasses(tmp_path):
    state, event = fixture(tmp_path)
    health = dict(id='health', kind='six_hour_native_health', event_sha256='health-hash')
    ready, waiting = actionable(tmp_path, state, [event, health], datetime(2026, 10, 10, 20, tzinfo=timezone.utc))
    assert ready == [health] and waiting == [event]
    assert not (state / 'acks').exists()


@pytest.mark.parametrize('receipt', ['COMPLETE.json', 'DISPOSITION.json'])
def test_dependency_completion_or_failure_releases(tmp_path, receipt):
    state, event = fixture(tmp_path)
    write(tmp_path / 'packets' / 'replay' / receipt, {'fixture': True})
    assert actionable(tmp_path, state, [event], datetime(2026, 10, 10, 20, tzinfo=timezone.utc)) == ([event], [])


def test_deadline_releases_and_changed_hash_fails(tmp_path):
    state, event = fixture(tmp_path)
    assert actionable(tmp_path, state, [event], datetime(2026, 10, 10, 22, tzinfo=timezone.utc)) == ([event], [])
    with pytest.raises(ValueError, match='hash differs'):
        actionable(tmp_path, state, [dict(event, event_sha256='changed')])


def test_sparse_shift_reference_keeps_native_history():
    import numpy as np
    from ghostscale.validation.soundingline.v21 import world as W, oracle as O
    from ghostscale.validation.soundingline.v21.checker import scalar_mixture
    native = W.enumerate_world(71000000)
    changed = W.enumerate_world(71000000, 'changed-tool')
    chosen = W.sampled(changed, 2, 'review-mixed-history-control')
    x, _ = W.inputs(native, chosen, 'context', 4, 'review-mixed-history-control',
                    history_tier='complete', history_observed=True)
    expected, _ = O.posterior(x, 'context', [W.law(71000000, 'changed-tool')],
                             history_tier='complete', history_observed=True,
                             shift='changed-tool', history_shift='native')
    actual = scalar_mixture([changed], x, 'context', W.CONTEXTS, 'complete', False, True,
                            history_worlds=[native])
    assert np.allclose(actual, np.exp(expected), atol=1e-12, rtol=1e-10)
