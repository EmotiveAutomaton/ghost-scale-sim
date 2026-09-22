import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import support_transfer as T, forward_support as S, rollout as R, local_world as L, missing_tool as M
from ghostscale.validation.soundingline.v18_3.io import write, read, file_digest


def test_known_answer_controls_and_empty_strata():
    assert all(T.controls().values())
    assert len(T.summarize([], {})) == 15
    assert all(r['loss'] is None for r in T.summarize([], {}))


def test_support_and_change_crossing_uses_conditional_mass():
    rows = [dict(query_seen=False, all_primitives_seen=True, withheld_composition=True,
                 changed=True, probability_mass=.25, loss=4., squared_error=1., true_probability=.1),
            dict(query_seen=True, all_primitives_seen=True, withheld_composition=False,
                 changed=False, probability_mass=.75, loss=0., squared_error=0., true_probability=1.)]
    cells = {(x['change'], x['subset']): x for x in T.summarize(rows, {})}
    assert cells['all', 'all']['loss'] == 1
    assert cells['changed', 'heldout-seen-primitives']['loss'] == 4
    assert cells['stay', 'heldout-composition']['loss'] is None


@pytest.fixture
def parent(tmp_path):
    root = tmp_path / 'parent'; inputs = root / 'inputs'; (inputs / 'models').mkdir(parents=True)
    rr = L.enumerate_world(L.law(190961))
    selected = [next(r for r in rr if S.withheld(R.query(r))), next(r for r in rr if not S.withheld(R.query(r)))]
    write(inputs / 'auxiliary/selection-190301.json', dict(paths=selected))
    counts, direct = R.fit(selected); keys = sorted(direct)
    np.savez(inputs / 'models/190301-2048.npz', transition_counts=counts, direct_keys=np.array(keys), direct_counts=np.array([direct[q] for q in keys]))
    pins = {p.relative_to(inputs).as_posix(): file_digest(p) for p in inputs.rglob('*') if p.is_file()}
    cfg = dict(arms=list(S.ARMS), epsilon=S.EPSILON, lineages=[190962], training_draws=[190301], input_files=pins)
    result = S.run(root, dict(design=cfg), lambda **kw: None); write(root / 'SUMMARY.json', result)
    return root


def transfer_fixture(tmp_path, parent):
    root = tmp_path / 'transfer'; inputs = root / 'inputs'; inputs.mkdir(parents=True)
    for folder in ('models', 'forecasts'): shutil.copytree(parent / folder, inputs / folder)
    shutil.copyfile(parent / 'SUMMARY.json', inputs / 'SUMMARY.json')
    pins = {p.relative_to(inputs).as_posix(): file_digest(p) for p in inputs.rglob('*') if p.is_file()}
    cfg = dict(arms=list(T.ARMS), rules=list(M.RULES), epsilon=S.EPSILON, lineages=[190962], training_draws=[190301], queries=640, input_files=pins)
    return root, cfg


def test_complete_transfer_reproduces_parent_and_separates_true_law(tmp_path, parent):
    root, cfg = transfer_fixture(tmp_path, parent)
    result = T.run(root, dict(design=cfg), lambda **kw: None)
    assert result['fits'] == 0 and result['paths'] == 27648 and result['queries'] == 640
    assert read(root / 'PARENT_REPRODUCTION.json')['passed']
    assert all(file_digest(root / 'inputs' / n) == h for n, h in cfg['input_files'].items())
    true = [r for r in result['cells'] if r['arm'] == 'true-law-oracle' and r['population_mass']]
    assert all(abs(r['loss'] + np.log(249/256)) < 1e-12 for r in true)
    wrong = [r for r in result['cells'] if r['arm'] == 'oracle-exact' and r['rule'] == 'presentation-tool' and r['change'] == 'changed' and r['population_mass']]
    assert wrong and all(abs(r['loss'] - np.log(256)) < 1e-12 for r in wrong)
    for packet in read(root / 'reader/QUERIES.json'): R.validate_visible(packet)


def test_altered_parent_predictions_cannot_pass_reproduction(tmp_path, parent):
    root, cfg = transfer_fixture(tmp_path, parent)
    p = root / 'inputs/forecasts/190301-original.npz'
    with np.load(p) as saved: arrays = {k: saved[k] for k in saved.files}
    arrays['direct'][0] = arrays['direct'][0][::-1] + .01
    np.savez(p, **arrays); cfg['input_files']['forecasts/190301-original.npz'] = file_digest(p)
    with pytest.raises(ValueError, match='parent forecasts'): T.run(root, dict(design=cfg), lambda **kw: None)
