import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import transfer_review as V, support_transfer as T
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from test_v19_support_transfer import parent, transfer_fixture


def test_independent_known_answers():
    assert all(V.controls().values())


def test_regroup_pairs_draws_within_lineage_and_preserves_empty_cells():
    cells = []
    for lineage in (1, 2):
        for draw in (10, 20):
            for rule in V.RULES:
                for mode in ('original', 'composition-holdout'):
                    for change in V.CHANGES:
                        for subset in V.V.SUBSETS:
                            for arm in V.ARMS:
                                value = (lineage + draw/10 + (arm != 'direct') + (rule != 'original')) if subset == 'all' else None
                                cells.append(dict(lineage=lineage, draw=draw, rule=rule, mode=mode, change=change, subset=subset, arm=arm,
                                                  **{m: value for m in V.METRICS}))
    result = V.regroup(cells, [1, 2], [10, 20], dict(bootstrap_seed=190501, bootstrap_resamples=20))
    for row in result['contrasts']:
        for metric in row['metrics'].values():
            if row['subset'] == 'all':
                assert metric['lineage_values'] == [1., 1.] and metric['complete_lineages'] == 2
                assert metric['low'] == metric['high'] == 1
            else: assert metric['mean'] is None and metric['lineage_values'] == []


def test_complete_native_transfer_and_corrupt_forecast_rejected(tmp_path, parent):
    original, cfg = transfer_fixture(tmp_path, parent)
    summary = T.run(original, dict(design=cfg), lambda **kw: None)
    write(original / 'SUMMARY.json', summary); write(original / 'PLAN.json', dict(design=cfg))
    review = tmp_path / 'review'; (review / 'inputs').mkdir(parents=True)
    shutil.copytree(original, review / 'inputs/original')
    shutil.copytree(parent / 'inputs/auxiliary', review / 'inputs/training')
    plan = dict(design=dict(input_files={}, target_plan_sha256=file_digest(original / 'PLAN.json'), bootstrap_seed=190501, bootstrap_resamples=20))
    result = V.run(review, plan, lambda **kw: None)
    assert all(result['controls'].values()) and result['paths'] == 27648 and result['queries'] == 640
    assert result['cells'] == 480 and result['rows'] == 20480 and result['max_error'] < 1e-10
    regroup = read(review / 'INDEPENDENT_REGROUP.json')
    for row in regroup['means']:
        if row['arm'] == 'true-law-oracle' and row['metrics']['loss']['mean'] is not None:
            assert abs(row['metrics']['loss']['mean'] + np.log(249/256)) < 1e-12
    path = review / 'inputs/original/forecasts/190301-original.npz'
    with np.load(path) as saved: arrays = {k: saved[k] for k in saved.files}
    arrays['learned-exact'][0, 0] += .01; np.savez(path, **arrays)
    with pytest.raises(ValueError): V.run(review, plan, lambda **kw: None)
