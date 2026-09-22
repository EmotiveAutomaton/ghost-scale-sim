"""Independent correction checks; campaign verification runs only after these."""
from itertools import product
import gzip
import math
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import restoration_review as V
from ghostscale.validation.soundingline.v19 import provenance_restoration as P
from ghostscale.validation.soundingline.v19 import source_omission as O, unknown_change as U, local_world as L
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_scalar_products_corrections_and_reset():
    assert all(V.controls().values())
    law = np.random.default_rng(198122).dirichlet(np.ones(8), size=(16, 4))
    rows = [dict(step=i, source_step=i, source_id=str(i), context=i%4, endpoint=i%8) for i in range(1, 33)]
    for i in (3, 7, 15, 16, 23, 27): rows[i] = dict(rows[i-1], step=i+1)
    spec = V.correction_roster(rows)
    assert spec == dict(available_pairs=P.available_pairs(rows), conditions=P.conditions(P.available_pairs(rows)))
    supplied = V.omit_identity(rows)
    for c, arm, kind in product(spec['conditions'], V.ARMS, ('purpose', 'skill')):
        observed = V.apply_equivalences(supplied, c['pairs'])
        assert observed == P.correct(supplied, c['pairs'])
        cur, joint, hs = V.product_checkpoints(law, observed, arm, 32, kind, [32])[32]
        hs, prior = V.hypothesis_roster(arm, 32, kind)
        selected = V.distinct(observed); selected = selected[-16:] if arm == 'reset-16' else selected
        weights = np.asarray([p*math.prod(law[V.complement(m, k) if k != 'none' and r['source_step'] > t else m,
            r['context'], r['endpoint']] for r in selected) for (k, t, m), p in zip(hs, prior)])
        weights /= math.fsum(weights); current = np.zeros(16)
        for (k, t, m), w in zip(hs, weights): current[V.complement(m, k) if k != 'none' and 32 > t else m] += w
        assert np.allclose(joint, weights, atol=1e-13, rtol=0)
        assert np.allclose(cur, current, atol=1e-13, rtol=0)
    for pairs in ([[2, 1]], [[4, 3], [4, 3]], [[33, 1]], [[1, 2]]):
        with pytest.raises(ValueError): V.apply_equivalences(supplied, pairs)
    with pytest.raises(ValueError, match='empty support'):
        V.product_checkpoints(np.zeros_like(law), supplied, 'static', 32, 'purpose', [32])


def test_paired_regroup_and_aliases():
    rows = [dict(step=1, source_step=1, source_id='a', context=0, endpoint=0),
            dict(step=2, source_step=1, source_id='a', context=0, endpoint=0)]
    roster = {'32-1-32': V.correction_roster(rows)}; cells = []
    for lineage, draw, arm, condition in product((11, 12), (101, 102), V.ARMS, (0, 1)):
        cells.append(dict(lineage=lineage, draw=draw, arm=arm, condition=condition, length=32, kind='purpose',
            switched=True, duplicates=True, step=32, **{m: lineage+draw+condition for m in V.METRICS}))
    cfg = dict(bootstrap_seed=190501, bootstrap_resamples=100)
    result = V.regroup(cells, cfg, roster)
    assert len(result['means']) == 10 and len(result['contrasts']) == 200
    for r in result['contrasts']:
        delta = r['condition']-r['baseline_condition']
        assert r['mean'] == r['low'] == r['high'] == delta and r['draw_means'] == [delta, delta]
        if r['comparison'] == 'ascending-minus-descending': assert r['identity'] and delta == 0
    with pytest.raises(ValueError, match='duplicate'): V.regroup(cells+[cells[0]], cfg, roster)


def test_complete_native_fixture_and_corruption(tmp_path):
    lineage = 190969; records = L.enumerate_world(L.law(lineage))
    parent = tmp_path/'parent'; p = parent/'inputs'/f'lineage-{lineage}_points.json.gz'; p.parent.mkdir(parents=True)
    p.write_bytes(gzip.compress(canonical(records), mtime=0))
    cfg = dict(lineages=[lineage], draws=[190201, 190202], lengths=[32], paths_per_lineage=13824, input_files={p.name: file_digest(p)})
    write(parent/'SUMMARY.json', U.run(parent, {'design': cfg}, lambda **kw: None))
    omission = tmp_path/'omission'; (omission/'inputs').mkdir(parents=True)
    shutil.copyfile(p, omission/'inputs'/p.name); shutil.copytree(parent, omission/'inputs/parent')
    cfg['input_files'] = {q.relative_to(omission/'inputs').as_posix(): file_digest(q) for q in (omission/'inputs').rglob('*') if q.is_file()}
    write(omission/'SUMMARY.json', O.run(omission, {'design': cfg}, lambda **kw: None))
    root = tmp_path/'review/inputs/original'; (root/'inputs').mkdir(parents=True)
    shutil.copyfile(p, root/'inputs'/p.name)
    for source, name in ((parent, 'parent'), (omission, 'omission')):
        dest = root/'inputs'/name; dest.mkdir(); shutil.copytree(source/'raw', dest/'raw')
    streams = V.zipped(parent/'raw'/f'{lineage}-observations_points.json.gz'); roster = {}
    for stream in streams:
        for step in V.checkpoints(32):
            spec = V.correction_roster(stream['observations'][:step]); key = f'32-{int(stream["duplicates"])}-{step}'
            assert key not in roster or roster[key] == spec
            roster[key] = spec
    write(root/'inputs/CORRECTION_ROSTER.json', roster)
    cfg['input_files'] = {q.relative_to(root/'inputs').as_posix(): file_digest(q) for q in (root/'inputs').rglob('*') if q.is_file()}
    write(root/'PLAN.json', {'design': cfg})
    result = P.run(root, {'design': cfg}, lambda **kw: None); write(root/'SUMMARY.json', result)
    review = tmp_path/'review'; plan = dict(design=dict(input_files={}, target_plan_sha256=file_digest(root/'PLAN.json'), bootstrap_seed=190501, bootstrap_resamples=20))
    got = V.verify(review, plan, lambda **kw: None)
    assert got['streams'] == 256 and got['rows'] == 18560 and got['cells'] == 1160 and got['max_error'] < 1e-10
    grouped = read(review/'INDEPENDENT_REGROUP.json')
    assert len(grouped['means']) == 580 and len(grouped['contrasts']) == 9800
    for row in grouped['means']:
        matched = [r for r in result['cells'] if all(r[k] == row[k] for k in (*V.AXES, 'condition', 'arm'))]
        for m in V.METRICS: assert abs(row[m]['mean']-math.fsum(r[m] for r in matched)/2) < 1e-10
    packet = read(root/'PUBLIC_PACKET.json'); public = {r['input_sha256']: r for r in packet['cases']}
    bad = dict(packet, cases=[dict(packet['cases'][0], maker=0), *packet['cases'][1:]])
    with pytest.raises(ValueError, match='reader projection'): V.check_packet(bad, public)
    # Fail before full reconstruction on a corrupt frozen correction template.
    roster['32-0-8']['conditions'][0]['pairs'] = [[2, 1]]
    write(root/'inputs/CORRECTION_ROSTER.json', roster, immutable=False)
    with pytest.raises(ValueError, match='correction roster'): V.verify(review, plan, lambda **kw: None)
