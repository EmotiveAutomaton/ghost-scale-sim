"""Bounded supplied source-equivalence correction on retained observations."""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import time
import numpy as np
from ..v18_3.io import canonical, digest, file_digest, read, write
from . import source_omission as O, unknown_change as U, transient_filter as T
from .crossed_review import population
from .local_world import CONTEXTS


def available_pairs(observations):
    """Evaluator prepares equivalences; the reader receives only chosen pairs."""
    T.unique_sources(observations)
    first = {}; pairs = []
    for row in observations:
        source = row['source_id']
        if source in first:
            pairs.append([row['step'], first[source]])
        else:
            first[source] = row['step']
    return pairs


def conditions(pairs):
    """Deduplicate identical interventions, retaining all budget/order aliases."""
    grouped = {}
    for order in ('ascending', 'descending'):
        ordered = sorted(pairs, reverse=order == 'descending')
        for budget in (0, 1, 4, 'all'):
            chosen = sorted(ordered if budget == 'all' else ordered[:budget])
            key = tuple(map(tuple, chosen))
            if key not in grouped:
                grouped[key] = dict(condition=len(grouped), pairs=chosen, aliases=[])
            grouped[key]['aliases'].append(dict(order=order, budget=budget))
    return list(grouped.values())


def correct(observations, pairs, rename=False):
    """Apply observed equal-source statements; never infer them from timestamps."""
    rows = [dict(r) for r in observations]
    if [r['step'] for r in rows] != list(range(1, len(rows)+1)):
        raise ValueError('nonconsecutive observation times')
    if len(T.unique_sources(rows)) != len(rows):
        raise ValueError('correction input must have distinct supplied identities')
    targets = set()
    for copy, original in pairs:
        if not 1 <= original < copy <= len(rows) or copy in targets:
            raise ValueError('invalid or duplicate source pair')
        targets.add(copy)
        if any(rows[copy-1][k] != rows[original-1][k]
               for k in ('source_step', 'context', 'endpoint')):
            raise ValueError('conflicting source correction')
        rows[copy-1]['source_id'] = rows[original-1]['source_id']
    if rename:
        # A bijection of the ORIGINAL distinct labels, without equivalence.
        rows = [dict(r, source_id=f'renamed-{r["step"]:03d}') for r in observations]
    T.unique_sources(rows)
    return rows


def controls():
    a = dict(step=1, source_step=1, source_id='source-001', context=0, endpoint=0)
    original = [a, dict(a, step=2)]
    omitted = O.supplied_observations(original)
    fixed = correct(omitted, [[2, 1]])
    law = np.full((16, 4, 8), 1/8)
    flat = U.filter_checkpoints(law, fixed, 'unknown-time-type', 32, 'purpose', [2])[2][0]
    law[:8, 0] = [.8, .2, 0, 0, 0, 0, 0, 0]
    law[8:, 0] = [.2, .8, 0, 0, 0, 0, 0, 0]
    post = U.filter_checkpoints(law, fixed, 'static', 32, 'purpose', [2])[2][0]
    return dict(O.controls(), **{
        'positive:restored_single_likelihood': bool(np.allclose(post[:8], .8/8)),
        'placebo:neutral_restoration': bool(np.allclose(flat, 1/16)),
        'positive:full_source_identity': fixed == original,
        'positive:empty_correction_identity': correct(omitted, []) == omitted,
        'positive:deduplicated_interventions': len(conditions([[2, 1]])) == 2,
    })


def run(root, plan, pulse):
    cfg = plan['design']; checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()):
        raise ValueError('restoration controls failed')
    for n, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h:
            raise ValueError('input differs')
    roster = read(root/'inputs/CORRECTION_ROSTER.json')
    cells = []; packets = {}; total_rows = total_streams = total_conditions = 0
    identities = defaultdict(int)
    for lineage in cfg['lineages']:
        pulse(phase='restoration-native-support', lineage=lineage)
        records = json.loads(gzip.decompress((root/'inputs'/f'lineage-{lineage}_points.json.gz').read_bytes()))
        if len(records) != cfg['paths_per_lineage']:
            raise ValueError('path denominator')
        population(records, lineage, 'original')
        table = T.endpoint_law(records); write(root/'evaluator'/f'{lineage}-law.json', table.tolist())
        parent = root/'inputs/parent'; omitted = root/'inputs/omission'
        streams = json.loads(gzip.decompress((parent/'raw'/f'{lineage}-observations_points.json.gz').read_bytes()))
        expected = list(product(cfg['draws'], range(16), cfg['lengths'], ('purpose','skill'), (False,True), (False,True)))
        actual = [tuple(s[k] for k in ('draw','maker','length','kind','switched','duplicates')) for s in streams]
        if actual != expected: raise ValueError('parent stream roster differs')
        def index(folder):
            rows = json.loads(gzip.decompress((folder/'raw'/f'{lineage}-forecasts_points.json.gz').read_bytes()))
            indexed = {(r['stream'], r['step'], r['arm']): r for r in rows}
            if len(indexed) != len(rows): raise ValueError('duplicate parent rows')
            return indexed
        parents = index(parent); omissions = index(omitted)
        rows = []; acc = defaultdict(list); joints = defaultdict(list); mapping = defaultdict(list)
        for si, stream in enumerate(streams):
            pulse(phase='provenance-correction', lineage=lineage, stream=si)
            length = stream['length']; kind = stream['kind']; total_streams += 1
            supplied = O.supplied_observations(stream['observations'])
            for step in T.checkpoints(length):
                available = available_pairs(stream['observations'][:step])
                key = f'{length}-{int(stream["duplicates"])}-{step}'
                spec = roster[key]
                if spec['available_pairs'] != available or spec['conditions'] != conditions(available):
                    raise ValueError('frozen correction roster differs')
                for condition in spec['conditions']:
                    cid = condition['condition']; pairs = condition['pairs']; total_conditions += 1
                    observed = correct(supplied[:step], pairs)
                    visible = dict(contexts=list(CONTEXTS), observations=supplied[:step], source_equivalences=pairs)
                    ident = digest(visible); packets[ident] = dict(input_sha256=ident, inputs=visible)
                    for arm in U.ARMS:
                        start = time.process_time()
                        current, joint, hs = U.filter_checkpoints(table, observed, arm, length, kind, [step])[step]
                        joint = joint.reshape(-1)
                        with (root/'TIMING.jsonl').open('a', encoding='utf-8') as f:
                            f.write(json.dumps(dict(lineage=lineage, stream=si, step=step, condition=cid, arm=arm,
                                                   cpu_seconds=time.process_time()-start), sort_keys=True)+'\n')
                        truth = int(U.FLIPS[kind][stream['maker']]) if stream['switched'] and step > length//2 else stream['maker']
                        values, forecast = T.score(table, current, truth)
                        baseline = omissions[si, step, arm]
                        if not pairs:
                            if (current.tolist() != baseline['posterior'] or forecast.tolist() != baseline['forecast']
                                    or any(values[m] != baseline[m] for m in T.METRICS)):
                                raise ValueError('zero correction identity differs')
                            identities['zero'] += 1
                            # Label-only renaming receives no equivalence information.
                            named = U.filter_checkpoints(table, correct(supplied[:step], [], rename=True), arm, length, kind, [step])[step]
                            if not np.array_equal(named[0], current) or not np.array_equal(named[1].reshape(-1), joint):
                                raise ValueError('label-only identity differs')
                            identities['label_only'] += 1
                        if pairs == available:
                            baseline = parents[si, step, arm]
                            if (current.tolist() != baseline['posterior'] or forecast.tolist() != baseline['forecast']
                                    or any(values[m] != baseline[m] for m in T.METRICS)):
                                raise ValueError('full correction identity differs')
                            identities['full'] += 1
                        jkey = f'{length}-{kind}-{arm}'; ji = len(joints[jkey])
                        joints[jkey].append(joint); mapping[jkey].append([si, step, cid])
                        retained = T.unique_sources(observed)
                        if arm == 'reset-16': retained = retained[-16:]
                        true_sources = len({stream['observations'][r['step']-1]['source_id'] for r in retained})
                        types = {k: math.fsum(float(w) for (x,t,m),w in zip(hs,joint) if x == k)
                                 for k in ('none','purpose','skill')}
                        row = dict(stream=si, draw=stream['draw'], initial_maker=stream['maker'], actual_maker=truth,
                                   length=length, kind=kind, switched=stream['switched'], duplicates=stream['duplicates'],
                                   step=step, condition=cid, revealed_pairs=len(pairs), arm=arm,
                                   supplied_sources=len(T.unique_sources(observed)), retained_true_sources=true_sources,
                                   posterior=current.tolist(), forecast=forecast.tolist(), joint_array=jkey, joint_row=ji,
                                   type_mass=types, **values)
                        rows.append(row)
                        acc[stream['draw'],length,kind,stream['switched'],stream['duplicates'],step,cid,arm].append(row)
        raw = root/'raw'; raw.mkdir(exist_ok=True)
        np.savez_compressed(raw/f'{lineage}-joint_points.npz', **{k:np.asarray(v,dtype=np.float64) for k,v in sorted(joints.items())})
        meta = {}
        for key in sorted(joints):
            length,kind,arm = key.split('-',2); hs,prior = U.hypotheses(arm,int(length),kind)
            meta[key] = dict(rows=mapping[key], hypotheses=hs, prior=prior.tolist())
        write(root/'evaluator'/f'{lineage}-joint-map.json', meta)
        (raw/f'{lineage}-forecasts_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
        total_rows += len(rows)
        for key, rr in sorted(acc.items()):
            if len(rr) != 16: raise ValueError('maker denominator')
            cells.append(dict(lineage=lineage, **dict(zip(('draw','length','kind','switched','duplicates','step','condition','arm'),key)),
                              makers=16, **{m:math.fsum(r[m] for r in rr)/16 for m in T.METRICS}))
    for key, value in (('expected_streams',total_streams),('expected_rows',total_rows),('expected_cells',len(cells))):
        if key in cfg and cfg[key] != value: raise ValueError('planned denominator differs')
    write(root/'IDENTITIES.json', dict(passed=True, rows=dict(identities),
          scope='zero/full/label-only controls are identities, not independent scientific replicates'))
    write(root/'PUBLIC_PACKET.json', dict(schema='v19.provenance-restoration.reader.1', cases=[packets[k] for k in sorted(packets)],
          role='observed endpoints/timestamps and supplied source equivalences only; evaluator identity and law excluded'))
    return dict(controls=checks, cells=cells, streams=total_streams, conditions=total_conditions, rows=total_rows,
                public_packets=len(packets), fits=0, identities=dict(identities),
                scope='supplied provenance correction on saved streams; order/budget aliases deduplicated; laws supplied; no learned detection or historical process claim')
