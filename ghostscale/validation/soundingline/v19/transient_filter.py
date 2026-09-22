"""Supplied-law filtering across a declared possible purpose switch; no fits."""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import time
import numpy as np
from ..v18_3.io import canonical, digest, file_digest, write
from ..v18_3.world import rng
from .local_world import MAKERS, CONTEXTS

ARMS = ('static', 'reset-16', 'coherent-mixture')
FLIP = np.array([MAKERS.index((1-m[0], *m[1:])) for m in MAKERS])
METRICS = ('endpoint_loss', 'state_loss', 'true_state_mass', 'coverage90', 'size90')


def endpoint_law(records):
    table = np.zeros((16, 4, 8))
    for row in records:
        endpoint = sum(int(x) << i for i, x in enumerate(row['final']))
        table[row['maker_index'], row['context_index'], endpoint] += row['probability'] * 64
    if not np.allclose(table.sum(axis=-1), 1., atol=1e-12, rtol=0):
        raise ValueError('incomplete native endpoint law')
    return table / table.sum(axis=-1, keepdims=True)


def normalize(logweights):
    peak = np.max(logweights)
    if not np.isfinite(peak):
        raise ValueError('no compatible filter hypothesis')
    weights = np.exp(logweights - peak)
    return weights / weights.sum()


def unique_sources(observations):
    seen = {}; result = []
    for row in observations:
        key = row['source_id']
        identity = (row['source_step'], row['context'], row['endpoint'])
        if key in seen:
            if seen[key] != identity: raise ValueError('conflicting duplicate source')
        else:
            seen[key] = identity; result.append(row)
    return result


def posterior(table, observations, arm, switch_at, current_step):
    if arm not in ARMS: raise ValueError('unknown filter')
    sources = unique_sources(observations)
    if arm == 'reset-16': sources = sources[-16:]
    weights = np.zeros((2 if arm == 'coherent-mixture' else 1, 16))
    with np.errstate(divide='ignore'):
        for row in sources:
            likelihood = table[:, row['context'], row['endpoint']]
            weights[0] += np.log(likelihood)
            if arm == 'coherent-mixture':
                weights[1] += np.log(likelihood[FLIP] if row['source_step'] > switch_at else likelihood)
    joint = normalize(weights)
    result = joint[0].copy()
    if arm == 'coherent-mixture':
        result += joint[1, FLIP] if current_step > switch_at else joint[1]
    return result, joint


def score(table, posterior_state, actual):
    prediction = np.einsum('m,mce->ce', posterior_state, table)
    truth = table[actual]
    loss = -float(np.sum(truth * np.log(np.maximum(prediction, 1e-300))) / 4)
    order = np.argsort(-posterior_state, kind='stable')
    cutoff = posterior_state[order[np.searchsorted(np.cumsum(posterior_state[order]), .9)]]
    covered = posterior_state >= cutoff - 1e-14
    return dict(endpoint_loss=loss, state_loss=-math.log(max(float(posterior_state[actual]), 1e-300)),
        true_state_mass=float(posterior_state[actual]), coverage90=float(covered[actual]),
        size90=float(covered.sum())), prediction


def make_stream(table, lineage, draw, maker, length, switched, duplicates):
    # Common uniforms and context order across the switch and duplicate arms.
    random = rng('v19-transient-filter', lineage, draw, maker, length)
    uniforms = random.random(length); rows = []
    for index, u in enumerate(uniforms):
        step = index + 1
        if duplicates and step % 4 == 0:
            rows.append(dict(rows[-1], step=step)); continue
        actual = int(FLIP[maker]) if switched and step > length // 2 else maker
        context = (index + index // 4) % 4
        endpoint = min(int(np.searchsorted(np.cumsum(table[actual, context]), u, side='right')), 7)
        rows.append(dict(step=step, source_step=step, source_id=f'source-{step:03d}', context=context, endpoint=endpoint))
    return rows


def checkpoints(length):
    return [8, 16, 17, 20, 32] if length == 32 else [16, 32, 64, 65, 80, 128]


def controls():
    table = np.full((16, 4, 8), 1/8)
    for m in range(16):
        table[m, :, :] = .1 / 7
        table[m, :, MAKERS[m][0]] = .9
    a = dict(step=1, source_step=1, source_id='a', context=0, endpoint=0)
    b = dict(step=2, source_step=2, source_id='b', context=0, endpoint=1)
    p, _ = posterior(table, [a], 'static', 1, 1)
    duplicate, _ = posterior(table, [a, dict(a, step=2)], 'static', 1, 2)
    mixture, _ = posterior(table, [a, b], 'coherent-mixture', 1, 2)
    # Direct two-path enumeration for every initial maker, independent of log filter.
    initial = table[:, 0, 0] * table[:, 0, 1]
    switching = table[:, 0, 0] * table[FLIP, 0, 1]
    expected = (initial + switching[FLIP]) / (initial.sum() + switching.sum())
    bad = False
    try: unique_sources([a, dict(a, endpoint=1)])
    except ValueError: bad = True
    neutral, _ = posterior(np.full((16, 4, 8), 1/8), [a, b], 'coherent-mixture', 1, 2)
    older = [dict(a, step=i, source_step=i, source_id=str(i)) for i in range(1, 20)]
    last, _ = posterior(table, older, 'reset-16', 10, 19)
    reference, _ = posterior(table, older[-16:], 'static', 10, 19)
    return {'live:informative_observation':float(p[:8].sum())>.9,
        'positive:mixture_enumeration':bool(np.allclose(mixture, expected, atol=1e-14, rtol=0)),
        'positive:purpose_flip_involution':bool(np.array_equal(FLIP[FLIP], np.arange(16))),
        'positive:conflicting_duplicate_rejected':bad,
        'placebo:duplicate_identity':bool(np.array_equal(p, duplicate)),
        'placebo:uninformative_uniform':bool(np.allclose(neutral, 1/16, atol=1e-14, rtol=0)),
        'positive:last_sixteen_sources':bool(np.array_equal(last, reference))}


def run(root, plan, pulse):
    cfg = plan['design']; checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('transient filter controls failed')
    for name, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/name) != h: raise ValueError('native input differs')
    all_cells = []; packets = []; observations_count = rows_count = 0
    for lineage in cfg['lineages']:
        pulse(phase='freeze-observations', lineage=lineage)
        records = json.loads(gzip.decompress((root/'inputs'/f'lineage-{lineage}_points.json.gz').read_bytes()))
        if len(records) != cfg['paths_per_lineage']: raise ValueError('path denominator')
        table = endpoint_law(records)
        streams = []
        for draw, maker, length, switched, duplicates in product(cfg['draws'], range(16), cfg['lengths'], (False,True), (False,True)):
            stream = make_stream(table, lineage, draw, maker, length, switched, duplicates)
            streams.append(dict(draw=draw, maker=maker, length=length, switched=switched, duplicates=duplicates, observations=stream))
        # Persist the entire population of observations before running any filter.
        raw = root/'raw'; raw.mkdir(exist_ok=True)
        write(root/'evaluator'/f'{lineage}-law.json', table.tolist())
        (raw/f'{lineage}-observations_points.json.gz').write_bytes(gzip.compress(canonical(streams),mtime=0))
        rows = []; accumulator = defaultdict(list)
        for stream_index, stream in enumerate(streams):
            pulse(phase='filter', lineage=lineage, stream=stream_index)
            maker = stream['maker']; length = stream['length']; observations_count += length
            for step in checkpoints(length):
                observed = stream['observations'][:step]
                actual = int(FLIP[maker]) if stream['switched'] and step > length//2 else maker
                for arm in ARMS:
                    start = time.process_time()
                    p, joint = posterior(table, observed, arm, length//2, step)
                    values, prediction = score(table, p, actual)
                    elapsed = time.process_time()-start
                    row = dict(stream=stream_index, draw=stream['draw'], initial_maker=maker, actual_maker=actual,
                        length=length, switched=stream['switched'], duplicates=stream['duplicates'], step=step, arm=arm,
                        unique_sources=len(unique_sources(observed)), posterior=p.tolist(), joint=joint.tolist(), forecast=prediction.tolist(), **values)
                    rows.append(row)
                    key=(stream['draw'],length,stream['switched'],stream['duplicates'],step,arm)
                    accumulator[key].append(row)
                    with (root/'TIMING.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(dict(lineage=lineage,stream=stream_index,step=step,arm=arm,cpu_seconds=elapsed),sort_keys=True)+'\n')
            visible = dict(contexts=list(CONTEXTS), observations=stream['observations'])
            packets.append(dict(input_sha256=digest(visible), inputs=visible))
        (raw/f'{lineage}-forecasts_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0)); rows_count += len(rows)
        for key, rr in sorted(accumulator.items()):
            all_cells.append(dict(lineage=lineage,**dict(zip(('draw','length','switched','duplicates','step','arm'),key)),makers=len(rr),
                **{metric:math.fsum(r[metric] for r in rr)/len(rr) for metric in METRICS}))
    unique = {p['input_sha256']:p for p in packets}
    write(root/'PUBLIC_PACKET.json', dict(schema='v19.transient.reader.1',cases=[unique[k] for k in sorted(unique)],
        role='observed endpoints and source identities only; supplied-law filters are evaluator references'))
    return dict(controls=checks,cells=all_cells,observations=observations_count,rows=rows_count,streams=len(packets),public_packets=len(unique),fits=0,
        scope='supplied-law purpose-switch rulers; fixed possible switch time and uniform maker prior; no historical path or human intent conclusion')
