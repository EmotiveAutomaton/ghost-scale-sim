"""Finite full-simplex witnesses for current marginal versus future schedule.

Cartesian blocks retain every unordered pair without repeating identical laws.
These point hypotheses are not claimed reachable from the campaign prior.
"""
from collections import defaultdict
from itertools import combinations, product
import gzip
import math
import numpy as np
from ..v18_3.io import canonical, read, write, file_digest
from .unknown_change import hypotheses, state_indices


def partitions(hs, checkpoint, future):
    if future < checkpoint:
        raise ValueError('future precedes checkpoint')
    current = state_indices(hs, checkpoint)
    later = state_indices(hs, future)
    groups = defaultdict(list)
    for i, (a, b) in enumerate(zip(current, later)):
        groups[int(a), int(b)].append(i)
    rows = [dict(current=a, future=b, members=ids) for (a, b), ids in sorted(groups.items())]
    blocks = []
    for i, j in combinations(range(len(rows)), 2):
        a, b = rows[i], rows[j]
        if a['current'] == b['current']:
            blocks.append(dict(left=i, right=j, pairs=len(a['members'])*len(b['members'])))
    return dict(checkpoint=checkpoint, future=future, groups=rows, blocks=blocks,
                pairs=sum(b['pairs'] for b in blocks))


def direct_state(hypothesis, step):
    # Independent tuple-level transition; never uses producer state_indices/FLIPS.
    kind, time, maker = hypothesis
    bits = list(product(range(2), repeat=4))[maker]
    if kind != 'none' and step > time:
        values = list(bits)
        axis = {'purpose': 0, 'skill': 1}[kind]
        values[axis] = 1-values[axis]
        bits = tuple(values)
    return list(product(range(2), repeat=4)).index(bits)


def check_partition(hs, row):
    expected = defaultdict(list)
    for i, h in enumerate(hs):
        expected[direct_state(h, row['checkpoint']), direct_state(h, row['future'])].append(i)
    observed = {(g['current'], g['future']): g['members'] for g in row['groups']}
    if len(observed) != len(row['groups']) or dict(expected) != observed:
        raise ValueError('partition membership')
    seen = set()
    for b in row['blocks']:
        i, j = b['left'], b['right']
        if not 0 <= i < j < len(row['groups']) or (i, j) in seen:
            raise ValueError('pair block identity')
        seen.add((i, j)); a, z = row['groups'][i], row['groups'][j]
        if a['current'] != z['current'] or a['future'] == z['future']:
            raise ValueError('pair state')
        if b['pairs'] != len(a['members'])*len(z['members']):
            raise ValueError('pair count')
    required = {(i, j) for i, a in enumerate(row['groups']) for j, b in enumerate(row['groups'])
                if i < j and a['current'] == b['current'] and a['future'] != b['future']}
    if seen != required or row['pairs'] != sum(b['pairs'] for b in row['blocks']):
        raise ValueError('pair coverage')


def distances(table):
    a = np.asarray(table, dtype=float)
    if a.shape != (16, 4, 8) or not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a.sum(-1), 1, rtol=0, atol=1e-12):
        raise ValueError('endpoint law')
    answer = []
    for left, right in combinations(range(16), 2):
        for context in range(4):
            delta = a[left, context]-a[right, context]
            tv = float(np.abs(delta).sum()/2)
            # Separate scalar contraction of each point-mass witness.
            p = [math.fsum((1. if m == left else 0.)*float(a[m, context, e]) for m in range(16)) for e in range(8)]
            q = [math.fsum((1. if m == right else 0.)*float(a[m, context, e]) for m in range(16)) for e in range(8)]
            scalar = math.fsum(abs(x-y) for x, y in zip(p, q))/2
            if abs(tv-scalar) > 1e-14:
                raise ValueError('scalar forecast')
            answer.append(dict(left=left, right=right, context=context,
                total_variation=tv, max_absolute_difference=float(np.abs(delta).max()),
                identical_binary64=bool(np.array_equal(a[left, context], a[right, context])),
                above_1e_12=bool(np.abs(delta).max() > 1e-12)))
    return answer


def controls():
    hs = [('none', 0, 0), ('purpose', 8, 0), ('skill', 8, 0)]
    before = partitions(hs, 8, 8); after = partitions(hs, 8, 9)
    check_partition(hs, before); check_partition(hs, after)
    stationary = partitions([('none', 0, m) for m in range(16)], 8, 128)
    law = np.full((16, 4, 8), .125)
    return {'live:future_schedule_separates': after['pairs'] == 3,
        'placebo:same_time_no_pairs': before['pairs'] == 0,
        'placebo:stationary_no_pairs': stationary['pairs'] == 0,
        'placebo:forecast_aliases_retained': all(r['identical_binary64'] for r in distances(law)),
        'positive:change_after_boundary': direct_state(hs[1], 8) == 0 and direct_state(hs[1], 9) == 8,
        'positive:skill_change': direct_state(hs[2], 9) == 4}


def run(root, plan, pulse):
    cfg = plan['design']; checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()):
        raise ValueError('marginal transition controls')
    for name, bound in cfg['input_files'].items():
        if file_digest(root/'inputs'/name) != bound:
            raise ValueError('input differs')
    all_partitions = {}; roster = {}; all_counts = []
    for length in cfg['lengths']:
        hs, _ = hypotheses('unknown-time-type', length, 'purpose')
        # Independent ordering and entire roster, including stationary hypotheses.
        reference = [('none', 0, m) for m in range(16)]
        reference += [(k, t, m) for k in ('purpose', 'skill') for t in range(8, length-7) for m in range(16)]
        if hs != reference:
            raise ValueError('hypothesis roster')
        roster[str(length)] = hs; rows = []
        for checkpoint in cfg['checkpoints']:
            if checkpoint > length: continue
            for future in range(checkpoint, length+1):
                if (future-checkpoint) % 16 == 0:
                    pulse(phase='complete-future-partitions', length=length, checkpoint=checkpoint, future=future)
                row = partitions(hs, checkpoint, future); check_partition(hs, row)
                # Same-time and stationary checks for every time, not just fixtures.
                if future == checkpoint and row['pairs']:
                    raise ValueError('same-time identity')
                if partitions(hs[:16], checkpoint, future)['pairs']:
                    raise ValueError('stationary identity')
                rows.append(row)
        all_partitions[length] = rows
        target = root/'raw'/f'{length}-partitions_points.json.gz'; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(canonical(rows), mtime=0))
        all_counts.append(dict(length=length, hypotheses=len(hs), checkpoint_future_cells=len(rows),
            pair_blocks=sum(len(r['blocks']) for r in rows), pair_instances=sum(r['pairs'] for r in rows)))
    write(root/'HYPOTHESES.json', roster)
    cells = []
    for lineage in cfg['lineages']:
        pulse(phase='complete-future-forecasts', lineage=lineage)
        table = read(root/'inputs'/f'{lineage}-law.json'); metric_rows = distances(table)
        index = {(r['left'], r['right'], r['context']): r for r in metric_rows}
        write(root/'evaluator'/f'{lineage}-forecast-pairs.json', metric_rows)
        for length, rows in all_partitions.items():
            for row in rows:
                for context in range(4):
                    weighted = []; aliases = significant = 0; maximum = 0.
                    for block in row['blocks']:
                        a = row['groups'][block['left']]['future']; b = row['groups'][block['right']]['future']
                        score = index[min(a, b), max(a, b), context]; n = block['pairs']
                        weighted.append(n*score['total_variation']); maximum = max(maximum, score['total_variation'])
                        aliases += n*score['identical_binary64']; significant += n*score['above_1e_12']
                    cells.append(dict(lineage=lineage, length=length, checkpoint=row['checkpoint'], future=row['future'], context=context,
                        pairs=row['pairs'], exact_binary64_forecast_aliases=aliases, differences_above_1e_12=significant,
                        mean_total_variation=math.fsum(weighted)/row['pairs'] if row['pairs'] else None, max_total_variation=maximum))
    (root/'raw'/'forecast_summary_points.json.gz').write_bytes(gzip.compress(canonical(cells), mtime=0))
    write(root/'EVIDENCE_ROLES.json', dict(reader='no new reader input; no observations acquired',
        evaluator='supplied laws, complete schedules and full-simplex point-mass witnesses; not reachable-posterior claims',
        enumeration='Every block denotes all member cross-products as unordered hypothesis pairs. Distinct future states only; forecast aliases retained. Laws are factored once per lineage/state pair/context.'))
    checks.update(positive_complete_partition_reconstruction=True, positive_scalar_forecast_reconstruction=True,
        placebo_all_same_time_and_stationary=True)
    return dict(controls=checks,counts=all_counts,forecast_cells=len(cells),lineages=cfg['lineages'],fits=0,
        scope='finite supplied-law full-simplex structural diagnostic; reachability unresolved; no historical process or human-intent claim')
