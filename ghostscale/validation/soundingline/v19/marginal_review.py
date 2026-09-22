"""Independent scalar review of exhaustive current/future marginal witnesses.

No producer, transition, partition or regroup function is imported here.
The law audit is inherited from the bound parent, not repeated by this review.
"""
from collections import defaultdict
from itertools import combinations, product
import gzip
import json
import math
from ..v18_3.io import canonical, read, write


def state(h, step):
    kind, change, maker = h
    bits = list(product((0, 1), repeat=4))[maker]
    if kind == 'none' or step <= change:
        return maker
    axis = {'purpose': 0, 'skill': 1}[kind]
    return sum((1-b if i == axis else b) * 2**(3-i) for i, b in enumerate(bits))


def partition(hs, step, future, saved):
    members = defaultdict(list)
    for i, h in enumerate(hs):
        members[state(h, step), state(h, future)].append(i)
    groups = [dict(current=k[0], future=k[1], members=v) for k, v in sorted(members.items())]
    blocks = [dict(left=i, right=j, pairs=len(a['members'])*len(b['members']))
              for i, a in enumerate(groups) for j, b in enumerate(groups)
              if i < j and a['current'] == b['current']]
    answer = dict(checkpoint=step, future=future, groups=groups, blocks=blocks,
                  pairs=sum(b['pairs'] for b in blocks))
    if answer != saved:
        raise ValueError('partition reconstruction')
    # Complement count independently establishes the full unordered-pair count.
    by_current = defaultdict(list)
    for (current, later), ids in members.items():
        by_current[current].append(len(ids))
    expected = sum(math.comb(sum(ns), 2)-sum(math.comb(n, 2) for n in ns)
                   for ns in by_current.values())
    if expected != answer['pairs']:
        raise ValueError('complement count')
    return answer


def law_distances(law):
    if len(law) != 16 or any(len(m) != 4 or any(len(p) != 8 or
            any(not math.isfinite(x) or x < 0 for x in p) or
            abs(math.fsum(p)-1) > 1e-12 for p in m) for m in law):
        raise ValueError('invalid law')
    return [dict(left=a, right=b, context=c,
                 total_variation=math.fsum(abs(x-y) for x, y in zip(law[a][c], law[b][c]))/2,
                 max_absolute_difference=max(abs(x-y) for x, y in zip(law[a][c], law[b][c])),
                 identical_binary64=law[a][c] == law[b][c],
                 above_1e_12=max(abs(x-y) for x, y in zip(law[a][c], law[b][c])) > 1e-12)
            for a, b in combinations(range(16), 2) for c in range(4)]


def compare(actual, expected, tolerance=1e-14):
    if type(actual) is not type(expected):
        raise ValueError('type differs')
    if isinstance(expected, dict):
        if actual.keys() != expected.keys(): raise ValueError('keys differ')
        return max((compare(actual[k], v, tolerance) for k, v in expected.items()), default=0.)
    if isinstance(expected, list):
        if len(actual) != len(expected): raise ValueError('length differs')
        return max((compare(a, b, tolerance) for a, b in zip(actual, expected)), default=0.)
    if isinstance(expected, float):
        error = abs(actual-expected)
        if not math.isfinite(error) or error > tolerance: raise ValueError('numeric differs')
        return error
    if actual != expected: raise ValueError('value differs')
    return 0.


def review(parent, output):
    cfg = read(parent/'PLAN.json')['design']; summary = read(parent/'SUMMARY.json')
    saved_hs = read(parent/'HYPOTHESES.json'); partitions = {}; counts = []
    for length in cfg['lengths']:
        hs = [['none', 0, m] for m in range(16)] + [list(h) for h in
              product(('purpose', 'skill'), range(8, length-7), range(16))]
        if saved_hs[str(length)] != hs: raise ValueError('roster differs')
        rows = json.loads(gzip.decompress((parent/'raw'/f'{length}-partitions_points.json.gz').read_bytes()))
        times = [(s, t) for s in cfg['checkpoints'] if s <= length for t in range(s, length+1)]
        if [(r['checkpoint'], r['future']) for r in rows] != times: raise ValueError('time coverage')
        partitions[length] = [partition(hs, s, t, r) for (s, t), r in zip(times, rows)]
        counts.append(dict(length=length, hypotheses=len(hs), checkpoint_future_cells=len(rows),
                           pair_blocks=sum(len(r['blocks']) for r in rows),
                           pair_instances=sum(r['pairs'] for r in rows)))
    if counts != summary['counts']: raise ValueError('counts differ')
    cells = []; maximum_error = 0.; aggregates = defaultdict(list)
    for lineage in cfg['lineages']:
        scores = law_distances(read(parent/'inputs'/f'{lineage}-law.json'))
        maximum_error = max(maximum_error, compare(read(parent/'evaluator'/f'{lineage}-forecast-pairs.json'), scores))
        lookup = {(r['left'], r['right'], r['context']): r for r in scores}
        for length, rows in partitions.items():
            for r in rows:
                for context in range(4):
                    contributions = []
                    for block in r['blocks']:
                        a, b = sorted(r['groups'][block[x]]['future'] for x in ('left', 'right'))
                        contributions.append((block['pairs'], lookup[a, b, context]))
                    n = r['pairs']
                    row = dict(lineage=lineage, length=length, checkpoint=r['checkpoint'], future=r['future'], context=context,
                               pairs=n, exact_binary64_forecast_aliases=sum(k*s['identical_binary64'] for k, s in contributions),
                               differences_above_1e_12=sum(k*s['above_1e_12'] for k, s in contributions),
                               mean_total_variation=math.fsum(k*s['total_variation'] for k, s in contributions)/n if n else None,
                               max_total_variation=max((s['total_variation'] for _, s in contributions), default=0.))
                    cells.append(row); aggregates[length, r['checkpoint']].append(row)
    old = json.loads(gzip.decompress((parent/'raw/forecast_summary_points.json.gz').read_bytes()))
    maximum_error = max(maximum_error, compare(old, cells))
    if len(cells) != summary['forecast_cells']: raise ValueError('forecast denominator')
    regroup = []
    for (length, step), rows in sorted(aggregates.items()):
        n = sum(r['pairs'] for r in rows)
        regroup.append(dict(length=length, checkpoint=step, forecast_cells=len(rows), pair_law_context_instances=n,
                            binary64_aliases=sum(r['exact_binary64_forecast_aliases'] for r in rows),
                            above_1e_12=sum(r['differences_above_1e_12'] for r in rows),
                            mean_total_variation=math.fsum((r['mean_total_variation'] or 0)*r['pairs'] for r in rows)/n if n else None,
                            max_total_variation=max(r['max_total_variation'] for r in rows)))
    output.mkdir(parents=True, exist_ok=False)
    (output/'reconstructed_cells_points.json.gz').write_bytes(gzip.compress(canonical(cells), mtime=0))
    result = dict(passed=True, counts=counts, forecast_cells=len(cells), scalar_law_pairs=len(cfg['lineages'])*480,
                  max_absolute_error=maximum_error, regroup=regroup,
                  scope='exhaustive finite supplied-law full-simplex structural counterexample; reachability unresolved; inherited parent law validity')
    write(output/'NUMERICAL_REVIEW.json', result)
    return result
