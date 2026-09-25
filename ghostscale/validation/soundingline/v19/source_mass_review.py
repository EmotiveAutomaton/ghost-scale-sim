"""Independent subset-enumeration and population audit for source mass storage."""
from bisect import bisect_right
from collections import defaultdict
from fractions import Fraction
from math import fsum, lcm
import gzip
import json
import zipfile


def optimum(times, costs, capacity, values):
    """Meet in the middle; exhaustive halves, with exact objective and mask ties."""
    n = len(times)
    rank = {t: i for i, t in enumerate(sorted(times))}
    def half(indices):
        rows = [(0, 0, 0)]
        for i in indices:
            rows += [(c+costs[i], v+values[i], m | (1 << rank[times[i]]))
                     for c, v, m in rows if c+costs[i] <= capacity]
        return rows
    left, right = half(range(n//2)), sorted(half(range(n//2, n)))
    capacities = [r[0] for r in right]
    best = []
    running = (-1, -1)
    for _, value, mask in right:
        running = max(running, (value, mask))
        best.append(running)
    return max((v+best[bisect_right(capacities, capacity-c)-1][0],
                m | best[bisect_right(capacities, capacity-c)-1][1])
               for c, v, m in left)


def rebuild_selection(s):
    times, costs, capacity, prior = (s[k] for k in ('times','costs','capacity_bytes','prior'))
    assert len(times) == len(costs) == len(set(times))
    assert all(type(c) is int and c >= 0 for c in costs) and capacity >= 0
    raw = [Fraction(1) if prior == 'uniform' else Fraction(t) if prior == 'time-proportional'
           else Fraction(1, t) for t in times]
    assert prior in ('uniform','time-proportional','reciprocal-time')
    den = lcm(*(v.denominator for v in raw))
    values = [int(v*den) for v in raw]
    rank = {t: i for i, t in enumerate(sorted(times))}
    score, mask = optimum(times, costs, capacity, values)
    selected = {'optimal':[i for i,t in enumerate(times) if mask & (1 << rank[t])]}
    for policy in ('recent','probability','probability-per-byte'):
        available = set(range(len(times))); used = 0; chosen = []
        while available:
            if policy == 'recent': i = max(available, key=lambda i: times[i])
            elif policy == 'probability': i = max(available, key=lambda i: (raw[i],times[i]))
            else: i = max(available, key=lambda i: (costs[i] == 0, raw[i]/costs[i] if costs[i] else 0,times[i]))
            available.remove(i)
            if used+costs[i] <= capacity:
                chosen.append(i); used += costs[i]
        selected[policy] = chosen
    result = {}
    normalizer = sum(values) or 1
    for policy, chosen in selected.items():
        mass = sum(values[i] for i in chosen); used = sum(costs[i] for i in chosen)
        result[policy] = dict(selected_times=sorted((times[i] for i in chosen),reverse=True),
            used_bytes=used,unused_bytes=capacity-used,mass_numerator=mass,
            mass_denominator=normalizer,retained_mass=mass/normalizer,gap_to_optimum=(score-mass)/normalizer)
    return result


def controls():
    sample = dict(times=[1,2,3],costs=[10,20,30],capacity_bytes=50,prior='time-proportional')
    answer = rebuild_selection(sample)
    tie = rebuild_selection(dict(sample,times=[1,2],costs=[10,10],capacity_bytes=10,prior='uniform'))
    return {'live:known_optimum':answer['optimal']['selected_times']==[3,2],
        'placebo:empty_capacity':rebuild_selection(dict(sample,capacity_bytes=0))['optimal']['retained_mass']==0,
        'positive:zero_cost':rebuild_selection(dict(sample,costs=[0,0,0],capacity_bytes=0))['optimal']['retained_mass']==1,
        'positive:recent_tie':tie['optimal']['selected_times']==[2]}


def verify(root):
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    structures=read(root/'inputs/STRUCTURES.json'); population=read(root/'inputs/POPULATION.json')
    with zipfile.ZipFile(root/'inputs/PARENT_PROVENANCE.zip') as z:
        specs=json.loads(z.read('SCHEDULES.json'))
        for key, st in structures.items():
            spec=specs[key]; group_count=len(spec['signatures']); rebuilt=[]
            for t in range(1,spec['checkpoint']+1):
                past_by_group=defaultdict(set)
                for group, (kind,change,maker) in zip(spec['membership'],spec['hypotheses']):
                    if t>change and kind=='purpose':maker = maker+8 if maker<8 else maker-8
                    if t>change and kind=='skill':maker = maker+4 if maker%8<4 else maker-4
                    past_by_group[group].add(maker)
                entries=sum(len(v) for v in past_by_group.values() if len(v)>1)
                metadata=3+2*sum(map(len,past_by_group.values()))
                rebuilt.append(8*entries+4*metadata)
            assert rebuilt==st['source_costs'] and st['weighted_overhead']==8*(group_count+45)
            assert st['old_overhead']+56==st['weighted_overhead']
        lookup={}
        for name in z.namelist():
            if not name.startswith('bindings/'):continue
            binding=json.loads(z.read(name));lineage,evidence,length,cp,_=name.split('/')[-1].split('-')
            times=defaultdict(list)
            for index,t,context,endpoint,source in binding['sources']:times[index].append(t)
            for i,r in enumerate(binding['rows']):
                rr=dict(lineage=int(lineage),evidence=evidence,length=int(length),checkpoint=int(cp),
                    structure=length+'-'+cp,times=times[i],**{k:r[k] for k in ('draw','initial_maker','kind','switched','duplicates','stream','report_sources')})
                k=tuple(rr[n] for n in ('lineage','evidence','length','checkpoint','draw','initial_maker','kind','switched','duplicates'))
                assert k not in lookup;lookup[k]=rr
        assert len(lookup)==len(population)==28672
        for r in population:
            key=tuple(r[n] for n in ('lineage','evidence','length','checkpoint','draw','initial_maker','kind','switched','duplicates'))
            assert lookup.pop(key)==r
        assert not lookup
    selections=read(root/'SELECTIONS.json'); assert len(selections)==84
    answers={}; keys={}
    for s in selections:
        answers[s['id']]=rebuild_selection(s)
        assert answers[s['id']]==s['result'],s['id']
        key=(tuple(s['times']),tuple(s['costs']),s['capacity_bytes'],s['prior'])
        assert key not in keys;keys[key]=s['id']
    rows=json.loads(gzip.decompress((root/'raw/mass_frontier_points.json.gz').read_bytes()))
    assert len(rows)==688128
    groups=defaultdict(list); position=0
    axes=('lineage','evidence','length','checkpoint','draw','budget','prior','policy')
    metrics=('retained_mass','gap_to_optimum','retained_sources','fixed_bytes','total_budget_bytes','total_used_bytes','used_bytes','unused_bytes')
    for r in population:
        st=structures[r['structure']];times=r['times'];costs=[st['source_costs'][t-1] for t in times]
        for budget in ('half','quarter'):
            threshold=r['checkpoint']//2 if budget=='half' else 3*r['checkpoint']//4
            recent=[i for i,t in enumerate(times) if t>threshold];count=len(recent)
            ordered=sorted(range(len(times)),key=times.__getitem__)
            spaced=[ordered[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity=min(sum(costs[i] for i in recent),sum(costs[i] for i in spaced))
            for prior in ('uniform','time-proportional','reciprocal-time'):
                sid=keys[(tuple(times),tuple(costs),capacity,prior)]
                for policy in ('recent','probability','probability-per-byte','optimal'):
                    a=answers[sid][policy]
                    expected=dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,prior=prior,policy=policy,
                        selection_id=sid,fixed_bytes=st['weighted_overhead'],total_budget_bytes=st['weighted_overhead']+capacity,
                        total_used_bytes=st['weighted_overhead']+a['used_bytes'],retained_sources=len(a['selected_times']),
                        **{k:v for k,v in a.items() if k!='selected_times'})
                    actual=rows[position];position+=1
                    assert actual==expected,position
                    groups[tuple(actual[k] for k in axes)].append(actual)
    def aggregate(groups, names, expected):
        result=[]
        for key, rr in sorted(groups.items()):
            assert len(rr)==expected
            result.append(dict(zip(names,key),rows=len(rr),**{m:fsum(r[m] for r in rr)/len(rr) for m in metrics}))
        return result
    strata=aggregate(groups,axes,128)
    law_groups=defaultdict(list)
    for r in strata:law_groups[tuple(r[k] for k in axes if k!='draw')].append(r)
    law=aggregate(law_groups,tuple(k for k in axes if k!='draw'),2)
    population_groups=defaultdict(list)
    for r in law:population_groups[tuple(r[k] for k in axes if k not in ('draw','lineage'))].append(r)
    cells=aggregate(population_groups,tuple(k for k in axes if k not in ('draw','lineage')),8)
    return dict(passed=True,numerical_acceptance=True,original_rows=position,source_rosters=len(population),
        structures=len(structures),distinct_allocations=len(selections),paired_strata=strata,law_strata=law,
        equal_law_cells=cells,all_exact_integer_and_rational_identities=True,
        checks='structural set recount;parent source-only roster reconstruction;exhaustive meet-in-the-middle exact subsets;independent greedy choices;every original row;two draws averaged within each of eight laws',
        scope='finite supplied source-mass storage allocation;no forecast optimum or historical correspondence')
