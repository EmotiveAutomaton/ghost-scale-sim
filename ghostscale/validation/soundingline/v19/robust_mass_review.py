"""Independent rational and original-population review of robust source mass."""
from collections import defaultdict
from fractions import Fraction
from math import fsum
import gzip
import json
import zipfile
from .source_mass_review import rebuild_selection

PRIORS = ('uniform', 'time-proportional', 'reciprocal-time')


def reconstruct(records):
    assert len(records) == 3 and {r['prior'] for r in records} == set(PRIORS)
    first = records[0]
    times, costs, capacity = (first[k] for k in ('times', 'costs', 'capacity_bytes'))
    rank = dict(zip(sorted(times), range(len(times))))
    library = {}; optima = []
    for prior in PRIORS:
        r = next(r for r in records if r['prior'] == prior)
        assert (r['times'], r['costs'], r['capacity_bytes']) == (times, costs, capacity)
        rebuilt = rebuild_selection(r)
        assert rebuilt == r['result']
        a = rebuilt['optimal']; optima.append(Fraction(a['mass_numerator'], a['mass_denominator']))
        for policy, answer in rebuilt.items():
            selected = answer['selected_times']; mask = sum(2**rank[t] for t in selected)
            if mask not in library:
                library[mask] = dict(mask=mask, selected_times=selected, used_bytes=answer['used_bytes'], identities=[])
            library[mask]['identities'].append(prior+'/'+policy)
    rational = lambda x: [x.numerator, x.denominator]
    rows = []
    for mask, entry in sorted(library.items()):
        masses = []
        for prior in PRIORS:
            weights = {t: Fraction(1) if prior == 'uniform' else Fraction(t) if prior == 'time-proportional' else Fraction(1,t) for t in times}
            masses.append(sum((weights[t] for t in entry['selected_times']), Fraction()) / (sum(weights.values()) or 1))
        regrets = [a-b for a,b in zip(optima,masses)]
        rows.append(dict(entry, identities=sorted(entry['identities']), masses=list(map(rational,masses)),
            worst_mass=rational(min(masses)), prior_regrets=list(map(rational,regrets)), maximum_regret=rational(max(regrets))))
    value = max(Fraction(*r['worst_mass']) for r in rows)
    ties = [r['mask'] for r in rows if Fraction(*r['worst_mass']) == value]
    return dict(times=times, costs=costs, capacity_bytes=capacity, candidates=rows,
        selected_mask=max(ties), optimal_masks=ties, minimum_mass=rational(value),
        best_prior_mass=list(map(rational,optima)),
        prior_optimal_masks=[[r['mask'] for r in rows if Fraction(*r['masses'][j]) == a] for j,a in enumerate(optima)])


def provenance(root):
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
    return structures,population


def controls():
    r=dict(times=[1,2],costs=[1,1],capacity_bytes=1)
    records=[dict(r,prior=p,result=rebuild_selection(dict(r,prior=p))) for p in PRIORS]
    x=reconstruct(records)
    return {'live:known_robust_tie':x['optimal_masks']==[1,2],
        'positive:recent_tie_choice':x['selected_mask']==2,
        'placebo:no_added_mass':x['minimum_mass']==[1,3]}


def verify(root):
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    structures,population=provenance(root)
    parent=read(root/'inputs/PARENT_SELECTIONS.json'); assert len(parent)==84
    groups=defaultdict(list)
    for r in parent:groups[(tuple(r['times']),tuple(r['costs']),r['capacity_bytes'])].append(r)
    selections=read(root/'SELECTIONS.json');assert len(selections)==len(groups)==28
    answers={}; lookup={}; comparisons=[]
    for s in selections:
        key=(tuple(s['times']),tuple(s['costs']),s['capacity_bytes'])
        answer=reconstruct(groups.pop(key));assert dict(answer,id=s['id'])==s
        assert key not in lookup and s['id'] not in answers
        lookup[key]=s['id'];answers[s['id']]=answer
        chosen=next(r for r in answer['candidates'] if r['mask']==answer['selected_mask'])
        recent=next(r for r in answer['candidates'] if 'uniform/recent' in r['identities'])
        comparisons.append(dict(selection_id=s['id'],robust_mask=chosen['mask'],recent_mask=recent['mask'],
            robust_worst_mass=float(Fraction(*chosen['worst_mass'])),recent_worst_mass=float(Fraction(*recent['worst_mass'])),
            improvement=float(Fraction(*chosen['worst_mass'])-Fraction(*recent['worst_mass']))))
    assert not groups
    rows=json.loads(gzip.decompress((root/'raw/robust_mass_points.json.gz').read_bytes())); assert len(rows)==57344
    axes=('lineage','evidence','length','checkpoint','draw','budget')
    metrics=('minimum_mass','maximum_regret','retained_sources','fixed_bytes','total_budget_bytes','total_used_bytes',
        'recent_minimum_mass','mass_gain_over_recent','mass_uniform','mass_time','mass_reciprocal')
    grouped=defaultdict(list); position=0; pairs={}
    for r in population:
        st=structures[r['structure']];times=r['times'];costs=[st['source_costs'][t-1] for t in times]
        pair=tuple(r[k] for k in ('lineage','structure','draw','initial_maker','kind','switched','duplicates'))
        if r['evidence']=='aware':pairs[pair]=times
        else:assert pairs[pair]==times
        for budget in ('half','quarter'):
            threshold=r['checkpoint']//2 if budget=='half' else 3*r['checkpoint']//4
            recent=[i for i,t in enumerate(times) if t>threshold];count=len(recent)
            ordered=sorted(range(len(times)),key=times.__getitem__)
            spaced=[ordered[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity=min(sum(costs[i] for i in recent),sum(costs[i] for i in spaced))
            sid=lookup[(tuple(times),tuple(costs),capacity)];s=answers[sid]
            chosen=next(a for a in s['candidates'] if a['mask']==s['selected_mask'])
            recent_choice=next(a for a in s['candidates'] if 'uniform/recent' in a['identities'])
            expected=dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,selection_id=sid,
                fixed_bytes=st['weighted_overhead'],total_budget_bytes=st['weighted_overhead']+capacity,
                total_used_bytes=st['weighted_overhead']+chosen['used_bytes'],retained_sources=len(chosen['selected_times']),
                minimum_mass=float(Fraction(*chosen['worst_mass'])),masses=[float(Fraction(*x)) for x in chosen['masses']],
                prior_regrets=[float(Fraction(*x)) for x in chosen['prior_regrets']],maximum_regret=float(Fraction(*chosen['maximum_regret'])),
                unique_candidates=len(s['candidates']),optimal_masks=s['optimal_masks'],selected_mask=s['selected_mask'])
            actual=rows[position];position+=1;assert actual==expected,position
            base=float(Fraction(*recent_choice['worst_mass']))
            enriched=dict(actual,recent_minimum_mass=base,mass_gain_over_recent=actual['minimum_mass']-base,
                **dict(zip(('mass_uniform','mass_time','mass_reciprocal'),actual['masses'])))
            grouped[tuple(actual[k] for k in axes)].append(enriched)
    def aggregate(groups,names,expected):
        result=[]
        for key,rr in sorted(groups.items()):
            assert len(rr)==expected
            result.append(dict(zip(names,key),rows=len(rr),**{m:fsum(r[m] for r in rr)/len(rr) for m in metrics}))
        return result
    strata=aggregate(grouped,axes,128); by_law=defaultdict(list)
    for r in strata:by_law[tuple(r[k] for k in axes if k!='draw')].append(r)
    law=aggregate(by_law,tuple(k for k in axes if k!='draw'),2); by_cell=defaultdict(list)
    for r in law:by_cell[tuple(r[k] for k in axes if k not in ('draw','lineage'))].append(r)
    cells=aggregate(by_cell,tuple(k for k in axes if k not in ('draw','lineage')),8)
    return dict(passed=True,numerical_acceptance=True,original_rows=position,source_rosters=len(population),
        structures=len(structures),distinct_allocations=len(selections),candidate_count=sum(len(s['candidates']) for s in selections),
        paired_strata=strata,law_strata=law,equal_law_cells=cells,allocation_comparisons=comparisons,
        strict_gain_allocations=sum(r['improvement']>0 for r in comparisons),checks=controls(),
        scope='candidate-library maximin retained source mass;not unrestricted minimax,forecast optimality or historical correspondence')
