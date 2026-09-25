"""Exact supplied source-mass allocation under structural byte budgets."""
from fractions import Fraction
from functools import lru_cache
from math import lcm
import gzip
from ..v18_3.io import read,write,canonical,file_digest

PRIORS=('uniform','time-proportional','reciprocal-time')
POLICIES=('recent','probability','probability-per-byte','optimal')


def weights(times,prior):
    if prior not in PRIORS:raise ValueError('prior')
    values=[Fraction(1) if prior=='uniform' else Fraction(t) if prior=='time-proportional' else Fraction(1,t) for t in times]
    denominator=lcm(*(v.denominator for v in values))
    return [int(v*denominator) for v in values]


@lru_cache(maxsize=4096)
def allocate(times,costs,capacity,prior):
    if len(times)!=len(costs) or len(set(times))!=len(times) or any(type(t)!=int or t<=0 for t in times):raise ValueError('times')
    if type(capacity)!=int or capacity<0 or any(type(k)!=int or k<0 for k in costs):raise ValueError('costs/capacity')
    w=weights(times,prior);rank={t:i for i,t in enumerate(sorted(times))}
    bits=[1<<rank[t] for t in times];normalizer=sum(w)
    if not times:return {p:dict(selected_times=[],used_bytes=0,unused_bytes=capacity,mass_numerator=0,mass_denominator=1,retained_mass=0.,gap_to_optimum=0.) for p in POLICIES}
    frontier={0:(0,0)}
    for cost,value,bit in zip(costs,w,bits):
        for used,(score,mask) in list(frontier.items()):
            new=used+cost;candidate=(score+value,mask|bit)
            if new<=capacity and candidate>frontier.get(new,(-1,-1)):frontier[new]=candidate
    best=max(frontier.values());masks={}
    for policy in POLICIES[:-1]:
        if policy=='recent':order=sorted(range(len(times)),key=lambda i:-times[i])
        elif policy=='probability':order=sorted(range(len(times)),key=lambda i:(-w[i],-times[i]))
        else:order=sorted(range(len(times)),key=lambda i:(costs[i]!=0,-Fraction(w[i],costs[i]) if costs[i] else Fraction(0),-times[i]))
        remaining=capacity;mask=0
        for i in order:
            if costs[i]<=remaining:mask|=bits[i];remaining-=costs[i]
        masks[policy]=mask
    masks['optimal']=best[1];result={}
    for policy,mask in masks.items():
        chosen=[i for i in range(len(times)) if mask&bits[i]];mass=sum(w[i] for i in chosen);used=sum(costs[i] for i in chosen)
        result[policy]=dict(selected_times=sorted((times[i] for i in chosen),reverse=True),used_bytes=used,unused_bytes=capacity-used,mass_numerator=mass,mass_denominator=normalizer,retained_mass=mass/normalizer,gap_to_optimum=(best[0]-mass)/normalizer)
    return result


def structural_costs(spec):
    hs=spec['hypotheses'];mapping=spec['membership'];groups=len(spec['signatures'])
    if len(hs)!=len(mapping) or set(mapping)!=set(range(groups)):raise ValueError('structural mapping')
    costs=[]
    for time in range(1,spec['checkpoint']+1):
        pairs=set()
        for group,(kind,change,maker) in zip(mapping,hs):
            if kind not in ('none','purpose','skill') or not 0<=maker<16:raise ValueError('state')
            past=maker^({'purpose':8,'skill':4}.get(kind,0) if time>change else 0)
            pairs.add((group,past))
        counts={g:0 for g in range(groups)}
        for g,_ in pairs:counts[g]+=1
        costs.append(8*sum(v for v in counts.values() if v>1)+4*(3+2*len(pairs)))
    return costs,8*(groups+13+32)


def controls():
    live=allocate((1,2,3),(10,20,30),50,'time-proportional')
    return {'live:positive_mass':live['optimal']['retained_mass']>0,
        'placebo:empty_capacity':allocate((1,2),(10,20),0,'uniform')['optimal']['retained_mass']==0,
        'positive:full_capacity':allocate((1,2),(10,20),30,'uniform')['optimal']['retained_mass']==1,
        'positive:recent_tie':allocate((1,2),(10,10),10,'uniform')['optimal']['selected_times']==[2]}


def run(root,plan,pulse):
    cfg=plan['design']
    if cfg['source_priors']!=list(PRIORS) or cfg['policies']!=list(POLICIES):raise ValueError('design')
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    population=read(root/'inputs/POPULATION.json');structures=read(root/'inputs/STRUCTURES.json')
    rows=[];selections={};paired={}
    for i,r in enumerate(population):
        if i%256==0:pulse(phase='source-mass-frontier',row=i)
        st=structures[r['structure']];times=tuple(r['times']);costs=tuple(st['source_costs'][t-1] for t in times)
        if len(times)!=r['report_sources'] or any(t>st['checkpoint'] for t in times):raise ValueError('roster')
        pair=(r['lineage'],r['structure'],r['draw'],r['initial_maker'],r['kind'],r['switched'],r['duplicates'])
        if r['evidence']=='aware':paired[pair]=times
        elif paired[pair]!=times:raise ValueError('paired source times')
        overhead=st['weighted_overhead'];cp=st['checkpoint']
        for fraction,threshold in [('half',cp//2),('quarter',3*cp//4)]:
            recent=[j for j,t in enumerate(times) if t>threshold]
            count=len(recent);order=sorted(range(len(times)),key=lambda j:times[j])
            spaced=[order[((2*j+1)*len(times))//(2*count)] for j in range(count)]
            capacity=min(sum(costs[j] for j in recent),sum(costs[j] for j in spaced))
            for prior in PRIORS:
                key=canonical(dict(times=times,costs=costs,capacity=capacity,prior=prior)).decode()
                if key not in selections:
                    selections[key]=dict(id=len(selections),times=times,costs=costs,capacity_bytes=capacity,prior=prior,result=allocate(times,costs,capacity,prior))
                selection=selections[key]
                for policy,answer in selection['result'].items():
                    rows.append(dict(**{k:v for k,v in r.items() if k!='times'},budget=fraction,prior=prior,policy=policy,selection_id=selection['id'],fixed_bytes=overhead,total_budget_bytes=overhead+capacity,total_used_bytes=overhead+answer['used_bytes'],retained_sources=len(answer['selected_times']),**{k:v for k,v in answer.items() if k!='selected_times'}))
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/mass_frontier_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',list(selections.values()))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='retained source times and structural byte costs;supplied source priors;exact mass optimum',scope='storage allocation only;no endpoint likelihood,forecast error,learned provenance,process correspondence or human intent'))
    return dict(controls=checks,population_rows=len(population),rows=len(rows),unique_selections=len(selections),numerical_acceptance=False,scope='finite supplied source-probability storage frontier;paired laws and draws retained')
