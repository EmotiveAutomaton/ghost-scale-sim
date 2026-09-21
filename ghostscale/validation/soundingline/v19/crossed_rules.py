"""Exact joint-process support when separately supplied rule changes compose."""
from collections import defaultdict
import gzip
import time
import numpy as np
from ..v18_3.io import canonical, write
from . import local_world as L, missing_tool as M

RULES=('original','presentation-tool','inverted-repair','both')
FAMILIES={'base':(0,), 'tool-expanded':(0,1), 'repair-expanded':(0,2),
          'single-changes':(0,1,2), 'complete':(0,1,2,3)}
ARMS=tuple(FAMILIES)+('known-rule-oracle',)


def execute(a,b,op,maker,rule):
    if rule not in RULES:raise ValueError('unadmitted composition rule')
    if rule in ('inverted-repair','both') and op=='repair-evidence':
        return (a[0],(1-(a[0]^maker[2])) if maker[1] else 1-a[1],a[2])
    return M.execute(a,b,op,maker,'presentation-tool' if rule in ('presentation-tool','both') else 'original')


def enumerate_rule(world,rule):
    records=[]
    for ci,context in enumerate(L.CONTEXTS):
        for mi,maker in enumerate(L.MAKERS):
            frontier=[(tuple(context['initial']),tuple(context['initial']),[],1.)]
            for step in range(3):
                following=[]
                for a,b,history,probability in frontier:
                    for goal,op,weight in L.choices(world,maker,a,step,context):
                        after=execute(a,b,op,maker,rule)
                        event=dict(step=step,goal=goal,operation=op,before=list(a),after=list(after),undo_buffer=list(b),
                            dependency_edges=[[0,1]],
                            tool_proposal=list(after) if op=='accept-tool' else None,perceived_claim=a[0]^maker[2])
                        following.append((after,a,history+[event],probability*weight))
                frontier=following
            for a,b,history,probability in frontier:
                records.append(dict(context_index=ci,maker_index=mi,maker=list(maker),initial=context['initial'],
                    requested_purpose=context['requested_purpose'],final=list(a),steps=history,probability=probability/64))
    return records


def posterior(groups,key,indices):
    joint=defaultdict(float)
    for i in indices:
        for process,value in groups[i].get(key,{}).get('mass',{}).items():joint[process]+=value
    total=sum(joint.values())
    return {k:v/total for k,v in joint.items()} if total else {}


def controls():
    a=(0,1,0);m=(0,1,0,0);b=(1,0,1)
    def endpoint(ops,rule):
        x=y=a
        for op in ops:y,x=x,execute(x,y,op,m,rule)
        return x
    return {'live:tool_change':execute(a,b,'accept-tool',m,'both')==(0,1,1),
        'live:repair_change':execute((1,1,0),b,'repair-evidence',m,'both')==(1,0,0),
        'positive:composed_execution':endpoint(('accept-tool','repair-evidence'),'both')==(0,1,1),
        'positive:undo_restores':endpoint(('accept-tool','undo'),'both')==a,
        'placebo:inspect_identity':all(execute(a,b,'inspect',m,r)==a for r in RULES),
        'positive:missing_support':M.score({('b',):1.},{('a',):1.})['infinite_loss_mass']==1,
        'positive:family_union':posterior([{'x':{'mass':{('a',):2.}}},{'x':{'mass':{('b',):1.}}}],'x',(0,1))=={('a',):2/3,('b',):1/3}}


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls()
    if not all(checks.values()) or cfg['rules']!=list(RULES) or cfg['arms']!=list(ARMS):raise ValueError('crossed rule admission failed')
    write(root/'CONTROLS.json',checks);(root/'raw').mkdir();public={};cells=[];enumeration=[];timing=[];total_rows=0
    for lineage in cfg['lineages']:
        start=time.process_time();world=L.law(lineage);populations=[]
        for rule in RULES:
            pulse(phase='crossed-rule-enumeration',lineage=lineage,rule=rule)
            rr=enumerate_rule(world,rule)
            if len(rr)!=13824 or abs(sum(r['probability'] for r in rr)-1)>1e-10:raise ValueError('population mismatch')
            populations.append(rr)
            (root/'raw'/f'{lineage}-{rule}_points.json.gz').write_bytes(gzip.compress(canonical(rr),mtime=0))
            enumeration.append(dict(lineage=lineage,rule=rule,paths=len(rr),world=world))
        for tier in M.TIERS:
            pulse(phase='crossed-rule-conditioning',lineage=lineage,tier=tier)
            pairs=[M.grouped(rr,tier) for rr in populations];groups=[p[0] for p in pairs];rows=[];posteriors=[]
            for _,packets in pairs:public.update(packets)
            for truth_index,group in enumerate(groups):
                for key,truth in sorted(group.items()):
                    mass=sum(truth['mass'].values());target={k:v/mass for k,v in truth['mass'].items()}
                    for arm in ARMS:
                        pred=posterior(groups,key,(truth_index,) if arm=='known-rule-oracle' else FAMILIES[arm])
                        metrics=M.score(target,pred)
                        if arm in ('complete','known-rule-oracle') and metrics['coverage']!=1:raise ValueError('complete family lost truth')
                        rows.append(dict(lineage=lineage,truth_rule=RULES[truth_index],tier=tier,frame=key,arm=arm,probability_mass=mass,**metrics))
                        posteriors.append(dict(truth_rule=RULES[truth_index],frame=key,arm=arm,
                            truth=[[list(k[0]),list(k[1]),v] for k,v in sorted(target.items())],
                            prediction=[[list(k[0]),list(k[1]),v] for k,v in sorted(pred.items())]))
            for rule in RULES:
                for arm in ARMS:
                    rr=[r for r in rows if r['truth_rule']==rule and r['arm']==arm];mass=sum(r['probability_mass'] for r in rr)
                    if abs(mass-1)>1e-10:raise ValueError('conditional population mass')
                    cells.append(dict(lineage=lineage,truth_rule=rule,tier=tier,arm=arm,frames=len(rr),
                        **{k:float(sum(r['probability_mass']*r[k] for r in rr)/mass) for k in
                           ('abstain','coverage','infinite_loss_mass','finite_loss_contribution','entropy','hypotheses')}))
            for label,data in (('scores',rows),('posterior',posteriors)):
                (root/'raw'/f'{lineage}-{tier}-{label}_points.json.gz').write_bytes(gzip.compress(canonical(data),mtime=0))
            total_rows+=len(rows)
        timing.append(dict(lineage=lineage,cpu_seconds=time.process_time()-start))
    write(root/'PUBLIC_PACKET.json',dict(schema='v19.crossed-rules.export.1',frames=public,role='anonymous visible evidence; supplied rules and evaluation truth are separate'))
    write(root/'ENUMERATION.json',enumeration);write(root/'TIMING.jsonl',dict(measurements=timing,accounting='included in native charge'))
    return dict(controls=checks,cells=cells,rows=total_rows,paths=len(cfg['lineages'])*55296,lineages=len(cfg['lineages']),
        fits=0,anonymous_frames=len(public),scope='supplied-family composition diagnostic; constructed method; miniature - architecture untested; no learned law invention')
