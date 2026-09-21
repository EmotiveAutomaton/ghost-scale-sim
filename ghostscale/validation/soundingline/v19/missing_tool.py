"""Exact support under one supplied in-alphabet omitted execution rule."""
from collections import defaultdict
import gzip
import time
import numpy as np
from ..v18_3.io import canonical, digest, write
from . import local_world as L

RULES=('original','presentation-tool')
TIERS=('E0','E1','E2-sparse','E2-full')
ARMS=('old-exact','expanded-exact','expanded-template','known-rule-oracle')


def execute(artifact,previous,operation,maker,rule):
    if rule not in RULES:raise ValueError('unadmitted tool rule')
    if rule=='presentation-tool' and operation=='accept-tool':
        if not maker[1]:raise ValueError('unreachable tool use')
        return (artifact[0],artifact[1],1-artifact[2])
    return L.execute(artifact,previous,operation,maker)


def enumerate_rule(world,rule):
    records=[]
    for ci,context in enumerate(L.CONTEXTS):
        for mi,maker in enumerate(L.MAKERS):
            frontier=[(tuple(context['initial']),tuple(context['initial']),[],1.)]
            for step in range(3):
                following=[]
                for artifact,previous,history,probability in frontier:
                    for goal,operation,weight in L.choices(world,maker,artifact,step,context):
                        after=execute(artifact,previous,operation,maker,rule)
                        event=dict(step=step,goal=goal,operation=operation,before=list(artifact),after=list(after),
                            undo_buffer=list(previous),dependency_edges=[[0,1]],
                            tool_proposal=list(after) if operation=='accept-tool' else None,
                            perceived_claim=artifact[0]^maker[2])
                        following.append((after,artifact,history+[event],probability*weight))
                frontier=following
            for artifact,previous,history,probability in frontier:
                records.append(dict(context_index=ci,maker_index=mi,maker=list(maker),initial=context['initial'],
                    requested_purpose=context['requested_purpose'],final=list(artifact),steps=history,
                    probability=probability/64))
    return records


def identity(record):
    return tuple(e['goal'] for e in record['steps']),tuple(e['operation'] for e in record['steps'])


def grouped(records,tier):
    groups={};packets={}
    for r in records:
        packet=L.project(r,tier);L.validate_public(packet);key=digest(packet);packets[key]=packet
        if key not in groups:groups[key]=dict(mass=defaultdict(float),count=defaultdict(float))
        ident=identity(r);groups[key]['mass'][ident]+=r['probability'];groups[key]['count'][ident]+=1.
    return groups,packets


def posterior(groups,key,arm,truth_rule):
    selected=[0] if arm=='old-exact' else [truth_rule] if arm=='known-rule-oracle' else [0,1]
    field='count' if arm=='expanded-template' else 'mass';joint=defaultdict(float);law_mass=np.zeros(2)
    for index in selected:
        for process,value in groups[index].get(key,{}).get(field,{}).items():
            joint[process]+=value;law_mass[index]+=value
    total=float(law_mass.sum())
    if not total:return {},law_mass
    return {k:v/total for k,v in joint.items()},law_mass/total


def score(target,prediction):
    if not np.isclose(sum(target.values()),1):raise ValueError('unnormalized truth')
    if prediction and not np.isclose(sum(prediction.values()),1):raise ValueError('unnormalized prediction')
    missing=sum(v for k,v in target.items() if prediction.get(k,0)<=0)
    finite=-sum(v*np.log(prediction[k]) for k,v in target.items() if prediction.get(k,0)>0)
    return dict(abstain=not bool(prediction),coverage=float(1-missing),infinite_loss_mass=float(missing),
        finite_loss_contribution=float(finite),entropy=float(-sum(v*np.log(v) for v in prediction.values())),
        hypotheses=len(prediction))


def controls():
    a=(0,1,0);m=(0,1,0,0);old=execute(a,a,'accept-tool',m,'original');new=execute(a,a,'accept-tool',m,'presentation-tool')
    forbidden=False
    try:execute(a,a,'accept-tool',(0,0,0,0),'presentation-tool')
    except ValueError:forbidden=True
    target={('a',):.25,('b',):.75};known=score(target,target);absent=score(target,{})
    return {'live:changed_tool':old==(1,1,0) and new==(0,1,1),
        'placebo:non_tool_identity':all(execute(a,(1,0,1),op,m,'original')==execute(a,(1,0,1),op,m,'presentation-tool') for op in L.OPERATIONS if op!='accept-tool'),
        'positive:skill_restriction':forbidden,'positive:proper_score':bool(abs(known['finite_loss_contribution']+(.25*np.log(.25)+.75*np.log(.75)))<1e-14),
        'positive:zero_support_abstains':absent['abstain'] and absent['infinite_loss_mass']==1 and absent['finite_loss_contribution']==0}


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('missing-tool controls failed')
    if cfg['rules']!=list(RULES) or cfg['tiers']!=list(TIERS) or cfg['arms']!=list(ARMS):raise ValueError('unadmitted comparison')
    rows=[];posteriors=[];public={};enumeration=[];timing=[]
    (root/'raw').mkdir(exist_ok=True)
    for lineage in cfg['lineages']:
        began=time.process_time();pulse(phase='paired-rule-enumeration',lineage=lineage)
        world=L.law(lineage);records=[enumerate_rule(world,rule) for rule in RULES]
        for rule,rr in zip(RULES,records):
            if len(rr)!=13824 or abs(sum(r['probability'] for r in rr)-1)>1e-10:raise ValueError('path population mismatch')
            (root/'raw'/f'{lineage}-{rule}_points.json.gz').write_bytes(gzip.compress(canonical(rr),mtime=0))
            enumeration.append(dict(lineage=lineage,rule=rule,paths=len(rr),mass=sum(r['probability'] for r in rr),world=world))
        for tier in TIERS:
            pulse(phase='exact-family-conditioning',lineage=lineage,tier=tier)
            grouped_pairs=[grouped(rr,tier) for rr in records];groups=[p[0] for p in grouped_pairs]
            for _,packets in grouped_pairs:public.update(packets)
            for truth_rule,truth_groups in enumerate(groups):
                for key,group in sorted(truth_groups.items()):
                    mass=sum(group['mass'].values());target={k:v/mass for k,v in group['mass'].items()}
                    for arm in ARMS:
                        pred,laws=posterior(groups,key,arm,truth_rule);metrics=score(target,pred)
                        if arm=='known-rule-oracle' and (metrics['abstain'] or metrics['infinite_loss_mass']!=0):raise ValueError('oracle support failed')
                        rows.append(dict(lineage=lineage,truth_rule=RULES[truth_rule],tier=tier,frame=key,arm=arm,
                            probability_mass=mass,**metrics,correct_rule_probability=float(laws[truth_rule]),
                            original_family_compatible=key in groups[0]))
                        posteriors.append(dict(lineage=lineage,truth_rule=RULES[truth_rule],tier=tier,frame=key,arm=arm,
                            truth=[[list(k[0]),list(k[1]),v] for k,v in sorted(target.items())],
                            prediction=[[list(k[0]),list(k[1]),v] for k,v in sorted(pred.items())],rule_probabilities=laws.tolist()))
        timing.append(dict(lineage=lineage,cpu_seconds=time.process_time()-began))
    for label,data in (('missing-tool',rows),('posterior',posteriors)):
        (root/'raw'/f'{label}_points.json.gz').write_bytes(gzip.compress(canonical(data),mtime=0))
    write(root/'PUBLIC_PACKET.json',dict(schema='v19.missing-tool.export.1',frames=public,
        role='anonymous evidence only; rule labels, lineage pairing, probabilities and true processes excluded'))
    write(root/'ENUMERATION.json',enumeration);write(root/'TIMING.jsonl',dict(measurements=timing,accounting='included in native charge'))
    cells=[]
    for lineage in cfg['lineages']:
        for rule in RULES:
            for tier in TIERS:
                for arm in ARMS:
                    rr=[r for r in rows if (r['lineage'],r['truth_rule'],r['tier'],r['arm'])==(lineage,rule,tier,arm)]
                    mass=sum(r['probability_mass'] for r in rr)
                    if abs(mass-1)>1e-10:raise ValueError('evidence population mass failed')
                    metrics=('abstain','coverage','infinite_loss_mass','finite_loss_contribution','entropy','hypotheses','correct_rule_probability','original_family_compatible')
                    cells.append(dict(lineage=lineage,truth_rule=rule,tier=tier,arm=arm,frames=len(rr),
                        **{k:float(sum(r['probability_mass']*r[k] for r in rr)/mass) for k in metrics}))
    checks.update({'positive:oracle_coverage':all(r['coverage']==1 for r in rows if r['arm']=='known-rule-oracle'),
        'positive:expanded_coverage':all(r['coverage']==1 for r in rows if r['arm'].startswith('expanded')),
        'positive:complete_mass':True})
    return dict(controls=checks,cells=cells,rows=len(rows),lineages=len(cfg['lineages']),paths=len(cfg['lineages'])*27648,
        anonymous_frames=len(public),fits=0,scope='exact supplied-family method diagnostic; miniature — architecture untested; no open-ended cause discovery')
