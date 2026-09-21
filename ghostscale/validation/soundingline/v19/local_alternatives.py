"""Finite local frame and jointness rivals; evidence changes are explicit."""
from collections import defaultdict
import gzip
import numpy as np
from ..v18_3.io import canonical,digest,write
from ..v18_3.world import rng
from . import local_world as L

def identity(r):
    return (tuple(e['goal'] for e in r['steps']),tuple(e['operation'] for e in r['steps']))

def distribution(records,prior='neutral',relation=None):
    selected=[r for r in records if relation is None or (r['steps'][0]['goal']==r['steps'][2]['goal'])==relation]
    if not selected:return {}
    weights=np.array([r['probability']*(3 if prior=='self-like' and r['maker'][0]==0 else 1) for r in selected])
    weights/=weights.sum();out=defaultdict(float)
    for r,p in zip(selected,weights):out[identity(r)]+=float(p)
    return dict(out)

def marginals(p):
    goals=defaultdict(float);process=defaultdict(float)
    for (g,o),w in p.items():goals[g]+=w;process[o]+=w
    return goals,process

def metrics(p,true,legal):
    if not p:return dict(unknown=True,coverage=False,loss=None,infinite=True,unsupported_mass=0.,contradiction=False,assertions=0)
    g,o=marginals(p);mass=p.get(true,0.)
    unsupported=sum(w for k,w in p.items() if k not in legal)
    best=max(p,key=p.get);single=len(p)==1
    return dict(unknown=False,coverage=mass>0,loss=float(-np.log(mass)) if mass>0 else None,infinite=mass<=0,
        goal_loss=float(-np.log(g[true[0]])) if g.get(true[0],0)>0 else None,
        process_loss=float(-np.log(o[true[1]])) if o.get(true[1],0)>0 else None,
        unsupported_mass=float(unsupported),contradiction=single and best!=true,assertions=int(single),
        unwarranted_single_account=single and len(legal)>1,
        localized_goal_accuracy=float(sum(a==b for a,b in zip(best[0],true[0]))/3),
        localized_operation_accuracy=float(sum(a==b for a,b in zip(best[1],true[1]))/3))

def rectangle(p):
    support=set(p);by=defaultdict(set)
    for g,o in support:by[g].add(o)
    keys=sorted(by)
    for i,g in enumerate(keys):
        for h in keys[i+1:]:
            shared=sorted(by[g]&by[h])
            if len(shared)>=2:
                a,b=shared[:2];positive={(g,a):.5,(h,b):.5};negative={(g,b):.5,(h,a):.5}
                assert marginals(positive)==marginals(negative)
                return dict(goals=[list(g),list(h)],processes=[list(a),list(b)],all_four_executable=True,
                    marginal_equality=True,diagonal_joint_probability=[1.,0.],evidence_identifies_association=False)
    return None

def controls():
    g1=('meaning',)*3;g2=('dependency',)*3;o1=('inspect',)*3;o2=('inspect','inspect','undo')
    p={(g1,o1):.5,(g2,o2):.5};g,o=marginals(p)
    independent={(a,b):u*v for a,u in g.items() for b,v in o.items()}
    return dict(live_joint_correlation=abs(sum(v for k,v in independent.items() if k not in p)-.5)<1e-12,
        placebo_rearrangement_identity=distribution([])=={},
        positive_unknown_abstention=metrics({},(g1,o1),set())['unknown'])

def run(root,plan,pulse):
    cfg=plan['design'];kind=cfg['handler'];rows=[];fixtures=[];public=[];truth=[];denominators=[]
    for lineage in cfg['lineages']:
        pulse(phase='alternative-enumeration',handler=kind,lineage=lineage)
        records=L.enumerate_world(L.law(lineage));weights=np.array([r['probability'] for r in records]);weights/=weights.sum()
        path=root/'evaluator'/f'lineage-{lineage}_points.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(gzip.compress(canonical(records),mtime=0))
        tiers=('E1',) if kind=='frame' else ('E0','E1','E2-sparse','E2-full')
        grouped={}
        for tier in tiers:
            groups=defaultdict(list)
            for r in records:groups[digest(L.project(r,tier))].append(r)
            grouped[tier]=groups
        denominators.append(dict(lineage=lineage,trajectories=len(records),mass=float(weights.sum())))
        for seed in cfg['sampling_seeds']:
            indices=rng('v19-alternative',kind,lineage,seed).choice(len(records),size=32,p=weights)
            for i,index in enumerate(indices):
                r=records[int(index)];true=identity(r)
                for tier in tiers:
                    packet=L.project(r,tier);L.validate_public(packet);group=grouped[tier][digest(packet)]
                    legal=set(identity(v) for v in group)
                    case_id=f'{lineage}-{seed}-{i}-{tier}'
                    public.append(dict(case_id=case_id,packet=packet,input_sha256=digest(packet)))
                    truth.append(dict(case_id=case_id,trajectory_index=int(index),true_goals=list(true[0]),true_operations=list(true[1])))
                    if kind=='frame':
                        relation=r['steps'][0]['goal']==r['steps'][2]['goal']
                        for prior in ('neutral','self-like'):
                            base=distribution(group,prior)
                            arms={name:base for name in ('base','literal-rearrangement','equal-length-reread','irrelevant','corrected')}
                            arms['truthful-new-relation']=distribution(group,prior,relation)
                            arms['wrong-new-relation']=distribution(group,prior,not relation)
                            for arm,p in arms.items():
                                rows.append(dict(lineage=lineage,seed=seed,index=i,tier=tier,prior=prior,arm=arm,
                                    **metrics(p,true,legal),cue_information='new evaluator-selected relation' if 'new-relation' in arm else 'no additional accepted proposition',
                                    asserted_first_last_goal_equality=(relation if arm=='truthful-new-relation' else not relation) if 'new-relation' in arm else None,
                                    atomic_content='first and last selected local goal have the asserted equality' if 'new-relation' in arm else 'unchanged accepted base evidence',
                                    relation_entailed=all((a['steps'][0]['goal']==a['steps'][2]['goal'])==relation for a in group)))
                    else:
                        p=distribution(group);g,o=marginals(p)
                        independent={(a,b):u*v for a,u in g.items() for b,v in o.items()}
                        best=max(p,key=p.get)
                        arms={'coherent-mixture':p,'independent-marginals':independent,'single-best':{best:1.}}
                        for arm,q in arms.items():rows.append(dict(lineage=lineage,seed=seed,index=i,tier=tier,arm=arm,**metrics(q,true,legal)))
                        if not any(f['lineage']==lineage and f['tier']==tier for f in fixtures):
                            fixture=rectangle(p)
                            if fixture:fixtures.append(dict(lineage=lineage,tier=tier,case_id=case_id,**fixture))
                pulse(phase='alternative-case',handler=kind,lineage=lineage,seed=seed,completed=i+1)
        unknown=dict(schema='v19.local.public.1',tier='E0',inputs=dict(artifact=[2,0,0]))
        if not L.infer(unknown,records)['unknown']:raise ValueError('outside-family control failed')
    write(root/'PUBLIC_PACKET.json',dict(schema='v19.alternative.public.1',cases=public,
        role='base evidence only; experimental teacher relations and evaluator labels are not reader input'))
    write(root/'EVALUATOR_ONLY.json',dict(cases=truth,matched_marginal_fixtures=fixtures,denominators=denominators))
    (root/'raw').mkdir(exist_ok=True);(root/'raw/alternative_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    groups=defaultdict(list)
    for row in rows:groups[(row['tier'],row.get('prior','neutral'),row['arm'])].append(row)
    cells=[]
    for key,rr in sorted(groups.items()):
        losses=[r['loss'] for r in rr]
        cells.append(dict(zip(('tier','prior','arm'),key),cases=len(rr),lineages=len({r['lineage'] for r in rr}),
            mean_loss=None if any(x is None for x in losses) else float(np.mean(losses)),
            infinite_fraction=float(np.mean([r['infinite'] for r in rr])),
            coverage=float(np.mean([r['coverage'] for r in rr])),unsupported_mass=float(np.mean([r['unsupported_mass'] for r in rr])),
            contradiction_rate=float(np.mean([r['contradiction'] for r in rr]))))
    checks=controls()
    if kind=='frame':
        identities={}
        for row in rows:
            key=(row['lineage'],row['seed'],row['index'],row['prior'])
            if row['arm']=='base':identities[key]=row
        checks['identity_frames_preserve_loss']=all(row['loss']==identities[(row['lineage'],row['seed'],row['index'],row['prior'])]['loss']
            for row in rows if row['arm'] in ('literal-rearrangement','equal-length-reread','irrelevant','corrected'))
    else:checks['matched_marginal_fixture']=bool(fixtures) and all(f['marginal_equality'] for f in fixtures)
    return dict(controls=checks,handler=kind,cells=cells,scored_rows=len(rows),matched_marginal_fixtures=len(fixtures),
        uncertainty='eight development lineages, two trajectory sampling seeds; no fits; reused finite architecture',
        access='known finite executable hypothesis family; relation cues explicitly add evaluator-selected information',
        pursuit='correction and missing-family follow-on for frame; retain coherent packets if independent marginalization loses compatibility',
        warrant='exploratory constructed-method diagnostic; miniature — architecture untested; not learned-reader or human intent evidence')
