"""Reproduce retained paired estimates, inventory access and audit numerical laws."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from ..v18_3.io import read, write, file_digest, digest, canonical
from ..v18_3 import world as W
from ..v18_4 import neural_data as D
from ..v18_4 import feasible_bank as F
from .algebra import audit, bank_error, controls


ARMS = ('L2d-old-total-bank-1','L2d-diverse-bank-1')


def raw_blocks(base, arm, pins):
    root=Path(base)/arm
    for name, sha in pins[arm].items():
        if name!='raw_blocks' and file_digest(root/name)!=sha:
            raise ValueError('retained source identity changed: '+name)
    complete=read(root/'COMPLETE.json')
    if complete['summary_sha256']!=file_digest(root/'SUMMARY.json'):
        raise ValueError('retained completion mismatch')
    for name, sha in pins[arm]['raw_blocks'].items():
        path=root/'raw'/(Path(name).stem+'_points.json.gz')
        if file_digest(path)!=sha:raise ValueError('retained raw corruption')
        yield json.loads(gzip.decompress(path.read_bytes()))


def class_name(world):return world['rule']+'|endogenous-'+str(int(world['endogenous']))


def paired_interval(values, seed=190301):
    values=np.asarray(values,float)
    r=np.random.default_rng(seed)
    means=values[r.integers(len(values),size=(10000,len(values)))].mean(axis=1)
    return dict(lineages=len(values),mean=float(values.mean()),low=float(np.quantile(means,.025)),
        high=float(np.quantile(means,.975)),resamples=10000,seed=seed,
        uncertainty='paired coefficient-lineage resampling conditional on retained fits and draws')


def certify(root, plan, pulse):
    cfg=plan['design']; base=cfg['retained_root']; pins=cfg['retained_pins']
    records=[]; keys=[]; diagnostics={}; noise=defaultdict(list); reference_errors=[]
    for arm in ARMS:
        rows={}; identities={}; count=0
        for block in raw_blocks(base,arm,pins):
            pulse(phase='retained-analysis',arm=arm,completed_histories=count)
            for unit in block:
                payload=unit['payload']; w=payload['world']; index=unit['index']
                key=(unit['cell'],index,unit['support'])
                if key in rows:raise ValueError('duplicate retained history')
                identities[key]=digest({k:v for k,v in payload.items() if k!='banks'})
                if not all(unit['gates'].values()):raise ValueError('retained gate failed')
                # Recompute every score directly from retained probabilities and evaluator truth.
                truth=np.stack([W.artifact_matrix(w,c)[payload['state']] for c in F.QUERIES])
                recalculated=defaultdict(list)
                for name,p in unit['forecasts'].items():
                    p=np.asarray(p,float)
                    if np.min(p)<0 or not np.allclose(p.sum(1),1,atol=1e-6,rtol=0):
                        raise ValueError('invalid retained forecast')
                    if '|' in name:
                        reader,mode=name.split('|'); kind=reader.rsplit('-seed',1)[0]
                    else:kind,mode=name,'reference'
                    for label,selection in (('composition',slice(0,3)),('farther',slice(3,5))):
                        recalculated[kind+'|'+mode+'|'+label].append(F.score(truth[selection],p[selection])[0])
                scored={k:float(np.mean(v)) for k,v in recalculated.items()}
                if any(abs(scored[r['method']]-r['expected_loss'])>1e-10 for r in unit['rows']):
                    raise ValueError('retained score reconstruction mismatch')
                rows[key]=dict(index=index,cell=unit['cell'],support=unit['support'],world_class=class_name(w),loss=scored)
                world_id=digest(w)
                if world_id not in diagnostics:
                    atoms=np.concatenate([W.artifact_matrix(w,c) for c in D.TRAIN_QUERIES],axis=1)
                    target=np.concatenate([W.artifact_matrix(w,c) for c in F.QUERIES],axis=1)
                    diagnostic,mapping=audit(atoms,target)
                    diagnostics[world_id]=(atoms,target,mapping,dict(diagnostic,world_sha256=world_id,world_class=class_name(w),coefficient_index=index))
                atoms,target,mapping,_=diagnostics[world_id]
                posterior=W.posterior(W.packet(w,payload['history']))
                for reader, bank in payload['banks'].items():
                    kind=reader.rsplit('-seed',1)[0]
                    diagnostic=bank_error(atoms,target,posterior,bank,mapping)
                    noise[(arm,class_name(w),kind)].append(diagnostic)
                    reference_errors.append(diagnostic['exact_bank_max_readout_error'])
                count+=1
        if count!=1536:raise ValueError('incomplete retained population')
        records.append(rows);keys.append(identities)
    if keys[0]!=keys[1]:raise ValueError('unpaired retained populations')
    classes=sorted({r['world_class'] for r in records[0].values()})
    contrasts=[]; class_rows=[]
    for population in ['equal-class-pooled']+classes:
        eligible=[k for k,r in records[0].items() if population=='equal-class-pooled' or r['world_class']==population]
        for reader in ('question-only','direct','flat','split'):
            method=reader+'|hull|farther'; lineage=defaultdict(list)
            for key in eligible:
                lineage[key[1]].append(records[1][key]['loss'][method]-records[0][key]['loss'][method])
            if len(lineage)!=48:raise ValueError('coefficient lineage denominator')
            contrasts.append(dict(population=population,reader=reader,histories=len(eligible),
                **paired_interval([np.mean(lineage[i]) for i in sorted(lineage)])))
        if population!='equal-class-pooled':
            means={m:float(np.mean([records[1][k]['loss'][m] for k in eligible])) for m in records[1][eligible[0]]['loss']}
            class_rows.append(dict(world_class=population,histories=len(eligible),means=means,
                history_value=means['prior|reference|farther']-means['history-posterior|reference|farther'],
                split_remaining_gap=means['split|hull|farther']-means['history-posterior|reference|farther']))
    write(root/'LINEAGE_POINTS.json',dict(arms=[dict(arm=a,rows=list(r.values())) for a,r in zip(ARMS,records)]))
    write(root/'CERTIFICATES.json',dict(rows=[x[3] for x in diagnostics.values()],scope='fixed supplied laws; numerical rank, not an exact real-arithmetic rank proof'))
    write(root/'NOISE_DIAGNOSTICS.json',dict(rows=[dict(arm=k[0],world_class=k[1],reader=k[2],observations=len(v),
        means={m:float(np.mean([r[m] for r in v])) for m in v[0]}) for k,v in noise.items()],
        scope='empirical learned-bank error passed through the tighter-tolerance supplied linear law; not a learned decoder'))
    return dict(histories_per_arm=1536,lineages=48,retained_scores_reconstructed=3072,paired_rosters=True,
        contrasts=contrasts,classes=class_rows,certificate_worlds=len(diagnostics),controls=controls(),
        max_exact_bank_readout_error=max(reference_errors),
        access='realized program history, hidden rule and full finite policy matrices supplied; oracle-law diagnostic',
        reference_allocation='stipulated uniform 24-state prior; retained neural test states use only skill-positive parity halves; not test-allocation-optimal Bayes',
        pursuit='conditioning and learned-law access remain open',warrant='descriptive conditional retained-fit method analysis; miniature — architecture untested')


def packet_zero(root,plan,pulse):
    cfg=plan['design']; first=next(raw_blocks(cfg['retained_root'],ARMS[0],cfg['retained_pins']))[0]
    history=first['payload']['history']; last=history[-1]
    actual=list(last['program']); artifact=last['artifact']
    alternatives=[list(p) for p,a in zip(W.PROGRAMS,W.ARTIFACTS) if int(a)==artifact]
    public=[]; evaluator=[]
    for index,tier in enumerate(('E0','E1','E2-sparse','E2-full','E1-corrected')):
        pulse(phase='packet-zero',tier=tier)
        case=f'case-{index:03d}'
        inputs=dict(artifact=artifact,passages=[dict(id=f'unit-{j}',value=bool(artifact&(1<<j))) for j in range(4)])
        if tier!='E0':inputs['context']=last['context']
        if tier=='E2-sparse':inputs['observations']=[dict(step=0,operation=actual[0])] if actual else []
        if tier=='E2-full':inputs['observations']=[dict(step=i,operation=v) for i,v in enumerate(actual)]
        if tier=='E1-corrected':
            # Transparent controlled report injection, not a claim of recorded human testimony.
            inputs['reports']=[dict(status='retracted',field='budget',value=1 if last['context']['budget']!=1 else 2),
                dict(status='correction',field='budget',value=last['context']['budget'])]
        compatible=alternatives
        if tier=='E2-sparse' and actual:compatible=[p for p in alternatives if p and p[0]==actual[0]]
        if tier=='E2-full':compatible=[actual]
        public.append(dict(schema='v19.evidence.1',case_id=case,tier=tier,inputs=inputs,input_sha256=digest(inputs),
            baseline=dict(process_alternatives=compatible,support='legal-process compatibility only; uncalibrated',
                unknown=['local subordinate goals unavailable in this old executor','governing purpose','unobserved attention','endorsement','human values']),
            scope='retained four-bit V16 executor, not the new three-unit local world'))
        evaluator.append(dict(case_id=case,actual_program=actual,state=first['payload']['state'],world=first['payload']['world'],
            process_coverage=actual in compatible,constructed_misleading_report=tier=='E1-corrected'))
    write(root/'PUBLIC_PACKET.json',dict(schema='v19.packet-zero.1',cases=public))
    write(root/'EVALUATOR_ONLY.json',dict(schema='v19.evaluator.1',cases=evaluator))
    return dict(cases=len(public),tiers=[c['tier'] for c in public],all_true_processes_covered=all(c['process_coverage'] for c in evaluator),
        controls=controls(),pursuit='early downstream evidence-role integration available',
        warrant='executor-backed projection fixture, not local-goal recovery or calibrated belief')
