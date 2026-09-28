"""Whole-lineage paired review; rows and repeated fits are not independent worlds."""
from collections import defaultdict
import numpy as np
from .world import seed
from ..v18_3.io import read,write,file_digest
from .runtime import verify_complete
from .checker import reaggregate

def interval(values):
    a=np.asarray(values,float);rng=np.random.default_rng(seed('v20-lineage-bootstrap'))
    if not len(a):raise ValueError('empty contrast')
    boot=np.mean(rng.choice(a,(4096,len(a)),replace=True),axis=1)
    return dict(lineages=len(a),mean=float(a.mean()),lower=float(np.quantile(boot,.025)),upper=float(np.quantile(boot,.975)),unit='independent whole generator lineage; fit seeds averaged within lineage')

def review(roots):
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)));receipts=[];retention=[]
    for root in roots:
        complete=verify_complete(root);d=read(root/'PLAN.json')['design'];s=read(root/'SUMMARY.json')
        if not all(s['controls'].values()):raise ValueError('ruler failed')
        population={(r['lineage'],r['method'],r.get('update')):r for r in s.get('population',[])}
        observed=set()
        for p in (root/'raw').glob('*_points.npz'):
            with np.load(p) as data:
                if 'prediction' not in data:continue
                stored={k:data[k] for k in data.files if k not in ('prediction','truth')}
                reaggregate(data['prediction'],data['truth'],stored)
                detail=read(p.with_name(p.name.replace('_points.npz','.json')))
                identity=(detail['lineage'],detail['method'],detail.get('update'));observed.add(identity)
                metrics=population[identity]['metrics']
                for k,v in stored.items():
                    if not np.isclose(np.mean(v),metrics[k],rtol=2e-10,atol=2e-10):raise ValueError('summary differs from raw '+k)
                if d['branch']=='G4':
                    policy=read(root/'policies'/f"{detail['lineage']}.json")
                    cost=float(np.asarray(policy['costs'])[policy['choices'][detail['method']]].mean())
                    if not np.isclose(cost,metrics['acquisition_cost']) or not np.isclose(cost+metrics['log_loss'],metrics['net_log_loss']):raise ValueError('acquisition cost differs')
                if d['branch']=='G5':
                    with np.load(p.with_name(p.name.replace('_points.npz','-transport_points.npz'))) as t:
                        mass=t['candidate_mass'];unknown=t['unknown_mass'];mask=t['mask'];truth=data['truth']
                        if not np.allclose(mass.sum(1)+unknown,1) or (mass<0).any() or (unknown<0).any():raise ValueError('invalid candidate transport')
                        q=np.where(mask[np.arange(len(truth)),truth],mass[np.arange(len(truth)),truth],unknown)
                        loss=float(-np.log(np.maximum(q,1e-12)).mean());brier=float(((mass*mass).sum(1)+unknown*unknown-2*q+1).mean())
                        if not np.isclose(loss,metrics['candidate_event_log_loss']) or not np.isclose(brier,metrics['candidate_event_brier']):raise ValueError('candidate-event score differs')
        if observed!=set(population):raise ValueError('missing or extra raw forecast population')
        key=tuple((k,str(d.get(k))) for k in ['branch','tier','history','train_cases','shift','split','reliability','copied','irrelevant','candidate_budget'])
        for row in s.get('population',[]):
            method=row['method']+('/'+row['update'] if 'update' in row else '')
            for metric,v in row['metrics'].items():groups[(key,metric)][row['lineage']][method].append(v)
        retention.extend(s.get('rows',[]));receipts.append(dict(packet=root.name,complete_sha256=file_digest(root/'COMPLETE.json'),rows_reaggregated=True))
    contrasts=[]
    for (key,metric),lineages in groups.items():
        arms=sorted(set.intersection(*[set(v) for v in lineages.values()]))
        def contrast(a,b):
            return interval([np.mean(v[a])-np.mean(v[b]) for v in lineages.values()])
        if metric in ('log_loss','unsupported_attribution','brier','correct','net_log_loss'):
            for a in arms:
                for b in arms:
                    if a<b:contrasts.append(dict(configuration=dict(key),metric=metric,left=a,right=b,left_minus_right=contrast(a,b)))
        if metric=='log_loss' and all(a in arms for a in ['saved-conclusion/false-to-true','ledger-recompute/false-to-true','saved-conclusion/unchanged','ledger-recompute/unchanged','saved-conclusion/irrelevant','ledger-recompute/irrelevant']):
            for control in ('unchanged','irrelevant'):
                values=[]
                for v in lineages.values():
                    gain=lambda u:np.mean(v['saved-conclusion/'+u])-np.mean(v['ledger-recompute/'+u])
                    values.append(gain('false-to-true')-gain(control))
                contrasts.append(dict(configuration=dict(key),metric='correction-specific log-loss gain versus '+control,paired=interval(values)))
    return dict(passed=True,packets=receipts,contrasts=contrasts,retention=retention,
        claim_rule='Primary context structured reader must improve log loss by >=0.02 against both direct rivals AND template on untouched confirmation; unsupported attribution may not increase. Costs reported separately; no capacity-matched claim. World intervals are descriptive under this finite coefficient generator; no human inference.',
        status='numerically verified; documentary and bounded source replay required before scientific acceptance')
