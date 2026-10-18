"""D10 retained-data robust summaries; never changes the frozen N2 decision."""
from pathlib import Path
import math
import numpy as np
from scipy.special import expit
from . import family,registry
from .common import read,write,digest,interval
from .pipeline import Stream
POWERS=(1.,.75,.5)

def loss(logit,y):return float(np.logaddexp(0.,(1-2*y)*logit))
def independent_loss(logit,y):
    z=(1-2*y)*logit
    return max(z,0.)+math.log1p(math.exp(-abs(z)))

def robust_pool(means,standard_errors):
    means=np.asarray(means,float);se=np.asarray(standard_errors,float)
    if not np.isfinite(means).all() or not np.isfinite(se).all() or np.any(se<0):raise ValueError('nonfinite robust input')
    variances=se*se;floor=float(np.median(variances))
    if not math.isfinite(floor) or floor<=0:raise ValueError('unidentified variance floor')
    effective=np.maximum(variances,floor);weights=floor/effective;weights/=weights.sum()
    result=float(np.dot(weights,means))
    if not np.isfinite(weights).all() or not math.isfinite(result):raise ValueError('robust pooling overflow')
    # Separately rescaled scalar inverse-variance arithmetic.
    scalar=[floor/max(float(s)*float(s),floor) for s in se];denom=math.fsum(scalar)
    check=math.fsum(w*float(m) for w,m in zip(scalar,means,strict=True))/denom
    if not math.isclose(result,check,abs_tol=1e-12,rel_tol=1e-12):raise ValueError('independent pooled mean differs')
    return dict(mean=result,variance_floor=floor,floored_settings=int(np.sum(variances<floor)),maximum_weight=float(weights.max()),weights=weights.tolist(),finite=True)

def controls():
    value=robust_pool([0.,1.,2.],[1e-158,.1,.2]);assert math.isclose(value['mean'],(0+1+.25*2)/2.25,abs_tol=1e-14)
    for logit in (-512.,-1.,0.,1.,512.):
        for y in (0,1):assert math.isclose(loss(logit,y),independent_loss(logit,y),rel_tol=1e-14,abs_tol=1e-220)
    return dict(passed=True,variance_floor_prevents_tiny_error_domination=True,independent_softplus=True,finite_extreme_log_odds=True,no_score_floor=True)

def validate(plan):
    if len(plan['retained_packets'])!=5 or plan['family']!='N2':raise ValueError('complete retained N2 source set required')
    for item in plan['retained_packets']:
        root=Path(item['packet'])
        if digest(root/'COMPLETE.json')!=item['complete_sha256']:raise ValueError('retained completion changed')
        p=read(root/'PLAN.json')
        if p['family']!='N2' or p['batch'] not in (1,2):raise ValueError('wrong retained family')
    if digest(plan['validation_file'])!=plan['validation_sha256']:raise ValueError('validation bank changed')

def run(root,plan,pulse):
    validate(plan);root=Path(root);root.mkdir(parents=True,exist_ok=False);write(root/'CONTROLS.json',controls())
    variants=registry.variants('N2');tuning={};input_hashes={}
    with np.load(plan['validation_file']) as bank:x,y=bank['x'],bank['y'].reshape(-1)
    if len(x)!=4096:raise ValueError('complete frozen validation bank required')
    for i,v in enumerate(variants):
        indices=np.arange(i,len(x),len(variants));record=x[indices,:1024].reshape(-1,128,8)
        if not np.all(record[:,:,7].sum(1)==v['settings']['records']):raise ValueError('validation row assignment differs')
        if not np.allclose(x[indices,1024:],registry.variant_features(v),atol=0,rtol=0):raise ValueError('validation setting features differ')
        logits=v['settings']['beta']*np.sum((2*record[:,:,0]-1)*record[:,:,7],axis=1)
        risks={str(power):float(np.mean(np.logaddexp(0.,(1-2*y[indices])*power*logits))) for power in POWERS}
        selected=float(min(risks,key=risks.get));tuning[v['id']]=dict(exponent=selected,losses=risks,validation_rows=indices.tolist())
    write(root/'TEMPERING_FIT.json',dict(validation_sha256=plan['validation_sha256'],settings=tuning,selected_before_test_scan=True,powers=POWERS))
    source_summaries={}
    for item in plan['retained_packets']:
        packet=Path(item['packet']);p,summary=family.verify_packet(packet)
        source_summaries[packet.name]=(p,summary)
        input_hashes[packet.name]=dict(complete=item['complete_sha256'],replay=digest(packet/'REPLAY.json'))
    summaries=[];total=0
    for v in variants:
        groups={key:{} for key in ('original','tempered')};seen=set();stream=Stream(root/(v['id']+'_points.jsonl.gz'))
        try:
            for item in plan['retained_packets']:
                packet=Path(item['packet']);source_plan,source_summary=source_summaries[packet.name];batch=source_plan['batch'];folder=packet/'original'/v['id']
                weights=np.asarray(read(folder/'TIMING_FIT.json')['weights']);power=tuning[v['id']]['exponent']
                paths=[folder/(name+'_points.jsonl.gz') for name in ('public','truth','prediction')]
                for public,truth,pred in zip(*(family.rows(p) for p in paths),strict=True):
                    identity=public['identity'];_,variant,b,w,k=identity
                    if truth['identity']!=identity or (b,variant)!=(batch,v['id']) or (pred['world'],pred['case'])!=(w,k):raise ValueError('retained identity differs')
                    if (b,w,k) in seen or not 0<=w<16 or not 0<=k<64:raise ValueError('duplicate or invalid retained case')
                    seen.add((b,w,k));history=public['public']['history'];actual=truth['truth']['y'][0];beta=v['settings']['beta']
                    naive=beta*sum(2*r['choice']-1 for r in history)
                    weighted=beta*math.fsum((2*r['choice']-1)*float(weights[r['latency_bin']]) for r in history)
                    losses={name:loss(z,actual) for name,z in [('naive',naive),('weighted',weighted),('tempered',power*naive)]}
                    for name,z in [('naive',naive),('weighted',weighted),('tempered',power*naive)]:
                        if not math.isclose(losses[name],independent_loss(z,actual),rel_tol=1e-12,abs_tol=1e-220):raise ValueError('independent score differs')
                    for name,key,z in [('naive','everything-is-preference',naive),('weighted','response-time-weighted',weighted)]:
                        saved=np.array(pred['probabilities'][key])[0];expected=np.array([expit(-z),expit(z)])
                        if not np.allclose(expected,saved,atol=1e-11,rtol=1e-10):raise ValueError('retained prediction does not reproduce')
                        if not math.isclose(losses[name],pred['scores'][key]['log_loss'],abs_tol=1e-10,rel_tol=1e-10):raise ValueError('retained loss does not reproduce')
                    for name in groups:groups[name].setdefault((b,w),[]).append(losses['naive' if name=='original' else 'tempered']-losses['weighted'])
                    stream.write(dict(identity=identity,naive_log_odds=naive,weighted_log_odds=weighted,exponent=power,truth=actual,losses=losses))
        finally:stream.close()
        if len(seen)!=2048 or len(groups['original'])!=32 or any(len(xs)!=64 for xs in groups['original'].values()):raise ValueError('complete two-burn-in coverage required')
        rows={name:[math.fsum(xs)/len(xs) for _,xs in sorted(worlds.items())] for name,worlds in groups.items()}
        for item in plan['retained_packets']:
            source_plan,source_summary=source_summaries[Path(item['packet']).name];sv=next(x for x in source_summary['variants'] if x['variant']['id']==v['id'])
            worlds=source_plan.get('world_indices',list(range(16)));observed=[rows['original'][(source_plan['batch']-1)*16+w] for w in worlds]
            if not np.allclose(observed,sv['world_gains'],atol=1e-10,rtol=1e-10):raise ValueError('published world gain does not reproduce')
        item=dict(variant=v,structural=v['settings']['nondiagnostic']==0,exponent=tuning[v['id']]['exponent'],world_gains=rows,
            estimates={name:interval(values,['N2-D10',v['id'],name]) for name,values in rows.items()})
        summaries.append(item);total+=len(seen);pulse(phase='retained-N2-robust',variant=v['id'],cases=total)
    included=[x for x in summaries if not x['structural']];pooled={}
    for name in ('original','tempered'):
        means=[float(np.mean(x['world_gains'][name])) for x in included];errors=[]
        for item in included:
            values=np.asarray(item['world_gains'][name]);scale=float(np.max(np.abs(values)))
            errors.append(scale*float(np.std(values/scale,ddof=1))/math.sqrt(32) if scale else 0.)
        pooled[name]=robust_pool(means,errors)
    count=sum(item['estimates']['tempered']['mean']>.02 for item in included)
    write(root/'SUMMARY.json',dict(state='DESCRIPTIVE_REANALYSIS_CHECKED',family='N2',cases=total,settings=summaries,
        robust_pooled=pooled,above_margin=dict(count=count,total=120,share=count/120,margin=.02,formal_partial_conjunction_test=False),
        original_decision='STOPPED_BELOW_MARGIN',new_discovery_cases=0,source_receipts=input_hashes))


def reconstruct(root,plan):
    """Separate retained-row aggregation, scalar scores, quantiles and pooling."""
    from .common import rng
    root=Path(root);validate(plan);summary=read(root/'SUMMARY.json');checked={};pooled={name:[] for name in ('original','tempered')};above=0;total=0
    for item in summary['settings']:
        v=item['variant'];groups={name:{} for name in pooled};seen=set()
        for row in family.rows(root/(v['id']+'_points.jsonl.gz')):
            identity=row['identity'];_,name,b,w,k=identity
            if name!=v['id'] or b not in (1,2) or not 0<=w<16 or not 0<=k<64 or (b,w,k) in seen:raise ValueError('independent coverage mismatch')
            seen.add((b,w,k));values={key:independent_loss(logit,row['truth']) for key,logit in [('naive',row['naive_log_odds']),('weighted',row['weighted_log_odds']),('tempered',row['exponent']*row['naive_log_odds'])]}
            if any(not math.isclose(value,row['losses'][key],abs_tol=1e-220,rel_tol=1e-12) for key,value in values.items()):raise ValueError('retained scalar score mismatch')
            for key in groups:groups[key].setdefault((b,w),[]).append(values['naive' if key=='original' else 'tempered']-values['weighted'])
        if len(seen)!=2048:raise ValueError('independent complete coverage required')
        for name,worlds in groups.items():
            means=[math.fsum(values)/64 for _,values in sorted(worlds.items())];center=math.fsum(means)/32
            if not np.allclose(means,item['world_gains'][name],atol=1e-10,rtol=1e-10):raise ValueError('independent world means mismatch')
            indices=rng('bootstrap',['N2-D10',v['id'],name]).integers(32,size=(4096,32))
            boot=sorted(math.fsum(means[i] for i in draw)/32 for draw in indices)
            def quantile(q):
                position=q*(len(boot)-1);low=int(position)
                return boot[low]+(position-low)*(boot[min(low+1,len(boot)-1)]-boot[low])
            for key,value in [('mean',center),('lower',quantile(.025)),('upper',quantile(.975))]:
                if not math.isclose(value,item['estimates'][name][key],abs_tol=1e-10,rel_tol=1e-10):raise ValueError('independent interval mismatch')
            if not item['structural']:
                scale=max(abs(x) for x in means);variance=(scale*math.sqrt(math.fsum((x/scale-center/scale)**2 for x in means)/31)/math.sqrt(32))**2 if scale else 0.
                pooled[name].append((center,variance))
                if name=='tempered':above+=center>.02
        checked[v['id']]=dict(cases=len(seen),worlds=32);total+=len(seen)
    for name,rows in pooled.items():
        variances=sorted(v for _,v in rows);floor=(variances[59]+variances[60])/2
        weights=[floor/max(floor,v) for _,v in rows];value=math.fsum(w*m for w,(m,_) in zip(weights,rows,strict=True))/math.fsum(weights)
        if not math.isclose(value,summary['robust_pooled'][name]['mean'],abs_tol=1e-10,rel_tol=1e-10):raise ValueError('independent robust pool mismatch')
    if total!=253952 or above!=summary['above_margin']['count']:raise ValueError('independent scope/count mismatch')
    return dict(state='passed',cases=total,settings=len(checked),intervals_reconstructed=248,variants=checked,above_margin_count=above,independent_scalar_scores=True,independent_variance_floor_and_pool=True)
