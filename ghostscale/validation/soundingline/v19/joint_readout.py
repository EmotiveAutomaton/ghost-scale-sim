"""Matched learned readouts of complete local goal-operation paths."""
from collections import defaultdict
import gzip,time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from ..v18_3.world import rng
from ..v18_3.io import canonical,digest,write
from . import local_world as L,readout_data as D,readout_model as M

UNIVERSE=5832
ARMS=('raw-history','frozen-latent','learned-bank')

def label(r):
    gs=0;os=0
    for e in r['steps']:gs=3*gs+L.GOALS.index(e['goal']);os=6*os+L.OPERATIONS.index(e['operation'])
    return 216*gs+os

def decode(k):
    g,o=divmod(int(k),216);goals=[];operations=[]
    for _ in range(3):goals.insert(0,g%3);operations.insert(0,o%6);g//=3;o//=6
    return goals,operations

def objective(flat,x,y,classes):
    w=flat.reshape(x.shape[1],classes);z=x@w;logp=z-logsumexp(z,axis=1,keepdims=True);p=np.exp(logp)
    value=-logp[np.arange(len(y)),y].mean()+.005*np.sum(w[1:]**2)
    p[np.arange(len(y)),y]-=1;grad=x.T@p/len(y);grad[1:]+=.01*w[1:]
    return float(value),grad.ravel()

def fit(x,y,classes,pulse):
    calls=[0]
    def cb(w):
        calls[0]+=1
        if calls[0]%10==0:pulse(iteration=calls[0])
    result=minimize(objective,np.zeros(x.shape[1]*classes),args=(x,y,classes),jac=True,method='L-BFGS-B',callback=cb,
        options=dict(maxiter=200,maxfun=400,gtol=1e-7,ftol=1e-12,maxls=30))
    val,grad=objective(result.x,x,y,classes)
    if not np.isfinite(val) or not np.isfinite(result.x).all():raise ValueError('nonfinite joint head')
    return result.x.reshape(x.shape[1],classes),dict(success=bool(result.success),status=int(result.status),iterations=int(result.nit),function_calls=int(result.nfev),objective=val,maximum_gradient=float(abs(grad).max()),gradient_tolerance_met=bool(abs(grad).max()<=1e-7))

def forecast(x,w,budget):
    z=x@w;p=np.exp(z-logsumexp(z,axis=1,keepdims=True));return p*(1-1/(budget+1))

def metrics(pred,alphabet,target,budget):
    alphabet=np.asarray(alphabet,int);unknown=1/(budget+1);tail=unknown/(UNIVERSE-len(alphabet));target=dict(target)
    y=np.array([target.get(int(k),0.) for k in alphabet]);remaining=1-y.sum();keys=set(map(int,alphabet));unseen=sum(k not in keys for k in target)
    p=np.asarray(pred);top=int(np.argmax(p));chosen=int(alphabet[top]);g,o=decode(chosen)
    # Proper scores on the full fixed syntactic universe, not a coarsened unknown bin.
    loss=-float(y@np.log(np.maximum(p,1e-300)))-remaining*np.log(tail)
    squared=float(p@p)+(UNIVERSE-len(alphabet))*tail**2-2*(float(p@y)+tail*remaining)+sum(v*v for v in target.values())
    compatible=float(p[y>0].sum()+unseen*tail)
    candidates=list(np.argsort(-p,kind='stable'));covered=[];mass=0.
    for i in candidates:
        covered.append(int(alphabet[i]));mass+=p[i]
        if mass>=.9:break
    if mass<.9:raise ValueError('unknown mass unexpectedly needed for candidate set')
    correct_goals=np.zeros(3);correct_ops=np.zeros(3)
    for k,v in target.items():
        tg,to=decode(k);correct_goals+=v*(np.array(tg)==g);correct_ops+=v*(np.array(to)==o)
    return dict(loss=loss,squared_error=squared,compatible_mass=compatible,candidate_coverage=sum(target.get(k,0) for k in covered),candidate_size=len(covered),
        top_incompatible=chosen not in target,abstain=float(p[top])<.5,top_probability=float(p[top]),unknown_mass=unknown,truth_outside_alphabet=float(remaining),
        goal_accuracy=float(correct_goals.mean()),operation_accuracy=float(correct_ops.mean()),**{f'goal_{i}':float(v) for i,v in enumerate(correct_goals)},**{f'operation_{i}':float(v) for i,v in enumerate(correct_ops)})

def controls():
    x=np.column_stack((np.ones(20),np.tile([-1.,1.],10)));y=np.tile([0,1],10);w,receipt=fit(x,y,2,lambda **k:None);p=forecast(x,w,32)
    z=np.array([.2,-.1,.3,-.4]);value,grad=objective(z,x,y,2);eps=1e-6
    fd=np.array([(objective(z+np.eye(4)[i]*eps,x,y,2)[0]-objective(z-np.eye(4)[i]*eps,x,y,2)[0])/(2*eps) for i in range(4)])
    uniform=np.array([.5,.5])*(32/33);a=metrics(uniform,[0,1],[(0,.5),(1,.5)],32);b=metrics(uniform,[0,1],[(2,1.)],32)
    constant,cr=fit(np.ones((20,1)),y,2,lambda **k:None)
    return {'live:known_labels':bool((p.argmax(1)==y).all()),'placebo:constant_features':bool(np.allclose(forecast(np.ones((2,1)),constant,32),16/33)),
        'positive:gradient':bool(np.max(abs(fd-grad))<1e-8),'positive:fixed_universe':bool(np.isclose(uniform.sum()+1/33,1)),
        'positive:unknown_loss':bool(np.isclose(b['loss'],-np.log(1/33/(UNIVERSE-2))) and b['truth_outside_alphabet']==1),
        'positive:label_codec':all(label(dict(steps=[dict(goal=L.GOALS[g],operation=L.OPERATIONS[o]) for g,o in zip(*decode(k))]))==k for k in range(UNIVERSE)),
        **D.controls()}

def prepare(root,cfg,pulse):
    training={};references={};packets={};timing=[]
    for folder in ('raw','reader','training','models','forecasts','evaluator'):(root/folder).mkdir()
    for split,ls in [('train',cfg['train_lineages']),('development',cfg['development_lineages'])]:
        for lineage in ls:
            began=time.process_time();pulse(phase='joint-population',split=split,lineage=lineage);rr=L.enumerate_world(L.law(lineage));old=D.local_cache(rr)
            (root/'raw'/f'{split}-{lineage}_points.json.gz').write_bytes(gzip.compress(canonical(rr),mtime=0))
            if split=='train':training[lineage]=(rr,old)
            else:
                for tier in cfg['tiers']:
                    gr={}
                    for r in rr:
                        packet=L.project(r,tier);key=digest(packet);packets[key]=packet
                        if key not in gr:gr[key]=defaultdict(float)
                        gr[key][label(r)]+=r['probability']
                    references[lineage,tier]=gr
            timing.append(dict(phase='population-and-reference',split=split,lineage=lineage,cpu_seconds=time.process_time()-began))
    write(root/'reader/PACKETS.json',dict(schema='v19.joint-readout.reader.1',packets=packets))
    saved=[dict(lineage=l,tier=tier,frames=[dict(frame=key,mass=sum(v.values()),target=[[k,w/sum(v.values())] for k,w in sorted(v.items())]) for key,v in sorted(gr.items())]) for (l,tier),gr in references.items()]
    write(root/'evaluator/REFERENCES.json',saved)
    return training,references,packets,timing

def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()) or cfg['arms']!=list(ARMS):raise ValueError('joint readout admission failed')
    training,references,packets,timing=prepare(root,cfg,pulse);rows=[];fits=[]
    for draw in cfg['training_draws']:
        indices={}
        for l,(rr,_) in training.items():
            weights=np.array([r['probability'] for r in rr]);weights/=weights.sum();indices[l]=rng('v19-joint-readout-training',draw,l).choice(len(rr),size=cfg['paths_per_lineage'],p=weights).tolist()
        selected=[(l,training[l][0][indices[l][i]]) for i in range(cfg['paths_per_lineage']) for l in cfg['train_lineages']]
        write(root/'evaluator'/f'selection-{draw}.json',dict(indices=indices,order='interleaved lineages',identities=[[l,r['context_index'],r['maker_index']] for l,r in selected]))
        labels=np.array([label(r) for _,r in selected]);old=np.array([training[l][1][r['maker_index']] for l,r in selected])
        for tier in cfg['tiers']:
            x=np.array([D.local_features(L.project(r,tier)) for _,r in selected]);keys=sorted(k for k,p in packets.items() if p['tier']==tier);xt=np.array([D.local_features(packets[k]) for k in keys]);positions={k:i for i,k in enumerate(keys)}
            M.save_arrays(root/'training'/f'{draw}-{tier}.npz',x=x,old=old,joint_label=labels)
            for seed in cfg['fit_seeds']:
                pulse(phase='joint-old-fit',draw=draw,tier=tier,seed=seed);began=time.process_time()
                h,hw,hb=M.basis(x,seed);ht,_,_=M.basis(xt,seed);ow=M.fit(h,old);latent=np.linalg.svd(ow,full_matrices=False)[0]
                bank,invalid=M.probabilities(h@ow,[8]*4);bankt,invalidt=M.probabilities(ht@ow,[8]*4)
                stem=f'{draw}-{tier}-{seed}';M.save_arrays(root/'models'/f'{stem}-old.npz',history_weights=hw,history_bias=hb,old_head=ow,latent_basis=latent)
                timing.append(dict(phase='old-fit-and-encoding',draw=draw,tier=tier,seed=seed,cpu_seconds=time.process_time()-began,old_head_parameters=ow.size,old_invalid_fraction=float(invalidt.mean())))
                pairs={'raw-history':(x,xt),'frozen-latent':(h@latent,ht@latent),'learned-bank':(bank,bankt)}
                for budget in cfg['budgets']:
                    if budget>len(selected) or budget<16:raise ValueError('infeasible joint budget')
                    alphabet=np.unique(labels[:budget]);target=np.searchsorted(alphabet,labels[:budget]);write(root/'models'/f'{stem}-{budget}-alphabet.json',alphabet.tolist())
                    predictions={}
                    for arm,(xx,tt) in pairs.items():
                        pulse(phase='joint-target-fit',draw=draw,tier=tier,seed=seed,budget=budget,arm=arm);began=time.process_time();f,rw,rb=M.basis(xx,seed+17);ft,_,_=M.basis(tt,seed+17)
                        w,receipt=fit(f[:budget],target,len(alphabet),lambda **kw:pulse(phase='joint-optimizer',draw=draw,tier=tier,seed=seed,budget=budget,arm=arm,**kw));pred=forecast(ft,w,budget)
                        if not np.allclose(pred.sum(1),1-1/(budget+1)) or (pred<0).any():raise ValueError('invalid joint forecast')
                        n=f'{stem}-{budget}-{arm}';M.save_arrays(root/'models'/f'{n}.npz',readout_weights=rw,readout_bias=rb,target_head=w)
                        fits.append(dict(draw=draw,tier=tier,seed=seed,budget=budget,arm=arm,classes=len(alphabet),head_parameters=w.size,input_width=xx.shape[1],**receipt))
                        timing.append(dict(phase='target-fit-and-predict',draw=draw,tier=tier,seed=seed,budget=budget,arm=arm,cpu_seconds=time.process_time()-began))
                        predictions[arm]=pred
                    counts=np.bincount(target,minlength=len(alphabet))+1;mean=counts/counts.sum()*(1-1/(budget+1));predictions['matched-frequency']=np.tile(mean,(len(keys),1))
                    for arm,pred in predictions.items():
                        M.save_arrays(root/'forecasts'/f'{stem}-{budget}-{arm}.npz',probabilities=pred,alphabet=alphabet)
                        write(root/'forecasts'/f'{stem}-{budget}-{arm}-frames.json',keys)
                        for lineage in cfg['development_lineages']:
                            sums=defaultdict(float);total=0.
                            for key,joint in references[lineage,tier].items():
                                mass=sum(joint.values());truth=[(k,v/mass) for k,v in joint.items()];m=metrics(pred[positions[key]],alphabet,truth,budget);total+=mass
                                for k,v in m.items():sums[k]+=mass*v
                            if abs(total-1)>1e-10:raise ValueError('native query mass')
                            rows.append(dict(draw=draw,tier=tier,seed=seed,budget=budget,arm=arm,lineage=lineage,frames=len(references[lineage,tier]),**{k:float(v/total) for k,v in sums.items()}))
    (root/'raw/joint-readout_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    oracle=[]
    for (lineage,tier),gr in references.items():
        entropy=0.
        for v in gr.values():
            mass=sum(v.values());entropy-=sum(w*np.log(w/mass) for w in v.values())
        oracle.append(dict(lineage=lineage,tier=tier,loss=float(entropy),compatible_mass=1.))
    write(root/'TIMING.jsonl',dict(measurements=timing,accounting='component times included in native charge; source identity and auxiliary acquisition charged'))
    indexed={tuple(r[k] for k in ('draw','tier','seed','budget','arm','lineage')):r['loss'] for r in rows}
    contrasts=[];areas=[]
    def interval(values):
        v=np.asarray(values);random=np.random.default_rng(190501);samples=v[random.integers(len(v),size=(10000,len(v)))].mean(1)
        return dict(mean=float(v.mean()),low=float(np.quantile(samples,.025)),high=float(np.quantile(samples,.975)),lineage_values=v.tolist())
    for tier in cfg['tiers']:
        for baseline in ('raw-history','frozen-latent'):
            curves=[]
            for budget in cfg['budgets']:
                differences={(d,s,l):indexed[d,tier,s,budget,'learned-bank',l]-indexed[d,tier,s,budget,baseline,l] for d in cfg['training_draws'] for s in cfg['fit_seeds'] for l in cfg['development_lineages']}
                values=[np.mean([v for (d,s,ll),v in differences.items() if ll==l]) for l in cfg['development_lineages']];curves.append(values)
                contrasts.append(dict(tier=tier,baseline=baseline,budget=budget,**interval(values),
                    draw_means=[float(np.mean([v for (dd,s,l),v in differences.items() if dd==d])) for d in cfg['training_draws']],
                    feature_seed_means=[float(np.mean([v for (d,ss,l),v in differences.items() if ss==s])) for s in cfg['fit_seeds']]))
            if len(cfg['budgets'])>1:
                area=np.trapezoid(curves,x=np.log(cfg['budgets']),axis=0)/np.log(cfg['budgets'][-1]/cfg['budgets'][0])
                areas.append(dict(tier=tier,baseline=baseline,**interval(area)))
    return dict(controls=checks,rows=len(rows),cells=rows,fits=fits,exact_reference=oracle,universe=UNIVERSE,contrasts=contrasts,normalized_log_budget_area=areas,
        scope='exploratory joint historical labels, finite syntactic uniform unknown tail, complete natural development populations; fixed-feature seeds are not tiny-model settings; miniature — architecture untested')
