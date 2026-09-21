"""Independent reconstruction of the fixed-universe joint readout.

No imports from the producer's world, features, fitting or scoring modules.
Known-answer controls run before any completed scientific input is opened.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from ..v18_3.world import rng

OPS = ('edit-claim','repair-evidence','replace-presentation','accept-tool','inspect','undo')
GOALS = ('meaning','dependency','presentation')
UNIVERSE = 5832


def close(a, b, tol=1e-8):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid joint reconstruction shape/value')
    error = float(np.max(abs(a-b))) if a.size else 0.
    if error > tol: raise ValueError(f'joint reconstruction mismatch: {error}')
    return error


def arrays(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k] for k in z.files}


def execute(a, previous, op, skill, belief):
    out = list(a)
    if op == 0: out[0] ^= 1
    elif op == 1: out[1] = (a[0]^belief) if skill else 1-a[1]
    elif op == 2: out[2] ^= 1
    elif op == 3:
        if not skill: raise ValueError('illegal tool')
        out[0] ^= 1; out[1] = out[0]
    elif op == 5: out = list(previous)
    return tuple(out)


def label(record):
    goals = sum(GOALS.index(e['goal'])*3**(2-i) for i,e in enumerate(record['steps']))
    operations = sum(OPS.index(e['operation'])*6**(2-i) for i,e in enumerate(record['steps']))
    return 216*goals+operations


def digits(k):
    g, o = divmod(int(k),216)
    return [g//9,g//3%3,g%3], [o//36,o//6%6,o%6]


def packet(r, tier):
    a = dict(artifact=r['final'])
    if tier != 'E0': a.update(initial=r['initial'],requested_purpose=r['requested_purpose'])
    if tier.startswith('E2'):
        events = r['steps'][:1] if tier == 'E2-sparse' else r['steps']
        a['observations'] = [{k:e[k] for k in ('step','operation','before','after','tool_proposal')} for e in events]
    return dict(schema='v19.local.public.1',tier=tier,inputs=a)


def features(p):
    a=p['inputs']; expected={'artifact'}
    if p['tier'] != 'E0': expected |= {'initial','requested_purpose'}
    if p['tier'].startswith('E2'): expected.add('observations')
    if set(p) != {'schema','tier','inputs'} or set(a) != expected: raise ValueError('private feature field')
    x=list(a['artifact'])+[int('initial' in a)]+a.get('initial',[0,0,0])+[a.get('requested_purpose',0)]
    events={e['step']:e for e in a.get('observations',[])}
    for i in range(3):
        e=events.get(i); x.append(int(e is not None))
        if e is None: x += [0]*15
        else:
            if set(e) != {'step','operation','before','after','tool_proposal'}: raise ValueError('private event field')
            x += [int(e['operation']==op) for op in OPS]+e['before']+e['after']+(e['tool_proposal'] or [0,0,0])
    return x


def basis(x, seed, saved=None):
    random=np.random.default_rng(seed)
    w=random.normal(size=(512,128))/np.sqrt(512); b=random.normal(scale=.2,size=128)
    if saved is not None: close(w,saved[0],0); close(b,saved[1],0)
    return np.column_stack((np.ones(len(x)),np.tanh(np.pad(x,((0,0),(0,512-x.shape[1])))@w+b)))


def ridge(x,y):
    # Centered slope solve independent of the producer's augmented normal equation.
    a=x[:,1:]; am=a.mean(0); ym=y.mean(0); ac=a-am
    slope=np.linalg.solve(ac.T@ac+.01*np.eye(a.shape[1]),ac.T@(y-ym))
    return np.vstack((ym-am@slope,slope))


def bank(raw):
    p=np.maximum(raw.reshape(len(raw),4,8),0)+1e-6
    return (p/p.sum(2,keepdims=True)).reshape(len(raw),32)


def softmax(z):
    a=np.exp(z-z.max(1,keepdims=True)); return a/a.sum(1,keepdims=True)


def objective(x,y,w):
    z=x@w; p=softmax(z)
    logs=z-z.max(1,keepdims=True)-np.log(np.exp(z-z.max(1,keepdims=True)).sum(1,keepdims=True))
    val=-logs[np.arange(len(y)),y].mean()+.005*np.square(w[1:]).sum()
    p[np.arange(len(y)),y]-=1
    grad=x.T@p/len(y); grad[1:]+=.01*w[1:]
    return float(val),grad


def score(p, alphabet, truth, budget):
    """Full-universe proper scores and stable known-alphabet candidate policy."""
    a=np.asarray(alphabet,int); tail=1/(budget+1)/(UNIVERSE-len(a)); y=np.array([truth.get(int(k),0) for k in a])
    known=set(map(int,a));outside=sum(v for k,v in truth.items() if k not in known)
    top=int(np.argmax(p)); chosen=int(a[top]); gs,ops=digits(chosen)
    order=np.argsort(-p,kind='stable'); cum=np.cumsum(p[order]); size=int(np.searchsorted(cum,.9))+1
    if size>len(a): raise ValueError('candidate policy requires tail')
    support=set(truth); coverage=sum(truth.get(int(a[i]),0) for i in order[:size])
    acc=np.zeros(6)
    for k,v in truth.items():
        g,o=digits(k); acc+=v*(np.array(g+o)==np.array(gs+ops))
    return dict(loss=float(-y@np.log(p)-outside*np.log(tail)),
        squared_error=float(p@p+(UNIVERSE-len(a))*tail**2-2*(p@y+tail*outside)+sum(v*v for v in truth.values())),
        compatible_mass=float(p[y>0].sum()+len(support-known)*tail),candidate_coverage=coverage,candidate_size=size,
        top_incompatible=chosen not in support,abstain=p[top]<.5,top_probability=float(p[top]),unknown_mass=1/(budget+1),
        truth_outside_alphabet=outside,goal_accuracy=float(acc[:3].mean()),operation_accuracy=float(acc[3:].mean()),
        **{f'goal_{i}':float(acc[i]) for i in range(3)},**{f'operation_{i}':float(acc[i+3]) for i in range(3)})


def controls():
    x=np.array([[1.,-1.],[1.,1.]]); y=np.array([0,1]); w=np.array([[.2,-.1],[.3,-.4]])
    _,g=objective(x,y,w); fd=np.zeros_like(w)
    for ij in np.ndindex(w.shape):
        step=np.zeros_like(w);step[ij]=1e-6
        fd[ij]=(objective(x,y,w+step)[0]-objective(x,y,w-step)[0])/2e-6
    s=score(np.array([16/33,16/33]),[0,1],{2:1.},32)
    live=score(np.array([.95,32/33-.95]),[0,1],{0:1.},32)
    corruption=False; hidden=False
    try: close([0,1],[1,0])
    except ValueError: corruption=True
    try: features(dict(schema='v19.local.public.1',tier='E0',inputs={'artifact':[0,1,0],'goal':0}))
    except ValueError: hidden=True
    return {'live:known_process':live['candidate_coverage']==1 and live['goal_accuracy']==1 and live['operation_accuracy']==1,
        'placebo:uniform_head':bool(np.allclose(softmax(np.zeros((2,3))),1/3)),
        'positive:finite_difference':bool(np.max(abs(fd-g))<1e-8),
        'positive:unseen_label':bool(abs(s['loss']-np.log(33*(UNIVERSE-2)))<1e-12),
        'positive:corruption_rejected':corruption,'positive:hidden_field_rejected':hidden,
        'positive:undo':execute((0,1,0),(1,0,1),5,1,0)==(1,0,1)}


def populations(root,cfg,pulse):
    training={}; references={}; public={}; count=0; max_error=0.
    makers=tuple(product(range(2),repeat=4))
    for split,ls in [('train',cfg['train_lineages']),('development',cfg['development_lineages'])]:
        for lineage in ls:
            pulse(phase='independent-joint-population',split=split,lineage=lineage)
            rr=json.loads(gzip.decompress((root/'raw'/f'{split}-{lineage}_points.json.gz').read_bytes()))
            if len(rr)!=13824: raise ValueError('incomplete native population')
            random=rng('v19-local-world',lineage); strength=random.uniform(.8,1.2); rate=random.uniform(.65,.85); routine_strength=random.uniform(.1,.3)
            seen=defaultdict(set); mass=defaultdict(float); old=np.zeros((16,4,8)); groups={tier:defaultdict(lambda:defaultdict(float)) for tier in cfg['tiers']}
            for r in rr:
                ci,mi=r['context_index'],r['maker_index']; purpose,skill,belief,routine=makers[mi]
                initial=((0,1,0),(1,0,1))[ci//2]; request=ci%2
                assert r['maker']==list(makers[mi]) and r['initial']==list(initial) and r['requested_purpose']==request
                art=previous=initial; probability=1/64; seq=[]
                for i,e in enumerate(r['steps']):
                    weights=np.array([1.5 if purpose==0 else .6,1.4 if art[0]^belief!=art[1] else .5,1.8 if purpose==1 else .5])
                    weights[2 if routine else 0]+=routine_strength;weights[2 if request else 0]+=.2;weights=weights**strength
                    gi=GOALS.index(e['goal']); op=OPS.index(e['operation']); action=(3 if skill else 0,1,2)[gi]; alternative=5 if gi==1 and i==2 else 4
                    assert op in (action,alternative)
                    probability*=weights[gi]/weights.sum()*(rate if op==action else 1-rate)
                    after=execute(art,previous,op,skill,belief)
                    assert e==dict(step=i,goal=GOALS[gi],operation=OPS[op],before=list(art),after=list(after),undo_buffer=list(previous),dependency_edges=[[0,1]],tool_proposal=list(after) if op==3 else None,perceived_claim=art[0]^belief)
                    previous,art=art,after;seq.append(2*gi+int(op==alternative))
                assert len(seq)==3 and list(art)==r['final'] and tuple(seq) not in seen[ci,mi]
                seen[ci,mi].add(tuple(seq)); mass[ci,mi]+=probability;max_error=max(max_error,close(probability,r['probability'],1e-14))
                old[mi,ci,art[0]+2*art[1]+4*art[2]]+=64*probability
                if split=='development':
                    for tier in cfg['tiers']:
                        p=packet(r,tier); key=digest(p);public[key]=p;groups[tier][key][label(r)]+=probability
                count+=1
            assert len(seen)==64 and all(len(v)==216 for v in seen.values());close(list(mass.values()),np.full(64,1/64),1e-12)
            close(old.sum(2),np.ones((16,4)),1e-12)
            if split=='train': training[lineage]=(rr,old.reshape(16,32))
            else:
                for tier,gr in groups.items(): references[lineage,tier]=gr
    assert public==read(root/'reader/PACKETS.json')['packets']
    saved=read(root/'evaluator/REFERENCES.json'); assert len(saved)==len(references)
    for r in saved:
        gr=references[r['lineage'],r['tier']];assert [f['frame'] for f in r['frames']]==sorted(gr)
        for f in r['frames']:
            joint=gr[f['frame']]; mass=sum(joint.values());close(mass,f['mass']);close([[k,v/mass] for k,v in sorted(joint.items())],f['target'])
    return training,references,public,count,max_error


def interval(v):
    v=np.asarray(v); r=np.random.default_rng(190501); b=v[r.integers(len(v),size=(10000,len(v)))].mean(1)
    return dict(mean=float(v.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),lineage_values=v.tolist())


def run(out,plan,pulse):
    from contextlib import redirect_stdout,redirect_stderr
    import pytest
    pulse(phase='independent-joint-known-answer-tests')
    with (out/'review-tests.log').open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
        test_code=int(pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(out/'test-temp'),
            'tests/test_v19_joint_review.py','tests/test_v19_replay_dispatch.py']))
    write(out/'TESTS.json',dict(exit_code=test_code,log_sha256=file_digest(out/'review-tests.log')))
    if test_code: raise ValueError('independent tests failed before scientific input access')
    checks={k:bool(v) for k,v in controls().items()};write(out/'CONTROLS.json',checks)
    if not all(checks.values()): raise ValueError('independent controls failed before scientific inputs')
    for n,h in plan['design']['input_files'].items(): assert file_digest(out/'inputs'/n)==h,n
    root=out/'inputs/original'; target=read(root/'PLAN.json'); done=read(root/'COMPLETE.json'); cfg=target['design']
    assert file_digest(root/'PLAN.json')==done['plan_sha256']==plan['design']['target_plan_sha256']
    assert file_digest(root/'COMPLETE.json')==plan['design']['target_complete_sha256']
    assert cfg['train_lineages']==list(range(193000,193032)) and cfg['development_lineages']==list(range(190000,190016))
    assert cfg['fit_seeds']==list(range(190101,190106)) and cfg['training_draws']==[190201,190202] and cfg['budgets']==[32,128,512,2048]
    for n,h in {**done['files'],**done.get('execution_measurements',{})}.items():assert file_digest(root/n)==h,n
    tr,refs,public,path_count,prob_error=populations(root,cfg,pulse)
    summary=read(root/'SUMMARY.json');raw=json.loads(gzip.decompress((root/'raw/joint-readout_points.json.gz').read_bytes()))
    fields=('draw','tier','seed','budget','arm','lineage'); saved_rows={tuple(r[k] for k in fields):r for r in raw}
    assert len(saved_rows)==len(raw)==summary['rows'];fit_fields=fields[:-1];fits={tuple(r[k] for k in fit_fields):r for r in summary['fits']}
    rows=[];gradients=[];forecast_error=score_error=0.
    for draw in cfg['training_draws']:
        sel=read(root/'evaluator'/f'selection-{draw}.json');indices={}
        for l,(rr,old) in tr.items():
            weights=np.array([r['probability'] for r in rr]); weights/=weights.sum()
            indices[l]=rng('v19-joint-readout-training',draw,l).choice(len(rr),size=64,p=weights).tolist()
            assert indices[l]==sel['indices'][str(l)]
        selected=[(l,tr[l][0][indices[l][i]]) for i in range(64) for l in cfg['train_lineages']]
        assert sel['identities']==[[l,r['context_index'],r['maker_index']] for l,r in selected]
        labels=np.array([label(r) for l,r in selected]);old=np.array([tr[l][1][r['maker_index']] for l,r in selected])
        for tier in cfg['tiers']:
            x=np.array([features(packet(r,tier)) for l,r in selected],float);keys=sorted(k for k,p in public.items() if p['tier']==tier)
            xt=np.array([features(public[k]) for k in keys],float);pos={k:i for i,k in enumerate(keys)}
            data=arrays(root/'training'/f'{draw}-{tier}.npz');close(x,data['x'],0);close(labels,data['joint_label'],0);close(old,data['old'])
            for seed in cfg['fit_seeds']:
                pulse(phase='independent-joint-features',draw=draw,tier=tier,seed=seed)
                stem=f'{draw}-{tier}-{seed}';acq=arrays(root/'models'/f'{stem}-old.npz')
                h=basis(x,seed,(acq['history_weights'],acq['history_bias']));ht=basis(xt,seed)
                close(ridge(h,old),acq['old_head'],1e-6);ow=acq['old_head'];u,s,v=np.linalg.svd(ow,full_matrices=False)
                close(u,acq['latent_basis'],1e-8)
                pairs={'raw-history':(x,xt),'frozen-latent':(h@u,ht@u),'learned-bank':(bank(h@ow),bank(ht@ow))}
                for budget in cfg['budgets']:
                    alphabet=np.unique(labels[:budget]);y=np.searchsorted(alphabet,labels[:budget]);close(alphabet,read(root/'models'/f'{stem}-{budget}-alphabet.json'),0)
                    predictions={}
                    for arm,(xx,tt) in pairs.items():
                        pulse(phase='independent-joint-head',draw=draw,tier=tier,seed=seed,budget=budget,arm=arm)
                        model=arrays(root/'models'/f'{stem}-{budget}-{arm}.npz');f=basis(xx,seed+17,(model['readout_weights'],model['readout_bias']));ft=basis(tt,seed+17)
                        w=model['target_head'];assert w.shape==(129,len(alphabet))
                        obj,grad=objective(f[:budget],y,w);receipt=fits[draw,tier,seed,budget,arm]
                        close([obj,float(abs(grad).max())],[receipt['objective'],receipt['maximum_gradient']],1e-8)
                        assert receipt['head_parameters']==w.size and receipt['classes']==len(alphabet) and receipt['input_width']==xx.shape[1]
                        assert receipt['gradient_tolerance_met']==bool(abs(grad).max()<=1e-7) and receipt['iterations']<=200
                        gradients.append(dict(draw=draw,tier=tier,seed=seed,budget=budget,arm=arm,maximum_gradient=float(abs(grad).max()),gradient_tolerance_met=receipt['gradient_tolerance_met'],solver_success=receipt['success']))
                        predictions[arm]=softmax(ft@w)*budget/(budget+1)
                    counts=np.bincount(y,minlength=len(alphabet))+1
                    predictions['matched-frequency']=np.tile(counts/counts.sum()*budget/(budget+1),(len(keys),1))
                    for arm,pred in predictions.items():
                        sf=arrays(root/'forecasts'/f'{stem}-{budget}-{arm}.npz');close(sf['alphabet'],alphabet,0)
                        assert read(root/'forecasts'/f'{stem}-{budget}-{arm}-frames.json')==keys
                        forecast_error=max(forecast_error,close(pred,sf['probabilities']))
                        close(pred.sum(1),np.full(len(pred),budget/(budget+1)))
                        # Adjudicate discrete ties on the actual saved forecast after
                        # independent numerical reconstruction, not rounded recomputation.
                        pred=sf['probabilities']
                        for l in cfg['development_lineages']:
                            gr=refs[l,tier];sums=defaultdict(float);total=0.
                            for key,joint in gr.items():
                                mass=sum(joint.values());truth={k:v/mass for k,v in joint.items()};metrics=score(pred[pos[key]],alphabet,truth,budget);total+=mass
                                for k,v in metrics.items():sums[k]+=mass*v
                            close(total,1.)
                            row=dict(draw=draw,tier=tier,seed=seed,budget=budget,arm=arm,lineage=l,frames=len(gr),**{k:float(v/total) for k,v in sums.items()})
                            original=saved_rows[tuple(row[k] for k in fields)]
                            score_error=max(score_error,close([row[k] for k in sums],[original[k] for k in sums],1e-7));rows.append(row)
    assert len(rows)==len(raw);lookup={tuple(r[k] for k in fields):r for r in rows}
    for r in summary['cells']:
        rebuilt=lookup[tuple(r[k] for k in fields)];close([v for k,v in r.items() if k not in fields],[rebuilt[k] for k in r if k not in fields],1e-7)
    for r in summary['exact_reference']:
        entropy=-sum(w*np.log(w/sum(joint.values())) for joint in refs[r['lineage'],r['tier']].values() for w in joint.values())
        close(entropy,r['loss'])
    contrasts=[];areas=[]
    for tier,baseline in product(cfg['tiers'],('raw-history','frozen-latent')):
        curves=[]
        for budget in cfg['budgets']:
            d=np.array([[[lookup[draw,tier,seed,budget,'learned-bank',l]['loss']-lookup[draw,tier,seed,budget,baseline,l]['loss'] for l in cfg['development_lineages']] for seed in cfg['fit_seeds']] for draw in cfg['training_draws']])
            values=d.mean((0,1));curves.append(values)
            contrasts.append(dict(tier=tier,baseline=baseline,budget=budget,**interval(values),draw_means=d.mean((1,2)).tolist(),feature_seed_means=d.mean((0,2)).tolist()))
        areas.append(dict(tier=tier,baseline=baseline,**interval(np.array([1/6,1/3,1/3,1/6])@np.array(curves))))
    for new,old in [(contrasts,summary['contrasts']),(areas,summary['normalized_log_budget_area'])]:
        assert len(new)==len(old)
        for a,b in zip(new,old):
            for k in a:
                if k in ('tier','baseline','budget'):assert a[k]==b[k]
                else:close(a[k],b[k],1e-7)
    write(out/'INDEPENDENT_REGROUP.json',dict(contrasts=contrasts,normalized_log_budget_area=areas,gradient_receipts=gradients))
    (out/'review_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    checks['positive:complete_reconstruction']=True
    return dict(controls=checks,paths=path_count,rows=len(rows),target_heads=len(gradients),max_probability_error=prob_error,max_forecast_error=forecast_error,max_score_error=score_error,
        target_plan_sha256=plan['design']['target_plan_sha256'],scope='independent native path/probability, label, feature, saved-fit gradient, forecast, process-score and paired-area reconstruction; solver trajectory checked by separate full replays; event adjudication pending')
