"""E04 executable-route versus historical-route supplied-law diagnostic."""
import time,hashlib,importlib,types,sys,json
from pathlib import Path
import numpy as np
from . import world as W
from .metrics import check_scores,mean_scores
from ..v18_3.io import write,file_digest,digest

READERS=('shortest','goal-plausible','historical','direct-enumeration')
WITNESSES=('none','inspection','complete')

def programs(goal):
    own='one' if goal else 'zero';other='zero' if goal else 'one'
    return [(own,),(other,'flip'),(own,'flip','flip'),(other,'inspect','flip')]

def execute(program):
    value=0
    for op in program:
        if op in ('zero','one'):value=int(op=='one')
        elif op=='flip':value^=1
        elif op!='inspect':raise ValueError(op)
    return value

def law(world):
    rng=np.random.default_rng(W.seed('v21-e04-law',world));p=np.empty((2,4,4))
    for goal in range(2):
        for maker in range(4):
            alpha=np.ones(4);alpha[(maker+goal)%4]=5;p[goal,maker]=rng.dirichlet(alpha)
    return p

def posterior(policy,history,goal,witness):
    lp=np.full(4,-np.log(4))
    for g,r in history:lp+=np.log(policy[int(g),:,int(r)])
    prior=np.exp(lp-lp.max());joint=prior[:,None]*policy[goal]
    allowed=np.ones(4,bool)
    if witness is not None:
        allowed=(np.arange(4)==3)==bool(witness[1]) if witness[0]=='inspection' else np.arange(4)==witness[1]
    joint*=allowed;current=joint.sum(0);current/=current.sum();maker=joint.sum(1);maker/=maker.sum()
    return current,maker@policy[1-goal]

def retain(root,name,p,truth,**extras):
    # Keep uncapped infinities in NPZ; JSON reports null plus their frequency.
    with np.errstate(divide='ignore'):lp=np.log(p)
    saved=np.exp(lp);q=saved[np.arange(len(truth)),truth];mode=saved.argmax(1)
    a=dict(log_loss=-lp[np.arange(len(truth)),truth],capped_log_loss=-np.log(np.maximum(q,1e-12)),brier=(saved*saved).sum(1)-2*q+1,correct=(mode==truth).astype(float),confident_error=((saved.max(1)>=.9)&(mode!=truth)).astype(float),zero_mass=(q==0).astype(float))
    check_scores(lp,truth,a);(root/'raw').mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/'raw'/f'{name}_points.npz',logp=lp,truth=truth,**a,**extras)
    return mean_scores(a)

def controls():
    p=np.full((2,4,4),.25);history=np.array([[0,3],[1,2]])
    a=posterior(p,history,0,None)[0];b=posterior(p,[],0,None)[0]
    from .routes_check import execute as scalar_execute,reference
    specific=law(71000790);q=posterior(specific,history,0,('complete',2))[0]
    return dict(live_all_routes_execute=all(execute(r)==g==scalar_execute(r) for g in (0,1) for r in programs(g)),placebo_history_uninformative=bool(np.array_equal(a,b)),positive_complete_witness=bool(q[2]==1),live_exact_alias=len(set(programs(0)))==4 and len({execute(r) for r in programs(0)})==1,positive_independent_posterior=bool(np.allclose(posterior(specific,history,0,None),reference(specific,history,0,None),atol=1e-12)))

def fixtures(root,sl):
    from ..v20.integration import load
    module=load('v21_routes_process',sl/'soundingline/process_record.py');pkg=types.ModuleType('v21_routes_stage13');pkg.__path__=[str(sl/'runners/stage13')];sys.modules[pkg.__name__]=pkg
    exporter=importlib.import_module(pkg.__name__+'.export_tompathy');common=importlib.import_module(pkg.__name__+'.common')
    names=('soundingline/process_record.py','runners/stage13/export_tompathy.py','runners/stage13/common.py','runners/stage13/scoring.py');before={n:file_digest(sl/n) for n in names};cases=[]
    for i in range(12):
        goal=i%2;actual=(i//2)%4;tier=WITNESSES[i%3];witness=None if tier=='none' else ('inspection',int(actual==3)) if tier=='inspection' else ('complete',actual)
        identity=digest(['v21-e04-fixture',i])[:24];lineage=digest(['v21-e04-fixture-group',i//3])[:24];text=f'Final binary artifact: {goal}'
        context=dict(artifact=goal,current_witness=witness,allowed_routes=[list(x) for x in programs(goal)],past_records=[])
        case=module.ProcessCase(case_id=identity,lineage_id=lineage,domain='constructed',medium='binary-program',brief_id='synthetic',declared_context=context,participants={'human':'constructed'},route_family='finite-executable-routes',events=[],artifact_final=text);case.validate()
        artifact=dict(id=identity,revision=hashlib.sha256(text.encode()).hexdigest(),regions=[dict(id='region-0',anchor=dict(exact=text,index=0,start=0,prefix='',suffix=''))])
        reading={'region-0':dict(visibility='I1' if tier=='none' else 'I2',goals=[],processes=[dict(label=f'Executable method {r}',probability=.25) for r in range(4)])}
        excerpt=json.dumps(context,sort_keys=True,separators=(',',':'));source='Ghost V21 constructed route observations';ctx=[dict(id=common.digest([source,excerpt]),source=source,excerpt=excerpt,selected=True)]
        outer=dict(case_id=identity,scope='constructed observations; actual route retained evaluator-only',observed_events=[],unknown_review_scope=True,governing_purpose='not inferred from reenactment')
        exported,sidecar=exporter.export(artifact,reading,context=ctx,outer=outer);base=root/'reader'/identity
        write(base/'observed.json',context);write(base/'offline-packet.json',exported);write(base/'sidecar.json',sidecar);write(base/'candidates.json',dict(target='actual executed method',routes=[list(x) for x in programs(goal)]))
        q,_=posterior(law(71000791),[],goal,witness);write(root/'evaluator'/f'{identity}.json',dict(actual_method=actual,actual_program=list(programs(goal)[actual]),supplied_law_posterior=q.tolist(),scope='supplied-law diagnostic; evaluator only; offline placeholder reading is uniform'))
        cases.append(dict(case_id=identity,witness=tier,canonical_process_case=True,offline_export=True))
    assert before=={n:file_digest(sl/n) for n in names};write(root/'CONSUMER.json',dict(passed=True,sources=before,cases=cases,coverage='12 executable-route alias/inspection fixtures; no omission/selection/retraction coverage',browser_tested=False));return cases

def run(root,d,pulse):
    from .routes_check import reference
    population=[];timings=[];maxerr=0.;prefix_checks=0
    for world in d['lineages']:
        start=time.process_time();policy=law(world);rng=np.random.default_rng(W.seed('v21-e04-cases',world));records=[];states=[];goals=[];truth=[];future_truth=[]
        for case in range(d['test_cases']):
            maker=int(rng.integers(4));past=[]
            for _ in range(max(d['histories'])):
                g=int(rng.integers(2));past.append([g,int(rng.choice(4,p=policy[g,maker]))])
            g=int(rng.integers(2));r=int(rng.choice(4,p=policy[g,maker]));f=int(rng.choice(4,p=policy[1-g,maker]))
            records.append(past);states.append(maker);goals.append(g);truth.append(r);future_truth.append(f)
        records=np.asarray(records);goals=np.asarray(goals);truth=np.asarray(truth);future_truth=np.asarray(future_truth)
        identity=f"case-{W.seed('v21-e04-blind-id',world):016x}";(root/'evaluator').mkdir(parents=True,exist_ok=True);(root/'reader').mkdir(exist_ok=True)
        np.savez_compressed(root/'evaluator'/f'{identity}_points.npz',world=world,maker=states,method=truth,future_method=future_truth)
        for history in d['histories']:
            for tier in WITNESSES:
                qs={r:[] for r in READERS};fs={r:[] for r in READERS};witnesses=np.full(len(truth),-1) if tier=='none' else (truth==3).astype(int) if tier=='inspection' else truth
                np.savez_compressed(root/'reader'/f'{identity}-{history}-{tier}_points.npz',artifact=goals,past_records=records[:,:history],witness=witnesses)
                for i,g in enumerate(goals):
                    witness=None if tier=='none' else (tier,int(witnesses[i]));past=records[i,:history];prefix_checks+=1
                    q,f=posterior(policy,past,int(g),witness);direct,df=reference(policy,past,int(g),witness);err=max(float(np.max(abs(q-direct))),float(np.max(abs(f-df))));maxerr=max(maxerr,err);assert err<1e-12
                    base,bf=posterior(policy,[],int(g),witness);short=int(np.flatnonzero(base>0)[0]);plausible=int(base.argmax())
                    for reader,qq,ff in [('shortest',np.eye(4)[short],bf),('goal-plausible',np.eye(4)[plausible],bf),('historical',q,f),('direct-enumeration',direct,df)]:qs[reader].append(qq);fs[reader].append(ff)
                for reader in READERS:
                    q=np.asarray(qs[reader]);f=np.asarray(fs[reader]);selected=q.argmax(1);success=np.array([execute(programs(int(g))[int(r)])==g for g,r in zip(goals,selected)]);lengths=np.array([len(programs(int(g))[int(r)]) for g,r in zip(goals,selected)])
                    metrics=retain(root,f'{world}-{history}-{tier}-{reader}-route',q,truth,production_success=success,selected_length=lengths);future=retain(root,f'{world}-{history}-{tier}-{reader}-future',f,future_truth)
                    population.append(dict(lineage=world,history=history,witness=tier,reader=reader,route=metrics,future=future,production_success=float(success.mean()),selected_length=float(lengths.mean())))
                pulse(phase='route-consumer',lineage=world,history=history,witness=tier)
        timings.append(dict(lineage=world,total_cpu_seconds=time.process_time()-start))
    cases=fixtures(root/'fixtures',Path(d['soundingline'])) if d.get('export_fixtures') else []
    return dict(card='G21-E04',population=population,timings=timings,controls=controls(),independent_max_probability_error=maxerr,prefix_checks=prefix_checks,fixtures=cases,scope='Native finite supplied-law route/history ruler; eight development worlds, no fits or confirmation.',limitations=['Exact route catalog and policy are supplied, not learned.','Shortest and goal-plausible ignore past history; their loss is an evidence restriction, not matched learned architecture superiority.','Direct enumeration shares all supplied information; exact agreement is a control.','Execution success, actual route correspondence and future method prediction are separate targets.','Deterministic reconstructor log loss can be infinite; raw arrays preserve it and JSON gives null plus infinite fraction.','No native-to-shift comparison or duplicated-product reader exists. Paired prefixes and repeated baselines are controls, not independent worlds.','Miniature — architecture untested.'])
