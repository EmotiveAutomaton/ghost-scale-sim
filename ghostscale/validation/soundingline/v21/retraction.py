"""D05 source-keyed incremental factors versus independent raw recomputation."""
import time,json,hashlib,types,sys,importlib
from pathlib import Path
import numpy as np
from . import world as W
from .routes import retain
from ..v18_3.io import write,file_digest,digest

READERS=('frozen','append-only','raw-recompute','incremental-ledger')
UPDATES=('true-correction','false-correction','retraction','irrelevant-addition','unchanged-repeat')

class Ledger:
    """Delete old source factors with zero counters; never divide by zero."""
    def __init__(self,prior,records):
        self.prior=np.array(prior,float);self.logs=np.zeros(2);self.zeros=np.zeros(2,dtype=int);self.records={};self.factor_visits=0
        for source,(value,reliability) in records.items():self.update((source,value,reliability))
    def factor(self,value,reliability):
        return np.ones(2) if value<0 else np.array([reliability if t==value else 1-reliability for t in (0,1)])
    def change(self,record,sign):
        p=self.factor(*record);positive=p>0;self.zeros+=sign*(~positive);self.logs[positive]+=sign*np.log(p[positive]);self.factor_visits+=1
    def update(self,update):
        source,value,reliability=update;old=self.records.get(source);new=None if value is None else (value,reliability)
        if old==new:return
        if old is not None:self.change(old,-1);del self.records[source]
        if new is not None:self.change(new,1);self.records[source]=new
    def posterior(self):
        lp=np.log(self.prior)+self.logs;lp[self.zeros>0]=-np.inf
        if not np.isfinite(lp).any():return self.prior.copy(),True
        p=np.exp(lp-np.max(lp));return p/p.sum(),False
    def payload(self):
        return dict(prior=self.prior.tolist(),records=self.records,log_sum=self.logs.tolist(),zero_counts=self.zeros.tolist())

def bytes_used(value):return len(json.dumps(value,sort_keys=True,separators=(',',':')).encode())

def controls():
    from .retraction_check import posterior,updated
    prior=[.4,.6];records={0:(0,1.),1:(1,1.)};a=Ledger(prior,records);_,conflict=a.posterior();a.update((0,None,1.));p,after=a.posterior()
    b=Ledger(prior,{0:(1,.8)});original=b.posterior()[0];b.update((0,1,.8));repeat=np.array_equal(original,b.posterior()[0]);b.update((5,-1,.8));irrelevant=np.array_equal(original,b.posterior()[0])
    records={i:(i%2,.8) for i in range(16)};a=Ledger(prior,records);a.update((0,1,.8));reference,_=posterior(prior,updated(records,(0,1,.8)))
    return dict(live_conflict_detected=bool(conflict),positive_retraction_restores_support=bool(not after and np.array_equal(p,[0.,1.])),placebo_irrelevant=bool(irrelevant),positive_repeat_idempotent=bool(repeat),positive_independent_raw=bool(np.max(abs(a.posterior()[0]-reference))<1e-12))

def fixtures(root,sl):
    from ..v20.integration import load
    from .retraction_check import posterior,updated
    module=load('v21_retraction_process',sl/'soundingline/process_record.py');pkg=types.ModuleType('v21_retraction_stage13');pkg.__path__=[str(sl/'runners/stage13')];sys.modules[pkg.__name__]=pkg
    exporter=importlib.import_module(pkg.__name__+'.export_tompathy');common=importlib.import_module(pkg.__name__+'.common')
    names=('soundingline/process_record.py','runners/stage13/export_tompathy.py','runners/stage13/common.py','runners/stage13/scoring.py');before={n:file_digest(sl/n) for n in names};cases=[]
    for i in range(12):
        reliability=(.6,.8,1.)[i%3];records={0:(0,reliability),1:(1,reliability)};update=(0,None,reliability);identity=digest(['v21-d05-fixture',i])[:24]
        observed=dict(reports=[dict(source=k,value=v[0],reliability=v[1]) for k,v in records.items()],update=dict(source=0,action='retract'),target='latent binary state');text='Two source reports disagree; source zero is retracted.'
        case=module.ProcessCase(case_id=identity,lineage_id=digest(['v21-d05-group',i//3])[:24],domain='constructed',medium='source-linked-reports',brief_id='synthetic',declared_context=observed,participants={'human':'constructed'},route_family='context-retraction',events=[],artifact_final=text);case.validate()
        art=dict(id=identity,revision=hashlib.sha256(text.encode()).hexdigest(),regions=[dict(id='region-0',anchor=dict(exact=text,index=0,start=0,prefix='',suffix=''))]);reading={'region-0':dict(visibility='I2',goals=[],processes=[dict(label='Revised evidence set',probability=.5)])}
        excerpt=json.dumps(observed,sort_keys=True,separators=(',',':'));source='Ghost V21 constructed source updates';ctx=[dict(id=common.digest([source,excerpt]),source=source,excerpt=excerpt,selected=True)]
        exported,sidecar=exporter.export(art,reading,context=ctx,outer=dict(case_id=identity,scope='constructed reports and retraction only; evaluator separate',observed_events=[],unknown_review_scope=True,governing_purpose='retract source evidence without inventing new testimony'))
        base=root/'reader'/identity
        for name,value in [('observed.json',observed),('offline-packet.json',exported),('sidecar.json',sidecar),('candidates.json',dict(choices=[0,1]))]:write(base/name,value)
        write(root/'evaluator'/f'{identity}.json',dict(actual_state=i%2,prior=[.5,.5],before=posterior([.5,.5],records),after=posterior([.5,.5],updated(records,update)),scope='evaluator-only; placeholder reading is not inference'))
        cases.append(dict(case_id=identity,canonical_process_case=True,offline_export=True))
    assert before=={n:file_digest(sl/n) for n in names};write(root/'CONSUMER.json',dict(passed=True,sources=before,cases=cases,coverage='12 retraction fixtures; repeated reliability controls, not 12 independent scientific worlds',browser_tested=False));return cases

def run(root,d,pulse):
    from .retraction_check import posterior,updated
    rows=[];timings=[];maxerr=0.;(root/'reader').mkdir(parents=True,exist_ok=True);(root/'evaluator').mkdir(exist_ok=True)
    for world in d['lineages']:
        rng=np.random.default_rng(W.seed('v21-d05-law',world));prior=np.array([rng.uniform(.2,.8),0]);prior[1]=1-prior[0]
        rng=np.random.default_rng(W.seed('v21-d05-cases',world));truth=rng.choice(2,size=d['test_cases'],p=prior);noise=rng.random((d['test_cases'],max(d['history_lengths'])))
        identity=f"case-{W.seed('v21-d05-blind-id',world):016x}";np.savez_compressed(root/'evaluator'/f'{identity}_points.npz',world=world,truth=truth,prior=prior,noise=noise)
        for h in d['history_lengths']:
          for reliability in d['reliabilities']:
            base=np.where(noise[:,:h]<reliability,truth[:,None],1-truth[:,None])
            for update_index,kind in enumerate(UPDATES):
                vals=base.copy()
                if kind in ('true-correction','retraction'):vals[:,0]=1-truth
                elif kind=='false-correction':vals[:,0]=truth
                updates=[(0,int(y),reliability) if kind=='true-correction' else (0,int(1-y),reliability) if kind=='false-correction' else (0,None,reliability) if kind=='retraction' else (h,-1,reliability) if kind=='irrelevant-addition' else (0,int(row[0]),reliability) for y,row in zip(truth,vals)]
                # Opaque update index is never an intervention label in the reader payload.
                blind=f'{identity}-{h}-{int(reliability*100)}-{W.seed("v21-d05-view",world,h,reliability,update_index):016x}'
                write(root/'reader'/f'{blind}.json',dict(reports=[[[int(v),reliability] for v in row] for row in vals],updates=updates))
                forecasts={r:[] for r in READERS};conflicts={r:[] for r in READERS};sizes={r:[] for r in READERS};visits={r:[] for r in READERS};clock={r:dict(initial_cpu_seconds=0.,update_cpu_seconds=0.) for r in READERS}
                for row,update in zip(vals,updates):
                    records={i:(int(v),reliability) for i,v in enumerate(row)}
                    t=time.process_time();initial,initial_conflict=posterior(prior,records);clock['frozen']['initial_cpu_seconds']+=time.process_time()-t
                    forecasts['frozen'].append(initial);conflicts['frozen'].append(initial_conflict);sizes['frozen'].append(bytes_used(dict(prior=prior.tolist(),posterior=initial)));visits['frozen'].append(0)
                    t=time.process_time();append=Ledger(prior,records);clock['append-only']['initial_cpu_seconds']+=time.process_time()-t
                    t=time.process_time()
                    if update[1] is not None:append.update((h+1,update[1],update[2]))
                    q,conflict=append.posterior();clock['append-only']['update_cpu_seconds']+=time.process_time()-t
                    forecasts['append-only'].append(q);conflicts['append-only'].append(conflict);sizes['append-only'].append(bytes_used(append.payload()));visits['append-only'].append(int(update[1] is not None))
                    t=time.process_time();raw=dict(records);clock['raw-recompute']['initial_cpu_seconds']+=time.process_time()-t
                    t=time.process_time();raw=updated(raw,update);q,conflict=posterior(prior,raw);clock['raw-recompute']['update_cpu_seconds']+=time.process_time()-t
                    forecasts['raw-recompute'].append(q);conflicts['raw-recompute'].append(conflict);sizes['raw-recompute'].append(bytes_used(dict(prior=prior.tolist(),records=raw)));visits['raw-recompute'].append(len(raw))
                    t=time.process_time();ledger=Ledger(prior,records);clock['incremental-ledger']['initial_cpu_seconds']+=time.process_time()-t;before=ledger.factor_visits
                    t=time.process_time();ledger.update(update);q,conflict=ledger.posterior();clock['incremental-ledger']['update_cpu_seconds']+=time.process_time()-t
                    forecasts['incremental-ledger'].append(q);conflicts['incremental-ledger'].append(conflict);sizes['incremental-ledger'].append(bytes_used(ledger.payload()));visits['incremental-ledger'].append(ledger.factor_visits-before)
                    err=float(np.max(abs(q-forecasts['raw-recompute'][-1])));maxerr=max(maxerr,err);assert err<1e-12 and conflict==conflicts['raw-recompute'][-1]
                    if kind in ('irrelevant-addition','unchanged-repeat'):assert np.max(abs(q-initial))<1e-12
                for reader in READERS:
                    name=f'{world}-{h}-{int(reliability*100)}-{kind}-{reader}';metrics=retain(root,name,np.array(forecasts[reader]),truth,conflict=np.array(conflicts[reader]),payload_bytes=np.array(sizes[reader]),factor_visits=np.array(visits[reader]))
                    rows.append(dict(lineage=world,history=h,reliability=reliability,update=kind,reader=reader,metrics=metrics,conflict_fraction=float(np.mean(conflicts[reader])),mean_payload_bytes=float(np.mean(sizes[reader])),mean_factor_visits=float(np.mean(visits[reader]))))
                    timings.append(dict(lineage=world,history=h,reliability=reliability,update=kind,reader=reader,**clock[reader]))
                pulse(phase='retraction-consumer',lineage=world,history=h,reliability=reliability,update=kind)
    cases=fixtures(root/'fixtures',Path(d['soundingline'])) if d.get('export_fixtures') else []
    return dict(card='G21-D05',population=rows,timings=timings,controls=controls(),independent_max_probability_error=maxerr,fixtures=cases,limitations=['Binary static latent-state and reliability/prior supplied; no original-world, learned or unknown-law superiority.','True correction/retraction deliberately corrupt source zero first; false correction starts with a true source zero. Conditions have different starting evidence.','Reliability one can conflict under deliberate corruption; posterior is undefined there and the explicit fallback is the supplied prior with a conflict flag.','Append-only repeats add evidence; retractions are ignored. This is a deliberately incapable comparator, not a modern revision algorithm.','Incremental log/zero counters and independent raw products are distinct; setup, update CPU and serialized payloads remain separate. One update per initialized history is measured.','Shared cases and nested histories are not independent worlds; native only, no product-of-factors ablation or confirmation.','Miniature — architecture untested.'])
