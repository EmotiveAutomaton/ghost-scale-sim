"""B05 supplied-law storage ruler with actual serialized byte ceilings.

Witnessed order/inspection are independent of endpoint emissions. Retaining them
is historical memory; a sufficient predictive state intentionally discards them.
Codecs, shared schema and supplied channel are explicit diagnostic assumptions.
"""
import hashlib,struct,time
import numpy as np
from . import world as W
from .drift import channels
from .metrics import check_scores

READERS=('raw-history','maker-posterior','predictive-head','episode-ledger','predictor-plus-ledger')
MAGIC={name:i for i,name in enumerate(READERS)}

def posterior(records,emission):
    lp=np.full(8,-np.log(8))
    for row in records:lp+=np.log(emission[:,2*int(row[1])+int(row[2])])
    p=np.exp(lp-lp.max());return p/p.sum()

def encode(kind,records,emission,budget,case_id):
    if budget<80:raise ValueError('declared B05 budgets start at 256')
    body=b''
    if kind in ('maker-posterior','predictor-plus-ledger'):body=posterior(records,emission).astype('<f8').tobytes()
    if kind=='predictive-head':body=(posterior(records,emission)@emission).astype('<f8').tobytes()
    size=24 if kind=='raw-history' else 8
    count=min(len(records),(budget-8-len(body))//size) if kind in ('raw-history','episode-ledger','predictor-plus-ledger') else 0
    selected=records[-count:] if count else records[:0]
    for row in selected:
        index,x,y,order,inspection=map(int,row)
        if kind=='raw-history':body+=hashlib.sha256(f'{case_id}:{index}'.encode()).digest()[:16]
        body+=struct.pack('<IBBBB',index,x,y,order,inspection)
    result=struct.pack('<4sBBH',b'GSM1',MAGIC[kind],0,count)+body
    assert len(result)<=budget
    return result

def decode(blob,emission,case_id):
    magic,tag,reserved,count=struct.unpack('<4sBBH',blob[:8]);assert magic==b'GSM1' and reserved==0
    kind=READERS[tag];offset=8;belief=None;prediction=None
    if kind in ('maker-posterior','predictor-plus-ledger'):belief=np.frombuffer(blob[offset:offset+64],dtype='<f8').copy();offset+=64
    if kind=='predictive-head':prediction=np.frombuffer(blob[offset:offset+32],dtype='<f8').copy();offset+=32
    records=[]
    for _ in range(count):
        identity=None
        if kind=='raw-history':identity=blob[offset:offset+16];offset+=16
        row=struct.unpack('<IBBBB',blob[offset:offset+8]);offset+=8
        if identity is not None:assert identity==hashlib.sha256(f'{case_id}:{row[0]}'.encode()).digest()[:16]
        records.append(row)
    assert offset==len(blob)
    if prediction is None:
        if belief is None:belief=posterior(records,emission)
        prediction=belief@emission
    assert np.isfinite(prediction).all() and np.all(prediction>=0) and abs(prediction.sum()-1)<1e-12
    return prediction,{int(r[0]):(int(r[3]),int(r[4])) for r in records}

def retain(root,name,p,truth,**extra):
    with np.errstate(divide='ignore'):lp=np.log(p)
    q=p[np.arange(len(truth)),truth];mode=p.argmax(1)
    a=dict(log_loss=-lp[np.arange(len(truth)),truth],capped_log_loss=-np.log(np.maximum(q,1e-12)),brier=(p*p).sum(1)-2*q+1,correct=(mode==truth).astype(float),confident_error=((p.max(1)>=.9)&(mode!=truth)).astype(float))
    check_scores(lp,truth,a);(root/'raw').mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/'raw'/f'{name}_points.npz',logp=lp,truth=truth,**a,**extra)
    return {k:float(v.mean()) for k,v in a.items()}

def controls():
    emission=np.full((8,4),.25);records=np.array([[i,0,1,i%2,(i//2)%2] for i in range(16)])
    p=posterior(records,emission);blob=encode('episode-ledger',records,emission,256,'fixture');q,ledger=decode(blob,emission,'fixture')
    twin=records.copy();twin[:,3:]=1-twin[:,3:]
    return dict(placebo_uninformative_maker=bool(np.allclose(p,1/8)),positive_witness_retrieval=bool(ledger[3]==tuple(records[3,3:])),live_predictive_alias=bool(encode('maker-posterior',records,emission,256,'fixture')==encode('maker-posterior',twin,emission,256,'fixture')),positive_ledger_distinguishes_alias=bool(blob!=encode('episode-ledger',twin,emission,256,'fixture')))

def run(root,d,pulse):
    from .memory_check import reference,maker_reference
    population=[];timings=[];maxerr=0.;alias_checks=0;byte_rows=[]
    for world in d['lineages']:
        start=time.process_time();emission=channels(world)['emission'];rng=np.random.default_rng(W.seed('v21-b05-streams',world))
        futures={};histories={};future_truth=[];history_truth=[];order_truth=[];inspection_truth=[]
        for stream in range(d['streams']):
            state=int(rng.integers(8));n=d['history_episodes'];obs=rng.choice(4,size=n+d['future_cases'],p=emission[state]);witnesses=rng.integers(2,size=(n,2))
            records=np.column_stack([np.arange(n),obs[:n]//2,obs[:n]%2,witnesses]);queries=rng.integers(n,size=d['history_queries'])
            future_truth.extend(obs[n:]);history_truth.extend(2*witnesses[queries,0]+witnesses[queries,1]);order_truth.extend(witnesses[queries,0]);inspection_truth.extend(witnesses[queries,1])
            case=f"case-{W.seed('v21-b05-blind-id',world,stream):016x}"
            (root/'reader').mkdir(parents=True,exist_ok=True);(root/'evaluator').mkdir(exist_ok=True)
            np.savez_compressed(root/'reader'/f'{case}_points.npz',records=records,queries=queries)
            np.savez_compressed(root/'evaluator'/f'{case}_points.npz',maker_state=state,future=obs[n:],history_truth=2*witnesses[queries,0]+witnesses[queries,1],world=world,stream=stream)
            twin=records.copy();twin[:,3:]=1-twin[:,3:]
            for budget in d['byte_budgets']:
              for kind in READERS:
                blob=encode(kind,records,emission,budget,case);q,ledger=decode(blob,emission,case)
                if kind in ('maker-posterior','predictive-head','predictor-plus-ledger'):
                    full=maker_reference(records,emission);assert np.allclose(full,posterior(records,emission),atol=1e-12)
                    assert np.allclose(q,full@emission,atol=1e-12)
                ref,refhistory,parsed=reference(blob,emission,queries,case)
                maxerr=max(maxerr,float(np.max(np.abs(ref-q))));assert np.allclose(ref,q,atol=1e-12)
                historical=np.full((len(queries),4),.25)
                for i,index in enumerate(queries):
                    if int(index) in ledger:
                        order,inspection=ledger[int(index)];historical[i]=0;historical[i,2*order+inspection]=1
                assert np.array_equal(historical,refhistory) and len(parsed)==len(ledger)
                if kind in ('maker-posterior','predictive-head'):assert blob==encode(kind,twin,emission,budget,case);alias_checks+=1
                memory=root/'memory'/f'{case}-{budget}-{kind}.bin';memory.parent.mkdir(exist_ok=True);memory.write_bytes(blob)
                key=(budget,kind);futures.setdefault(key,[]).extend([q]*d['future_cases']);histories.setdefault(key,[]).extend(historical)
                byte_rows.append(dict(world=world,stream=stream,budget=budget,reader=kind,serialized_bytes=len(blob),retained_episodes=len(ledger),witness_query_coverage=float(np.mean([int(index) in ledger for index in queries]))))
            pulse(phase='memory-consumer',lineage=world,stream=stream)
        for budget in d['byte_budgets']:
          for kind in READERS:
            key=(budget,kind);future=retain(root,f'{world}-{budget}-{kind}-future',np.array(futures[key]),np.array(future_truth))
            hp=np.array(histories[key]);order_p=hp[:,2:].sum(1);inspection_p=hp[:,[1,3]].sum(1)
            historical=retain(root,f'{world}-{budget}-{kind}-history',hp,np.array(history_truth),order_correct=((order_p>=.5)==order_truth).astype(float),inspection_correct=((inspection_p>=.5)==inspection_truth).astype(float))
            rows=[r for r in byte_rows if (r['world'],r['budget'],r['reader'])==(world,budget,kind)]
            population.append(dict(lineage=world,budget=budget,reader=kind,future=future,historical=historical,order_accuracy=float(((order_p>=.5)==order_truth).mean()),inspection_accuracy=float(((inspection_p>=.5)==inspection_truth).mean()),mean_bytes=float(np.mean([r['serialized_bytes'] for r in rows])),max_bytes=max(r['serialized_bytes'] for r in rows),mean_retained_episodes=float(np.mean([r['retained_episodes'] for r in rows])),witness_query_coverage=float(np.mean([r['witness_query_coverage'] for r in rows]))))
        timings.append(dict(lineage=world,total_cpu_seconds=time.process_time()-start))
    return dict(card='G21-B05',population=population,byte_rows=byte_rows,timings=timings,controls=controls(),independent_max_probability_error=maxerr,predictive_alias_checks=alias_checks,scope='Native static supplied-law predictive compression and exact historical witness ruler; no learned architecture superiority.',limitations=['The supplied eight-state endpoint channel and common codec/schema are shared external information, not charged to per-maker memory. Storage is actual per-maker serialized payload, not peak process RAM or full application cost.','Raw entries use 16-byte source digest plus index and four observed fields; compact ledger uses index plus fields and reconstructs source link by the declared hash. This format comparison is not an optimal compression lower bound.','Sufficient posterior and four-probability predictive head consume all permitted history once; raw/ledger inference sees only retained recent episodes. Hybrid stores full-history posterior plus as many recent ledger entries as fit.','Every historical order/inspection witness is observed in the input, but independent of endpoint emissions by construction. Predictive twins differ only in these witnesses; their identical predictive memory is an exact finite counterexample, not a human memory conclusion.','No execution shift is included; native-to-shift contrasts are unavailable for every reader. No fitted seeds, original-world headroom, F03 category or confirmation.','Distinct episodes are used once; no copied-evidence product reader exists in this miniature. Learned heads, dynamic maker changes and query-aware retention remain conditional.','Miniature — architecture untested.'])
