"""B06 finite supplied-law ruler for knowledge, skill and permission.

The matched direct table has the same complete distribution as exact inference.
Single-factor readouts are deliberately restricted posterior representations.
No learned architecture comparison or human trait interpretation is implied.
"""
import time
import numpy as np
from . import world as W
from .memory import retain

BITS=np.array([[(i>>j)&1 for j in range(3,-1,-1)] for i in range(16)])
READERS=('goal-only','skill-only','joint','direct-table')
REVEALS={'none':(), 'knowledge':(1,), 'skill':(2,), 'permission':(3,), 'all-factors':(1,2,3)}

def law(world):
    rng=np.random.default_rng(W.seed('v21-b06-law',world))
    return np.array([rng.uniform(.55,.65),rng.uniform(.85,.95)])

def likelihood(success):
    intended=BITS[:,0]*BITS[:,1]*BITS[:,3]
    py=np.where(intended,success[BITS[:,2]],1-success[BITS[:,2]])
    return np.column_stack([1-py,py])

def posterior(y,cues,success,reader):
    mass=likelihood(success)[:,y]/16
    for bit,value in cues.items():mass=np.where(BITS[:,bit]==value,mass,0)
    q=mass/mass.sum()
    if reader in ('joint','direct-table'):return q
    bit=0 if reader=='goal-only' else 2
    # Retain only the named factor's endpoint-conditioned marginal, plus exact
    # supplied observations; all other unavailable factors retain their prior.
    p=q[BITS[:,bit]==1].sum();result=np.ones(16)
    for j in range(4):
        prob=float(cues[j]) if j in cues else p if j==bit else .5
        result*=np.where(BITS[:,j],prob,1-prob)
    return result/result.sum()

def direct_table(success,reveal):
    # Enumerate the complete observed joint table before reading any test case.
    table={};channel=likelihood(success)
    for state in range(16):
      for y in (0,1):
        key=(y,*(int(BITS[state,b]) for b in reveal))
        table.setdefault(key,np.zeros(16))[state]+=channel[state,y]/16
    return {key:p/p.sum() for key,p in table.items()}

def controls():
    s=np.array([.6,.9]);l=likelihood(s);q=posterior(1,{},np.array([.5,.5]),'joint')
    # Knowledge and permission swaps have identical endpoint distributions.
    a=8+4; b=8+1
    return dict(placebo_uninformative_execution=bool(np.allclose(q,1/16)),
        live_knowledge_permission_alias=bool(np.array_equal(l[a],l[b])),
        positive_witness_breaks_alias=bool(posterior(1,{1:1},s,'joint')[b]==0 and posterior(1,{1:1},s,'joint')[a]>0),
        positive_complete_factor_support=bool(np.count_nonzero(posterior(1,{1:1,2:1,3:1},s,'joint'))==2))

def run(root,d,pulse):
    from .factors_check import reference,endpoint_reference
    population=[];timings=[];maxerr=0.;twins=[]
    for world in d['lineages']:
        start=time.process_time();success=law(world);channel=likelihood(success);rng=np.random.default_rng(W.seed('v21-b06-twins',world))
        # All 16 states share the same uniform execution draws within a twin
        # index. Goal-fixed changes of one other factor are exact interventions.
        coins=rng.random(d['twins']);truth=np.repeat(np.arange(16),d['twins']);u=np.tile(coins,16)
        intended=BITS[truth,0]*BITS[truth,1]*BITS[truth,3];worked=u<success[BITS[truth,2]];y=np.where(worked,intended,1-intended);twin_outcomes=y.copy()
        permutation=rng.permutation(len(truth));truth=truth[permutation];intended=intended[permutation];y=y[permutation]
        case=f"case-{W.seed('v21-b06-blind-id',world):016x}"
        (root/'reader').mkdir(parents=True,exist_ok=True);(root/'evaluator').mkdir(exist_ok=True)
        np.savez_compressed(root/'evaluator'/f'{case}_points.npz',world=world,truth=truth,execution_coins=coins,intended=intended,outcomes=y,permutation=permutation)
        for reveal,bits in REVEALS.items():
            observations=np.column_stack([y,*[BITS[truth,b] for b in bits]])
            np.savez_compressed(root/'reader'/f'{case}-{reveal}_points.npz',observations=observations)
            table=direct_table(success,bits)
            for reader in READERS:
                lookup={}
                for key in set(map(tuple,observations.tolist())):
                    cues=dict(zip(bits,key[1:]));q=table[key] if reader=='direct-table' else posterior(key[0],cues,success,reader)
                    ref=reference(key[0],cues,success,reader);err=float(np.max(abs(ref-q)));maxerr=max(maxerr,err);assert err<1e-12;lookup[key]=q
                p=np.array([lookup[tuple(row)] for row in observations]);metrics=retain(root,f'{world}-{reveal}-{reader}',p,truth)
                accuracy=[];bitloss=[]
                for bit in range(4):
                    pp=p[:,BITS[:,bit]==1].sum(1);actual=BITS[truth,bit];accuracy.append(float(((pp>=.5)==actual).mean()))
                    right=np.where(actual,p[:,BITS[:,bit]==1].sum(1),p[:,BITS[:,bit]==0].sum(1));bitloss.append(float(-np.log(right).mean()))
                future=p@channel
                # Expected fresh-outcome loss integrates the true endpoint law;
                # it is distinct from state/history accuracy and uses no labels.
                fl=float(np.mean(-np.sum(channel[truth]*np.log(future),axis=1)))
                for q in p[::max(1,len(p)//16)]:assert np.max(abs(endpoint_reference(q,success)-q@channel))<1e-12
                population.append(dict(lineage=world,reveal=reveal,reader=reader,metrics=metrics,factor_accuracy=accuracy,factor_log_loss=bitloss,expected_future_log_loss=fl))
            pulse(phase='factor-consumer',lineage=world,reveal=reveal)
        for state in range(16):
          for bit in (1,2,3):
            other=state^(1<<(3-bit))
            if state<other:
                assert BITS[state,0]==BITS[other,0]
                twins.append(dict(lineage=world,state=state,twin=other,changed_factor=bit,goal_fixed=True,outcome_disagreement=float(np.mean(twin_outcomes[state*d['twins']:(state+1)*d['twins']]!=twin_outcomes[other*d['twins']:(other+1)*d['twins']]))))
        timings.append(dict(lineage=world,total_cpu_seconds=time.process_time()-start))
    return dict(card='G21-B06',population=population,twins=twins,timings=timings,controls=controls(),independent_max_probability_error=maxerr,scope='Initial native supplied-law exact information and representation ruler; direct table has identical information, no fits.',limitations=['Goal, knowledge, skill and permission are binary constructed variables. Knowledge and permission enter intended action symmetrically and are exactly aliased without distinguishing observations.','All 16 counterfactual states and cases within a coefficient world are paired, not independent worlds. Exact factor reveals are privileged witnessed observations.','Goal-only and skill-only retain only their named posterior marginal and observed factors; discarded factors use their prior. Their limited output representation is not a learned model with matched capacity.','Direct table and joint inference share the full supplied distribution; equivalence tests representation, not learning or architectural superiority.','No original-world headroom gate, unknown-law integration, native-to-shift transfer, learned direct fitting, new F03 category or confirmation is inherited.','Distinct twin execution draws are shared counterfactual controls, not copied evidence multiplication.','Miniature — architecture untested.'])
