"""D04 supplied-law sequential order evidence with a complementary record pair."""
from functools import lru_cache
import time
import numpy as np
from . import world as W
from .acquisition import tables
from .studies import retain

POLICIES=('greedy-information','two-step-lookahead','fixed-pair-first','uncertainty-threshold')
STATES=np.array([(order,pad) for order in (0,1) for pad in (0,1)])
REL=.8
CHANNEL=np.empty((8,4,2))
for action in range(8):
    for state,(order,pad) in enumerate(STATES):
        if action<2:
            bit=(order^pad) if action==0 else pad
            CHANNEL[action,state]=[int(bit==b) for b in (0,1)]
        else:CHANNEL[action,state]=[REL if order==b else 1-REL for b in (0,1)]


def entropy(p):
    p=np.asarray(p);p=p[p>0];return float(-np.dot(p,np.log(p)))


class Planner:
    def __init__(self,order_prior):
        self.prior=np.repeat(np.asarray(order_prior)/2,2);self.evaluations=0

    @lru_cache(None)
    def posterior(self,history):
        self.evaluations+=1;p=self.prior.copy()
        for a,b in history:p*=CHANNEL[a,:,b]
        total=p.sum();assert total>0
        return p/total

    @lru_cache(None)
    def order(self,history):return self.posterior(history).reshape(2,2).sum(1)

    def transitions(self,history,action):
        p=self.posterior(history);out=[]
        for bit in (0,1):
            mass=float(p@CHANNEL[action,:,bit])
            if mass>1e-15:out.append((mass,tuple(sorted(history+((action,bit),)))))
        return out

    @lru_cache(None)
    def value(self,history,depth,fee):
        stop=entropy(self.order(history))
        if depth==0:return stop
        scores=[fee+sum(mass*self.value(h,depth-1,fee) for mass,h in self.transitions(history,a)) for a in range(8) if a not in dict(history)]
        return min([stop]+scores)

    def choice(self,history,remaining,fee,policy,threshold):
        available=[a for a in range(8) if a not in dict(history)]
        if not available or remaining==0:return -1
        if policy=='fixed-pair-first':return available[0]
        h=entropy(self.order(history))
        if policy=='uncertainty-threshold' and h<=threshold:return -1
        depth=min(2,remaining) if policy=='two-step-lookahead' else 1
        scores=[fee+sum(mass*self.value(hist,depth-1,fee) for mass,hist in self.transitions(history,a)) for a in available]
        best=min(scores)
        j=next(i for i,value in enumerate(scores) if abs(value-best)<1e-14)
        return available[j] if best<h-1e-12 else -1


def controls():
    p=Planner([.5,.5]);h=entropy(p.order(()))
    single=[h-sum(m*entropy(p.order(t)) for m,t in p.transitions((),a)) for a in (0,1)]
    pair=p.order(((0,1),(1,0)))
    return dict(live_complementary_pair=bool(np.allclose(pair,[0,1])),placebo_single_shares=bool(max(abs(x) for x in single)<1e-12),positive_lookahead=bool(p.choice((),2,.05,'two-step-lookahead',.1)==0),positive_greedy_noisy=bool(p.choice((),2,.05,'greedy-information',.1)==2),placebo_expensive_stop=bool(p.choice((),8,1.,'two-step-lookahead',.1)==-1))


def run(root,d,pulse):
    from .sequential_check import posterior as reference_posterior,choice as reference_choice
    population=[];timings=[];maxerr=0.;decision_checks=0
    for world in d['lineages']:
        start=time.process_time();rows=W.enumerate_world(world,d['shift'],d['regime']);table=tables(rows)
        sampled=W.sampled(rows,d['test_cases'],'v21-d04',world,d['shift'],d['regime']);x=W.project(sampled,'context');truth=sampled[:,10].astype(int)
        base=np.array([table['current'][table['index'][row.tobytes()]].sum(0) for row in x]);base/=base.sum(1)[:,None]
        priors=np.stack([base[:,W.LABELS[:,3]==g].sum(1) for g in (0,1)],axis=1)
        rng=np.random.default_rng(W.seed('v21-d04-reports',world,d['shift'],d['regime']));order=W.LABELS[truth,3];pad=rng.integers(2,size=len(truth))
        reports=np.column_stack([order^pad,pad,np.where(rng.random((len(truth),6))<REL,order[:,None],1-order[:,None])])
        (root/'reader').mkdir(parents=True,exist_ok=True);(root/'evaluator').mkdir(exist_ok=True)
        np.savez_compressed(root/'reader'/f'{world}_points.npz',evidence=x)
        np.savez_compressed(root/'evaluator'/f'{world}_points.npz',source_rows=sampled,truth=truth,base=base,order_prior=priors,potential_reports=reports,pad=pad)
        setup=time.process_time()-start;planners={};reference_cache={}
        for budget in d['budgets']:
          for fee in d['fees']:
            for policy in POLICIES:
                tick=time.process_time();pred=[];histories=[];counts=[];missed=[];computed=[]
                for i in range(len(truth)):
                    pk=tuple(priors[i]);planner=planners.setdefault(pk,Planner(pk));history=();before=planner.evaluations
                    while len(history)<budget:
                        remaining=budget-len(history);action=planner.choice(history,remaining,fee,policy,d['entropy_threshold'])
                        ck=(pk,history,remaining,fee,policy)
                        if ck not in reference_cache:reference_cache[ck]=reference_choice(pk,history,remaining,fee,policy,d['entropy_threshold'])
                        assert action==reference_cache[ck];decision_checks+=1
                        if action<0:break
                        history=tuple(sorted(history+((action,int(reports[i,action])),)))
                    post=planner.order(history);ref=reference_posterior(pk,history);err=float(np.max(np.abs(post-ref)));maxerr=max(maxerr,err);assert err<1e-12
                    scale=np.divide(post,priors[i],out=np.zeros(2),where=priors[i]>0)
                    q=base[i]*scale[W.LABELS[:,3]];assert abs(q.sum()-1)<1e-12
                    pred.append(q);h=np.full(8,-1,dtype=int)
                    for a,b in history:h[a]=b
                    histories.append(h);counts.append(len(history));missed.append(bool(budget>=2 and entropy(post)>1e-12));computed.append(planner.evaluations-before)
                with np.errstate(divide='ignore'):lp=np.log(np.array(pred))
                prefix=f'{world}-{budget}-{fee}-{policy}';metrics=retain(root,prefix,lp,truth)
                # Extend score arrays without changing retain's proper scores.
                path=root/'raw'/f'{prefix}_points.npz'
                with np.load(path) as z:saved={k:z[k] for k in z.files}
                saved.update(purchased_reports=np.array(histories),purchases=np.array(counts),net_loss=saved['log_loss']+fee*np.array(counts),missed_order_identification=np.array(missed),new_posterior_evaluations=np.array(computed))
                np.savez_compressed(path,**saved)
                population.append(dict(lineage=world,budget=budget,fee=fee,policy=policy,metrics=metrics,mean_purchases=float(np.mean(counts)),mean_net_loss=float(np.mean(saved['net_loss'])),missed_identification_fraction=float(np.mean(missed))))
                timings.append(dict(lineage=world,budget=budget,fee=fee,policy=policy,consumer_cpu_seconds=time.process_time()-tick))
                pulse(phase='sequential-acquisition',lineage=world,budget=budget,fee=fee,policy=policy)
        timings.append(dict(lineage=world,setup_cpu_seconds=setup,total_cpu_seconds=time.process_time()-start,shared_posterior_evaluations=sum(p.evaluations for p in planners.values())))
    return dict(card='G21-D04',population=population,controls=controls(),timings=timings,independent_max_probability_error=maxerr,decision_checks=decision_checks,scope='Supplied-law one-episode order-recovery acquisition ruler, no learned maker or task-performance policy.',limitations=['Exact generator law and record-channel reliability .8 are supplied; no learned acquisition or law-marginal ceiling.','Two complementary truthful shares reveal operation order together; each alone is independent of the full target. Six additional reports are independent noisy order readings.','Fixed rival buys the complementary pair first; it is not deprived of the useful pair.','Fees are stipulated nat-equivalent units per purchase. Compute counters include checking and export; shared memoization and loop order preclude a clean per-policy latency ranking.','Budgets share current cases and potential reports. Missed identification means positive remaining order entropy when the available complementary pair could resolve it within the allowed budget. It is not full historical identifiability.','Sequential query order is unnecessary for likelihood multiplication here; the saved action/answer vector records exactly what was purchased.','No fitted seeds, new F03 category, fresh confirmation or human-process inference.','Miniature — architecture untested.'])
