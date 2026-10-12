"""B04 finite supplied-law change-point ruler; separate from the fitted V21 world.

One 96-episode stream provides paired 32/96 prefixes. Change positions are absolute,
so the common 24-episode change is both 75% of 32 and 25% of 96. Later changes
are honestly future changes in the short prefix. No change time or role is an input.
"""
import time
import numpy as np
from . import world as W
from .studies import retain

STATES=np.array([(p,s,c) for p in (0,1) for s in (0,1) for c in (0,1)])
READERS=('static','all-fast','all-slow','role-mixture','direct-chronological')

def channels(world):
    rng=np.random.default_rng(W.seed('v21-b04-channel',world))
    preference=np.array([rng.uniform(.1,.3),rng.uniform(.7,.9)])
    skill=np.array([rng.uniform(.55,.65),rng.uniform(.85,.95)])
    context=np.array([rng.uniform(.15,.35),rng.uniform(.65,.85)])
    joint=np.zeros((8,128));emission=np.zeros((8,4))
    for state,(p,s,c) in enumerate(STATES):
      for x in (0,1):
       for intention in (0,1):
        for success in (0,1):
            # A preference is for matching the opportunity, independent of skill.
            y=intention if success else 1-intention
            mass=(context[c] if x else 1-context[c])*(preference[p] if intention==x else 1-preference[p])*(skill[s] if success else 1-skill[s])
            label=int(''.join(map(str,(p,s,c,x,intention,success,y))),2)
            joint[state,label]=mass;emission[state,2*x+y]+=mass
    return dict(preference=preference,skill=skill,context=context,joint=joint,emission=emission)

def transitions(kind):
    if kind=='role-mixture':
        result=np.zeros((3,8,8))
        for i,row in enumerate(STATES):
          for role in range(3):
            other=row.copy();other[role]^=1;j=np.flatnonzero((STATES==other).all(1))[0];result[role,i,j]=1/3
        return result
    return (np.ones((1,8,8))-np.eye(8)[None])/7

def infer(evidence,emission,reader):
    evidence=np.asarray(evidence,int);h=len(evidence)
    if reader in ('static','all-fast','all-slow'):
        p=np.full(8,1/8);rate={'static':0.,'all-fast':1/8,'all-slow':1/64}[reader]
        for t,obs in enumerate(evidence):
            if t:p=(1-rate)*p+rate/8
            p*=emission[:,obs];p/=p.sum()
        return (1-rate)*p+rate/8,None
    trans=transitions(reader);pre=np.full(8,1/8);post=np.zeros((len(trans),8))
    for t in range(h+1):
        if t:
            # Conditional on no previous change, transition t has this prior mass.
            hazard=(.75/95)/(.25+.75*(96-t)/95) if t<96 else 0.
            post+=hazard*np.einsum('i,rij->rj',pre,trans);pre*=1-hazard
        if t==h:break
        pre*=emission[:,evidence[t]];post*=emission[:,evidence[t]]
        total=pre.sum()+post.sum();pre/=total;post/=total
    roles=np.r_[pre.sum(),post.sum(1)]
    return pre+post.sum(0),roles

def controls():
    ch=channels(71000760);flat=np.full((8,4),.25)
    p,r=infer([0]*32,flat,'role-mixture')
    signal=np.full((8,8),.01);np.fill_diagonal(signal,.93)
    q,_=infer([0]*96,signal,'static')
    return dict(placebo_no_information=bool(np.allclose(p,np.full(8,1/8))),
                live_known_state=bool(q[0]>.999),
                positive_normalized_channel=bool(np.allclose(ch['joint'].sum(1),1) and np.allclose(ch['emission'].sum(1),1)),
                placebo_no_change_prior=bool(abs(r[0]-(.25+.75*(95-32)/95))<1e-12))

def run(root,d,pulse):
    from .drift_check import reference
    population=[];timings=[];maxerr=0.;prefix_checks=0
    for world in d['lineages']:
        start=time.process_time();ch=channels(world);n=d['streams'];rng=np.random.default_rng(W.seed('v21-b04-streams',world))
        initial=rng.integers(8,size=n);coins=rng.random((n,97,3))
        for position in d['change_positions']:
          for changed_role in (-1,0,1,2):
            states=np.tile(initial[:,None],(1,97))
            if changed_role>=0:
                moved=STATES[initial].copy();moved[:,changed_role]^=1
                mapped=np.array([np.flatnonzero((STATES==row).all(1))[0] for row in moved]);states[:,position:]=mapped[:,None]
            traits=STATES[states];x=(coins[:,:,0]<ch['context'][traits[:,:,2]]).astype(int)
            match=coins[:,:,1]<ch['preference'][traits[:,:,0]];intention=np.where(match,x,1-x)
            success=(coins[:,:,2]<ch['skill'][traits[:,:,1]]).astype(int);y=np.where(success,intention,1-intention)
            obs=2*x+y
            (root/'reader').mkdir(parents=True,exist_ok=True);(root/'evaluator').mkdir(exist_ok=True)
            prefix=f'{world}-{position}-{changed_role}'
            # Reader filenames carry no intervention labels. Each horizon's file
            # contains only that permitted prefix; future episodes stay evaluator-only.
            reader_ids=[]
            for horizon in d['horizons']:
                reader_id=f"case-{W.seed('v21-b04-blind-id',world,position,changed_role,horizon):016x}"
                reader_ids.append(reader_id)
                np.savez_compressed(root/'reader'/f'{reader_id}_points.npz',observations=obs[:,:horizon])
            np.savez_compressed(root/'evaluator'/f'{prefix}_points.npz',states=states,intention=intention,success=success,observations=obs,initial=initial,coins=coins,reader_ids=np.array(reader_ids),horizons=np.array(d['horizons']))
            prefix_checks+=n
            for horizon in d['horizons']:
                truth=np.array([int(''.join(map(str,(*STATES[states[i,horizon]],x[i,horizon],intention[i,horizon],success[i,horizon],y[i,horizon]))),2) for i in range(n)])
                for reader in READERS:
                    probs=[];role_probs=[]
                    for i in range(n):
                        posterior,roles=infer(obs[i,:horizon],ch['emission'],reader)
                        ref=reference(obs[i,:horizon],ch['preference'],ch['skill'],ch['context'],reader)
                        err=float(np.max(np.abs(posterior-ref)));maxerr=max(maxerr,err);assert err<1e-12
                        probs.append(posterior@ch['joint'])
                        if roles is not None:role_probs.append(roles)
                    with np.errstate(divide='ignore'):lp=np.log(probs)
                    name=f'{prefix}-{horizon}-{reader}';metrics=retain(root,name,lp,truth)
                    if role_probs:np.savez_compressed(root/'evaluator'/f'{name}-roles_points.npz',role_probabilities=np.array(role_probs))
                    population.append(dict(lineage=world,change_position=position,changed_role=changed_role,horizon=horizon,reader=reader,change_observed=changed_role>=0 and position<horizon,change_at_forecast=changed_role>=0 and position==horizon,metrics=metrics))
                pulse(phase='role-drift',lineage=world,change_position=position,changed_role=changed_role,horizon=horizon)
        timings.append(dict(lineage=world,total_cpu_seconds=time.process_time()-start))
    return dict(card='G21-B04',population=population,controls=controls(),timings=timings,independent_max_probability_error=maxerr,paired_prefix_streams=prefix_checks,scope='Supplied-law finite binary preference/skill/opportunity-change diagnostic; no learned architecture superiority.',limitations=['Known finite channel and state vocabulary are shared by all readers; no law-marginal ceiling or trained direct model.','Direct chronological rival allows any different successor state; role mixture permits exactly one changed role. Different transition priors and capacity are explicit.','Static, all-fast and all-slow use declared reset rates 0, 1/8 and 1/64. Role/direct models use prior .25 no change, otherwise one uniformly located change in the 95 gaps. No true change time or role enters any reader.','32/96 histories are prefixes of the same 96-episode stream at the same absolute change position; matched fractional positions at different absolute times are distinct conditions. Future target episodes differ by horizon.','No-change rows repeat across positions and are duplicate controls, not independent evidence. All interventions share initial states and random draws; worlds, not streams, are uncertainty units.','The seven-bit target encodes current preference/skill/opportunity states, next opportunity, intention, execution success and artifact. This new miniature does not inherit old-world headroom or historical numbers.','No stable human values, learned change detection, execution-shift transfer, F03 category or fresh confirmation is established.','Miniature — architecture untested.'])
