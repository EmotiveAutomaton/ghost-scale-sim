"""Hidden renewal structures with exact boundary/goal/state filtering.

DESIGN CHECK: three independent episode lengths, each uniform on 1..3, give
3..9 pieces. Prefix symbols are independent of the shared world goal; only the
final checksum carries it (2% emission lapse). Repeated structures share that
goal, which is necessary for the declared across-structure records-needed task.
This is a new checksum library in hierarchy.py's three-episode length format,
not the old library or a test of actual humour. M2 uses the declared population
unigram surprisal; M3 has symmetric 0.1 switching and calibrated emissions.
"""
import math,itertools
import numpy as np
from numba import njit
from .common import rng,measure,sustained,posteriors,margin,scores,close

ALPHABET=6
PREFIX=np.array([.35,.25,.15,.1,.1,.05])
LENGTH=np.bincount([sum(x) for x in itertools.product((1,2,3),repeat=3)],minlength=10)/27
HAZARD=np.array([LENGTH[k+1]/LENGTH[k+1:].sum() for k in range(9)])
POP=5/6*PREFIX+1/36
SURPRISE=-np.log(POP)
MEAN_SURPRISE=float(POP@SURPRISE)


def parameters(s):
    average=(s['qc']+5*s['q0'])/6
    surprise=average*SURPRISE/MEAN_SURPRISE
    if np.max(surprise)>1:raise ValueError('surprise-rate calibration exceeds one')
    spread=.8*min(average,1-average)
    return surprise,np.array([average-spread,average+spread])


def generate(s,identity):
    r=rng(identity);goal=int(r.integers(6));play=int(r.integers(2));share=r.integers(0,2,s['listeners']);rows=[];ends=[]
    surprise,state=parameters(s)
    for structure in range(s['structures']):
        lengths=r.integers(1,4,3);length=int(lengths.sum());checksum=0
        for k in range(length):
            complete=k==length-1
            if complete:
                symbol=int((goal-checksum)%6)
                if r.random()<.02:symbol=int(r.integers(6))
            else:symbol=int(r.choice(6,p=PREFIX));checksum=(checksum+symbol)%6
            if r.random()<.1:play=1-play
            p=(s['qc'] if complete else s['q0']) if s['law']=='M1' else surprise[symbol] if s['law']=='M2' else state[play]
            if not complete:p=1-(1-p)*(1-s['fake'])
            marker=int(r.random()<p)
            listener=[int(r.random()<(s['q0']+s['overlap']*(s['qc']-s['q0']) if complete and q and s['law']=='M1' else s['q0'])) for q in share]
            rows.append([symbol,marker,listener,int(complete),play])
        ends.append(len(rows)-1)
    return dict(goal=goal,listener_share=share.tolist(),records=rows,ends=ends)


@njit
def forward(symbols,markers,qc,q0,fake,law,surprise,state,reverse=False):
    # Producer scatters in ascending order; review gathers contributions in the
    # opposite state order with independently accumulated normalizers.
    n=len(symbols);q=np.zeros((6,9,6,2));q[:,0,0,:]=.5
    logs=np.zeros((n,6));boundary=np.zeros((n,6));acc=np.zeros(6)
    for t in range(n):
        x=symbols[t];y=markers[t];nxt=np.zeros_like(q);end=np.zeros(6)
        for z in range(6*9*6*2):
            ix=6*9*6*2-1-z if reverse else z
            play=ix%2;checksum=(ix//2)%6;k=(ix//12)%9;g=ix//108
            mass=q[g,k,checksum,play]
            if mass==0:continue
            for npy in range(2):
                base=mass*(.9 if play==npy else .1)
                for done in range(2):
                    cp=HAZARD[k] if done else 1-HAZARD[k]
                    if cp==0:continue
                    content=(.98*(x==(g-checksum)%6)+.02/6) if done else PREFIX[x]
                    marker=1.
                    if law>=0:
                        prob=(qc if done else q0) if law==0 else surprise[x] if law==1 else state[npy]
                        if not done:prob=1-(1-prob)*(1-fake)
                        marker=prob if y else 1-prob
                    value=base*cp*content*marker
                    if done:nxt[g,0,0,npy]+=value;end[g]+=value
                    else:nxt[g,k+1,(checksum+x)%6,npy]+=value
        for g in range(6):
            total=0.
            for k in range(9):
                for ch in range(6):
                    for pl in range(2):total+=nxt[g,k,ch,pl]
            if total>0:
                acc[g]+=math.log(total);logs[t,g]=acc[g];boundary[t,g]=end[g]/total
                nxt[g]/=total
            else:acc[g]=-np.inf;logs[t,g]=-np.inf;boundary[t,g]=0.
        q=nxt
    return logs,boundary


def analyze(raw,s,independent=False):
    symbols=np.array([r[0] for r in raw['records']]);marks=np.array([r[1] for r in raw['records']]);surprise,state=parameters(s)
    out={};logs_by={};goal=raw['goal'];prior=np.ones(6)/6;ends=raw['ends']
    for reader,law in [('N',-1),('M',('M1','M2','M3').index(s['law'])),('Mc',0)]:
        logs,bound=forward(symbols,marks,s['qc'],s['q0'],s['fake'],law,surprise,state,independent);logs_by[reader]=logs
        if np.any(np.all(np.isneginf(logs),axis=1)):
            out[reader]=dict(true_goal_probability=[0.,0.],top_log_score=None,records=None,
                boundary_half_brier=[0.,0.],boundary_accuracy=[0.,0.],model_contradicted=1.)
            continue
        p=posteriors(logs,prior,independent);yes=np.arange(6)==goal
        m=margin(logs,prior,yes,~yes);prob=(p*bound).sum(1);truth=np.array([r[3] for r in raw['records']])
        out[reader]=dict(true_goal_probability=[scores(p[-1],goal)['posterior'],1.],top_log_score=float(np.log(p[-1,goal])),
            records=sustained(m[ends]),boundary_half_brier=[float(np.mean((prob-truth)**2)),1.],boundary_accuracy=[float(np.mean((prob>.5)==truth)),1.],model_contradicted=0.)
    listener=[]
    for j,truth in enumerate(raw['listener_share']):
        lm=np.array([r[2][j] for r in raw['records']]);evidence=[]
        for sharing in (0,1):
            target=s['q0']+s['overlap']*(s['qc']-s['q0']) if sharing and s['law']=='M1' else s['q0']
            logs,_=forward(symbols,lm,target,s['q0'],0.,0,surprise,state,independent)
            from scipy.special import logsumexp
            evidence.append(logsumexp(logs,axis=1)-math.log(6))
        listener.append(sustained((evidence[truth]-evidence[1-truth])[ends]))
    out['contrasts']=dict(marker_ratio=[out['N']['records'],out['M']['records']],misread_gain=out['Mc']['top_log_score']-out['N']['top_log_score'] if out['Mc']['top_log_score'] is not None else None,listener_records=listener)
    return out


def block(s,identity,independent=False):
    raw=[];values={};out=[]
    for maker in range(s['makers']):
        o=generate(s,[identity,maker]);raw.append(o)
        for reader,stats in analyze(o,s,independent).items():
            for key,value in stats.items():
                v=values.setdefault((reader,key),[])
                if key=='listener_records':v.extend(value)
                else:v.append(value)
    for (reader,key),v in values.items():measure(out,key,v,'extended_mean' if key in ('top_log_score','misread_gain') else 'ratio_of_sums' if key in ('true_goal_probability','boundary_half_brier','boundary_accuracy') else 'median_ratio' if key=='marker_ratio' else 'median' if key.endswith('records') else 'mean',reader=reader)
    return dict(observations=raw,measures=out)


def controls():
    s=dict(structures=2,makers=1,law='M1',qc=.2,q0=.2,fake=0.,listeners=1,overlap=1.)
    o=generate(s,['excluded-marker-control']);v=analyze(o,s);w=analyze(o,s,True);close(v,w,tol=1e-12)
    assert abs(v['M']['top_log_score']-v['N']['top_log_score'])<1e-12
    s.update(qc=1.,q0=0.,structures=1);o=generate(s,['excluded-noiseless']);a=analyze(o,s)
    assert a['M']['boundary_half_brier'][0]<1e-24 and a['M']['boundary_accuracy'][0]==1
    assert abs(sum(LENGTH)-1)<1e-12 and abs(sum(np.arange(10)*LENGTH)-6)<1e-12
    s.update(qc=.8,q0=0.,law='M3');o=generate(s,['excluded-contradicted-library']);o['records'][0][1]=1
    # A completion-only library assigns zero support to a first-piece marker,
    # since the shortest structure has three pieces. Do not floor that score.
    a=analyze(o,s);assert a['Mc']['model_contradicted']==1 and a['Mc']['top_log_score'] is None
    return dict(passed=True,no_information_marker_equal=True,noiseless_boundary_exact=True,independent_reversed_recursion=True,zero_support_retained=True)


def design_check(s,identity):
    shares=[]
    for k in range(32):
        o=generate(s,[identity,'excluded',k]);shares.append(len(o['ends'])/len(o['records'])>=.1)
    surprise,state=parameters(s)
    return dict(passed=float(np.mean(shares))>=.25,case_share=float(np.mean(shares)),floor=.25,
        mean_completion_surprisal=float(np.mean(SURPRISE)),mean_noncompletion_surprisal=float(PREFIX@SURPRISE),
        average_marker_rates=dict(M1=(s['qc']+5*s['q0'])/6,M2=float(POP@surprise),M3=float(state.mean())),
        expected_signs=dict(N='baseline',M='true marker law cannot lose in expectation',Mc='misspecification can mislead; size open'),scientific_draws=0)


def calibrate(s):
    """Excluded horizon/qc calibration, frozen before the primary's cost probe."""
    from ..v26.day_common import statistic,safe_number
    attempts=[];chosen=None
    for horizon in (s['structures'],2*s['structures'],4*s['structures']):
        rows=[]
        for qc in (.4,.6,.8,1.):
            config=s|dict(structures=horizon,qc=qc,law='M1',fake=0.,q0=.2)
            n=[];m=[]
            for k in range(32):
                stats=analyze(generate(config,['excluded-calibration',horizon,k]),config)
                n.append(stats['N']['records']);m.append(stats['M']['records'])
            a=statistic(n,'median');b=statistic(m,'median');ratio=a/b if math.isfinite(a) and math.isfinite(b) else None
            rows.append(dict(qc=qc,N_median=safe_number(a),M_median=safe_number(b),ratio=ratio,N_censored=math.isinf(a),M_censored=math.isinf(b)))
        attempts.append(dict(structures=horizon,rows=rows))
        standard=next(x for x in rows if x['qc']==.8)
        if not standard['N_censored'] and not standard['M_censored']:chosen=dict(structures=horizon,qc=.8);break
        if horizon==4*s['structures'] and not standard['N_censored']:
            admissible=sorted([x for x in rows if not x['M_censored']],key=lambda x:abs(x['qc']-.8))
            if admissible:chosen=dict(structures=horizon,qc=admissible[0]['qc'])
    if chosen is None:chosen=dict(structures=4*s['structures'],qc=.8)
    rows=attempts[-1]['rows'];ratios=[x['ratio'] for x in rows]
    same_side=all(v is not None and v>=1.5 for v in ratios) or all(v is not None and v<1.5 for v in ratios)
    return dict(chosen=chosen,attempts=attempts,worlds_per_cell=32,scientific_draws=0,
                scale_only_if_same_side=same_side,censoring_is_not_case_failure=True)
