"""R4: regenerate W4 counts, exchangeable bridges, complementary-tail ranking."""
import math
from decimal import Decimal,localcontext
import numpy as np
from .common import rng,record,close
from ..v25 import population as old,population_precision as precision,pieces
from ..v25.common import softmax

LEVELS=(128,192,256,384,512,768,1024)

def tail_scores(e,counts,precision_digits):
    logs=precision.log_laws(tuple(tuple(str(float(v)) for v in row) for row in e),precision_digits);answer=[]
    with localcontext() as ctx:
        ctx.prec=precision_digits;d=Decimal
        for count in counts:
            ll=[sum((d(int(count[si,k]))*logs[si][h][k] for si,a in enumerate(pieces.TRAIN) for k in range(len(a))),d(0)) for h in range(36)]
            top=max(ll);v=[(x-top).exp() for x in ll];total=sum(v,d(0));tails=[sum((v[h] for h in range(36) if not old.TAILS[h,j]),d(0)) for j in range(6)]
            answer.append(float((total/min(tails)).ln())) # -log complementary tail, never 1-rounded flag
    return np.array(answer)

def calibration(values):
    values=np.asarray(values);cut=float(np.sort(values)[math.ceil(.95*len(values))-1]);above=sum(values>cut);ties=sum(values==cut)
    credit=(.05*len(values)-above)/ties
    if not -1e-12<=credit<=1+1e-12:raise ValueError('specificity credit')
    return cut,float(credit)

def detect(v,c):return (v>c[0]).astype(float)+c[1]*(v==c[0])

def block(setting,identity,independent=False):
    b=identity[-1]+setting.get('reference_offset',0);original_identity=['v25-w4-study-1','W4',0,b]
    saved=old.block(dict(alpha=10,placebo=False),original_identity,independent=False);e=np.array(saved['public']['effects']);truth=np.array(saved['truth']['weights']);main=truth[:64];excluded=np.array(saved['truth']['null_excluded_population_member']);r=rng('R4-exchangeable-bridge',b)
    cumulative=np.zeros((640,4,6),dtype=np.int16);previous=0;counts={};bridges=[]
    for row in saved['observations']:
        end=row['records'];delta=np.array(row['category_counts'],dtype=np.int16);remainder=delta.copy();last=previous
        for n in [x for x in LEVELS if previous<x<end]+[end]:
            take=np.zeros_like(delta)
            for si,a in enumerate(pieces.TRAIN):
                trials=(n+3-si)//4-(last+3-si)//4
                for maker in range(640):
                    take[maker,si,:len(a)]=r.multivariate_hypergeometric(remainder[maker,si,:len(a)].astype(np.int64),trials) if n<end else remainder[maker,si,:len(a)]
            cumulative+=take;remainder-=take;last=n
            if n in LEVELS:counts[n]=cumulative.copy();bridges.append(dict(records=n,counts=cumulative.tolist()))
        assert np.all(remainder==0);previous=end
    out=[];raw=[];law=[pieces.laws(e,a,[(i,3,-1) for i in range(36)],weights=old.WEIGHTS) for a in pieces.TRAIN]
    for n in setting.get('prefixes',LEVELS):
        count=counts[n];ll=sum((count[:,si,:len(a)]@np.log(law[si]).T for si,a in enumerate(pieces.TRAIN)),np.zeros((640,36)));p=softmax(ll)
        rounded=precision.rounded_flags(p,e,count,main,excluded,10,False,'individual');score50=tail_scores(e,count[:576],50);score80=tail_scores(e,count[:576],80);close(score50.tolist(),score80.tolist(),tol=1e-12)
        if not np.array_equal(np.argsort(score50,kind='stable'),np.argsort(score80,kind='stable')):raise ValueError('50/80 digit tail order disagreement')
        allrates={}
        for arm,score,panel in [('a',rounded,slice(64,320)),('b',score80,slice(64,320)),('c',score80,slice(320,576))]:
            if arm=='c' and setting.get('omit_exact_mean'):continue
            cut=calibration(score[panel]);positive=old.DEVIATION[main]>=2;rate=float(np.mean(detect(score[:64],cut)[positive])) if positive.any() else None
            allrates[arm]=rate
            if arm=='a' and n in (512,1024):
                expected=next(x['metrics'] for x in saved['readouts'] if x['labels']==dict(reader='individual',history=n))
                if abs(rate-expected['detection_ge_two'])>1e-12 or cut!=(expected['threshold'],expected['threshold_tie_mass']):raise ValueError('arm a did not reproduce W4')
            out.append(record(dict(detection=rate,threshold=cut[0],tie_credit=cut[1],tie_panel_share=float(np.mean(score[panel]==cut[0])),calibration_false=float(np.mean(detect(score[panel],cut))),exact_mean_false=float(np.mean(detect(score[320:576],cut)))),reader=arm,history=n))
        out.append(record(dict(primary=allrates['b']-allrates['a'] if allrates['a'] is not None else None),reader='contrasts',history=n))
        raw.append(dict(records=n,rounded=rounded.tolist(),negative_log_tail=score80.tolist()))
    return dict(public=dict(original_identity=original_identity,effects=e.tolist(),truth=truth.tolist(),rule='a unchanged eight-decimal ties; b/c -log exact complementary tail checked at50/80 digits'),observations=bridges,rankings=raw,readouts=out)

def controls():
    old.controls();v=np.r_[np.zeros(244),np.ones(12)];c=calibration(v);assert abs(np.mean(detect(v,c))-.05)<1e-12
    # Tail ranking retains differences erased by 1-probability subtraction.
    assert float(1-1e-30)==float(1-1e-40) and -math.log(1e-30)<-math.log(1e-40)
    return dict(passed=True,uniform_prior_equivalence=True,matched_specificity=True,tail_preserves_saturated_difference=True)

def design_check(settings):
    # Original retained worlds, no new scientific draws; cheap double-precision W4 rule.
    rows=[]
    for b in range(16):
        data=old.block(dict(alpha=10,placebo=False),['v25-w4-study-1','W4',0,b]);m=next(x['metrics'] for x in data['readouts'] if x['labels']==dict(reader='individual',history=1024));rows.append(m)
    # Exact retained W4 setting0 values, independently read at setup. The prose
    # "threshold rounds to1" describes the median: 14/16 worlds, not all16.
    expected=[1.,.99999918,1.,1.,1.,1.,1.,1.,1.,1.,1.,.99999999,1.,1.,1.,1.]
    thresholds=[x['threshold'] for x in rows];share=sum(v==1. for v in thresholds)/16
    return dict(passed=thresholds==expected and share>=.25 and abs(np.mean([x['threshold_tie_mass'] for x in rows])-.4044633103545362)<1e-12,seed_domain='retained-W4-identities',threshold_mean=float(np.mean(thresholds)),threshold_one_share=share,mean_tie_credit=float(np.mean([x['threshold_tie_mass'] for x in rows])),note='Packet threshold1 is the median,14/16 worlds; tie mass .40 is randomized credit. Original failed all16 assumption retained; no worlds/settings changed.')
