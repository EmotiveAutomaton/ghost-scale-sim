from itertools import combinations,product
import copy
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import retrospective_quotient as Q
from ghostscale.validation.soundingline.v18_3.io import write,file_digest,read


def law():
    x=np.random.default_rng(718).uniform(.1,1,(16,4,8));return x/x.sum(-1,keepdims=True)


def spec():
    hs=[('none',0,0),('purpose',1,8),('purpose',2,8),('skill',1,4),('none',0,1),('purpose',4,1),('skill',2,5)]
    makers=list(product((0,1),repeat=4))
    def state(h,t):
        k,c,m=h;bits=list(makers[m])
        if k!='none' and t>c:bits[{'purpose':0,'skill':1}[k]]=1-bits[{'purpose':0,'skill':1}[k]]
        return makers.index(tuple(bits))
    paths=[tuple(state(h,t) for t in range(3,7)) for h in hs]
    groups=sorted(set(paths));return dict(hypotheses=hs,length=6,checkpoint=3,signatures=groups,membership=[groups.index(p) for p in paths])


def brute(s,law):
    # Direct per-hypothesis Bayes, then direct complete future forecasts. Does
    # not use the producer's structure, masks, kernels or aggregation.
    hs=s['hypotheses'];makers=list(product((0,1),repeat=4));mapping=np.array(s['membership']);ng=len(s['signatures'])
    def state(h,t):
        kind,change,maker=h;bits=list(makers[maker])
        if kind!='none' and t>change:
            axis={'purpose':0,'skill':1}[kind];bits[axis]=1-bits[axis]
        return makers.index(tuple(bits))
    result=dict(report_comparisons=0,both_impossible=0,one_impossible=0,both_possible=0,tolerance_update_witnesses=0,tolerance_forecast_witnesses=0,max_quotient_distance=0.,max_forecast_coordinate_difference=0.)
    for i,j in combinations(range(len(hs)),2):
        if mapping[i]!=mapping[j] or all(state(hs[i],t)==state(hs[j],t) for t in range(1,s['checkpoint']+1)):continue
        for k in range(len(hs)):
            if mapping[k]==mapping[i]:continue
            for t,c,e in product(range(1,s['checkpoint']+1),range(4),range(8)):
                result['report_comparisons']+=1;posts=[]
                for member in (i,j):
                    p=np.zeros(len(hs));p[member]=.5;p[k]=.5
                    p*=np.array([law[state(h,t),c,e] for h in hs]);den=float(p.sum())
                    posts.append(None if den==0 else p/den)
                if any(p is None for p in posts):
                    result['both_impossible' if all(p is None for p in posts) else 'one_impossible']+=1;continue
                result['both_possible']+=1
                group=[np.array([sum(p[mapping==g]) for g in range(ng)]) for p in posts]
                delta=float(np.abs(group[0]-group[1]).sum()/2)
                mx=0.
                for future in range(s['checkpoint'],s['length']+1):
                    predicted=[sum(float(w)*law[state(h,future)] for h,w in zip(hs,p)) for p in posts]
                    mx=max(mx,float(np.abs(predicted[0]-predicted[1]).max()))
                result['tolerance_update_witnesses']+=int(delta>1e-12);result['tolerance_forecast_witnesses']+=int(mx>1e-12)
                result['max_quotient_distance']=max(result['max_quotient_distance'],delta)
                result['max_forecast_coordinate_difference']=max(result['max_forecast_coordinate_difference'],mx)
    return result


@pytest.mark.parametrize('mode',['random','uniform','impossible','disjoint'])
def test_exhaustive_scalar_matches_complete_factorization(mode):
    p=law()
    if mode=='uniform':p[:]=1/8
    if mode=='impossible':p[:,:,:2]=0;p/=p.sum(-1,keepdims=True)
    if mode=='disjoint':
        p[:]=0
        for i in range(16):p[i,:,i%8]=1
    s=spec();arrays,meta=Q.structure(s);actual=Q.evaluate(arrays,p);expected=brute(s,p)
    for k,v in expected.items():
        if k.startswith('max'):assert actual[k]==pytest.approx(v,abs=2e-15)
        else:assert actual[k]==v
    assert meta['report_comparisons']==actual['report_comparisons']
    assert len(arrays['pairs'])==meta['pair_count']
    for t,b in enumerate(meta['pair_classes_by_time']):
        classes=np.array(b['classes']);ix=b['membership'];pairs=arrays['pairs']
        expected=np.column_stack((arrays['pair_groups'],arrays['past'][t,pairs[:,0]],arrays['past'][t,pairs[:,1]]))
        assert np.array_equal(classes[ix],expected)
        assert np.array_equal(np.bincount(ix),b['multiplicity'])
    for t,b in enumerate(meta['third_classes_by_time']):
        assert np.array_equal(np.array(b['classes'])[b['membership']],np.column_stack((arrays['mapping'],arrays['past'][t])))
        assert sum(b['multiplicity'])==len(s['hypotheses'])


def test_stationary_no_pairs_and_same_time_identity():
    s=dict(hypotheses=[('none',0,m) for m in range(16)],length=4,checkpoint=2,signatures=[[m]*3 for m in range(16)],membership=list(range(16)))
    arr,meta=Q.structure(s);assert Q.evaluate(arr,law())['report_comparisons']==0
    arr,_=Q.structure(Q.fixture());p=law();kernel=Q.likelihood_kernel(p)
    states=arr['past'][-1];assert states[0]==states[1]
    assert np.all(kernel['quotient_distance'][states[0],states[1]]==0)
    for offset in range(arr['schedules'].shape[1]):
        a,b=arr['schedules'][arr['mapping'][0],offset],arr['schedules'][arr['mapping'][1],offset]
        assert a==b and np.all(kernel['quotient_distance'][a,b]==0)


def test_permuted_members_and_group_ids():
    s=spec();p=law();base=Q.evaluate(Q.structure(s)[0],p)
    order=[6,2,4,0,5,1,3];s['hypotheses']=[s['hypotheses'][i] for i in order];s['membership']=[s['membership'][i] for i in order]
    n=len(s['signatures']);s['signatures']=s['signatures'][::-1];s['membership']=[n-1-i for i in s['membership']]
    assert Q.evaluate(Q.structure(s)[0],p)==base


@pytest.mark.parametrize('bad',['mapping','shape','duplicate','hypothesis'])
def test_corrupt_structure_fails(bad):
    s=copy.deepcopy(spec())
    if bad=='mapping':s['membership'][0]=100
    if bad=='shape':s['membership'].pop()
    if bad=='duplicate':s['signatures'].append(s['signatures'][0])
    if bad=='hypothesis':s['hypotheses'][0]=('none',0,15)
    with pytest.raises(ValueError):Q.structure(s)


@pytest.mark.parametrize('bad',['negative','nan','mass','shape'])
def test_corrupt_law_fails(bad):
    p=law()
    if bad=='negative':p[0,0,0]=-.1
    if bad=='nan':p[0,0,0]=np.nan
    if bad=='mass':p[0,0,0]+=.1
    if bad=='shape':p=p[:15]
    with pytest.raises(ValueError):Q.likelihood_kernel(p)


def test_controls_and_full_handler(tmp_path):
    assert all(Q.controls().values());s=Q.fixture();inputs=tmp_path/'inputs';inputs.mkdir()
    write(inputs/'SCHEDULES.json',{'3-2':s});write(inputs/'7-law.json',law().tolist())
    cfg=dict(input_files={p.name:file_digest(p) for p in inputs.iterdir()},lineages=[7],lengths=[3],checkpoints=[2])
    result=Q.run(tmp_path,dict(design=cfg),lambda **kw:None)
    assert result['cells'][0]['report_comparisons']==64
    with np.load(tmp_path/'evaluator/7-kernel.npz') as z:assert len(z['quotient_distance'])==16
    assert read(tmp_path/'EVIDENCE_ROLES.json')['reader'].startswith('no new reader inputs')
    write(inputs/'7-law.json',np.full((16,4,8),1/8).tolist(),immutable=False)
    with pytest.raises(ValueError,match='input binding'):Q.run(tmp_path,dict(design=cfg),lambda **kw:None)
