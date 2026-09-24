import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import reachable_retrospective as Q


def fixture():
    return dict(hypotheses=[['none',0,0],['purpose',1,8],['none',0,1],['skill',3,2]],
                length=4,checkpoint=2,signatures=[[0,0,0],[1,1,1],[2,2,6]],membership=[0,0,1,2])


def law(mode):
    p=np.random.default_rng(324).uniform(.01,1,(16,4,8));p/=p.sum(-1,keepdims=True)
    if mode=='uniform':p[:]=1/8
    if mode=='zeros':p[:,:,:3]=0;p/=p.sum(-1,keepdims=True)
    if mode=='disjoint':
        p[:]=0
        for s in range(16):p[s,:,s%8]=1
    return p


@pytest.mark.parametrize('mode',['random','uniform','zeros','disjoint'])
def test_direct_all_hypothesis_bayes_and_every_future_coordinate(mode):
    s=fixture();prior=np.array([.2,.3,.4,.1]);W=np.array([[.6,.1,.2,.1],[0,.5,0,.5],[.2,.3,.4,.1]])
    st=Q.prepare(s,prior);p=law(mode);raw,full=Q.evaluate(W,st,p,retain_full=True)
    for row,w in enumerate(W):
        for time in range(1,3):
            for context in range(4):
                for endpoint in range(8):
                    e=8*context+endpoint
                    past=[h[2]^({'purpose':8,'skill':4}.get(h[0],0) if time>h[1] else 0) for h in s['hypotheses']]
                    numer=np.array([w[j]*p[state,context,endpoint] for j,state in enumerate(past)])
                    denominator=float(sum(numer));assert raw['report_probability'][row,time-1,e]==pytest.approx(denominator,abs=2e-15)
                    if not denominator:assert not raw['possible'][row,time-1,e];continue
                    exact=np.array([sum(numer[j] for j,g in enumerate(s['membership']) if g==i)/denominator for i in range(3)])
                    rival=np.array([sum(w[j] for j,g in enumerate(s['membership']) if g==i)*sum(prior[j]*p[past[j],context,endpoint] for j,g in enumerate(s['membership']) if g==i)/sum(prior[j] for j,g in enumerate(s['membership']) if g==i) for i in range(3)])
                    rival/=sum(rival)
                    np.testing.assert_allclose(full[time-1]['exact'][row,e],exact,atol=2e-15,rtol=0)
                    np.testing.assert_allclose(full[time-1]['rival'][row,e],rival,atol=2e-15,rtol=0)
                    for off in range(3):
                        forecast=sum((exact[g]-rival[g])*p[sig[off]] for g,sig in enumerate(s['signatures']))
                        np.testing.assert_allclose(full[time-1]['differences'][row,e,off].reshape(4,8),forecast,atol=2e-15,rtol=0)
    assert raw['privileged_max_error'].max()<1e-12
    if mode=='uniform':assert np.nanmax(raw['quotient_tv'])<1e-12
    assert np.nanmax(raw['quotient_tv'][2])<1e-12


def test_lossless_factors_reconstruct_all_updates():
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);W=np.array([[.6,.1,.2,.1]]);p=law('random')
    raw,full=Q.evaluate(W,st,p,retain_full=True)
    for t in range(2):
        numer=np.zeros((1,32,3))
        for g in range(3):
            ix=np.flatnonzero(st['mapping']==g)
            if len(ix)==1:numer[:,:,g]=W[:,ix[0],None]*p[st['past'][t,ix[0]]].reshape(1,32)
            else:numer[:,:,g]=raw['mixed_exact_numerator'][:,t,:,list(raw['mixed_groups']).index(g)]
        np.testing.assert_allclose(numer/raw['report_probability'][:,t,:,None],full[t]['exact'],atol=1e-15,rtol=0)


def test_identity_stationary_copy_and_permutation():
    st=Q.prepare(fixture(),[.2,.3,.4,.1]);W=np.array([[.6,.1,.2,.1]]);p=law('random');base,_=Q.evaluate(W,st,p)
    s=fixture();order=[3,1,0,2];s['hypotheses']=[s['hypotheses'][i] for i in order];s['membership']=[s['membership'][i] for i in order]
    reordered,_=Q.evaluate(W[:,order],Q.prepare(s,np.array([.2,.3,.4,.1])[order]),p)
    np.testing.assert_allclose(base['max_forecast_difference'],reordered['max_forecast_difference'],atol=1e-15,rtol=0)
    np.testing.assert_array_equal(Q.copied_report(base['initial_group_weights']),base['initial_group_weights'])
    s=dict(hypotheses=[['none',0,i] for i in range(16)],checkpoint=2,length=4,signatures=[[i]*3 for i in range(16)],membership=list(range(16)))
    raw,_=Q.evaluate(np.full((1,16),1/16),Q.prepare(s,np.full(16,1/16)),p)
    assert np.nanmax(raw['quotient_tv'])<1e-12 and all(Q.controls().values())


@pytest.mark.parametrize('bad',['mapping','schedule','prior','weight','law'])
def test_deliberate_corruption_fails(bad):
    s=fixture();prior=[.2,.3,.4,.1];W=np.array([[.6,.1,.2,.1]]);p=law('random')
    if bad=='mapping':s['membership'][0]=2
    if bad=='schedule':s['signatures'][0][0]=3
    if bad=='prior':prior[0]=-.1
    if bad=='weight':W[0,0]=-.1
    if bad=='law':p[0,0,0]=float('nan')
    with pytest.raises(ValueError):Q.evaluate(W,Q.prepare(s,prior),p)


@pytest.mark.parametrize('handler',['reachable','sufficient'])
def test_complete_handler_population_bindings_and_parent_identity(tmp_path,handler):
    import gzip
    from itertools import product
    from ghostscale.validation.soundingline.v18_3.io import write,canonical,file_digest
    # Native 16 stationary and two changed hypotheses; all256 row identities.
    hs=[['none',0,i] for i in range(16)]+[['purpose',1,8],['skill',1,4]]
    s=dict(hypotheses=hs,length=3,checkpoint=2,signatures=[[i,i] for i in range(16)],membership=list(range(16))+[0,0])
    inputs=tmp_path/'inputs';inputs.mkdir();write(inputs/'SCHEDULES.json',{'3-2':s})
    p=law('random');W=np.full((256,18),1/18);st=Q.prepare(s,[.5/16]*16+[.25]*2)
    pred=Q.forecast_array(Q.group(W,st),st['signatures'],p)[:,0].reshape(256,4,8)
    rows=[]
    for i,identity in enumerate(product([1,2],range(16),('purpose','skill'),(False,True),(False,True))):
        rows.append(dict(zip(('draw','initial_maker','kind','switched','duplicates'),identity),arm='unknown-time-type',length=3,step=2,stream=str(i),actual_maker=0,joint_array='x-unknown-time-type',joint_row=i,forecast=pred[i].tolist()))
    for e in ('aware','omitted'):
        base=inputs/e;(base/'raw').mkdir(parents=True);(base/'evaluator').mkdir()
        write(base/'evaluator/7-law.json',p.tolist());write(base/'evaluator/7-joint-map.json',{'x-unknown-time-type':dict(hypotheses=hs,rows=[[str(i),2] for i in range(256)])})
        (base/'raw/7-forecasts_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
        np.savez_compressed(base/'raw/7-joint_points.npz',**{'x-unknown-time-type':W})
    cfg=dict(lineages=[7],draws=[1,2],batch_rows=32,input_files={x.relative_to(inputs).as_posix():file_digest(x) for x in inputs.rglob('*') if x.is_file()})
    from ghostscale.validation.soundingline.v19 import retrospective_sufficient as S
    module=Q if handler=='reachable' else S
    result=module.run(tmp_path,dict(design=cfg),lambda **kw:None)
    assert result['rows']==512 and result['reports']==512*64 and not result['unavailable_checkpoints']
    # The second run fails the frozen input binding before reusing evidence.
    write(inputs/'aware/evaluator/7-law.json',law('uniform').tolist(),immutable=False)
    with pytest.raises(ValueError,match='input binding'):module.run(tmp_path,dict(design=cfg),lambda **kw:None)
