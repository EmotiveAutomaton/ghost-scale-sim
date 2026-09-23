from itertools import product
import gzip
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import metadata_disclosure as D, transfer_review as V
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_known_live_null_and_tie():
    assert all(D.controls().values())
    legal=[(s,b,2,1,4,4) for s,b in product(range(2),repeat=2)]
    before,tables,entropy,choice=D.distributions(legal,[0,1,2,3],[.1,.2,.3,.4])
    assert np.allclose(before[:4],[.1,.2,.3,.4])
    assert np.allclose(tables['skill'][0][:4],[1/3,2/3,0,0])
    assert np.allclose(tables['belief'][0][:4],[.25,0,.75,0])
    expected={}
    for field,axis in (('skill',0),('belief',1)):
        ent=0.
        for bit in (0,1):
            ix=[i for i,q in enumerate(legal) if q[axis]==bit];w=np.array([.1,.2,.3,.4])[ix];p=w/w.sum()
            ent+=float(w.sum()*(-p*np.log(p)).sum())
        expected[field]=ent
        assert abs(ent-entropy[field])<1e-12
    assert choice==min(expected,key=expected.get)


def test_zero_likelihood_reply_has_declared_fallback():
    legal=[(s,b,2,1,4,4) for s,b in product(range(2),repeat=2)]
    _,t,_,_=D.distributions(legal,[0,1,2,3],[.5,.5,0,0])
    assert np.allclose(t['skill'][1],[0,0,.5,.5,0,0,0,0])
    with pytest.raises(ValueError,match='conditional'):D.distributions(legal,[0,1,2,3],[.5]*4)


def test_choice_is_constant_for_visible_group_and_independent_of_realized_target():
    qs=[(s,b,2,1,4,4) for s,b in product(range(2),repeat=2)]
    groups=D.P.membership(qs)['omit-both'];targets=np.array([V.endpoint(q,'original') for q in qs])
    a,rows=D.group_laws(qs,groups,targets,np.array([.1,.2,.3,.4]),'original')
    b,_=D.group_laws(qs,groups,(targets+1)%8,np.array([.1,.2,.3,.4]),'original')
    for model in D.MODELS:
        assert len(set(a[model]['choices']))==1
        for name in ('none','skill','belief','entropy-choice','choices'):
            assert np.array_equal(a[model][name],b[model][name])
        assert not np.array_equal(a[model]['all-fields'],b[model]['all-fields'])


def fixture(root, corrupt=False):
    base=root/'inputs';(base/'raw').mkdir(parents=True)
    qs=[(s,b,2,1,4,4) for s,b in product(range(2),repeat=2)]
    truth=[dict(query=list(q),targets={r:V.endpoint(q,r) for r in D.P.M.RULES}) for q in qs]
    if corrupt:truth[0]['targets']['original']=(truth[0]['targets']['original']+1)%8
    write(base/'QUERY_TRUTH.json',truth);write(base/'MEMBERSHIP.json',D.P.membership(qs))
    for rule in D.P.M.RULES:
        records=[dict(maker=[0,q[0],q[1],0],initial=list(D.P.R.ARTIFACTS[q[2]]),steps=[dict(operation=D.P.R.L.OPERATIONS[o]) for o in q[3:]],final=list(D.P.R.ARTIFACTS[V.endpoint(q,rule)]),probability=w) for q,w in zip(qs,[.1,.2,.3,.4])]
        (base/'raw'/f'1-{rule}_points.json.gz').write_bytes(gzip.compress(canonical(records),mtime=0))
    return dict(policies=list(D.POLICIES),models=list(D.MODELS),costs=list(D.COSTS),rules=list(D.P.M.RULES),queries=4,lineages=[1],
        input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()})


def test_full_fixture_proper_mixture_scores_costs_and_reader_roles(tmp_path):
    cfg=fixture(tmp_path);result=D.run(tmp_path,dict(design=cfg),lambda **kw:None)
    assert len(result['cells'])==144 and result['fits']==0
    lookup={(r['rule'],r['model'],r['policy'],r['weighting'],r['cost']):r for r in result['cells']}
    for r in result['cells']:
        if r['policy']=='all-fields':assert r['finite_loss_contribution']==0 and r['residual_ambiguous_mass']==0 and r['net_finite_loss']==2*r['cost']
        assert abs(r['net_finite_loss']-r['finite_loss_contribution']-r['cost']*r['fields_requested'])<1e-12
        if r['policy']=='equal-request':
            for metric in ('finite_loss_contribution','infinite_loss_mass','squared_error','true_probability'):
                sides=[lookup[(r['rule'],r['model'],p,r['weighting'],r['cost'])][metric] for p in ('skill','belief')]
                assert abs(r[metric]-sum(sides)/2)<1e-12
    for p in (tmp_path/'reader').glob('*.json'):
        row=read(p);assert set(row)<={'initial','operations','requested_field','disclosed_value'}
    bad=tmp_path/'bad';cfg=fixture(bad,corrupt=True)
    with pytest.raises(ValueError,match='truth'):D.run(bad,dict(design=cfg),lambda **kw:None)
