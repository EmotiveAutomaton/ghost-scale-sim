"""Known-answer public mechanics and frozen-distribution restriction controls."""
from itertools import product
import copy
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import joint_support as S, joint_review as V
from ghostscale.validation.soundingline.v18_3.io import write,digest,file_digest


@pytest.fixture(scope='module')
def index():
    return S.legal_index()[0]


def packet(tier='E0',initial=(0,1,0),ops=(4,4,4),skill=0,belief=0):
    art=prev=initial;events=[]
    for i,op in enumerate(ops):
        after=V.execute(art,prev,op,skill,belief)
        events.append(dict(step=i,operation=V.OPS[op],before=list(art),after=list(after),tool_proposal=list(after) if op==3 else None))
        prev,art=art,after
    a=dict(artifact=list(art))
    if tier!='E0':a.update(initial=list(initial),requested_purpose=0)
    if tier.startswith('E2'):a['observations']=events[:1] if tier=='E2-sparse' else events
    return dict(schema='v19.local.public.1',tier=tier,inputs=a)


def test_complete_witness_retains_all_goals(index):
    p=packet('E2-full',ops=(0,1,5));mask=S.support(p,index)
    assert mask.sum()==27
    assert set(np.flatnonzero(mask)%216)=={11}
    assert set(np.flatnonzero(mask)//216)==set(range(27))


def test_artifact_aliases_and_hidden_start_union(index):
    mask=S.support(packet(),index)
    assert mask[0] and mask[172]  # edit-edit-edit from another start; inspect-inspect-inspect
    assert mask.sum()>27
    assert np.all(S.support(packet('E1'),index)<=mask)


def test_undo_and_tool(index):
    for ops in ((0,1,5),(3,5,3),(5,5,5)):
        mask=S.support(packet('E2-full',ops=ops,skill=1),index)
        assert mask.sum()==27 and mask[sum(op*6**(2-i) for i,op in enumerate(ops))]


def test_request_does_not_filter_goals(index):
    a=packet('E1');b=copy.deepcopy(a);b['inputs']['requested_purpose']=1
    assert np.array_equal(S.support(a,index),S.support(b,index))


@pytest.mark.parametrize('field',['goal','maker','policy','probability'])
def test_hidden_fields_rejected(index,field):
    p=packet();p['inputs'][field]=0
    with pytest.raises(ValueError):S.support(p,index)


def test_inconsistent_witness_empty(index):
    p=packet('E2-full');p['inputs']['artifact']=[1,1,1]
    assert not S.support(p,index).any()


def test_controls_and_uniform():
    assert all(S.controls().values())
    p=np.ones((1,5))/5;mask=np.array([[1,0,1,0,1]],bool)
    q,_,_=S.restrict(p,mask)
    assert np.array_equal(q,mask/3)


def test_fallback_retained():
    p=np.array([[1.,0.]])
    for mask in (np.array([[False,True]]),np.array([[False,False]])):
        q,z,valid=S.restrict(p,mask)
        assert not valid[0] and z[0]==0 and np.array_equal(q,p)


def test_false_exclusion_fails():
    refs=[dict(lineage=1,tier='E0',frames=[dict(frame='a',mass=1.,target=[[0,1.]])])]
    with pytest.raises(ValueError,match='true path excluded'):
        S.reference_arrays(refs,'E0',['a'],[1],np.zeros((1,5832),bool))


def test_full_universe_scores_reproduce_original():
    a=np.array([0,1,11]);p=S.expand(np.array([[.4,.3,32/33-.7]]),a,32)
    truth={0:.2,11:.4,333:.4}
    refs=[dict(lineage=1,tier='E0',frames=[dict(frame='a',mass=1.,target=list(truth.items()))])]
    r=S.reference_arrays(refs,'E0',['a'],[1],np.ones((1,5832),bool))
    m=S.scores(p,a,r);expected=V.score(p[0,a],a,truth,32)
    for k,v in expected.items():assert np.isclose(m[k][0],v,rtol=0,atol=1e-12),k
    mask=np.zeros_like(p,bool);mask[0,[0,11,333]]=True;q,_,_=S.restrict(p,mask)
    m=S.scores(q,a,r)
    assert np.isclose(m['loss'][0],-sum(v*np.log(q[0,k]) for k,v in truth.items()))
    assert np.isclose(m['compatible_mass'][0],1)


def test_unknown_only_support_uses_tail():
    p=S.expand(np.array([[32/33]]),[0],32);mask=np.zeros_like(p,bool);mask[0,[1,2]]=True
    q,_,valid=S.restrict(p,mask);assert valid[0] and np.allclose(q[0,[1,2]],.5)
    r=S.reference_arrays([dict(lineage=1,tier='E0',frames=[dict(frame='a',mass=1.,target=[[1,1.]])])],'E0',['a'],[1],mask)
    m=S.scores(q,[0],r)
    assert m['candidate_size'][0]==2 and m['candidate_coverage'][0]==1 and m['top_incompatible'][0]==0


def test_complete_small_handler(tmp_path):
    root=tmp_path/'packet';base=root/'inputs';(base/'reader').mkdir(parents=True);(base/'evaluator').mkdir();(base/'forecasts').mkdir()
    cfg=dict(tiers=['E0','E2-full'],budgets=[32,128],training_draws=[1],fit_seeds=[2],development_lineages=[3],bootstrap_seed=191003,bootstrap_resamples=10000)
    write(base/'PLAN.json',dict(design=cfg))
    packets={};refs=[];cells=[]
    for tier in cfg['tiers']:
        p=packet(tier);key=digest(p);packets[key]=p
        truth={172:.75,388:.25};refs.append(dict(lineage=3,tier=tier,frames=[dict(frame=key,mass=1.,target=list(truth.items()))]))
        for budget,arm in product(cfg['budgets'],S.ARMS):
            stem=f'1-{tier}-2-{budget}-{arm}';pred=np.array([[.6,budget/(budget+1)-.6]]);alphabet=np.array([172,388])
            np.savez_compressed(base/'forecasts'/(stem+'.npz'),probabilities=pred,alphabet=alphabet)
            write(base/'forecasts'/(stem+'-frames.json'),[key])
            cells.append(dict(draw=1,tier=tier,seed=2,budget=budget,arm=arm,lineage=3,frames=1,**{k:float(v) for k,v in V.score(pred[0],alphabet,truth,budget).items()}))
    write(base/'reader/PACKETS.json',dict(packets=packets));write(base/'evaluator/REFERENCES.json',refs);write(base/'SUMMARY.json',dict(cells=cells))
    cfg=dict(cfg,input_files={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()})
    result=S.run(root,dict(design=cfg),lambda **kw:None)
    assert result['original_cells_reproduced']==16 and result['rows']==36 and all(result['controls'].values())
    assert result['enumeration']['executions']==5456
