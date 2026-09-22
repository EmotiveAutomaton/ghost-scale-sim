import gzip
from itertools import product
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import input_privilege as P
from ghostscale.validation.soundingline.v19 import transfer_review as V
from ghostscale.validation.soundingline.v18_3.io import canonical, write, read, file_digest


def test_known_answer_live_and_null_controls():
    assert all(P.controls().values())


def test_legal_completions_keep_zero_policy_support_and_exclude_unskilled_tool():
    q=(0,0,2,0,4,4)
    c=P.completions(P.visible(q,'omit-both'))
    assert len(c)==4 and {v[:2] for v in c}==set(product(range(2),repeat=2))
    assert len(P.completions(P.visible((1,0,2,3,4,4),'omit-both')))==2


def test_independent_executor_for_all_metadata_and_short_operation_compositions():
    for s,b,a,ops,rule in product(range(2),range(2),range(8),product(range(6),repeat=3),P.M.RULES):
        if s==0 and 3 in ops:continue
        q=(s,b,a,*ops)
        assert P.T.oracle(q,rule)==V.endpoint(q,rule)


def test_independent_grouping_probability_entropy_and_mass():
    qs=[(s,b,2,1,4,4) for s,b in product(range(2),repeat=2)]
    mass=np.array([.1,.2,.3,.4]);targets=np.array([V.endpoint(q,'original') for q in qs])
    for view,groups in P.membership(qs).items():
        rows,u,n=P.evaluate(qs,groups,targets,mass,'original')
        for i,q in enumerate(qs):
            matching=[j for j,r in enumerate(qs) if P.visible(q,view)==P.visible(r,view)]
            expected=np.array([sum(mass[j] for j in matching if targets[j]==k)/sum(mass[j] for j in matching) for k in range(8)])
            assert np.allclose(n[i],expected)
        assert math.isclose(sum(r['native_mass'] for r in rows),1)
        score=P.proper(n,targets,mass)
        assert math.isclose(score['finite_loss_contribution'],sum(r['native_mass']*r['native_entropy'] for r in rows))
        assert score['infinite_loss_mass']==0


def test_membership_corruption_and_probability_failures():
    qs=[(1,0,2,1,4,4),(1,1,2,1,4,4)]
    groups=P.membership(qs)['omit-belief'];row=next(iter(groups.values()))
    with pytest.raises(ValueError,match='duplicate'):P.membership(qs*2)
    with pytest.raises(ValueError,match='population'):P.evaluate(qs,groups,[0,2],[1.,1.],'original')
    row['indices']=[0]
    with pytest.raises(ValueError,match='coverage'):P.evaluate(qs,groups,[0,2],[.5,.5],'original')
    row['indices']=[0,1];row['legal_completions']=row['legal_completions'][:1]
    with pytest.raises(ValueError,match='legal'):P.evaluate(qs,groups,[0,2],[.5,.5],'original')


def test_zero_native_target_is_explicit_under_equal_query_scores():
    p=np.array([[1.,0,0,0,0,0,0,0],[1.,0,0,0,0,0,0,0]])
    result=P.proper(p,np.array([0,1]),np.array([.5,.5]))
    assert result['infinite_loss_mass']==.5 and result['finite_loss_contribution']==0


def test_complete_native_fixture_and_reader_allowlist(tmp_path):
    base=tmp_path/'inputs';(base/'raw').mkdir(parents=True)
    qs=[(s,b,2,1,4,4) for s,b in product(range(2),repeat=2)]
    truth=[dict(query=list(q),targets={r:V.endpoint(q,r) for r in P.M.RULES}) for q in qs]
    write(base/'QUERY_TRUTH.json',truth);write(base/'MEMBERSHIP.json',P.membership(qs))
    for rule in P.M.RULES:
        records=[dict(maker=[0,q[0],q[1],0],initial=list(P.R.ARTIFACTS[q[2]]),steps=[dict(operation=P.R.L.OPERATIONS[o]) for o in q[3:]],final=list(P.R.ARTIFACTS[V.endpoint(q,rule)]),probability=.25) for q in qs]
        (base/'raw'/f'1-{rule}_points.json.gz').write_bytes(gzip.compress(canonical(records),mtime=0))
    pins={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}
    cfg=dict(views=list(P.VIEWS),rules=list(P.M.RULES),queries=4,lineages=[1],input_files=pins)
    result=P.run(tmp_path,dict(design=cfg),lambda **kw:None)
    assert all(result['controls'].values()) and len(result['cells'])==32 and result['group_rows']==18
    for p in (tmp_path/'reader').glob('*.json'):
        r=read(p);assert set(r)<=set(('skill','belief_error','initial','operations'))
        assert {'initial','operations'}<=set(r)
    for cell in result['cells']:
        if cell['view']=='full':assert cell['finite_loss_contribution']==0 and cell['legal_ambiguous_groups']==0
    for n,h in pins.items():assert file_digest(base/n)==h
