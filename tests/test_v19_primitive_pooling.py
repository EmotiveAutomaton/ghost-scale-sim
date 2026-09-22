import gzip
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import primitive_pooling as P
from ghostscale.validation.soundingline.v19 import rollout_transfer as T
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_known_semantics_and_negative_pool_controls():
    assert all(P.controls().values())


def test_empirical_pooling_and_single_prior_independent_scalar_sum():
    counts=np.ones((2,2,8,8,6,8))
    for a,u in product(range(8),range(8)):
        counts[1,0,a,u,3,(a+u)%8] += a+u+1
    for arm,axis in (('pool-undo',3),('pool-current-wrong',2)):
        revised,info=P.pooled_counts(counts,[],arm)
        for a,u,t in product(range(8),range(8),range(8)):
            key=[1,0,a,u,3]
            expected=1.
            for i in range(8):
                source=key.copy();source[axis]=i
                expected += counts[tuple(source)][t]-1
            assert revised[1,0,a,u,3,t] == expected
        assert info['total_tool_groups']==16
        assert np.array_equal(revised[0],counts[0])
        assert np.array_equal(revised[:,:,:,:,5],counts[:,:,:,:,5])


def test_alias_reports_keep_distinct_inputs_but_deduplicate_sources():
    counts=np.ones((2,2,8,8,6,8));obs=[((1,0,2,u,3),3) for u in range(8)]
    revised,info=P.pooled_counts(counts,obs*2,'pool-undo')
    assert info['unique_inputs']==8 and info['observed_groups']==1
    assert revised[1,0,2,0,3,3]==9 and revised[1,0,2,0,3].sum()==16
    with pytest.raises(ValueError,match='conflicting'):
        P.pooled_counts(counts,[obs[0],(obs[0][0],4)],'pool-undo')
    with pytest.raises(ValueError,match='unit-prior'):
        P.pooled_counts(counts*.5,[],'unpooled')


def test_wrong_pool_retains_disagreement_and_prior_once():
    counts=np.ones((2,2,8,8,6,8))
    revised,info=P.pooled_counts(counts,[((1,0,0,0,3),2),((1,0,1,0,3),3)],'pool-current-wrong')
    assert info['observed_groups']==1 and np.array_equal(revised[1,0,0,0,3],[1,1,2,2,1,1,1,1])


def test_complete_handler_public_roles_and_fixed_unpooled_identity(tmp_path):
    root=tmp_path/'job';inputs=root/'inputs';inputs.mkdir(parents=True)
    qs=[(1,0,2,3,4,4),(1,1,7,3,5,4),(0,1,3,0,1,5)]
    counts=np.ones((2,2,8,8,6,8)); counts[1,0,2,2,3,3]+=2
    write(inputs/'QUERY_TRUTH.json',[dict(query=list(q),targets={r:T.oracle(q,r) for r in P.F.M.RULES}) for q in qs])
    for folder in ('models','forecasts','raw'):(inputs/folder).mkdir()
    for mode in ('original','composition-holdout'):
        np.savez(inputs/'models'/f'1-{mode}.npz',transition_counts=counts)
        baseline=np.array([P.F.S.smooth(P.F.R.propagate(P.F.R.normalized(counts),q)) for q in qs])
        np.savez(inputs/'forecasts'/f'1-{mode}.npz',queries=qs,**{'learned-exact':baseline})
        write(inputs/'forecasts'/f'1-{mode}-support.json',[dict(query_seen=False,withheld_composition=False,all_primitives_seen=False) for q in qs])
    for rule in P.F.M.RULES:
        rows=[dict(maker=[0,q[0],q[1],0],initial=list(P.F.R.ARTIFACTS[q[2]]),steps=[dict(operation=P.F.L.OPERATIONS[o]) for o in q[3:]],final=list(P.F.R.ARTIFACTS[T.oracle(q,rule)]),probability=1/3) for q in qs]
        (inputs/'raw'/f'2-{rule}_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    pins={p.relative_to(inputs).as_posix():file_digest(p) for p in inputs.rglob('*') if p.is_file()}
    cfg=dict(arms=list(P.ARMS),budgets=list(P.BUDGETS),orders=list(P.F.ORDERS),feedback=list(P.F.FEEDBACK),epsilon=1/32,saved_prior=1,input_files=pins,queries=len(qs),lineages=[2],training_draws=[1])
    result=P.run(root,dict(design=cfg),lambda **kw:None)
    assert result['fits']==0 and result['score_rows']==288 and all(result['controls'].values())
    for row in read(root/'evaluator/FEEDBACK_MAP.json'):
        public=read(root/'reader'/f"{row['reader_id']}.json")
        assert len(public)==row['budget'] and all(set(r)=={'skill','belief_error','current','undo_buffer','operation','next_artifact'} for r in public)
    for row in read(root/'evaluator/GROUP_COUNTS.json'):
        assert row['unique_inputs']==row['budget'] and row['prior_per_outcome_per_group']==1
    for n,h in pins.items():assert file_digest(inputs/n)==h
