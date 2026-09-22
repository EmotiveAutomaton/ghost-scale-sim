import gzip
import json
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import primitive_feedback as F
from ghostscale.validation.soundingline.v19 import rollout as R, forward_support as S, rollout_transfer as T
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_known_answer_and_placebo_controls():
    assert all(F.controls().values())


def test_duplicate_input_not_double_evidence_and_conflict_rejected():
    counts = np.ones((2,2,8,8,6,8)); key = F.roster('forward')[0]
    assert np.array_equal(F.update(counts, [(key,2)]*3, 'add-one'), F.update(counts, [(key,2)], 'add-one'))
    with pytest.raises(ValueError, match='conflicting'): F.update(counts, [(key,2),(key,3)], 'add-one')
    with pytest.raises(ValueError, match='unit-prior'): F.update(counts*.5, [], 'unchanged')


def test_replacement_preserves_known_undo_and_stay():
    table = np.zeros((2,2,8,8,6,8))
    from ghostscale.validation.soundingline.v19 import local_world as L
    for skill,belief,a,b,op in product(range(2),range(2),range(8),range(8),range(6)):
        if not skill and op==3: table[skill,belief,a,b,op]=.125
        else: table[skill,belief,a,b,op,R.code(L.execute(R.ARTIFACTS[a],R.ARTIFACTS[b],L.OPERATIONS[op],(0,skill,belief,0)))]=1
    for key in F.roster('forward'): table[key]=np.eye(8)[F.observation(key,'presentation-tool')]
    for q in ((1,0,2,3,5,4),(1,0,2,4,4,4),(1,0,2,3,4,4)):
        assert F.path_sum(table,q)[T.oracle(q,'presentation-tool')]==1
        assert np.array_equal(F.path_sum(table,q),R.propagate(table,q))


def test_summaries_separate_native_weights_from_equal_queries():
    diag=[dict(query_seen=True,withheld_composition=False,all_primitives_seen=True)]*2
    cells=F.cells_for(np.array([2.,0.]),np.zeros(2),np.ones(2),np.array([.9,.1]),diag,np.array([True,False]),{})
    both={r['weighting']:r for r in cells if r['change']=='all' and r['subset']=='all'}
    assert both['native']['loss']==1.8 and both['equal-query']['loss']==1
    assert all(r['loss'] is None for r in cells if r['subset']=='heldout-composition')


def test_complete_handler_and_observed_only_budget_exports(tmp_path):
    root=tmp_path/'job'; inputs=root/'inputs'; inputs.mkdir(parents=True)
    q=(1,0,2,3,4,4); counts=np.ones((2,2,8,8,6,8))
    write(inputs/'QUERY_TRUTH.json',[dict(query=list(q),targets={rule:T.oracle(q,rule) for rule in F.M.RULES})])
    (inputs/'models').mkdir();(inputs/'forecasts').mkdir();(inputs/'raw').mkdir()
    for mode in ('original','composition-holdout'):
        np.savez(inputs/'models'/f'1-{mode}.npz',transition_counts=counts)
        np.savez(inputs/'forecasts'/f'1-{mode}.npz',queries=np.array([q]),**{'learned-exact':np.array([S.smooth(R.propagate(R.normalized(counts),q))])})
        write(inputs/'forecasts'/f'1-{mode}-support.json',[dict(query_seen=False,withheld_composition=False,all_primitives_seen=False)])
    for rule in F.M.RULES:
        r=dict(maker=[0,1,0,0],initial=list(R.ARTIFACTS[2]),steps=[dict(operation=F.L.OPERATIONS[o]) for o in q[3:]],final=list(R.ARTIFACTS[T.oracle(q,rule)]),probability=1.)
        (inputs/'raw'/f'2-{rule}_points.json.gz').write_bytes(gzip.compress(canonical([r]),mtime=0))
    pins={p.relative_to(inputs).as_posix():file_digest(p) for p in inputs.rglob('*') if p.is_file()}
    cfg=dict(arms=list(F.ARMS),budgets=list(F.BUDGETS),orders=list(F.ORDERS),feedback=list(F.FEEDBACK),epsilon=S.EPSILON,saved_prior=1,input_files=pins,queries=1,lineages=[2],training_draws=[1])
    result=F.run(root,dict(design=cfg),lambda **kw:None)
    assert result['fits']==0 and read(root/'PARENT_REPRODUCTION.json')['passed']
    for row in read(root/'evaluator/FEEDBACK_MAP.json'):
        public=read(root/'reader'/f"{row['reader_id']}.json")
        assert len(public)==row['budget']
        assert all(set(p)=={'skill','belief_error','current','undo_buffer','operation','next_artifact'} for p in public)
    for n,h in pins.items(): assert file_digest(inputs/n)==h


def test_nonuniform_three_step_path_sum():
    table=R.normalized(np.arange(1,2*2*8*8*6*8+1,dtype=float).reshape(2,2,8,8,6,8))
    for q in ((1,0,2,3,5,4),(0,1,7,0,1,5)):
        assert np.allclose(F.path_sum(table,q),R.propagate(table,q),atol=1e-12,rtol=0)
