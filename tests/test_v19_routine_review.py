import copy
import gzip
from itertools import product
import pytest
from ghostscale.validation.soundingline.v19 import routine_review as V
from ghostscale.validation.soundingline.v19 import routine_revision as R, local_world as L
from ghostscale.validation.soundingline.v18_3.io import canonical, read, write, file_digest


def test_independent_known_answers_and_masking():
    assert all(V.controls().values())
    for m in L.MAKERS:
        for request in (0,1):
            rows=[V.distribution(m,(1,0,a),(0,1,b),request,True,'visible-task',(1.,.75,.2)) for a,b in product(range(2),repeat=2)]
            assert all(r==rows[0] for r in rows)
            assert m[1] or all(op!='accept-tool' for _,op in rows[0])


def test_policy_and_outcome_corruption_is_rejected():
    m=(0,1,0,0);a=(0,0,0);b=(1,1,1);coeff=(1.,.75,.2)
    outcome=dict(goal=None,operation='undo',probability=1.,**V.outcome(m,a,b,0,'native','undo'))
    # Correct legal outcome with the wrong policy is still invalid.
    row=dict(arm='native',outcomes=[outcome],**{k:outcome[k] for k in V.METRICS})
    with pytest.raises(ValueError,match='support'):V.row_check(row,m,a,b,0,False,coeff)
    row=dict(arm='keep',outcomes=[dict(goal=None,operation='inspect',probability=1.,**V.outcome(m,a,b,0,'keep','inspect'))])
    row.update({k:row['outcomes'][0][k] for k in V.METRICS})
    V.row_check(row,m,a,b,0,False,coeff)
    row['outcomes'][0]['after']=[1,1,1]
    with pytest.raises(ValueError,match='artifact'):V.row_check(row,m,a,b,0,False,coeff)


def test_full_native_fixture_and_weighted_regroup(tmp_path):
    original=tmp_path/'inputs/original';(original/'inputs').mkdir(parents=True)
    lineage=190968;records=L.enumerate_world(L.law(lineage))
    p=original/'inputs'/f'lineage-{lineage}_points.json.gz'
    p.write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(arms=list(R.ARMS),lineages=[lineage],paths_per_lineage=13824,operation_cost=.05,inspection_cost=.01,input_files={p.name:file_digest(p)})
    write(original/'PLAN.json',{'design':cfg})
    summary=R.run(original,{'design':cfg},lambda **kw:None);write(original/'SUMMARY.json',summary)
    review=dict(input_files={},target_plan_sha256=file_digest(original/'PLAN.json'),bootstrap_seed=190501,bootstrap_resamples=20)
    verified=V.run(tmp_path,{'design':review},lambda **kw:None)
    assert all(verified['controls'].values()) and verified['paths']==13824 and verified['cells']==160
    assert verified['max_error']<1e-12
    regroup=read(tmp_path/'INDEPENDENT_REGROUP.json')
    assert len(regroup['contrasts'])==864 and len(regroup['means'])==180
    # Reconstruct pooled values from the original native strata and their masses.
    for row in regroup['means']:
        if row['purpose'] is not None:continue
        subset=[r for r in summary['cells'] if r['arm']==row['arm'] and r['changed_request']==row['changed_request'] and r['hidden_presentation']==row['hidden_presentation']]
        for metric in V.METRICS:
            expected=sum(r['population_mass']*r[metric] for r in subset)/sum(r['population_mass'] for r in subset)
            assert abs(row[metric]-expected)<1e-12
