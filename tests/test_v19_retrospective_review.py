from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import retrospective_review as R
from ghostscale.validation.soundingline.v19 import retrospective_quotient as Q
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from test_v19_retrospective_quotient import spec,law,brute


@pytest.fixture(scope='module')
def complete():
    s=spec();p=law();a,m=R.reconstruct(s);k,_=R.scalar_kernel(p)
    return s,p,a,m,k,R.reconstruct_row(a,p,k)


@pytest.mark.parametrize('mode',['random','uniform','impossible','disjoint'])
def test_independent_against_exhaustive_scalar(mode):
    s=spec();p=law()
    if mode=='uniform':p[:]=.125
    if mode=='impossible':p[:,:,:2]=0;p/=p.sum(-1,keepdims=True)
    if mode=='disjoint':
        p[:]=0
        for i in range(16):p[i,:,i%8]=1
    a,m=R.reconstruct(s);aa,mm=Q.structure(s);R.check_structure(a,m,aa,mm)
    k,error=R.scalar_kernel(p);R.check_kernel(k,Q.likelihood_kernel(p));assert error<1e-12
    row=R.reconstruct_row(a,p,k);R.check_row(row,Q.evaluate(aa,p))
    for key,value in brute(s,p).items():
        if key.startswith('max'):assert row[key]==pytest.approx(value,abs=2e-15)
        else:assert row[key]==value


def test_permutations_keep_complete_denominators(complete):
    s,p,_,_,k,row=complete;s=deepcopy(s);order=[6,2,4,0,5,1,3]
    s['hypotheses']=[s['hypotheses'][i] for i in order];s['membership']=[s['membership'][i] for i in order]
    s['signatures']=s['signatures'][::-1];s['membership']=[len(s['signatures'])-1-g for g in s['membership']]
    a,m=R.reconstruct(s);aa,mm=Q.structure(s);R.check_structure(a,m,aa,mm)
    assert R.reconstruct_row(a,p,k)==row


def test_stationary_and_same_time(complete):
    _,p,_,_,k,_=complete
    s=dict(hypotheses=[('none',0,m) for m in range(16)],length=4,checkpoint=2,signatures=[[m]*3 for m in range(16)],membership=list(range(16)))
    a,m=R.reconstruct(s);assert m['report_comparisons']==0
    assert R.reconstruct_row(a,p,k)['exact_forecast_witnesses']==0
    for m in range(16):assert not k['quotient_distance'][m,m].any()


@pytest.mark.parametrize('field',['pairs','pair_groups','past','schedules','mapping','masks','blocks'])
def test_structure_corruption_fails(complete,field):
    _,_,a,m,_,_=complete;bad={k:v.copy() for k,v in a.items()};bad[field].flat[0]+=1
    with pytest.raises(ValueError,match='structure'):R.check_structure(a,m,bad,m)


@pytest.mark.parametrize('field',['pair_classes_by_time','third_classes_by_time','pair_count','pair_third_time_comparisons','report_comparisons'])
def test_binding_corruption_fails(complete,field):
    _,_,a,m,_,_=complete;bad=deepcopy(m)
    if field.endswith('by_time'):bad[field][0]['multiplicity'][0]+=1
    else:bad[field]+=1
    with pytest.raises(ValueError,match='bindings'):R.check_structure(a,m,a,bad)


@pytest.mark.parametrize('field',['first_group_weight','second_group_weight','first_report_probability','second_report_probability','feasible','quotient_distance'])
def test_kernel_corruption_fails(complete,field):
    k=complete[4];bad={n:v.copy() for n,v in k.items()}
    if field=='feasible':bad[field].flat[0]=not bad[field].flat[0]
    else:bad[field].flat[0]+=.001
    with pytest.raises(ValueError,match='kernel'):R.check_kernel(k,bad)


@pytest.mark.parametrize('field',['report_comparisons','both_impossible','one_impossible','both_possible','exact_update_witnesses','tolerance_update_witnesses','exact_forecast_witnesses','tolerance_forecast_witnesses','max_quotient_distance','max_forecast_coordinate_difference'])
def test_summary_corruption_fails(complete,field):
    row=complete[5];bad=dict(row);bad[field]+=1
    with pytest.raises(ValueError,match='summary'):R.check_row(row,bad)


@pytest.mark.parametrize('field',['negative','mass','nan','shape'])
def test_invalid_law(field):
    p=law()
    if field=='negative':p[0,0,0]=-.1
    if field=='mass':p[0,0,0]+=.1
    if field=='nan':p[0,0,0]=np.nan
    if field=='shape':p=p[:15]
    with pytest.raises(ValueError):R.scalar_kernel(p)


def test_complete_synthetic_handler_and_corrupted_input(tmp_path):
    producer=tmp_path/'producer';(producer/'inputs').mkdir(parents=True)
    write(producer/'inputs/SCHEDULES.json',{'6-3':spec()});write(producer/'inputs/7-law.json',law().tolist())
    cfg=dict(input_files={p.name:file_digest(p) for p in (producer/'inputs').iterdir()},lengths=[6],checkpoints=[3],lineages=[7])
    plan=dict(design=cfg);write(producer/'PLAN.json',plan)
    result=Q.run(producer,plan,lambda **kw:None);write(producer/'SUMMARY.json',result)
    checker=tmp_path/'checker';(checker/'inputs').mkdir(parents=True)
    shutil.copytree(producer,checker/'inputs/original');shutil.copytree(producer/'inputs',checker/'inputs/parent')
    design=dict(input_files={p.relative_to(checker/'inputs').as_posix():file_digest(p) for p in (checker/'inputs').rglob('*') if p.is_file()},target_plan_sha256=file_digest(producer/'PLAN.json'))
    output=R.run(checker,dict(design=design),lambda **kw:None)
    assert output['passed'] and output['summary_rows']==1 and not output['numerical_acceptance']
    assert read(checker/'RECONSTRUCTED_POINTS.json')==result['cells']
    write(checker/'inputs/parent/7-law.json',np.full((16,4,8),.125).tolist(),immutable=False)
    with pytest.raises(ValueError,match='input binding'):R.run(checker,dict(design=design),lambda **kw:None)


def test_controls():assert all(R.controls().values())
