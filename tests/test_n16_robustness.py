import copy
from datetime import datetime, timezone, timedelta
import math

import pytest

from ghostscale.validation.soundingline.v23.common import write
from ghostscale.validation.soundingline.v25 import n16_robustness as n
from runners import run_n16_robustness as run


def fixture(p=.8,q=.2):
    return dict(posteriors=[dict(scenario=s,policy='P'+str(k),mass=[[q if s=='correct' and k==1 else p,1-(q if s=='correct' and k==1 else p)]]*6) for s in ('correct','omit-0','omit-1','high-reliability') for k in range(1,6)])


def blocks(count=2):
    return [dict(block_index=i,excluded=False,cases=n.CASES,scientific_identity=n.IDENTITY,cells=[dict(source=s,condition=c,policy=p,probes=b,cases=960 if s!='pooled' else 3840,mean=float(i)) for s in (*n.SOURCES,'pooled') for c in n.CONDITIONS for p in n.POLICIES for b in n.BUDGETS]) for i in range(count)]


def test_known_answers_and_native_controls():
    assert n.controls()['passed']


def test_chooser_error_keeps_correct_model_fixed_comparator():
    saved=fixture()
    # Omitted chooser's P1 must not replace the common correct-model P1 reference.
    for row in saved['posteriors']:
        if row['scenario']=='omit-0' and row['policy']=='P1': row['mass']=[[.99,.01]]*6
    got=n.contrasts(saved,0)
    assert got['omit-truth|P4|2']==pytest.approx(math.log(3.7))
    assert got['correct|P4|2']==got['omit-truth|P4|2']


def test_independent_arithmetic_preserves_losses_and_support():
    for p,q in ((.02,.8),(.8,.02),(.4,.4),(1e-40,.9),(.9,1e-40)):
        a=n.contrasts(fixture(p,q),0);b=n.contrasts(fixture(p,q),0,True)
        assert a==pytest.approx(b,abs=1e-12)
        assert min(a.values())>=math.log(.1)-1e-12
    assert max(n.contrasts(fixture(.02,.8),0).values())<0


def test_missing_condition_and_zero_reference_fail():
    missing=fixture();missing['posteriors']=[r for r in missing['posteriors'] if r['scenario']!='omit-1']
    with pytest.raises(ValueError,match='roster'):n.contrasts(missing,0)
    with pytest.raises(ValueError,match='support'):n.contrasts(fixture(.8,0),0)


def test_normalization_roundoff_preserves_values_and_rejects_invalid_mass():
    saved=fixture(1.,1.)
    for row in saved['posteriors']:row['mass']=[[1.0000000000000002,0.]]*6
    assert all(abs(v)<1e-12 for v in n.contrasts(saved,0).values())
    assert all(abs(v)<1e-12 for v in n.contrasts(saved,0,True).values())
    assert saved['posteriors'][0]['mass'][0][0]==1.0000000000000002
    saved['posteriors'][0]['mass'][0]=[1.01,0.]
    saved['posteriors'][0]['mass'][1]=[1.01,0.]
    with pytest.raises(ValueError,match='normalized'):n.contrasts(saved,0)


def test_new_contrasts_match_the_retained_stress_estimand():
    from ghostscale.validation.soundingline.v24.n16_followup import calculate,SCENARIOS
    saved=fixture(.02,.8);reference,_=calculate(saved,dict(mechanism=0))
    result=n.contrasts(saved,0)
    mapping={'truth-omitted':'omit-truth','other-omitted':'omit-other'}
    ordered=[result[f'{mapping.get(s,s)}|{p}|{b}'] for s in SCENARIOS for p in n.POLICIES for b in n.BUDGETS]
    assert ordered==pytest.approx(reference,abs=1e-12)


def test_whole_block_intervals_do_not_count_cases_as_replicates():
    result=n.aggregate_blocks(blocks())
    assert result['state']=='INTERIM_DESCRIPTIVE_ONLY'
    assert result['cells'][0]['blocks']==2
    assert result['cells'][0]['standard_error']==pytest.approx(.5)
    assert result['cells'][0]['mean']==.5
    assert result['confirmation'] is False
    assert n.aggregate_blocks(blocks(64))['state']=='COMPLETE_FIXED_SIZE_DESCRIPTIVE'


@pytest.mark.parametrize('fault',['excluded','identity','gap','missing-cell','duplicate-cell'])
def test_invalid_analysis_coverage_fails(fault):
    rows=blocks()
    if fault=='excluded':rows[0]['excluded']=True
    elif fault=='identity':rows[0]['scientific_identity']='v24-n16-confirmation-1'
    elif fault=='gap':rows[1]['block_index']=2
    elif fault=='missing-cell':rows[0]['cells'].pop()
    else:rows[0]['cells'][-1]=copy.deepcopy(rows[0]['cells'][0])
    with pytest.raises(ValueError):n.aggregate_blocks(rows)


def test_duplicate_balanced_case_is_rejected():
    records=[]
    values={f'{c}|{p}|{b}':0. for c in n.CONDITIONS for p in n.POLICIES for b in n.BUDGETS}
    for law in range(16):
        for variant in n.n16.model.variants():
            for case in range(5):records.append(dict(identity=[0,law,variant['id'],case],source=variant['source'],mixture_gain=0.,contrasts=values))
    plan=dict(block_index=0,kind='robustness',science_identity=n.IDENTITY)
    assert n.summarize(records,plan)['primary_mean']==0
    records[-1]=records[0]
    with pytest.raises(ValueError,match='duplicate'):n.summarize(records,plan)


def test_slot_reservation_and_deadline_limits(tmp_path,monkeypatch):
    current=datetime.now(timezone.utc)
    write(tmp_path/'ACCEPTANCE.json',dict(drain=(current+timedelta(hours=1)).isoformat(),extension_cpu_seconds=1000,protected_cpu_seconds=100,prior_charged_cpu_seconds=500,total_ceiling_seconds=1500))
    monkeypatch.setattr(run.science,'validate',lambda p:None)
    monkeypatch.setattr(run,'verify_source',lambda p:None)
    (tmp_path/'accounting').mkdir()
    plan=dict(kind='excluded',priority='below-normal',numerical_threads=1,measured_wall_seconds=10,cpu_cap_seconds=200)
    run.admission(tmp_path,plan,1,current)
    with pytest.raises(ValueError,match='Gear 2'):run.admission(tmp_path,plan,2,current)
    with pytest.raises(ValueError,match='safe block'):run.admission(tmp_path,plan,0,current+timedelta(minutes=59,seconds=50))
    write(tmp_path/'accounting/other.json',dict(state='RESERVED',reserved_cpu_seconds=800))
    with pytest.raises(ValueError,match='ceiling'):run.admission(tmp_path,plan,0,current)
