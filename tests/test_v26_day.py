"""Excluded finite fixtures; no scientific queue or predecessor state is touched."""
from datetime import datetime,timezone
from functools import lru_cache
import importlib
import numpy as np
import pytest
from ghostscale.validation.soundingline.v26 import day
from ghostscale.validation.soundingline.v26.common import close
from ghostscale.validation.soundingline.v26.day_common import bar_verdict,statistic,summarize

@lru_cache
def controls(f):return importlib.import_module('ghostscale.validation.soundingline.v26.'+day.MODULES[f]).controls()

@pytest.mark.parametrize('family,fixture',[
    ('G1','clipped_deficit_fixture'),('G1','above_setpoint_zero'),('G1','balanced_raw_alias'),('G1','unequal_urgency'),('G1','brute_four_record_initial_and_outcome_enumeration'),('G1','scatter_gather_equal'),('G1','raw_slope_setpoint_prior'),('G1','deterministic_known_path'),('G1','positive_floors'),
    ('G2','no_competing_cue_alias'),('G2','enumerated_frame_sequence'),('G2','no_full_service_alias'),('G2','positive_floor'),
    ('G3','no_gate_identity'),('G3','singleton_honest_transparent'),('G3','unfittable_no_update'),('G3','opaque_log_odds'),('G3','first_gate_record_14'),
    ('G4','closed_form_offer_acceptance'),('G4','future_lock_alias'),('G4','equal_class_priors'),('G4','positive_floor'),
    ('G5','linear_alias'),('G5','asymmetric_slopes'),('G5','smooth_linear_limit'),('G5','equal_class_priors'),('G5','positive_floor')])
def test_known_answer(family,fixture):assert controls(family)[fixture]

def fixture_plan():
    accepted=day.supplement('2026-11-07T00:00:00+00:00','packet','prior',{'studies':42*3600})
    plan=dict(family='G1',kind='study',pool='studies',lane='primary',cpu_cap_seconds=100.,pass_caps=dict(original=30.,replay=30.,review=30.),projected_wall_seconds=200.,numerical_threads=1,priority='below-normal',design_check_sha256='x',controls_sha256='y')
    return accepted,plan,datetime(2026,11,7,1,tzinfo=timezone.utc)

def test_admission_valid_after_friday_target_and_carries_prior_cost():
    a,p,t=fixture_plan();assert day.admission(p,a,t,{'studies':42*3600})
    with pytest.raises(ValueError,match='pool ceiling'):day.admission(p,a,t,{'studies':86*3600-99})
    with pytest.raises(ValueError,match='two shared'):day.admission(p,a,t,{},2)
    with pytest.raises(ValueError,match='outside new-start'):day.admission(p,a,datetime(2026,11,9,4,tzinfo=timezone.utc),{})
    with pytest.raises(ValueError,match='drain margin'):day.admission(p|dict(projected_wall_seconds=3*3600),a,datetime(2026,11,9,3,tzinfo=timezone.utc),{})

def test_family_shedding_texture_and_no_followups():
    a,p,t=fixture_plan()
    with pytest.raises(ValueError,match='shedding required'):day.admission(p,a,t,{'family:G1':2.6*3600-99})
    assert day.admission(p|dict(shedding_exhausted=True),a,t,{'family:G1':2.6*3600-99})
    with pytest.raises(ValueError,match='trigger 1'):day.admission(p|dict(shedding_exhausted=True),a,t,{'family:G1':3.9*3600-99})
    tx=p|dict(kind='texture',lane='texture',pool='followup_healing')
    assert day.admission(tx,a,t,{})
    with pytest.raises(ValueError,match='texture ceiling'):day.admission(tx,a,t,{'texture:G1':6*3600-99})
    with pytest.raises(ValueError,match='uncommissioned'):day.admission(p|dict(kind='followup'),a,t,{})
    with pytest.raises(ValueError,match='healing limit'):day.admission(p|dict(kind='healing',pool='followup_healing',healing_number=2),a,t,{})

def test_censoring_is_not_drop_or_horizon_imputation_and_bars_use_intervals():
    assert np.isinf(statistic([1,None,None],'median'))
    row=summarize([[None]*4 for _ in range(16)],'median','excluded-fixture')
    assert row['estimate'] is None and row['estimate_censored'] and row['upper_censored']
    assert bar_verdict(row,64,'at_most')=='not met'
    assert bar_verdict(dict(interval95=[.04,.06]),.05,'above')=='unresolved at this sample size'

@pytest.mark.parametrize('family',['G2','G3','G4','G5'])
def test_separate_reconstruction_of_excluded_short_trajectory(family):
    card=next(c for c in day.cards() if c['id']==family);s=card['settings'][0]|dict(records=16,makers=1,worlds_per_block=1,sources=2)
    module=importlib.import_module('ghostscale.validation.soundingline.v26.'+day.MODULES[family])
    close(module.block(s,['excluded-reconstruction',family]),module.block(s,['excluded-reconstruction',family],True),tol=1e-9)

def test_monitor_fixed_cadence_and_final_deadline(tmp_path):
    from runners.watch_v26_day import scheduled
    from ghostscale.validation.soundingline.v26.common import write,read
    write(tmp_path/'ACCEPTANCE.json',dict(opened='2026-11-07T00:00:00+00:00',new_starts_close='2026-11-09T04:00:00+00:00',drain='2026-11-09T06:00:00+00:00'))
    scheduled(tmp_path,datetime(2026,11,7,11,tzinfo=timezone.utc));assert not list((tmp_path/'events').glob('*'))
    scheduled(tmp_path,datetime(2026,11,7,12,tzinfo=timezone.utc));assert read(tmp_path/'events/day-health12-01.json')['kind']=='twelve_hour_health'
    scheduled(tmp_path,datetime(2026,11,9,6,tzinfo=timezone.utc));assert (tmp_path/'STOP').exists()
    assert len(list((tmp_path/'events').glob('day-health12*')))==4

def test_failed_predecessor_does_not_deadlock_later_primaries(tmp_path):
    from runners.run_v26_day import primary_terminal
    from ghostscale.validation.soundingline.v26.common import write
    write(tmp_path/'packets/v26-day-20261106-g1-probe/STATUS.json',dict(state='FAILED'))
    assert primary_terminal(tmp_path,'G1',dict(dispositions=[]))
    assert not primary_terminal(tmp_path,'G2',dict(dispositions=[]))

def test_g2_conflict_is_for_the_cued_competitor():
    from ghostscale.validation.soundingline.v26.day_identity import design_check
    card=next(c for c in day.cards() if c['id']=='G2');result=design_check(card['settings'][0],'excluded-denominator-control')
    assert .25<result['checks']['conflict_share']<.40


def test_contamination_after_real_gate_has_independent_full_horizon_check():
    from ghostscale.validation.soundingline.v26.day_closure import library,trajectory,review_trajectories,reconstruct_one
    card=next(c for c in day.cards() if c['id']=='G3')
    setting=card['settings'][0]|dict(theta=.5,delta=.1)
    items=np.array([[1,0]*32],dtype=np.int8);truth=np.array([0]);lib=library()
    args=(items,truth,1,.5,.1,16,1.,False,*lib)
    result,when=trajectory(*args);review,rwhen=review_trajectories(*args)
    slow,swhen=reconstruct_one(items[0],0,1,setting)
    assert when[0]==rwhen[0]==swhen==14
    assert np.allclose(result,review,atol=1e-12) and np.allclose(result[0],slow,atol=1e-12)
    undrifted,_=trajectory(items,truth,1,.5,0.,16,1.,False,*lib)
    assert abs(result[0,1]-undrifted[0,1])<1e-12
    assert abs(result[0,0]-undrifted[0,0])>.001


def test_report_conjunction_and_uncertain_contamination_threshold():
    from ghostscale.validation.soundingline.v26.day_reporting import conjunction,featured,primary_readout
    assert conjunction([dict(verdict='met'),dict(verdict='unresolved at this sample size')])=='unresolved at this sample size'
    assert conjunction([dict(verdict='not met'),dict(verdict='unresolved at this sample size')])=='not met'
    summary=dict(family='G3',settings=[dict(theta=.7,delta=.05),dict(theta=.7,delta=.1)],cells=[])
    for i,ci in enumerate(([-.01,.01],[.01,.03])):
        summary['cells'].append(dict(setting=i,labels=dict(source='M',truth=0),metric='D_minus_K_logscore',estimate=sum(ci)/2,interval95=ci,estimate_censored=False,lower_censored=False,upper_censored=False))
    wording=primary_readout(summary,featured(summary))
    assert 'above zero: 0.1' in wording and 'unresolved tested levels: 0.05' in wording
