import copy,json,subprocess,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import pytest
from ghostscale.validation.soundingline.v25 import wb1,misspec
from ghostscale.validation.soundingline.v25.common import close,write,digest


def payload(n=7):
    return dict(schema=wb1.SCHEMA,complete=True,controls_passed=True,effect_basis='standardized_mean',
        source=dict(project='SoundingLine',claim_id='SL-WB1-fixture',result_sha256='0'*64),
        classes=[dict(id=f'class-{i}',events=10+i,effects=dict(a=float(i),b=2.*i,c=-4.*i,d=1.)) for i in range(n)])


def test_class_cap_spread_and_terciles_are_deterministic():
    p=payload();result=wb1.calibrated_library(p,wb1.CUTOFF)
    assert result['class_ids']==[f'class-{i}' for i in range(6,0,-1)]
    assert result['measures']==['c','b','a']
    assert result['effects'][0]==[0,1,1] and result['effects'][-1]==[1,0,0]
    p['classes'].reverse();assert wb1.calibrated_library(p,wb1.CUTOFF)==result


@pytest.mark.parametrize('change',[
    lambda p:p.update(complete=False),lambda p:p.update(controls_passed=False),
    lambda p:p.update(raw_text='not an aggregate'),
    lambda p:p['classes'][0]['effects'].update(a=float('nan')),
    lambda p:p['classes'][0].update(events=0),
    lambda p:p['classes'][0]['effects'].pop('a'),
    lambda p:p['classes'][0].update(id=p['classes'][1]['id'])])
def test_invalid_or_incomplete_delivery_cannot_calibrate(change):
    p=payload();change(p)
    with pytest.raises(ValueError):wb1.calibrated_library(p,wb1.CUTOFF)


def test_cutoff_and_equal_effects_do_not_invent_information():
    p=payload(1)
    for key in p['classes'][0]['effects']:p['classes'][0]['effects'][key]=3.
    assert wb1.calibrated_library(p,wb1.CUTOFF)['effects']==[[0,0,0]]
    with pytest.raises(ValueError):wb1.calibrated_library(p,wb1.CUTOFF+timedelta(microseconds=1))
    with pytest.raises(ValueError):wb1.calibrated_library(p,datetime(2026,10,27))


@pytest.mark.parametrize('n',[1,3,6])
def test_constructed_full_world_reconstruction_and_true_reader_null(n):
    setting=dict(beta=3,calibrated_library=wb1.calibrated_library(payload(n),wb1.CUTOFF))
    original=misspec.block(setting,['excluded-wb1-fixture',n])
    independent=misspec.block(setting,['excluded-wb1-fixture',n],True)
    assert original['truth_and_observations']==independent['truth_and_observations']
    close(original,independent,tol=1e-9)
    assert len(original['truth_and_observations'])==8
    assert all(len(x['observations'])==512 for x in original['truth_and_observations'])
    assert all(abs(x['metrics']['primary'])<1e-12 for x in original['readouts'] if x['labels']['reader']=='true')
    assert original['public']['merge_available']==(n>=2)


def test_fresh_process_replay_and_controls(tmp_path):
    assert wb1.controls()['passed'] and misspec.controls()['passed']
    script="from ghostscale.validation.soundingline.v25 import wb1; from ghostscale.validation.soundingline.v25.common import canonical; import json,sys; print(canonical(wb1.block(json.loads(sys.argv[1]),['excluded-fresh-process-fixture'])).decode())"
    setting=dict(beta=1,calibrated_library=wb1.calibrated_library(payload(3),wb1.CUTOFF))
    first=subprocess.check_output([sys.executable,'-B','-c',script,json.dumps(setting)])
    second=subprocess.check_output([sys.executable,'-B','-c',script,json.dumps(setting)])
    assert first==second


def test_verified_probe_continues_same_calibrated_grid_with_measured_caps(tmp_path,monkeypatch):
    from runners import prepare_v25_wb1 as preparation,run_v25
    from ghostscale.validation.soundingline.v25 import contract
    campaign=tmp_path/'campaign';parent=campaign/'packets/v25-w5-calibrated-probe-1';parent.mkdir(parents=True)
    source=tmp_path/'source';source.mkdir()
    setting=dict(beta=1,calibrated_library=wb1.calibrated_library(payload(3),wb1.CUTOFF))
    plan=dict(family='W5',kind='probe',pool='studies',calibration_variant='wb1-calibrated',settings=[setting,dict(setting,beta=3)],source_root=str(source),numerical_threads=1,priority='below-normal')
    write(parent/'PLAN.json',plan);write(parent/'CONTROLS_BEFORE_DATA.json',dict(passed=True))
    complete=dict(state='VERIFIED',plan_sha256=digest(parent/'PLAN.json'))
    for key,name in [('summary_sha256','original/SUMMARY.json'),('replay_sha256','REPLAY.json'),('review_sha256','INDEPENDENT_REVIEW.json')]:write(parent/name,{});complete[key]=digest(parent/name)
    write(parent/'COMPLETE.json',complete)
    write(parent/'PASS_COSTS.json',dict(includes_world_io=True,setting_seconds=dict(original=[1.,2.],replay=[1.,2.],review=[2.,4.])))
    write(campaign/'accounting'/(parent.name+'.json'),dict(pass_cpu_seconds=dict(original=4.,replay=4.,review=8.),cpu_seconds=16,pool='studies',family='W5'))
    write(campaign/'ACCEPTANCE.json',contract.acceptance('2026-10-25T19:57:34+00:00','fixture'))
    monkeypatch.setattr(preparation,'verify_source',lambda p:None)
    prepared=run_v25.prepare(campaign,'W5','study',parent=parent,source=source)
    result=json.loads((prepared/'PLAN.json').read_text())
    assert prepared.name=='v25-w5-calibrated-study-1'
    assert result['settings']==plan['settings'] and result['blocks']==16
    assert result['pass_caps']==dict(original=107.5,replay=107.5,review=205.)
    assert result['parent_complete_sha256']==digest(parent/'COMPLETE.json')
