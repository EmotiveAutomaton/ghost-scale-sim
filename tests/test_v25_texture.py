import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v25 import texture,population,followups
from ghostscale.validation.soundingline.v25.common import close


def test_population_64_reproduces_parent_observations_and_every_readout():
    setting=dict(population_size=64,alpha=3,placebo=False)
    identity=['excluded-population64-equivalence']
    parent=population.block(setting,identity)
    new=texture.population_size(setting,identity)
    for key in ('truth','observations','readouts'):close(parent[key],new[key],tol=1e-12)


def test_population_decimal_boundary_preserves_observations_and_reconstructs():
    # Retained failure: scalar and matrix scores straddled .999999995 at 1024.
    setting=dict(population_size=8,alpha=3,placebo=False)
    identity=['v25-w4-followup-1','W4',0,14]
    original=texture.population_size(setting,identity)
    independent=texture.population_size(setting,identity,True)
    close(original,independent,tol=1e-9)
    row=next(r for r in original['readouts'] if r['labels']==dict(reader='estimated',history=1024))
    assert row['metrics']['detection_ge_two']==pytest.approx(.10774410774410774,abs=1e-12)


def test_population_precision_uses_counts_not_the_rounded_input_score(monkeypatch):
    from ghostscale.validation.soundingline.v25.population_precision import rounded_flags
    from ghostscale.validation.soundingline.v25 import pieces
    # At zero observations the exact posterior is uniform, independent of a
    # near-boundary double-precision screening value supplied by a caller.
    p=np.full((1,36),1/36)
    e=pieces.library('partial',np.random.default_rng(1))
    expected=np.round(population.flags(np.full((1,36),1/36)),8)
    monkeypatch.setattr(population,'flags',lambda p:np.array([.499999995]))
    actual=rounded_flags(p,e,np.zeros((1,4,6),dtype=int),[0,1],[],3,False,'individual')
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('size',[8,256])
def test_population_roster_independent_nulls_and_reconstruction(size):
    setting=dict(followup='population-size',population_size=size,alpha=30,placebo=False)
    original=followups.block(setting,['excluded-population-size',size])
    independent=followups.block(setting,['excluded-population-size',size],True)
    close(original,independent,tol=1e-9)
    assert len(original['truth']['weights'])==2*size+512
    assert len(original['truth']['null_excluded_population_member'])==512
    assert max(original['truth']['null_excluded_population_member'])<size
    for row in original['readouts']:
        if 'calibration_false_detection' in row['metrics']:
            assert row['metrics']['calibration_false_detection']==pytest.approx(.05,abs=1e-12)
        assert all(v is None or math.isfinite(v) for v in row['metrics'].values())


@pytest.mark.parametrize('habit,end',[(False,False),(False,True),(True,False),(True,True)])
def test_paired_affect_readers_reconstruct_and_keep_exact_alias(habit,end):
    setting=dict(followup='affect-channels',habit_affect=habit,end_distress=end,noise_sd=.25)
    identity=['excluded-affect-channel',habit,end]
    original=followups.block(setting,identity)
    independent=followups.block(setting,identity,True)
    close(original,independent,tol=1e-9)
    assert original['truth_and_observations']==independent['truth_and_observations']
    assert len(original['truth_and_observations'])==6
    for row in original['readouts']:
        if row['labels']['reader']=='neither' or (habit==end and row['labels']['reader']=='distress'):
            if 'primary' in row['metrics']:assert abs(row['metrics']['primary'])<1e-12
            if 'not_within_32' in row['metrics']:assert row['metrics']['not_within_32']==1
        assert all(v is None or math.isfinite(v) for v in row['metrics'].values())


def test_gaussian_measure_detects_a_broken_channel(monkeypatch):
    real=texture.gaussian_logs
    monkeypatch.setattr(texture,'gaussian_logs',lambda signal,means,sd,channels,independent=False:np.zeros(len(means)))
    with pytest.raises(AssertionError):texture.controls()
    monkeypatch.setattr(texture,'gaussian_logs',real)
    assert texture.controls()['passed']


def test_verified_probe_prepares_only_its_frozen_texture_grid(tmp_path,monkeypatch):
    from ghostscale.validation.soundingline.v23.common import read,write,digest
    from ghostscale.validation.soundingline.v25 import contract
    from runners import prepare_v25_texture as prep,run_v25
    parent=tmp_path/'packets/texture-probe';parent.mkdir(parents=True);source=tmp_path/'source';source.mkdir()
    protocol=dict(identity='fixture-followup',settings=[dict(followup='population-size',population_size=8,alpha=3,placebo=False)],blocks=32)
    path=tmp_path/'protocols/frozen.json';write(path,protocol)
    prior=dict(texture_card='W4F1',texture_phase='cost_probe',protocol_file=path.name,protocol_sha256=digest(path),source_root=str(source),family='W4',kind='followup',generation=1,pool='followup_healing',numerical_threads=1,priority='below-normal')
    write(parent/'PLAN.json',prior);write(parent/'CONTROLS_BEFORE_DATA.json',dict(passed=True))
    complete=dict(plan_sha256=digest(parent/'PLAN.json'))
    for key,name in [('summary_sha256','original/SUMMARY.json'),('replay_sha256','REPLAY.json'),('review_sha256','INDEPENDENT_REVIEW.json')]:
        write(parent/name,{});complete[key]=digest(parent/name)
    write(parent/'COMPLETE.json',complete)
    write(parent/'PASS_COSTS.json',dict(includes_world_io=True,setting_seconds=dict(original=[1,2],replay=[1,2],review=[2,4])))
    write(tmp_path/'accounting/texture-probe.json',dict(state='COMPLETE',pool='followup_healing',kind='followup',cpu_seconds=16,pass_cpu_seconds=dict(original=4,replay=4,review=8)))
    write(tmp_path/'ACCEPTANCE.json',contract.acceptance('2026-10-25T19:57:34+00:00','fixture'))
    monkeypatch.setattr(prep,'verify_source',lambda p:None)
    result=read(run_v25.prepare(tmp_path,'W4','study',parent=parent,source=source)/'PLAN.json')
    assert result['settings']==protocol['settings'] and result['blocks']==32
    assert result['pass_caps']==dict(original=107.5,replay=107.5,review=205.)
    assert result['kind']=='followup' and result['texture_phase']=='study'
    assert result['parent_complete_sha256']==digest(parent/'COMPLETE.json')
