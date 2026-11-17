"""A conditional builder must not turn failed or tampered gates into science."""
from pathlib import Path
import pytest
from ghostscale.validation.soundingline.v31.common import write,digest
from ghostscale.validation.soundingline.v31.q_r1_production_builder import validate


def fixture(tmp_path,passed=True):
    c=tmp_path/'campaign';j=c/'jobs/Q_HIDDEN_R1_IO'
    write(j/'PLAN.json',dict(family='Q_HIDDEN_R1_IO'))
    configs=[dict(setting=dict(view=view,group_rate=g,move_rate=h),
                  result=dict(passed=True,production_particles=8192))
             for view in ('V1','V2') for g in (0.,.03,.1) for h in (.01,.05)]
    write(j/'output/VALIDATION.json',dict(passed=passed,configurations=configs))
    write(j/'WORKER_COMPLETE.json',dict(state='VERIFIED',scientific=False,plan_sha256=digest(j/'PLAN.json'),
          validation_sha256=digest(j/'output/VALIDATION.json')))
    write(j/'COMPLETE.json',dict(state='VERIFIED',plan_sha256=digest(j/'PLAN.json'),worker_receipt_sha256=digest(j/'WORKER_COMPLETE.json')))
    return c,j


def test_failed_parent_never_profiles(tmp_path,monkeypatch):
    c,j=fixture(tmp_path)
    write(j/'COMPLETE.json',dict(state='FAILED'),immutable=False)
    monkeypatch.setattr('runners.v31_build.admit',lambda *a,**k:pytest.fail('failed parent attempted admission'))
    result=validate(dict(repo=str(tmp_path),campaign=str(c)),tmp_path/'out')
    assert result['production_admitted'] is False
    assert result['disposition']=='HELD_AFTER_PARENT_VALIDATION_FAILURE'


def test_failed_accuracy_never_profiles(tmp_path,monkeypatch):
    c,j=fixture(tmp_path,False)
    monkeypatch.setattr('runners.v31_build.admit',lambda *a,**k:pytest.fail('failed accuracy attempted admission'))
    with pytest.raises(AssertionError):validate(dict(repo=str(tmp_path),campaign=str(c)),tmp_path/'out')


def test_changed_receipt_never_profiles(tmp_path,monkeypatch):
    c,j=fixture(tmp_path)
    write(j/'output/VALIDATION.json',dict(passed=True,configurations=[]),immutable=False)
    monkeypatch.setattr('runners.v31_build.admit',lambda *a,**k:pytest.fail('changed receipt attempted admission'))
    with pytest.raises(AssertionError):validate(dict(repo=str(tmp_path),campaign=str(c)),tmp_path/'out')


def test_passed_parent_still_uses_profile_gate(tmp_path,monkeypatch):
    c,j=fixture(tmp_path);calls=[]
    def profile(repo,campaign,family,pool,module,settings,protocol,order,extra):
        assert pool=='Q' and len(settings)==18
        assert {s['particles'] for s in settings}=={8192}
        assert {s['worlds'] for s in settings}=={200}
        assert {s['blocks'] for s in settings}=={8}
        assert {s['steps'] for s in settings}=={2000}
        assert extra['validation_sha256']==digest(j/'output/VALIDATION.json')
        calls.append(family)
    monkeypatch.setattr('runners.v31_build.admit',profile)
    result=validate(dict(repo=str(tmp_path),campaign=str(c)),tmp_path/'out')
    assert result['production_admitted'] and len(calls)==2
