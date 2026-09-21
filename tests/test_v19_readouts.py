import copy
import numpy as np
from ghostscale.validation.soundingline.v19 import readout_model as M
from ghostscale.validation.soundingline.v19 import readout_data as D
from ghostscale.validation.soundingline.v19 import local_world as L
from ghostscale.validation.soundingline.v19 import local_alternatives as A
from ghostscale.validation.soundingline.v18_3 import world as W

def test_readout_known_and_placebo():
    assert all(M.controls().values())
    assert all(D.controls().values())

def test_fixed_capacity_and_deterministic_model_archive(tmp_path):
    x=np.ones((4,7));z,weights,bias=M.basis(x,7)
    assert z.shape==(4,129) and weights.shape==(512,128)
    a=tmp_path/'a.npz';b=tmp_path/'b.npz'
    M.save_arrays(a,z=z,weights=weights,bias=bias);M.save_arrays(b,z=z,weights=weights,bias=bias)
    assert a.read_bytes()==b.read_bytes()
    with np.load(a,allow_pickle=False) as saved:assert np.array_equal(saved['z'],z)

def test_old_artifacts_omit_program_and_duplicate_is_inert():
    w=W.make_world(0,190901);r=W.rng('verification-only',0)
    h=D.D.history(w,2,r,8);changed=copy.deepcopy(h);changed[0]['program']=[99]
    assert D.old_features(h)==D.old_features(changed)
    p=D.old_posterior(w,h)
    assert np.allclose(D.old_posterior(w,h+[h[-1]]),p,atol=1e-14)
    assert len(D.old_features(h))<=512

def test_local_truth_is_separate_and_reference_normalizes():
    records=L.enumerate_world(L.law(-190901));old=D.local_cache(records)
    assert old.shape==(16,32) and np.allclose(old.reshape(16,4,8).sum(2),1)
    r=records[0];changed=copy.deepcopy(r);changed['steps'][0]['goal']='presentation'
    for tier in D.TIERS:
        assert D.local_features(L.project(r,tier))==D.local_features(L.project(changed,tier))
        assert len(D.local_features(L.project(r,tier)))<=512
    refs=D.local_reference(records,'E1',old)
    for bank,target in refs.values():
        assert np.allclose(bank.reshape(4,8).sum(1),1)
        assert abs(sum(target[:3])-1)<1e-12
        assert abs(sum(target[9:15])-1)<1e-12

def test_jointness_and_frame_controls():
    assert all(A.controls().values())
    g1=('meaning',)*3;g2=('dependency',)*3
    o1=('inspect',)*3;o2=('inspect','inspect','undo')
    p={(g1,o1):.25,(g1,o2):.25,(g2,o1):.25,(g2,o2):.25}
    assert A.rectangle(p)['marginal_equality']
    impossible=A.metrics({(g1,o1):1.},(g2,o2),set(p))
    assert impossible['infinite'] and impossible['contradiction']
    unknown=A.metrics({},(g1,o1),set())
    assert unknown['unknown'] and not unknown['assertions']

def test_readout_end_to_end_fixture(tmp_path):
    from ghostscale.validation.soundingline.v19.readouts import run
    cfg=dict(domain='old',train_lineages=[-190910],development_lineages=[-190911],
        training_draws=[190999],fit_seeds=[190998],budgets=[4,8])
    result=run(tmp_path,dict(design=cfg),lambda **kw:None)
    assert all(result['controls'].values())
    assert len(result['fits'])==8
    assert (tmp_path/'TIMING.jsonl').is_file()
    assert all(f['new_head_parameters']==129*32 for f in result['fits'])
    assert any(c['world_class']==-1 for c in result['cells'])
    assert {r['baseline'] for r in result['contrasts']}=={'raw-history','frozen-latent'}
    with np.load(tmp_path/'reader/development-190999-artifact-history.npz') as reader:
        assert set(reader.files)=={'x','ids'}

def test_local_readout_end_to_end_fixture(tmp_path):
    from ghostscale.validation.soundingline.v19.readouts import run
    cfg=dict(domain='local',train_lineages=[-190920],development_lineages=[-190921],
        training_draws=[190999],fit_seeds=[190998],budgets=[4,8])
    result=run(tmp_path,dict(design=cfg),lambda **kw:None)
    assert all(result['controls'].values())
    assert len(result['fits'])==32
    assert all(f['new_head_parameters']==129*27 for f in result['fits'])
    assert {c['tier'] for c in result['cells']}==set(D.TIERS)

def test_alternatives_end_to_end_fixtures(tmp_path):
    for kind in ('frame','jointness'):
        root=tmp_path/kind;root.mkdir()
        result=A.run(root,dict(design=dict(handler=kind,lineages=[-190930],sampling_seeds=[190998])),lambda **kw:None)
        assert all(result['controls'].values())
        assert result['scored_rows']>0
        if kind=='jointness':assert result['matched_marginal_fixtures']>0
