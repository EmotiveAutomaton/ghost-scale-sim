"""Known posterior targets, paired access, and retained supervision checks."""
import numpy as np
import pytest
from ghostscale.validation.soundingline.v18_4 import neural_data as D
from ghostscale.validation.soundingline.v18_4 import conditional_targets as C
from ghostscale.validation.soundingline.v18_3 import world as W
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest


def test_live_placebo_and_duplicate_source_invariance():
    assert all(C.controls().values())
    w=W.make_world(5,1987002);states=D.NEURAL_STATES
    o=W.observe(w,states[3],D.TRAIN_QUERIES[0],W.rng('L2b-copy'),'root')
    p=C.teacher(W.packet(w,[o]),D.TRAIN_QUERIES,states)
    assert np.array_equal(p,C.teacher(W.packet(w,[o,o,o]),D.TRAIN_QUERIES,states))
    assert np.allclose(p,C.reference_teacher(W.packet(w,[o]),D.TRAIN_QUERIES,states),atol=1e-12,rtol=0)


def test_exact_conditioning_and_cross_entropy_gradient_identity():
    # Enumerate a known two-state observation law, including its nonuniform posterior.
    prior=np.array([.5,.5]);likelihood=np.array([.8,.2]);posterior=prior*likelihood;posterior/=posterior.sum()
    bank=np.array([[.9,.1],[.2,.8]]);target=posterior@bank;q=np.array([.4,.6])
    assert np.allclose(target,[.76,.24])
    assert np.isclose(-target@np.log(q),posterior@(-bank@np.log(q)))
    assert np.allclose(q-target,posterior@(q-bank))
    # Prior-only averaging is a different target after informative evidence.
    assert not np.allclose(target,prior@bank)


def test_paired_iid_inputs_and_shared_realized_development():
    kw=dict(per_cell=8,support='all',namespace='v18.4-conditional-fixture',state_sampling='iid')
    a,ta=D.make_split('pilot',target_mode='realized',**kw)
    b,tb=D.make_split('pilot',target_mode='conditional',**kw)
    assert ta==tb
    assert all(np.array_equal(a[k],b[k]) for k in a if k!='target')
    assert not np.array_equal(a['target'],b['target'])
    assert np.allclose(b['target'].sum(1),1,atol=1e-6)
    for split in ('dev','test'):
        x,tx=D.make_split(split,target_mode='realized',**kw)
        y,ty=D.make_split(split,target_mode='conditional',**kw)
        assert tx==ty and all(np.array_equal(x[k],y[k]) for k in x)
    for cell in range(16):
        for i in range(8):
            assert ta[cell*8+i]['state']==int(W.rng(kw['namespace'],'iid-state','pilot',cell,i,False).choice(D.NEURAL_STATES))


def test_wrong_prior_stress_and_evaluator_noninterference():
    w=W.make_world(0,1987003);states=D.NEURAL_STATES
    h=D.history(w,states[6],W.rng('L2b-truth-separation'),8);payload=W.packet(w,h)
    answer=C.teacher(payload,D.TRAIN_QUERIES,states)
    evaluator=dict(realized_state=states[6],test_target=[0.]*16)
    evaluator.update(realized_state=states[-1],test_target=[1.]*16)
    assert np.array_equal(answer,C.teacher(payload,D.TRAIN_QUERIES,states))
    assert not np.allclose(answer,C.teacher(payload,D.TRAIN_QUERIES,(states[0],)))
    with pytest.raises(ValueError):D.make_split('pilot',8,target_mode='conditional')
    with pytest.raises(ValueError):C.prior_for([states[0],states[0]])


def test_prepare_audit_resume_and_corruption(tmp_path):
    kw=dict(train_per_cell=8,dev_per_cell=8,test_per_cell=8,pilot=True,support='all',
        namespace='v18.4-conditional-fixture',state_sampling='iid',target_mode='conditional')
    manifest=D.prepare(tmp_path,**kw)
    audit=read(tmp_path/'TARGET_AUDIT.json')
    assert audit['passed'] and audit['independent_target_rows']==160
    assert audit==C.audit(tmp_path)
    # Exercise the actual replay entry point on the retained capsule.
    from runners.replay_v18_4 import target_audit
    import shutil
    packet=tmp_path/'packet';packet.mkdir();shutil.copytree(tmp_path,packet/'data',ignore=shutil.ignore_patterns('packet'))
    assert target_audit(packet)==audit
    saved=read(packet/'data/TARGET_AUDIT.json');saved['max_teacher_error']=5e-13
    write(packet/'data/TARGET_AUDIT.json',saved,immutable=False)
    assert target_audit(packet)['passed']
    saved['sampled_state_counts']['train'][str(D.NEURAL_STATES[0])]+=1
    write(packet/'data/TARGET_AUDIT.json',saved,immutable=False)
    with pytest.raises(ValueError,match='audit changed'):target_audit(packet)
    assert manifest['target_mode']=='conditional'
    assert D.prepare(tmp_path,**kw)==manifest
    with pytest.raises(ValueError):D.prepare(tmp_path,**dict(kw,target_mode='realized'))
    path=tmp_path/'reader/TRAIN.npz'
    with np.load(path) as z:data={k:z[k].copy() for k in z.files}
    data['train_target'][0]=np.roll(data['train_target'][0],1)
    np.savez_compressed(path,**data)
    with pytest.raises(ValueError,match='capsule'):C.audit(tmp_path)
    with pytest.raises(ValueError,match='input changed'):D.prepare(tmp_path,**kw)


def test_reference_rejects_normalized_but_wrong_target(tmp_path):
    # Self-consistent file hashes must not turn a wrong conditional target valid.
    kw=dict(train_per_cell=8,dev_per_cell=8,test_per_cell=8,pilot=True,support='all',
        namespace='v18.4-conditional-corruption',state_sampling='iid',target_mode='conditional')
    D.prepare(tmp_path,**kw)
    path=tmp_path/'train-DATA.npz'
    with np.load(path) as z:data={k:z[k].copy() for k in z.files}
    data['target'][0]=np.ones(16)/16;np.savez_compressed(path,**data)
    receipt=read(tmp_path/'train-PART.json');receipt['data_sha256']=file_digest(path);write(tmp_path/'train-PART.json',receipt,immutable=False)
    path=tmp_path/'reader/TRAIN.npz'
    with np.load(path) as z:capsule={k:z[k].copy() for k in z.files}
    capsule['train_target']=data['target'];np.savez_compressed(path,**capsule)
    with pytest.raises(ValueError,match='supervision'):C.audit(tmp_path)


def test_other_studies_do_not_require_target_build_plan(tmp_path):
    from runners.replay_v18_4 import target_audit
    assert target_audit(tmp_path) is None
