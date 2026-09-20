import numpy as np
import pytest
torch=pytest.importorskip('torch')
from ghostscale.validation.soundingline.v18_4 import bank_worker as B,torch_worker as T


def test_bank_forward_raw_arithmetic_resume_and_corruption(tmp_path):
    import platform
    from datetime import datetime,timedelta,timezone
    from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
    public=tmp_path/'reader';public.mkdir();torch.manual_seed(91)
    model=T.Reader('flat',2,3,6);weight=public/'flat-seed701-ENCODER.pt'
    torch.save(dict(spec=model.spec,state=model.state_dict()),weight);original=file_digest(weight)
    r=np.random.default_rng(8);n=12;nq=3
    h=r.normal(size=(n,32,2)).astype(np.float32);q=r.normal(size=(n*nq,3)).astype(np.float32)
    np.savez_compressed(public/'TEST.npz',history=h,length=np.full(n,8),sample=np.repeat(np.arange(n),nq),query=q)
    maps={'bank_query':r.normal(size=(n,5,3)).astype(np.float32)}
    for mode in ('bank-full','bank-truncated','bank-passive'):
        width=17 if mode=='bank-passive' else 81
        m=np.zeros((n,nq,width,16));m[:,:,1:17]=np.eye(16);maps[mode]=m
    # An intentionally unstable supplied map must stay visible in raw outputs.
    maps['bank-truncated'][:,:,0,0]=-2
    np.savez_compressed(public/'MAPS.npz',**maps)
    manifest=dict(tests={'fixture':dict(name='TEST.npz',sha256=file_digest(public/'TEST.npz'))},
        maps={'fixture':dict(name='MAPS.npz',sha256=file_digest(public/'MAPS.npz'))},
        encoders={'flat-seed701':dict(name=weight.name,sha256=original)})
    write(public/'INPUTS.json',manifest)
    write(tmp_path/'CONFIG.json',dict(report_start=(datetime.now(timezone.utc)+timedelta(minutes=10)).isoformat(),cpu_ceiling_seconds=600,
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,threads=1,interop_threads=1)))
    output=tmp_path/'output';B.run(public/'INPUTS.json',output,tmp_path/'CONFIG.json')
    complete=read(output/'COMPLETE.json');assert len(complete['predictions'])==4
    for reader,conditions in complete['predictions'].items():
        e=conditions['fixture'];path=tmp_path/'replay.npz'
        B.forecast(weight,T.dataset(public/'TEST.npz'),B.load_maps(public/'MAPS.npz',nq),path,e['mode'])
        with np.load(path) as a,np.load(output/e['file']) as b:
            assert np.array_equal(a['probabilities'],b['probabilities'])
            assert np.array_equal(a['raw_probabilities'],b['raw_probabilities'])
    bad=complete['predictions']['flat-bank-truncated-seed701']['fixture'];assert bad['raw_invalid_rows']==n*nq
    data=T.dataset(public/'TEST.npz');saved=torch.load(weight,weights_only=True);model.load_state_dict(saved['state']);model.eval()
    with torch.no_grad():bank=torch.softmax(model.decode(model.encode(data['history'],data['length']),torch.from_numpy(maps['bank_query'][:,0])),1).numpy()
    with np.load(output/complete['predictions']['flat-bank-full-seed701']['fixture']['file']) as z:
        assert np.allclose(z['raw_probabilities'],np.repeat(bank,nq,axis=0),atol=1e-12)
    B.run(public/'INPUTS.json',output,tmp_path/'CONFIG.json');assert file_digest(weight)==original
    weight.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='input changed'):B.run(public/'INPUTS.json',output,tmp_path/'CONFIG.json')
    assert not torch.cuda.is_initialized()


def test_bank_capsule_scoring_and_extracted_replay(tmp_path):
    import platform
    from datetime import datetime,timedelta,timezone
    from ghostscale.validation.soundingline.v18_4 import neural_data as D,bank_data,neural_runtime
    from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
    from runners.replay_v18_4 import neural
    parent=tmp_path/'parent';parent.mkdir()
    D.prepare(parent/'data',train_per_cell=8,dev_per_cell=8,test_per_cell=8,pilot=True)
    manifest=read(parent/'data/reader/INPUTS.json');weight=parent/'neural/selected/BEST.pt';weight.parent.mkdir(parents=True)
    model=T.Reader('direct',manifest['history_features'],manifest['query_features'],6)
    torch.save(dict(spec=model.spec,state=model.state_dict()),weight)
    write(parent/'PLAN.json',{'fixture':True});write(parent/'SUMMARY.json',{'fixture':True})
    write(parent/'neural/COMPLETE.json',dict(predictions={'direct-seed701':{c:dict(selected='selected') for c in manifest['tests']}},fits={'selected':dict(best_sha256=file_digest(weight))}))
    write(parent/'COMPLETE.json',dict(files={'neural/selected/BEST.pt':file_digest(weight)}))
    write(parent/'INDEPENDENT_REPLAY.json',dict(passed=True,summary_sha256=file_digest(parent/'SUMMARY.json'),plan_sha256=file_digest(parent/'PLAN.json')))
    root=tmp_path/'bank';bank_data.prepare(root/'data',parent)
    config=dict(report_start=(datetime.now(timezone.utc)+timedelta(minutes=10)).isoformat(),cpu_ceiling_seconds=600,
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,threads=1,interop_threads=1))
    write(root/'TRAINING.json',config);write(root/'PLAN.json',dict(design=dict(study='E-bank',baseline_method='direct-original')))
    B.run(root/'data/reader/INPUTS.json',root/'neural',root/'TRAINING.json')
    summary=neural_runtime.score_E(root)
    assert summary['family']=='L2a' and len(summary['bank_diagnostics'])==15
    assert summary['checks']['independent_scalar_scores']>0
    (tmp_path/'check').mkdir()
    proof=neural(root,tmp_path/'check/PROOF.json');assert len(proof['forecast_replays'])==20
    assert proof['independent_bank_diagnostic_means']==75
    assert all(r['invalidity_reconstructed'] and r['max_raw_absolute_error']==0 for r in proof['forecast_replays'])
    # Resume detects changes in the supplied law capsule, not just frozen weights.
    maps=root/'data/reader/new-queries-MAPS.npz';maps.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='capsule changed'):bank_data.prepare(root/'data',parent)
