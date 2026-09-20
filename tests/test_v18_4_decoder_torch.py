"""Optional isolated CPU fixture, with no scientific queue ownership."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from ghostscale.validation.soundingline.v18_4 import decoder_worker as P,torch_worker as T


def test_frozen_features_and_complete_decoder_resume_replay(tmp_path):
    import platform
    from datetime import datetime,timedelta,timezone
    from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
    public=tmp_path/'reader';public.mkdir();torch.manual_seed(91)
    model=T.Reader('flat',2,3,6);best=public/'flat-seed701-ENCODER.pt'
    torch.save(dict(spec=model.spec,state=model.state_dict()),best);original=file_digest(best)
    def data(n,seed):
        r=np.random.default_rng(seed);h=np.zeros((n,32,2),np.float32);h[:,0]=r.normal(size=(n,2))
        q=r.normal(size=(n,3)).astype(np.float32);target=np.eye(16,dtype=np.float32)[(h[:,0,0]>0).astype(int)]
        return dict(history=h,length=np.ones(n,np.int64),sample=np.arange(n),query=q,target=target)
    train=data(2048,1);dev=data(64,2);test=data(64,3)
    np.savez_compressed(public/'TRAIN.npz',**{f'train_{k}':v for k,v in train.items()},**{f'dev_{k}':v for k,v in dev.items()})
    np.savez_compressed(public/'TEST.npz',**{k:v for k,v in test.items() if k!='target'})
    manifest=dict(train=dict(name='TRAIN.npz',sha256=file_digest(public/'TRAIN.npz')),
        tests={'fixture':dict(name='TEST.npz',sha256=file_digest(public/'TEST.npz'))},
        encoders={'flat-seed701':dict(name=best.name,sha256=original)},train_histories=2048,query_context_features=0)
    write(public/'INPUTS.json',manifest)
    config=dict(report_start=(datetime.now(timezone.utc)+timedelta(minutes=10)).isoformat(),cpu_ceiling_seconds=600,
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,threads=1,interop_threads=1),
        label_budgets=[128],random_features=16,alphas=[.1],temperatures=[.1])
    write(tmp_path/'CONFIG.json',config)
    out=tmp_path/'output';P.run(public/'INPUTS.json',out,tmp_path/'CONFIG.json')
    complete=read(out/'COMPLETE.json');assert len(complete['fits'])==12
    assert file_digest(best)==original and not torch.cuda.is_initialized()
    input_data=T.dataset(public/'TEST.npz')
    for name,entry in complete['fits'].items():
        path=tmp_path/'replay.npz';P.forecast(out/name/'READOUT.npz',input_data,path,public,entry['identity'])
        with np.load(path) as a,np.load(out/complete['predictions'][name]['fixture']['file']) as b:
            assert np.max(abs(a['probabilities']-b['probabilities']))<1e-12
    fit_hashes={name:file_digest(out/name/'READOUT.npz') for name in complete['fits']}
    P.run(public/'INPUTS.json',out,tmp_path/'CONFIG.json')
    assert fit_hashes=={name:file_digest(out/name/'READOUT.npz') for name in complete['fits']}
    # A corrupted frozen encoder must not be silently accepted on resume.
    best.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='capsule changed'):P.run(public/'INPUTS.json',out,tmp_path/'CONFIG.json')
