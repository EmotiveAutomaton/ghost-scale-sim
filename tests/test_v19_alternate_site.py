from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import interchange_alignment as A
from ghostscale.validation.soundingline.v19 import interchange_fixtures as F
from ghostscale.validation.soundingline.v19.readout_model import save_arrays
from ghostscale.validation.soundingline.v18_3.io import write,file_digest


def fixture_parameters():
    g=np.random.default_rng(6103)
    shapes={'embedding.weight':(33,32),'positions':(33,32),'encoder.self_attn.in_proj_weight':(96,32),
        'encoder.self_attn.in_proj_bias':(96,),'encoder.self_attn.out_proj.weight':(32,32),
        'encoder.self_attn.out_proj.bias':(32,),'encoder.linear1.weight':(64,32),'encoder.linear1.bias':(64,),
        'encoder.linear2.weight':(32,64),'encoder.linear2.bias':(32,),'head.weight':(32,32),'head.bias':(32,)}
    p={k:g.normal(size=s)*.1 for k,s in shapes.items()}
    for n in ('norm1','norm2'):
        p[f'encoder.{n}.weight']=g.uniform(.7,1.3,size=32);p[f'encoder.{n}.bias']=g.normal(size=32)*.1
    return p


def test_alternate_site_preserves_output_and_duplicate_identity():
    p=fixture_parameters();codes=np.tile(np.arange(32),(2,1));novel=np.ones((2,32),bool);novel[:,7]=False
    final,prob=F.reconstruct(p,codes,novel,'transformer')
    pre,got=F.reconstruct(p,codes,novel,'transformer','pre-final-normalization')
    assert np.allclose(F.final_normalize(p,pre),final,atol=1e-14,rtol=0)
    assert np.allclose(got,prob,atol=1e-14,rtol=0)
    assert np.array_equal(pre[:,6],pre[:,7])
    changed=codes.copy();changed[:,20:]=31-changed[:,20:]
    future,_=F.reconstruct(p,changed,novel,'transformer','pre-final-normalization')
    assert np.array_equal(pre[:,:20],future[:,:20])
    with pytest.raises(ValueError,match='site'):F.reconstruct(p,codes,novel,'recurrent','pre-final-normalization')


def test_normalization_null_and_live_coordinate():
    p=fixture_parameters();g=np.random.default_rng(6);h=g.normal(size=(4,32));q=np.zeros(32);q[0]=1
    baseline=F.decode(p,F.final_normalize(p,h))
    assert np.allclose(F.decode(p,F.final_normalize(p,h+3)),baseline,atol=1e-14,rtol=0)
    altered=A.swap(h,h[::-1],q)
    assert np.max(abs(F.decode(p,F.final_normalize(p,altered))-baseline))>.001
    p['head.weight'][:]=0
    assert np.array_equal(F.decode(p,F.final_normalize(p,altered)),F.decode(p,F.final_normalize(p,h)))


def test_full_alternate_pipeline_normalizes_every_arm_and_rejects_corruption(tmp_path):
    source=tmp_path/'inputs';source.mkdir();p=fixture_parameters()
    write(source/'INDEPENDENT_REVIEW.json',dict(passed=True,per_fit_capability=[dict(passed=True)]))
    makers=np.array(list(product((0,1),repeat=4)));codes=np.tile(np.arange(32),(16,1))
    codes=(codes+np.arange(16)[:,None])%32;novel=np.ones_like(codes,bool)
    final,prob=F.reconstruct(p,codes,novel,'transformer');pre,_=F.reconstruct(p,codes,novel,'transformer','pre-final-normalization')
    save_arrays(source/'inputs/models/10-20-transformer.npz',**p)
    for split,lineage in [('train',0),('development',1)]:
        stem=f'{split}-{lineage}-10'
        save_arrays(source/'reader'/f'{stem}.npz',codes=codes,novel=novel)
        save_arrays(source/'forecasts'/f'{stem}-20-transformer.npz',hidden=final)
        save_arrays(source/'evaluator'/f'{stem}.npz',maker=makers,maker_indices=np.arange(16))
    save_arrays(source/'evaluator/development-1-laws.npz',laws=prob[:,31])
    cfg=dict(input_files={q.relative_to(source).as_posix():file_digest(q) for q in source.rglob('*') if q.is_file()},
        train_lineages=[0],development_lineages=[1],training_draws=[10],fit_seeds=[20],kinds=['transformer'],lengths=[8,32],site='pre-final-normalization')
    result=A.run(tmp_path,{'design':cfg},lambda **kw:None)
    assert result['rows']==896 and result['probes']==2 and all(result['controls'].values())
    for length in (8,32):
        h=pre[:,length-1]
        for factor,change in product((0,2),(False,True)):
            donor=makers.copy();donor[:,1]=1-donor[:,1]
            if change:donor[:,factor]=1-donor[:,factor]
            di=donor@np.array([8,4,2,1]);d=A.arrays(tmp_path/'models'/f'10-20-transformer-{factor}.npz')
            d['wrong-factor']=A.arrays(tmp_path/'models'/f'10-20-transformer-{2-factor}.npz')['learned']
            saved=A.arrays(tmp_path/'forecasts'/f'1-10-20-transformer-{length}-{factor}-{int(change)}.npz')['probabilities']
            for ai,arm in enumerate(A.ARMS):
                altered=h if arm=='unchanged' else h[di] if arm=='full-state' else h+np.outer(np.sum((h[di]-h)*d[arm],axis=1),d[arm])
                assert np.allclose(saved[ai],F.decode(p,F.final_normalize(p,altered)),atol=1e-14,rtol=0)
    corrupt=tmp_path/'corrupt'
    save_arrays(corrupt/'forecasts/train-0-10-20-transformer.npz',hidden=final+.01)
    save_arrays(corrupt/'reader/train-0-10.npz',codes=codes,novel=novel)
    with pytest.raises(ValueError,match='reconstruction'):A.states_at_site(corrupt,'train-0-10',20,'transformer',p,cfg['site'])
