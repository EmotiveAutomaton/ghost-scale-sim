import numpy as np
from ghostscale.validation.soundingline.v19 import bounded_reset as B, recursive_update as U, readout_model as M
from ghostscale.validation.soundingline.v19.roll_in import arrays,run as roll_in
from ghostscale.validation.soundingline.v18_3.io import file_digest


def test_reset_does_not_fire_on_a_copy_or_at_initialization():
    assert all(B.controls().values())
    assert not B.reset_due(np.array([0,8]),np.array([True,False]),8).any()


def test_complete_reset_fixture_preserves_parent_and_charges_resets(tmp_path):
    import shutil
    parent=tmp_path/'b1';parent.mkdir()
    cfg=dict(train_lineages=[-9511],development_lineages=[-9512],training_draws=[-9513],fit_seeds=[9514],streams_per_lineage=2,lengths=[8,32])
    U.run(parent,dict(design=cfg),lambda **kw:None)
    rolled=tmp_path/'b2';rolled.mkdir();inputs={}
    for folder in ('evaluator','models','forecasts'):
        for p in (parent/folder).glob('*.npz'):
            n=p.relative_to(parent).as_posix();d=rolled/'inputs'/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d);inputs[n]=file_digest(d)
    roll_in(rolled,dict(design=dict(cfg,input_files=inputs)),lambda **kw:None)
    out=tmp_path/'reset';out.mkdir();inputs={}
    for prefix,source in [('evaluator',parent/'evaluator'),('models',parent/'models'),('roll-in-models',rolled/'models'),('forecasts',rolled/'forecasts')]:
        for p in source.glob('*.npz'):
            n=prefix+'/'+p.name;d=out/'inputs'/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d);inputs[n]=file_digest(d)
    result=B.run(out,dict(design=dict(cfg,lengths=[8,32,128],horizon=128,reset_period=8,input_files=inputs)),lambda **kw:None)
    assert result['fits']==0 and result['rows']==48 and all(result['controls'].values())
    for condition,expected in [('independent',15),('copied',7)]:
        a=arrays(out/'evaluator'/f'resets--9513-9514-{condition}.npz')
        assert np.all(a['reset_before_observation'].sum(1)==expected)
        assert not a['reset_before_observation'][:,[7,31,127]].any()
        data=arrays(out/'evaluator'/f'-9513-{condition}.npz')
        if condition=='copied':
            assert np.array_equal(data['codes'][:,1::2],data['codes'][:,::2])
            p=arrays(out/'forecasts'/f'-9513-9514-{condition}-rolled-in-reset.npz')['probabilities']
            assert np.array_equal(p[:,1::2],p[:,::2])
