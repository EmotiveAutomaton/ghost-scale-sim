import gzip
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import canonical,file_digest
from ghostscale.validation.soundingline.v19 import task_assessment as E, local_world as L
from ghostscale.validation.soundingline.v19 import roll_in as B, recursive_update as U


def test_task_known_answers_and_complete_mass(tmp_path):
    assert all(E.controls().values())
    p=tmp_path/'inputs/lineage--9401_points.json.gz';p.parent.mkdir()
    p.write_bytes(gzip.compress(canonical(L.enumerate_world(L.law(-9401))),mtime=0))
    summary=E.run(tmp_path,dict(design=dict(lineages=[-9401],input_files={p.name:file_digest(p)})),lambda **kw:None)
    assert summary['trajectories']==13824
    assert all(abs(r['coverage']-1)<1e-10 for r in summary['cells'])
    assert all(abs(r['mass']-.25)<1e-10 for r in summary['outcomes'])


def test_roll_in_complete_fixture_retains_one_step(tmp_path):
    source=tmp_path/'b1';source.mkdir()
    cfg=dict(train_lineages=[-9411],development_lineages=[-9412],training_draws=[-9413],fit_seeds=[9414],streams_per_lineage=2,lengths=[8,32])
    U.run(source,dict(design=cfg),lambda **kw:None)
    out=tmp_path/'b2';out.mkdir();inputs={}
    import shutil
    for folder in ('evaluator','models','forecasts'):
        for p in (source/folder).glob('*.npz'):
            n=p.relative_to(source).as_posix();dest=out/'inputs'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);inputs[n]=file_digest(dest)
    result=B.run(out,dict(design=dict(cfg,input_files=inputs)),lambda **kw:None)
    assert all(result['controls'].values()) and result['rows']==24
    assert all(f['one_step_reproduced'] and f['control_identity'] for f in result['fits'])
    with np.load(out/'evaluator/training-roll-in--9413-9414.npz') as a:
        assert np.array_equal(a['previous_bank'][2:,2::2],a['previous_bank'][2:,1:-1:2])
