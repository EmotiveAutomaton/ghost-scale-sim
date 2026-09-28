"""Native-card replay and distinct acquisition-fee grouping regressions."""
from pathlib import Path
import pytest
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import write,file_digest
from ghostscale.validation.soundingline.v20.runtime import scientific_files
from ghostscale.validation.soundingline.v20.readers import score
from ghostscale.validation.soundingline.v20.analysis import review
import runners.replay_v20 as replay_module

@pytest.mark.parametrize('native_card',[False,True])
def test_full_replay_reconstructs_native_admission(tmp_path,monkeypatch,native_card):
    original=tmp_path/'packet';original.mkdir()
    plan=dict(design=dict(branch='G0'),environment=dict(python='fixture'),source_archive_sha256='source')
    write(original/'PLAN.json',plan)
    if native_card:
        write(original/'CARD.json',dict(id='packet',design=plan['design'],environment=plan['environment'],plan_sha256=file_digest(original/'PLAN.json'),source_archive_sha256='source'))
    def study(root,design,pulse):
        write(root/'WITNESSES.json',dict(known=True));return dict(controls=dict(known=True))
    summary=study(original,plan['design'],lambda **kw:None)
    write(original/'SUMMARY.json',summary)
    write(original/'COMPLETE.json',dict(files=scientific_files(original)))
    monkeypatch.setattr(replay_module,'run_study',study)
    assert replay_module.replay(original,tmp_path/'replay')['passed']
    if native_card:
        complete={'files':scientific_files(original)}
        complete['files']['CARD.json']='deliberately-wrong'
        write(original/'COMPLETE.json',complete,immutable=False)
        with pytest.raises(ValueError,match='identity differs'):
            replay_module.replay(original,tmp_path/'bad-replay')

@pytest.mark.parametrize('field,values',[('fee',[.005,.08]),('review_observed',[False,True])])
def test_review_does_not_average_distinct_evidence_or_prices(tmp_path,field,values):
    roots=[]
    for index,value in enumerate(values):
        root=tmp_path/str(index);root.mkdir();roots.append(root)
        write(root/'PLAN.json',dict(design=dict(branch='G4',tier='context',**{field:value})))
        population=[]
        for method,probability in [('a',.9 if index==0 else .2),('b',.5)]:
            p=np.array([[probability,1-probability]]);y=np.array([0])
            # The scorer's seven-bit targets require the full declared label space.
            p=np.pad(p,((0,0),(0,126)));metrics=score(p,y)
            (root/'raw').mkdir(exist_ok=True)
            np.savez(root/'raw'/f'{method}_points.npz',prediction=p,truth=y,**metrics)
            write(root/'raw'/f'{method}.json',dict(lineage=24,method=method))
            means={k:float(v.mean()) for k,v in metrics.items()};means.update(acquisition_cost=0.,net_log_loss=means['log_loss'])
            population.append(dict(lineage=24,method=method,metrics=means))
        write(root/'policies/24.json',dict(costs=[0],choices={'a':[0],'b':[0]}))
        write(root/'SUMMARY.json',dict(controls={'known':True},population=population))
        write(root/'COMPLETE.json',dict(plan_sha256=file_digest(root/'PLAN.json'),files=scientific_files(root)))
    contrasts=[r for r in review(roots)['contrasts'] if r['metric']=='log_loss']
    assert len(contrasts)==2
    assert sorted(r['left_minus_right']['mean']>0 for r in contrasts)==[False,True]
