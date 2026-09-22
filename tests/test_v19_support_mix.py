import gzip
import json
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import support_mix as S, practice as P
from ghostscale.validation.soundingline.v19.readout_model import save_arrays
from ghostscale.validation.soundingline.v18_3.io import write


def test_controls_and_mass():
    assert all(S.controls().values())
    a=np.full(P.SHAPE,.5); b=a.copy(); a[0,0,2,2,0,6]+=12; b[0,1,5,5,1,6]+=12
    for weight in S.WEIGHTS:
        c=S.mix_counts(a,b,weight); assert c.sum()==a.sum() and np.all(c>=.5)
    with pytest.raises(ValueError): S.mix_counts(a,b,.3)
    with pytest.raises(ValueError): S.mix_counts(a,b+1,.5)


def test_independent_native_and_path_evaluation():
    world=P.W.law(-9621); law=S.native_kernel(world['action_rate'])
    assert np.allclose(law,P.native_kernel(world),rtol=0,atol=1e-12)
    pi=np.random.default_rng(762).integers(3,size=P.SHAPE[:-2])
    assert np.allclose(S.exhaustive_success(law,pi),P.evaluate(law,pi)[0],rtol=0,atol=1e-12)


def test_complete_fixture_and_corruption(tmp_path):
    root=tmp_path/'fixture'; parent=root/'inputs/parent'; parent.mkdir(parents=True)
    world=P.W.law(-9622); kernel=S.native_kernel(world['action_rate'])
    write(root/'inputs/ENUMERATION.json',dict(lineages=[dict(lineage=-9622,world=world)]))
    save_arrays(parent/'evaluator/law--9622.npz',kernel=kernel)
    rows=[]
    for condition in ('restricted-start','matched-start'):
        logs=[]
        for episode in range(4):
            ctx=episode%2 if condition=='matched-start' else 0; a=u=P.INITIAL[ctx]
            for step in range(3):
                goal=step%3; b=int(np.argmax(kernel[step,ctx,a,u,goal])); logs.append([step,ctx,a,u,goal,b]);u,a=a,b
        log=np.asarray(logs)
        for arm in ('active','replay','demonstration'):
            base=f'-9622-1-2-{condition}'; save_arrays(parent/'observed'/f'{base}-{arm}.npz',transitions=log)
            counts=S.reconstruct(log,4);pi,values=P.policy(counts);success=S.exhaustive_success(kernel,pi)
            save_arrays(parent/'models'/f'{base}-4-{arm}.npz',counts=counts,policy=pi,learned_values=values)
            rows.append(dict(lineage=-9622,draw=1,policy_seed=2,condition=condition,episodes=4,arm=arm,success=float(success.mean()),context_success=success.tolist()))
    write(parent/'POINTS.json',rows)
    design=dict(weights=list(S.WEIGHTS),lineages=[-9622],training_draws=[1],policy_seeds=[2],budgets=[4],input_files={})
    summary=S.run(root,dict(design=design),lambda **kw:None)
    assert summary['rows']==10 and summary['parent_tables']==6 and all(summary['controls'].values())
    points=json.loads(gzip.decompress((root/'raw/support_mix_points.json.gz').read_bytes()))
    assert all(p['feedback_mass']==12 for p in points)
    assert all(p['policy_changes_by_context']==[0,0] for p in points if p['matched_weight']==0)
    rows[0]['success']+=.1;write(parent/'POINTS.json',rows,immutable=False)
    other=tmp_path/'corrupt';shutil.copytree(root/'inputs',other/'inputs')
    with pytest.raises(ValueError,match='endpoint reproduction'):S.run(other,dict(design=design),lambda **kw:None)
