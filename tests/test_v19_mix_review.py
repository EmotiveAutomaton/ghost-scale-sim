import gzip
import json
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import mix_review as V, support_mix as S, practice as P
from ghostscale.validation.soundingline.v19.readout_model import save_arrays
from ghostscale.validation.soundingline.v18_3.io import write, file_digest


def fixture(root):
    original=root/'inputs/original';parent=original/'inputs/parent';parent.mkdir(parents=True)
    world=P.W.law(-9722);law=S.native_kernel(world['action_rate'])
    write(original/'inputs/ENUMERATION.json',dict(lineages=[dict(lineage=-9722,world=world)]))
    save_arrays(parent/'evaluator/law--9722.npz',kernel=law)
    rows=[]
    for condition in ('restricted-start','matched-start'):
        log=[]
        for episode in range(4):
            ctx=episode%2 if condition=='matched-start' else 0;a=u=P.INITIAL[ctx]
            for t in range(3):
                goal=t%3;b=int(np.argmax(law[t,ctx,a,u,goal]));log.append([t,ctx,a,u,goal,b]);u,a=a,b
        log=np.asarray(log)
        for arm in ('active','replay','demonstration'):
            stem=f'-9722-1-2-{condition}';save_arrays(parent/'observed'/f'{stem}-{arm}.npz',transitions=log)
            counts=S.reconstruct(log,4);pi,values=P.policy(counts);success=S.exhaustive_success(law,pi)
            save_arrays(parent/'models'/f'{stem}-4-{arm}.npz',counts=counts,policy=pi,learned_values=values)
            rows.append(dict(lineage=-9722,draw=1,policy_seed=2,condition=condition,episodes=4,arm=arm,success=float(success.mean()),context_success=success.tolist()))
    write(parent/'POINTS.json',rows)
    design=dict(weights=list(S.WEIGHTS),lineages=[-9722],training_draws=[1],policy_seeds=[2],budgets=[4],input_files={})
    write(original/'PLAN.json',dict(design=design))
    summary=S.run(original,dict(design=design),lambda **kw:None);write(original/'SUMMARY.json',summary)
    return dict(design=dict(input_files={},target_plan_sha256=file_digest(original/'PLAN.json'),bootstrap_seed=190501,bootstrap_resamples=100))


def test_known_laws_and_independent_policy():
    assert all(V.controls().values())
    law=P.native_kernel(P.W.law(-9721));assert np.allclose(law,V.kernel(P.W.law(-9721)['action_rate']))
    counts=np.random.default_rng(72).integers(1,30,size=P.SHAPE)+.5
    pi,values=P.policy(counts);assert V.policy_check(counts,pi,values)[0]<1e-12
    assert np.allclose(V.score(law,pi),P.evaluate(law,pi)[0],rtol=0,atol=1e-12)


def test_complete_fixture(tmp_path):
    plan=fixture(tmp_path);summary=V.run(tmp_path,plan,lambda **kw:None)
    assert summary['passed'] and summary['rows']==10 and summary['parent_tables']==6
    assert summary['means']==10 and summary['contrasts']==15 and summary['metrics']==12


@pytest.mark.parametrize('kind',['count','score','order'])
def test_corruption_rejected(tmp_path,kind):
    plan=fixture(tmp_path);original=tmp_path/'inputs/original'
    if kind=='count':
        p=next((original/'models').glob('*.npz'));z=V.arrays(p);z['counts'].flat[0]+=1;np.savez(p,**z)
    elif kind=='score':
        p=original/'raw/support_mix_points.json.gz';rows=json.loads(gzip.decompress(p.read_bytes()));rows[0]['success']+=.1
        p.write_bytes(gzip.compress(json.dumps(rows).encode()))
    else:
        p=next((original/'inputs/parent/observed').glob('*demonstration.npz'));z=V.arrays(p);z['transitions'][0,0]=2;np.savez(p,**z)
    with pytest.raises(ValueError):V.run(tmp_path,plan,lambda **kw:None)


def test_pairing_keeps_fits_inside_lineage():
    design=dict(lineages=[1,2],training_draws=[3,4],policy_seeds=[5,6],budgets=[4]);rows=[]
    for lineage in design['lineages']:
        for draw in design['training_draws']:
            for seed in design['policy_seeds']:
                for arm in ('active','demonstration'):
                    for weight in V.WEIGHTS:
                        value=lineage+draw+seed+weight*(1 if arm=='active' else 0)
                        rows.append(dict(lineage=lineage,draw=draw,policy_seed=seed,episodes=4,arm=arm,matched_weight=weight,
                            success=value,restricted_success=value,context_success=[value,value],visited_by_context=[0,0],
                            newly_visited_by_context=[0,0],policy_changes_by_context=[0,0],changed_seen_probabilities_by_context=[0,0]))
    result=V.regroup(rows,design,dict(bootstrap_seed=190501,bootstrap_resamples=100))
    adjacent=next(r for r in result['contrasts'] if r['contrast']=='adjacent-weight' and r['arm']=='active')
    metric=adjacent['metrics']['success'];assert metric['lineage_values']==[.25,.25] and metric['fit_means']==[.25]*4
    assert metric['low']==metric['high']==.25
    with pytest.raises(ValueError):V.regroup(rows+[rows[0]],design,dict(bootstrap_seed=190501,bootstrap_resamples=100))
