import copy
import gzip
import json
from itertools import product
from pathlib import Path
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import sufficient_review as R
from ghostscale.validation.soundingline.v19 import retrospective_sufficient as S
from ghostscale.validation.soundingline.v19 import reachable_retrospective as Q
from ghostscale.validation.soundingline.v18_3.io import write, file_digest, canonical
from test_v19_reachable_retrospective import fixture, law


@pytest.mark.parametrize('mode',['random','uniform','zeros','disjoint'])
@pytest.mark.parametrize('sparse',[False,True])
def test_independent_complete_batch(mode,sparse):
    spec=fixture();W=np.array([[.6,.1,.2,.1],[0,.5,0,.5]])
    if sparse:W[0]=[1,0,0,0]
    p=law(mode);st=R.structure(spec)
    raw,supports,metrics=S.evaluate(W,Q.prepare(spec,[.2,.3,.4,.1]),p)
    storage,possible,error=R.verify_batch(W,st,p,raw)
    assert storage['full_weight_values']==4 and storage['group_weight_values']==3
    assert storage['reports']==64 and error<1e-12
    assert possible.tolist()==sum((raw[f'{t}-possible'].sum(-1) for t in (1,2))).tolist()
    # Scalar full-hypothesis posterior, complete future coordinate expansion.
    for row,w in enumerate(W):
        for t in range(1,3):
            for c,e in product(range(4),range(8)):
                numerator=np.array([sum(w[h]*p[st['past'][t-1][h],c,e] for h in m) for m in st['members']])
                d=numerator.sum()
                if not d:continue
                exact=numerator/d
                s=st['supports'][t-1];pairs=np.array(s['pairs']);mask=np.array(s['mixed'])
                masses=np.empty(len(pairs));masses[mask]=raw[f'{t}-mixed_joint_mass'][row]
                masses[~mask]=raw['group_weights'][row,pairs[~mask,0]]
                other=np.array([sum(masses[j]*p[state,c,e] for j,(g,state) in enumerate(pairs) if g==group) for group in range(3)])
                other/=other.sum();tv=.5*abs(other-exact).sum()
                for future in st['signatures'].T:
                    assert abs(sum((other[g]-exact[g])*p[state] for g,state in enumerate(future))).max()<=tv+1e-15


@pytest.mark.parametrize('field',['group_weights','1-mixed_joint_mass','2-mixed_joint_mass','1-report_probability','1-possible','1-updated_group_tv','1-report_probability_error'])
def test_corrupt_evidence_rejected(field):
    spec=fixture();W=np.array([[.6,.1,.2,.1]]);p=law('random')
    raw,_,_=S.evaluate(W,Q.prepare(spec,[.2,.3,.4,.1]),p)
    if raw[field].dtype==bool:raw[field].flat[0]=not raw[field].flat[0]
    else:raw[field].flat[0]+=.01
    with pytest.raises(ValueError):R.verify_batch(W,R.structure(spec),p,raw)


@pytest.mark.parametrize('field',['membership','signatures','checkpoint','hypotheses'])
def test_corrupt_structure_rejected(field):
    spec=fixture()
    if field=='membership':spec[field][0]=1
    elif field=='signatures':spec[field][0][0]=15
    elif field=='checkpoint':spec[field]=0
    else:spec[field][0][0]='invented'
    with pytest.raises(ValueError):R.structure(spec)


def test_singleton_stationary_and_zero_cells():
    spec=dict(hypotheses=[['none',0,0],['none',0,8]],length=3,checkpoint=2,signatures=[[0,0],[8,8]],membership=[0,1])
    W=np.array([[1.,0.]]);p=law('zeros');st=R.structure(spec)
    raw,_,_=S.evaluate(W,Q.prepare(spec,[.5,.5]),p)
    storage,_,err=R.verify_batch(W,st,p,raw)
    assert storage['factored_joint_values']==2 and err==0
    assert all(R.controls().values())


def complete_fixture(tmp_path):
    original=tmp_path/'producer';(original/'inputs').mkdir(parents=True)
    hs=[['none',0,s] for s in range(16)]+[['purpose',1,8],['skill',3,2]]
    paths=[tuple(initial ^ ({'purpose':8,'skill':4}.get(kind,0) if t>change else 0) for t in range(2,5)) for kind,change,initial in hs]
    signatures=sorted(set(paths))
    spec=dict(hypotheses=hs,length=4,checkpoint=2,signatures=[list(p) for p in signatures],membership=[signatures.index(p) for p in paths])
    write(original/'inputs/SCHEDULES.json',{'4-2':spec})
    p=law('random');draws=[1,2]
    for evidence in ('aware','omitted'):
        base=original/'inputs'/evidence;(base/'raw').mkdir(parents=True);(base/'evaluator').mkdir()
        rows=[];W=[];bindings=[];st=R.structure(spec)
        for i,identity in enumerate(product(draws,range(16),('purpose','skill'),(False,True),(False,True))):
            w=np.arange(1.,19.) if evidence=='aware' else np.arange(18.,0.,-1.);w/=w.sum();W.append(w)
            forecast=sum(w[h]*p[st['signatures'][st['mapping'][h],0]] for h in range(len(hs)))
            rows.append(dict(arm='unknown-time-type',length=4,step=2,draw=identity[0],initial_maker=identity[1],kind=identity[2],switched=identity[3],duplicates=identity[4],stream=i,actual_maker=identity[1],joint_array='test-unknown-time-type',joint_row=i,forecast=forecast.tolist()))
            bindings.append([i,2])
        write(base/'evaluator/1-law.json',p.tolist())
        write(base/'evaluator/1-joint-map.json',{'test-unknown-time-type':dict(rows=bindings,hypotheses=spec['hypotheses'])})
        (base/'raw/1-forecasts_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
        np.savez_compressed(base/'raw/1-joint_points.npz',**{'test-unknown-time-type':W})
    design=dict(lineages=[1],draws=draws,lengths=[4],checkpoints=[2],batch_rows=32,input_files={p.relative_to(original/'inputs').as_posix():file_digest(p) for p in (original/'inputs').rglob('*') if p.is_file()})
    plan=dict(design=design);write(original/'PLAN.json',plan)
    summary=S.run(original,plan,lambda **kw:None);write(original/'SUMMARY.json',summary)
    review=tmp_path/'review';(review/'inputs').mkdir(parents=True)
    shutil.copytree(original,review/'inputs/original');shutil.copytree(original/'inputs',review/'inputs/parent')
    cfg=dict(target_plan_sha256=file_digest(original/'PLAN.json'),input_files={})
    return review,dict(design=cfg)


def test_complete_execution_and_pairing(tmp_path):
    root,plan=complete_fixture(tmp_path);result=R.run(root,plan,lambda **kw:None)
    assert result['rows']==512 and result['reports']==32768 and result['strata']==4
    assert result['batches']==16 and result['passed'] and not result['numerical_acceptance']


@pytest.mark.parametrize('kind',['timing','storage','bound','row','support'])
def test_complete_corruption_refused(tmp_path,kind):
    root,plan=complete_fixture(tmp_path);original=root/'inputs/original'
    if kind=='timing':
        p=original/'TIMING.jsonl';rows=p.read_text().splitlines();p.write_text('\n'.join(rows[:-1])+'\n')
    elif kind=='support':
        p=original/'evaluator/4-2-support.json';x=json.loads(p.read_text());x[0]['mapping'][0]=9;p.write_text(json.dumps(x))
    else:
        p=original/'raw/sufficient_summary_points.json.gz';rows=json.loads(gzip.decompress(p.read_bytes()))
        field={'storage':'factored_joint_values','bound':'all_future_coordinate_error_upper_bound','row':'draw'}[kind]
        rows[0][field]+=1;p.write_bytes(gzip.compress(canonical(rows),mtime=0))
    with pytest.raises((ValueError,KeyError)):R.run(root,plan,lambda **kw:None)
