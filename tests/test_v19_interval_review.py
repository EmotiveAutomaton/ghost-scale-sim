import gzip
import json
import shutil
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import interval_review as R, source_interval as I, retrospective_source as S, reachable_retrospective as Q
from ghostscale.validation.soundingline.v18_3.io import write, read, file_digest, canonical
from test_v19_reachable_retrospective import fixture, law
from test_v19_source_interval import example


@pytest.mark.parametrize('mode',['random','uniform','zeros','disjoint'])
@pytest.mark.parametrize('sparse',[False,True])
def test_scalar_reconstruction_and_direct_bayes(mode,sparse):
    spec=fixture();st=Q.prepare(spec,[.2,.3,.4,.1]);W=np.array([[.6,.1,.2,.1],[0,.5,0,.5]])
    if sparse:W[0]=[1,0,0,0]
    p=law(mode);ids=[]
    for row,w in enumerate(W):
        for t in (1,2):
            for c in range(4):
                for e in range(8):
                    if w@p[st['past'][t-1],c,e]>0:ids.append([row,t,c,e])
    parent=S.evaluate(W,st,p,ids);saved=I.factors(parent);result=R.reconstruct(parent,saved)
    expected=I.measures(saved)
    for key in result:np.testing.assert_allclose(result[key],expected[key],atol=1e-14,rtol=0)
    future=p[st['future']].reshape(len(st['mapping']),-1)
    for n,(row,t,c,old) in enumerate(ids):
        prior=W[row];likelihood=p[st['past'][t-1],c,old]
        for j,(lo,hi) in enumerate(I.INTERVALS):
            forecasts=[]
            for alpha in np.linspace(lo,hi,9):
                weights=prior*(alpha+(1-alpha)*likelihood);weights/=weights.sum()
                forecasts.append(weights@future)
            width=np.ptp(forecasts,axis=0).max()
            radius=max(abs(f-forecasts[4]).max() for f in forecasts)
            np.testing.assert_allclose(result['old_endpoint_width_max_future_difference'][n,j],width,atol=1e-14,rtol=0)
            np.testing.assert_allclose(result['old_endpoint_midpoint_worst_max_future_difference'][n,j],radius,atol=1e-14,rtol=0)


@pytest.mark.parametrize('field',list(I.factors(example())))
def test_every_raw_factor_corruption_refused(field):
    parent=example();saved={k:v.copy() for k,v in I.factors(parent).items()}
    if saved[field].dtype==bool:saved[field].flat[0]=not saved[field].flat[0]
    elif field=='source_rows':saved[field].flat[0]+=1
    else:saved[field].flat[0]+=.01
    with pytest.raises(ValueError):R.reconstruct(parent,saved)


def test_singleton_and_support_controls():
    parent=example();intervals=((0,0),(.5,.5),(1,1))
    values=R.reconstruct(parent,I.factors(parent,intervals),intervals)
    assert (values['old_endpoint_width_group_tv']==0).all()
    assert values['supported_endpoints'].tolist()==[[2,2,1]]
    assert all(R.controls().values())


@pytest.mark.parametrize('bad',[(),((1,0),),((-.1,1),),((0,1.1),),((float('nan'),1),)])
def test_invalid_intervals(bad):
    with pytest.raises(ValueError):R.reconstruct(example(),I.factors(example()),bad)


def complete_fixture(tmp_path):
    from test_v19_source_interval import complete_fixture as make
    original,p=make(tmp_path);summary=I.run(original,p,lambda **kw:None)
    write(original/'PLAN.json',p);write(original/'SUMMARY.json',summary)
    root=tmp_path/'interval-review';(root/'inputs').mkdir(parents=True)
    shutil.copytree(original,root/'inputs/original');shutil.copytree(original/'inputs/source',root/'inputs/parent')
    return root,dict(design=dict(target_plan_sha256=file_digest(original/'PLAN.json'),input_files={}))


def test_complete_checker(tmp_path):
    root,p=complete_fixture(tmp_path);result=R.run(root,p,lambda **kw:None)
    assert result['passed'] and not result['numerical_acceptance']
    assert result['sources']==768 and result['rows']==2048 and result['strata']==16 and result['raw_batches']==12


@pytest.mark.parametrize('bad',['input','parent_plan','acceptance','timing','extra_raw','missing_raw','source','denominator','summary','duplicate'])
def test_complete_corruption(tmp_path,bad):
    root,p=complete_fixture(tmp_path);original=root/'inputs/original';parent=root/'inputs/parent'
    if bad=='input':p['design']['input_files']={'original/PLAN.json':'wrong'}
    elif bad=='parent_plan':(parent/'PLAN.json').write_text('{}')
    elif bad=='acceptance':(parent/'FINAL_REVIEW.json').write_text('{"numerical_acceptance":false}')
    elif bad=='timing':(original/'TIMING.jsonl').write_text('')
    elif bad=='extra_raw':np.savez(original/'raw/extra_points.npz',x=[1])
    elif bad=='missing_raw':next((original/'raw').glob('*.npz')).unlink()
    elif bad=='source':
        f=next((parent/'evaluator').glob('*-bindings.json'));r=read(f);r['sources'][0][1]+=1;f.write_text(json.dumps(r))
    else:
        f=original/'raw/interval_summary_points.json.gz';rows=json.loads(gzip.decompress(f.read_bytes()))
        if bad=='denominator':rows[0]['report_sources']+=1
        elif bad=='summary':rows[0]['old_endpoint_width_group_tv']+=.01
        else:rows.append(rows[0])
        f.write_bytes(gzip.compress(canonical(rows),mtime=0))
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):R.run(root,p,lambda **kw:None)
