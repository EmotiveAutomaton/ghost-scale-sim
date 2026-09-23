import gzip
import json
import math
from itertools import product
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import witnessed_goal_factorization as F
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from test_v19_joint_factorization import fixture as original_fixture


def fixture(root):
    plan=original_fixture(root);base=root/'inputs';cfg=plan['design'];cfg['tiers']=['E2-full']
    packets=read(base/'reader/PACKETS.json')['packets'];keys=sorted(k for k,p in packets.items() if p['tier']=='E2-full')
    refs=read(base/'evaluator/REFERENCES.json');masks=np.arange(F.N)[None,:]%216==172
    ref=F.S.reference_arrays(refs,'E2-full',keys,cfg['development_lineages'],masks)
    out=base/'support';out.mkdir();np.savez_compressed(out/'E2-full-support.npz',masks=masks);write(out/'E2-full-frames.json',keys)
    rows=[]
    for draw,seed,budget in product(cfg['training_draws'],cfg['fit_seeds'],cfg['budgets']):
        for arm in F.ARMS:
            with np.load(base/'forecasts'/f'{draw}-E2-full-{seed}-{budget}-{arm}.npz') as z:a=z['alphabet'];q=F.distribution(z['probabilities'],a,budget)
            r=F.S.restrict(q,masks)[0]
            for label,prob in ((arm,q),(arm+'-restricted',r),*((('uniform-support',masks/27),) if arm=='matched-frequency' else ())):
                scores=F.S.scores(prob,a,ref)
                for li,l in enumerate(cfg['development_lineages']):rows.append(dict(tier='E2-full',draw=draw,seed=seed,budget=budget,arm=label,lineage=l,frames=1,**{k:float(v[li]) for k,v in scores.items()}))
    (out/'support_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    cfg['input_files']={p.relative_to(base).as_posix():file_digest(p) for p in base.rglob('*') if p.is_file()}
    return plan


def test_controls():assert all(F.controls().values())


def test_scalar_conditional_marginals_and_products():
    q=np.arange(1,F.N+1,dtype=float);q/=q.sum();op=71
    r,f,m,z=F.factorize(q[None],np.array([op]));labels=[216*g+op for g in range(27)]
    normal=math.fsum(q[k] for k in labels);cond=[q[k]/normal for k in labels]
    marg=[[math.fsum(cond[g] for g in range(27) if g//3**(2-t)%3==v) for v in range(3)] for t in range(3)]
    expected=[math.prod(marg[t][g//3**(2-t)%3] for t in range(3)) for g in range(27)]
    assert abs(z[0]-normal)<1e-15 and np.allclose(m[0],marg,rtol=0,atol=1e-15)
    assert np.allclose(r[0,labels],cond,rtol=0,atol=1e-15) and np.allclose(f[0,labels],expected,rtol=0,atol=1e-15)
    assert not np.any(f[0,np.arange(F.N)%216!=op])
    wrong=[math.prod(marg[t][g//3**t%3] for t in range(3)) for g in range(27)]
    assert np.max(abs(f[0,labels]-wrong))>1e-3


def test_equal_single_marginals_distinct_goal_paths():
    a=np.zeros((1,F.N));b=a.copy();a[0,[0,1728]]=.5;b[0,[432,1296]]=.5
    ra,fa,*_=F.factorize(a,np.array([0]));rb,fb,*_=F.factorize(b,np.array([0]))
    assert np.array_equal(fa,fb) and not np.array_equal(ra,rb)
    assert abs(-.5*math.log(fa[0,0])-.5*math.log(fa[0,1728])-math.log(4))<1e-12


def test_unknown_and_point_identity():
    q=F.distribution(np.array([[32/33]]),np.array([0]),32)
    r,f,_,_=F.factorize(q,np.array([1]));assert np.allclose(r,f,rtol=0,atol=1e-15)
    assert np.allclose(r[0,216*np.arange(27)+1],1/27,rtol=0,atol=1e-15)
    q[:]=0;q[0,5831]=1
    assert np.array_equal(F.factorize(q,np.array([215]))[1],q)


@pytest.mark.parametrize('problem',['zero','wrong-witness','negative','nan','mass','shape','float-op','out-of-range'])
def test_invalids(problem):
    q=np.ones((1,F.N))/F.N;ops=np.array([1])
    if problem in ('zero','wrong-witness'):q[:]=0;q[0,0]=1
    if problem=='negative':q[0,0]=-1
    if problem=='nan':q[0,0]=np.nan
    if problem=='mass':q*=2
    if problem=='shape':q=q[0]
    if problem=='float-op':ops=np.array([1.])
    if problem=='out-of-range':ops=np.array([216])
    with pytest.raises(ValueError):F.factorize(q,ops)


def test_complete_handler(tmp_path):
    root=tmp_path/'packet';plan=fixture(root);s=F.run(root,plan,lambda **kw:None)
    assert s['rows']==192 and s['native_rows']==6 and s['packets']==1 and s['original_cells_reproduced']==64
    assert s['support_max_error']<1e-10 and len(s['contrasts'])==8 and len(s['normalized_log_budget_area'])==4 and all(s['controls'].values())


@pytest.mark.parametrize('problem',['support-score','mask','witness','private','hash','missing'])
def test_corruption(tmp_path,problem):
    root=tmp_path/'packet';plan=fixture(root);base=root/'inputs'
    if problem=='mask':
        p=base/'support/E2-full-support.npz';m=np.zeros((1,F.N),bool);np.savez_compressed(p,masks=m)
    elif problem in ('witness','private'):
        p=base/'reader/PACKETS.json';v=read(p);packet=next(v for v in v['packets'].values() if v['tier']=='E2-full')
        if problem=='witness':packet['inputs']['observations'][0]['operation']='undo'
        else:packet['inputs']['goal']=0
        write(p,v,immutable=False)
    else:
        p=base/'support/support_points.json.gz';v=json.loads(gzip.decompress(p.read_bytes()))
        if problem=='missing':v.pop()
        else:next(r for r in v if r['arm'].endswith('-restricted'))['loss']+=.1
        p.write_bytes(gzip.compress(canonical(v),mtime=0))
    if problem!='hash':plan['design']['input_files'][p.relative_to(base).as_posix()]=file_digest(p)
    with pytest.raises(ValueError):F.run(root,plan,lambda **kw:None)
