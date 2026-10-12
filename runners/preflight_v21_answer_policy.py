"""Complete engineering-only answer consumers and an independent saved-score check."""
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v21.answer_policy import run
from ghostscale.validation.soundingline.v21.answer_policy_check import check,reference_frontier
from ghostscale.validation.soundingline.v21.runtime import scientific_files
from ghostscale.validation.soundingline.v21.studies import retain


def fixture(base,cases=4,readers=1,fits=1):
    source=base/'source';source.mkdir(parents=True)
    rng=np.random.default_rng(71000850);truth=rng.integers(128,size=cases);population=[]
    for fit in range(fits):
        for reader in range(readers):
            p=rng.dirichlet(np.full(128,.4),size=cases)
            metrics=retain(source,f'{fit}-71000850-reader-{reader}',np.log(p),truth)
            population.append(dict(fit=fit,lineage=71000850,reader=f'reader-{reader}',metrics=metrics))
    write(source/'PLAN.json',dict(design=dict(lineages=[71000850],engineering_only=True)))
    write(source/'SUMMARY.json',dict(population=population))
    write(source/'COMPLETE.json',dict(plan_sha256=file_digest(source/'PLAN.json'),files=scientific_files(source)))
    return dict(handler='answer-policy',card='G21-A03',engineering_only=True,lineages=[71000850],forecast_packets=[dict(root=str(source),cell='fixture',complete_sha256=file_digest(source/'COMPLETE.json'))])


def preflight(base):
    d=fixture(base); hashes=[];observations=0
    for i in range(2):
        root=base/f'consumer-{i}';root.mkdir();write(root/'PLAN.json',dict(design=d));s=run(root,d,lambda **k:None);assert all(s['controls'].values());write(root/'SUMMARY.json',s)
        with np.load(base/'source/raw/0-71000850-reader-0_points.npz') as z:p=np.exp(z['logp']);truth=z['truth']
        for ratio in (1.,2.,5.,10.):
            for policy in ['adaptive']+[f'fixed-{k}' for k in range(8)]:
                path=root/'raw'/f'fixture-0-71000850-reader-0-{ratio}-{policy}_points.npz'
                with np.load(path) as z:a={k:z[k] for k in z.files}
                check(p,truth,a,ratio);observations+=len(truth)
                for j,prob in enumerate(p):
                    m,v,q=reference_frontier(prob);loss=[(7-k)/7+ratio*(1-q[k]) for k in range(8)]
                    k=int(policy.split('-')[1]) if policy!='adaptive' else next(k for k,x in enumerate(loss) if x<=min(loss)+1e-12)
                    assert a['mask'][j]==m[k] and a['value'][j]==v[k]
        files=scientific_files(root);write(root/'COMPLETE.json',dict(plan_sha256=file_digest(root/'PLAN.json'),files=files));hashes.append(files)
    assert hashes[0]==hashes[1]
    return dict(passed=True,engineering_only=True,complete_consumers=2,scientific_files=len(hashes[0]),policy_observations_checked=observations,full_scientific_replay=False)
