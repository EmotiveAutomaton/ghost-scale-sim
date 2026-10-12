"""One bounded B07 instrumentation completion; original identities stay immutable.

Reconstruct the original deterministic fits to retain their missing native
forecasts. Each original shifted forecast must reproduce before any new output
is usable. Native count readers receive the same k shifted labels by additive
counts, with the original one-half pseudocount retained. This is a separately
registered diagnostic, not a replacement primary rival or changed criterion.
"""
from pathlib import Path
import numpy as np
from ..v18_3.io import read, write, file_digest
from . import oracle as O
from .adaptation import policy_view, execution_table, execution_weight, modular, finetune
from .readers import Direct
from .studies import data, retain, controls
from .audit import counts, forecast


def independent_count_forecast(x, y, test, support, drop=()):
    """Independent probability-domain reconstruction, including pseudocounts."""
    probability = np.tile((np.bincount(y, minlength=128)+.5).astype(np.longdouble), (len(test),1))
    probability /= len(y)+64
    for feature in range(x.shape[1]):
        if feature % 13 in drop:
            continue
        for value in (-1,0,1):
            selected = test[:,feature] == value
            numerator = np.bincount(y[x[:,feature]==value], minlength=128)+.5
            denominator = np.bincount(y, minlength=128)+1.5
            probability[selected] *= numerator/denominator
    probability *= support
    probability /= probability.sum(axis=1,keepdims=True)
    return np.asarray(probability,float)


def run(root, design, pulse):
    original=Path(design['original_root']); native=Path(design['native_headroom_root'])
    shifted=Path(design['headroom_root']); bound=read(original/'COMPLETE.json')
    assert file_digest(original/'COMPLETE.json') == design['original_complete_sha256']
    assert file_digest(native/'COMPLETE.json') == design['native_complete_sha256']
    assert file_digest(shifted/'COMPLETE.json') == design['shifted_complete_sha256']
    source_design=read(original/'PLAN.json')['design']
    for key in ('training_lineages','validation_lineages','training_cases','validation_cases',
                'lineages','test_cases','fit_seeds','tier','history','history_tier','history_observed','regime'):
        assert design[key]==source_design[key], key
    assert read(shifted/'SUMMARY.json')['screen_admitted']
    assert all(f['early_stopped'] for f in read(shifted/'SUMMARY.json')['fit_records'])
    population=[]; max_error=0.; count_error=0.; reproduced=0
    (root/'models').mkdir(parents=True, exist_ok=False)
    for fit in design['fit_seeds']:
        x,y=data(design,'training',fit);vx,vy=data(design,'validation',fit)
        policy=Direct(policy_view(x),y,policy_view(vx),vy,fit_seed=fit,pulse=pulse)
        direct=Direct(x,y,vx,vy,fit_seed=fit,pulse=pulse)
        def fresh(n,step):
            return data(dict(design,training_lineages=[72000000+fit*100000+step],training_cases=n),'training',fit)
        fresh_model=Direct(x,y,vx,vy,fit_seed=fit,pulse=pulse,fresh=fresh)
        for name,model in [('policy',policy),('direct',direct),('fresh',fresh_model)]:
            model.save(root/'models'/f'{fit}-{name}_points.npz')
        with np.load(original/'training'/f'{fit}-adaptation_points.npz') as z:
            ax,ay=z['evidence'],z['labels']
        assert file_digest(original/'training'/f'{fit}-adaptation_points.npz')==bound['files'][f'training/{fit}-adaptation_points.npz']
        table=execution_table(x,y)
        for k in (0,16,64,256):
            weight=execution_weight(table,ax[:k],ay[:k]) if k else np.array([1.,0.])
            direct_k=finetune(direct,ax[:k],ay[:k]); fresh_k=finetune(fresh_model,ax[:k],ay[:k])
            nx,ny=np.concatenate([x,ax[:k]]),np.concatenate([y,ay[:k]])
            nb=counts(nx,ny)
            for world in design['lineages']:
                for label,source in [('native',native),('shifted',shifted)]:
                    pulse(phase='adaptation-controls',fit=fit,k=k,lineage=world,condition=label)
                    with np.load(source/'reader'/f'{world}_points.npz') as z:xx=z['evidence']
                    with np.load(source/'evaluator'/f'{world}_points.npz') as z: yy=z['truth'];support=np.isfinite(z['supplied_logp'])
                    predictions={'modular':modular(policy,table,xx,weight),
                        'direct-finetuned':direct_k.predict(xx),'fresh-law-finetuned':fresh_k.predict(xx)}
                    for reader in predictions:
                        predictions[reader]=O.normalize_log(np.where(support,predictions[reader],-np.inf))
                    for reader,drop in [('naive-bayes',()),('naive-bayes-without-evidence',(1,))]:
                        lp=forecast(nb,xx,drop,support=support)
                        independent=independent_count_forecast(nx,ny,xx,support,drop)
                        error=float(np.max(np.abs(independent-np.exp(lp))));count_error=max(count_error,error)
                        assert error<1e-10, 'count reference mismatch'
                        predictions[reader]=lp
                    for reader,lp in predictions.items():
                        if label=='shifted' and reader in ('modular','direct-finetuned','fresh-law-finetuned'):
                            old=original/'raw'/f'{fit}-{k}-{world}-{reader}_points.npz'
                            assert file_digest(old)==bound['files'][old.relative_to(original).as_posix()]
                            with np.load(old) as z:
                                assert np.array_equal(z['truth'],yy)
                                error=float(np.max(np.abs(np.exp(lp)-np.exp(z['logp']))))
                            max_error=max(max_error,error);assert error<1e-10,'original fit reproduction'
                            reproduced+=1
                            # Original shifted arrays are already retained and verified.
                            continue
                        metrics=retain(root,f'{fit}-{k}-{world}-{label}-{reader}',lp,yy)
                        population.append(dict(fit=fit,k=k,lineage=world,condition=label,reader=reader,metrics=metrics))
    write(root/'REPRODUCTION.json',dict(passed=True,original_forecast_blocks=reproduced,
        maximum_probability_error=max_error,independent_count_error=count_error))
    return dict(card='G21-B07',population=population,controls=dict(controls(),original_reproduced=True,
        independent_count_reference=True),scope='B07 missing-control supplement; no new target or primary criterion.',
        limitations=['Registered native/evidence-flip kernel library is supplied privilege.',
            'Exact support remains supplied to every arm.',
            'Count adaptation adds the same k labels to native counts; it is a diagnostic, not a replacement primary rival.'])
