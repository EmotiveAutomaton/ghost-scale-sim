"""C02 complete consumer on frozen matched headroom inputs."""
from pathlib import Path
import numpy as np
from ..v18_3.io import read,write,file_digest
from ..v20.readers import features
from . import world as W,oracle as O
from .joint_architectures import JointReader,KINDS
from .joint_check import check_saved
from .studies import retain


def run(root,d,pulse):
    native=Path(d['headroom_root']);shifted=Path(d['shifted_headroom_root'])
    for p,key in ((native,'headroom_complete_sha256'),(shifted,'shifted_headroom_complete_sha256')):
        assert file_digest(p/'COMPLETE.json')==d[key]
        complete=read(p/'COMPLETE.json')
        for name,h in complete['files'].items():assert file_digest(p/name)==h
        s=read(p/'SUMMARY.json');plan=read(p/'PLAN.json')['design']
        if not d.get('engineering_only'):
            assert plan['law_count']=='exact' and s['screen_admitted'] and s['headroom']['lower']>.02
            assert len(s['fit_records'])==2 and all(r['early_stopped'] for r in s['fit_records'])
        for field in ('tier','history','history_tier','history_observed','training_cases','training_lineages','validation_cases','validation_lineages','lineages','test_cases','fit_seeds','regime'):
            assert plan.get(field)==d.get(field),field
    population=[];fits=[];maximum=0.
    for fit in d['fit_seeds']:
        with np.load(native/'training'/f'{fit}_points.npz') as z:data={k:z[k] for k in z.files}
        with np.load(shifted/'training'/f'{fit}_points.npz') as z:
            for k in data:assert np.array_equal(data[k],z[k])
        x,y,vx,vy=[data[k] for k in ('evidence','labels','validation_evidence','validation_labels')]
        for budget in d['parameter_budgets']:
            for kind in KINDS:
                model=JointReader(features(x).shape[1],kind,budget,fit).fit(x,y,vx,vy,d.get('max_epochs',120),pulse)
                fits.append(model.metadata);model_dir=root/'models';model_dir.mkdir(parents=True,exist_ok=True)
                model.save(model_dir/f'{fit}-{budget}-{kind}_points.npz')
                for shift,source in (('native',native),('changed-tool',shifted)):
                    for world in d['lineages']:
                        with np.load(source/'reader'/f'{world}_points.npz') as z:xx=z['evidence']
                        with np.load(source/'evaluator'/f'{world}_points.npz') as z:yy=z['truth'];support=np.isfinite(z['supplied_logp'])
                        maximum=max(maximum,check_saved(model,xx))
                        lp=model.predict(xx);marginal=np.exp(lp)@W.LABELS
                        product=np.sum(np.where(W.LABELS[None,:,:],np.log(marginal)[:,None,:],np.log1p(-marginal)[:,None,:]),axis=2)
                        arms={'unmasked':lp,'common-support':O.normalize_log(np.where(support,lp,-np.inf)),'product-common-support':O.normalize_log(np.where(support,product,-np.inf))}
                        for arm,prediction in arms.items():
                            metrics=retain(root,f'{fit}-{budget}-{kind}-{shift}-{world}-{arm}',prediction,yy)
                            population.append(dict(fit=fit,budget=budget,architecture=kind,shift=shift,lineage=world,arm=arm,metrics=metrics,incompatible_mass=float((np.exp(prediction)*(~support)).sum(1).mean())))
        for shift,source in (('native',native),('changed-tool',shifted)):
            for world in d['lineages']:
                for name in ('converged-direct','naive-bayes','naive-bayes-without-evidence'):
                    with np.load(source/'raw'/f'{fit}-{world}-{name}_points.npz') as z:lp=z['logp'];yy=z['truth']
                    population.append(dict(fit=fit,budget=0,architecture=name,shift=shift,lineage=world,arm='common-support',metrics=retain(root,f'{fit}-reference-{name}-{shift}-{world}',lp,yy)))
    write(root/'FIT_METADATA.json',fits)
    return dict(card='G21-C02',population=population,fit_records=fits,independent_max_probability_error=maximum,controls=dict(independent_probabilities=maximum<1e-10,matched_training=True,source_headroom=True),scope='Finite constructed learned-method parameter frontier; both bit orders, all widths and convergence outcomes retained.',limitations=['Approximate parameter matching does not match function class or fit CPU; retain actual costs and counts.','All fitted heads share native training/validation; shifted forecasts use no shifted labels.','Exact support is supplied equally only to named projected arms.','Capped fits remain unresolved, never converged successes.','Development worlds and repeated control forecasts are not independent replications or confirmation.','Low-rank mixture has four components; AR uses logistic previous-bit interactions; pairwise head has unary and pair terms.','This initial original-law cell does not exhaust wider-law or label/diversity extensions.'])
