"""Complete V21 block consumers. Conditional cards are not implicit dispatch."""
from pathlib import Path
import time
import shutil
import numpy as np
from ..v18_3.io import read,write,file_digest
from ..v20 import readers as oldreaders,checker as oldchecker
from . import world as W,oracle as O,metrics as M
from .readers import Direct,SharedMaker,GroupedTable
from .audit import counts,forecast

def data(d,role,fit_seed=0):
    ids=d[role+'_lineages'];n=d[role+'_cases'];xs=[];ys=[]
    for i,lineage in enumerate(ids):
        rows=W.enumerate_world(lineage,d.get('training_shift','native'),d.get('regime','original'))
        count=n//len(ids)+(i<n%len(ids))
        chosen=W.sampled(rows,count,'v21',role,fit_seed,lineage,n)
        x,_=W.inputs(rows,chosen,d['tier'],d['history'],(role,fit_seed,lineage,n),
            d.get('observed',False),d.get('history_tier'),d.get('history_observed',False))
        xs.append(x);ys.append(chosen[:,10].astype(int))
    return np.concatenate(xs),np.concatenate(ys)

def retain(root,name,lp,y,**extras):
    a=M.scores(lp,y);M.check_scores(lp,y,a);path=root/'raw'/(name+'_points.npz');path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,logp=lp,truth=y,**a,**extras)
    return M.mean_scores(a)

def controls():
    p=np.full((2,128),-np.inf);p[0,0]=0;p[1,1]=0;a=M.scores(p,np.array([0,1]));M.check_scores(p,np.array([0,1]),a)
    independent=np.full((2,128),-np.log(128))
    return dict(revealing_evidence=bool(np.all(a['correct']==1)),
        irrelevant_update=bool(np.array_equal(O.normalize_log(independent),independent)))

def law_set(d):
    if d.get('law_count')=='exact':return W.all_laws(d.get('shift','native'),d.get('regime','original'))
    return [W.law(73000000+i,d.get('shift','native'),d.get('regime','original')) for i in range(d.get('law_count',400))]

def expectation(p,lp):
    with np.errstate(invalid='ignore'):return -np.sum(np.where(p>0,p*lp,0),axis=1)

def ceilings(root,d,pulse):
    """A01/A02 law uncertainty and observation aliases, no fitted reader."""
    population=[]
    for lineage in d['lineages']:
        pulse(phase='ceiling-world',lineage=lineage);w=W.law(lineage,d.get('shift','native'),d.get('regime','original'));rows=W.enumerate_law(w)
        check=oldchecker.verify_world(rows,w,W.MAKERS,W.CONTEXTS)
        chosen=W.sampled(rows,d['test_cases'],'v21-test',lineage,d.get('shift','native'))
        history_rows=W.enumerate_world(lineage,d['history_shift'],d.get('regime','original')) if d.get('history_shift') else rows
        x,_=W.inputs(history_rows,chosen,d['tier'],d['history'],('v21-test',lineage,d.get('shift','native')),
            d.get('observed',False),d.get('history_tier'),d.get('history_observed',False))
        args=dict(history_tier=d.get('history_tier'),observed=d.get('observed',False),history_observed=d.get('history_observed',False),shift=d.get('shift','native'),history_shift=d.get('history_shift'))
        supplied,_=O.posterior(x,d['tier'],[w],**args)
        if d['history']==0 and d.get('law_count')=='exact':
            meanlaw=dict(route=5 if d.get('regime')=='wide' else 4,select=4,revise=4,goal=4,
                inspect=5 if d.get('regime')=='wide' else 4,skill=6.5,shift=d.get('shift','native'))
            marginal,meta=O.posterior(x,d['tier'],[meanlaw],**args)
            meta.update(exact=True,method='analytic integration of independent multilinear coefficients',represented_laws=40500 if d.get('regime')=='wide' else 12500)
        else:marginal,meta=O.posterior(x,d['tier'],law_set(d),pulse=pulse,**args)
        p=np.exp(supplied);pm=np.exp(marginal);y=chosen[:,10].astype(int)
        es=expectation(p,supplied);em=expectation(p,marginal);bs=1-(p*p).sum(1);bm=bs+((p-pm)**2).sum(1)
        for name,lp in [('supplied-law',supplied),('law-marginal',marginal)]:
            metrics=retain(root,f'{lineage}-{name}',lp,y)
            population.append(dict(lineage=lineage,reader=name,metrics=metrics))
        path=root/'reader'/f'{lineage}_points.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,evidence=x)
        path=root/'evaluator'/f'{lineage}_points.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,truth=y,supplied_logp=supplied,marginal_logp=marginal,
            irreducible_nats=es,law_uncertainty_nats=em-es,irreducible_brier=bs,law_uncertainty_brier=bm-bs,
            bit_bayes_accuracy=np.maximum(pm@W.LABELS,1-pm@W.LABELS),supported_targets=(pm>0).sum(1))
        write(root/'checks'/f'{lineage}.json',dict(world=check,law_mixture=meta))
    return dict(card=d['card'],population=population,controls=controls(),law_count=d.get('law_count',400),
        ceiling_status='EXACT' if d.get('law_count')=='exact' else 'MONTE CARLO; do not use alone to admit an information-boundary claim',
        history_tier=d.get('history_tier',d['tier']),history_observed=d.get('history_observed',False),
        scope='Constructed method; supplied-law diagnostic is privileged; law-marginal uses the generator prior')

def headroom(root,d,pulse):
    # Same block includes the complete reference, fit, forecasts, independent scores,
    # error budget, validation curves and blind/evaluator separated exports.
    if d.get('ceiling_root'):
        src=Path(d['ceiling_root']);complete=read(src/'COMPLETE.json');original=read(src/'PLAN.json')['design']
        for field in ('tier','history','history_tier','history_observed','history_shift','observed','regime','shift','lineages','test_cases','law_count'):
            if original.get(field)!=d.get(field):raise ValueError('ceiling cell differs: '+field)
        for name,h in complete['files'].items():
            if file_digest(src/name)!=h:raise ValueError('ceiling evidence changed')
            dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src/name,dest)
        summary=read(src/'SUMMARY.json');(root/'SUMMARY.json').unlink()
        write(root/'CEILING_BINDING.json',dict(complete_sha256=file_digest(src/'COMPLETE.json')))
    else:summary=ceilings(root,d,pulse)
    rows=[];fit_records=[]
    for fit in d.get('fit_seeds',[0,1]):
        x,y=data(d,'training',fit);vx,vy=data(d,'validation',fit)
        train=root/'training'/f'{fit}_points.npz';train.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(train,evidence=x,labels=y,validation_evidence=vx,validation_labels=vy)
        model=Direct(x,y,vx,vy,fit_seed=fit,width=d.get('width',64),max_epochs=d.get('max_epochs',120),pulse=pulse)
        fit_records.append(dict(fit=fit,**model.metadata));model.save(root/'training'/f'{fit}-direct-model_points.npz')
        nb=counts(x,y)
        for lineage in d['lineages']:
            xx=np.load(root/'reader'/f'{lineage}_points.npz')['evidence'];e=np.load(root/'evaluator'/f'{lineage}_points.npz');yy=e['truth'];support=np.isfinite(e['supplied_logp'])
            law_loss=-e['marginal_logp'][np.arange(len(yy)),yy].mean()
            for name,lp in [('converged-direct',model.predict(xx,support)),('naive-bayes',forecast(nb,xx,support=support)),
                            ('naive-bayes-without-evidence',forecast(nb,xx,(1,),support=support))]:
                a=retain(root,f'{fit}-{lineage}-{name}',lp,yy)
                rows.append(dict(fit=fit,lineage=lineage,reader=name,metrics=a,headroom_nats=a['log_loss']-float(law_loss)))
                p=np.exp(e['supplied_logp']);q=np.exp(lp);pm=np.exp(e['marginal_logp'])
                extra=dict(reader_excess_nats=expectation(p,lp)-expectation(p,e['marginal_logp']),
                    reader_excess_brier=((q-p)**2).sum(1)-((pm-p)**2).sum(1))
                np.savez_compressed(root/'evaluator'/f'{fit}-{lineage}-{name}-budget_points.npz',**extra)
    world_values=[np.mean([r['headroom_nats'] for r in rows if r['lineage']==w and r['reader']=='converged-direct']) for w in d['lineages']]
    ci=M.bootstrap(world_values);converged=all(r['early_stopped'] for r in fit_records)
    gate=bool(ci['lower']>.02 and d.get('law_count')=='exact' and converged)
    summary.update(population=summary['population']+rows,fit_records=fit_records,headroom=ci,
        screen_admitted=gate,screen_disposition='ADMIT' if gate else ('PARITY' if ci['upper']<=.02 else 'UNRESOLVED'),
        practical_margin_nats=.02,headroom_gate_requires='exact law marginal, both validation-stopped fits, whole-world lower bound > .02')
    write(root/'VALIDATION_CURVES.json',fit_records);return summary

def learned(root,d,pulse):
    gate=Path(d['headroom_root']);receipt=read(gate/'COMPLETE.json');s=read(gate/'SUMMARY.json')
    if file_digest(gate/'SUMMARY.json')!=receipt['files']['SUMMARY.json']:raise ValueError('headroom summary changed')
    original=read(gate/'PLAN.json')['design']
    for field in ('tier','history','history_tier','history_observed','observed','regime','shift','training_lineages','validation_lineages','training_cases','validation_cases','lineages','test_cases','fit_seeds'):
        if original.get(field)!=d.get(field):raise ValueError('headroom cell differs: '+field)
    if not s.get('screen_admitted'):
        return dict(card=d['card'],state='PARITY' if s['headroom']['upper']<=.02 else 'UNRESOLVED',
            reason='GS-4 screen not admitted by exact headroom gate',headroom_sha256=file_digest(gate/'SUMMARY.json'),population=[],controls=controls())
    population=[];fits=[]
    for fit in d.get('fit_seeds',[0,1]):
        x,y=data(d,'training',fit);vx,vy=data(d,'validation',fit)
        direct=Direct(x,y,vx,vy,fit_seed=fit,pulse=pulse);latent=SharedMaker(x,y,fit_seed=fit,pulse=pulse);grouped=GroupedTable(x,y);nb=counts(x,y)
        compatibility=oldreaders.Readers(x,y,fit_seed=fit,pulse=pulse)
        def fresh(n,step):
            lineage=72000000+fit*100000+step;dd=dict(d,training_lineages=[lineage],training_cases=n)
            return data(dd,'training',fit)
        fresh_model=Direct(x,y,vx,vy,fit_seed=fit,pulse=pulse,fresh=fresh) if d.get('fresh_law',True) else None
        fs=dict(fit=fit,direct=direct.metadata,shared=latent.metadata,fixed_schedule=compatibility.metadata(),fresh=fresh_model.metadata if fresh_model else None);fits.append(fs)
        if latent.metadata['effective_states']<2:raise ValueError('shared-state manipulation collapsed')
        for lineage in d['lineages']:
            e=np.load(gate/'evaluator'/f'{lineage}_points.npz');xx=np.load(gate/'reader'/f'{lineage}_points.npz')['evidence'];yy=e['truth'];support=np.isfinite(e['supplied_logp'])
            methods={'converged-direct':direct.predict(xx,support),'shared-maker':latent.predict(xx,support),
                'grouped-empirical':grouped.predict(xx,support),'naive-bayes':forecast(nb,xx,support=support),
                'naive-bayes-without-evidence':forecast(nb,xx,(1,),support=support)}
            for name in ('direct-table','direct-mlp'):
                with np.errstate(divide='ignore'):methods['fixed-schedule-mlp' if name=='direct-mlp' else name]=np.log(compatibility.predict(xx,name,support))
            if fresh_model:methods['fresh-law-direct']=fresh_model.predict(xx,support)
            for name,lp in methods.items():population.append(dict(fit=fit,lineage=lineage,reader=name,metrics=retain(root,f'{fit}-{lineage}-{name}',lp,yy)))
        pulse(phase='learned-fit-retained',fit=fit)
    write(root/'FIT_METADATA.json',fits)
    return dict(card=d['card'],population=population,fit_records=fits,headroom_sha256=file_digest(gate/'SUMMARY.json'),controls=controls(),
        scope='Constructed learned method; fresh-law rival is generator-limit diagnostic with separately counted labels; latent maker IDs never supplied')

def factorization(root,d,pulse):
    src=Path(d['forecast_root']);complete=read(src/'COMPLETE.json');population=[]
    for file in sorted((src/'raw').glob('*converged-direct_points.npz')):
        if file_digest(file)!=complete['files'][file.relative_to(src).as_posix()]:raise ValueError('forecast changed')
        z=np.load(file);masked=z['logp'];y=z['truth'];fit,lineage=file.name.split('-')[:2]
        saved=np.load(src/'training'/f'{fit}-direct-model_points.npz');model=Direct.__new__(Direct)
        model.params=[saved[f'parameter_{i}'] for i in range(4)]
        xx=np.load(src/'reader'/f'{lineage}_points.npz')['evidence'];joint=model.predict(xx);p=np.exp(joint);m=p@W.LABELS
        with np.errstate(divide='ignore'):
            product=np.sum(np.where(W.LABELS[None,:,:],np.log(m)[:,None,:],np.log1p(-m)[:,None,:]),axis=2)
        support=np.isfinite(masked)
        for name,lp in [('unmasked-joint',joint),('unmasked-product-marginals',product),('common-projected-joint',masked),
                        ('projected-product',O.normalize_log(np.where(support,product,-np.inf))),
                        ('support-only',O.normalize_log(np.where(support,0.,-np.inf)))]:
            a=retain(root,file.stem+'-'+name,lp,y)
            population.append(dict(source=file.name,reader=name,metrics=a,incompatible_mass=float((np.exp(lp)*(~support)).sum(1).mean())))
        pulse(phase='same-reader-factorization',source=file.name)
    if not population:raise ValueError('no complete converged forecast consumer')
    return dict(card='G21-C01',population=population,controls=controls(),scope='Same retained predictor, no new fitting; mask and learned dependence separated')

def run_study(root,d,pulse):
    if d['handler']=='opportunities':
        from .opportunities import run
        return run(root,d,pulse)
    if d['handler']=='retraction':
        from .retraction import run
        return run(root,d,pulse)
    if d['handler']=='omission':
        from .omission import run
        return run(root,d,pulse)
    if d['handler']=='selection':
        from .selection import run
        return run(root,d,pulse)
    if d['handler']=='routes':
        from .routes import run
        return run(root,d,pulse)
    if d['handler']=='factors':
        from .factors import run
        return run(root,d,pulse)
    if d['handler']=='memory':
        from .memory import run
        return run(root,d,pulse)
    if d['handler']=='role-drift':
        from .drift import run
        return run(root,d,pulse)
    if d['handler']=='sequential':
        from .sequential import run
        return run(root,d,pulse)
    if d['handler']=='provenance':
        from .provenance import run
        return run(root,d,pulse)
    if d['handler']=='acquisition':
        from .acquisition import run
        return run(root,d,pulse)
    if d['handler']=='joint-architectures':
        from .joint_study import run
        return run(root,d,pulse)
    if d['handler']=='persistence':
        from .persistence import run
        return run(root,d,pulse)
    if d['handler']=='dependence':
        from .dependence import run
        return run(root,d,pulse)
    if d['handler']=='reliability':
        from .reliability import run
        return run(root,d,pulse)
    if d['handler']=='adaptation-controls':
        from .adaptation_controls import run
        return run(root,d,pulse)
    if d['handler']=='review-complete':
        from runners.queue_review_v21 import run
        return run(root,d,pulse)
    if d['handler']=='adaptation':
        from .adaptation import run
        return run(root,d,pulse)
    if d['handler']=='aliases':
        from .aliases import run
        return run(root,d,pulse)
    if d['handler']=='optimization':
        from .optimization import run
        return run(root,d,pulse)
    if d['handler']=='transport':
        from .transport import run
        return run(root,d,pulse)
    if d['handler']=='records':
        from .records import run
        return run(root,d,pulse)
    if d['handler']=='support':
        from .support import run
        return run(root,d,pulse)
    return {'ceiling':ceilings,'headroom':headroom,'learned':learned,'factorization':factorization}[d['handler']](root,d,pulse)
