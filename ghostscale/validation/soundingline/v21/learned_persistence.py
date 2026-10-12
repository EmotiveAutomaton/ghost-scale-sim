"""Candidate-specific B03 stress test with frozen native learned parameters.

Historical contexts are held fixed. The learned contamination mixture retains
their information while marginalizing the other historical fields. It knows no
generator maker identities; the supplied-rate arm knows only the intervention.
"""
from pathlib import Path
import numpy as np
from ..v18_3.io import read, write, file_digest
from . import world as W, oracle as O
from .readers import SharedMaker, Direct, GroupedTable
from .dependence import EpisodeGroups, ExactTable, choose_power
from .audit import counts, forecast
from .studies import retain

RATES=(1., .9, .5, 0.)


def context_tables(model):
    """Unknown episode mass is spread over eight contexts, explicitly."""
    mass=np.tile(model.emission[:,-1,None]/8,(1,8))
    for key,index in model.history.items():
        episode=np.frombuffer(key,dtype=np.int8)
        context=int(episode[3:6]@np.array([4,2,1]))
        mass[:,context]+=model.emission[:,index]
    assert np.allclose(mass.sum(1),1)
    return mass


def predict(model,x,support):
    cm=context_tables(model);background=model.prior@model.emission
    bc=model.prior@cm
    current=model._indices(x[:,:13],model.current)
    histories=x[:,13:].reshape(len(x),-1,13)
    indices=model._indices(histories.reshape(-1,13),model.history).reshape(len(x),-1)
    contexts=histories[:,:,3:6]@np.array([4,2,1])
    joint=np.log(model.joint[:,current,:].transpose(1,0,2))+np.log(model.prior)[None,:,None]
    masses=[]
    for rate in RATES:
        logweight=np.zeros((len(x),len(model.prior)))
        for step in range(histories.shape[1]):
            hi=indices[:,step];cx=contexts[:,step]
            bg=background[hi].copy();bg[hi==len(model.history)]/=8
            own=model.emission[:,hi].T.copy();own[hi==len(model.history)]/=8
            mixed=rate*own+(1-rate)*cm[:,cx].T*(bg/bc[cx])[:,None]
            logweight+=np.log(mixed)
        masses.append(O.logsumexp(joint+logweight[:,:,None],1))
    masses=np.array(masses)
    # Current legal support is common deployment-unavailable information.
    masses=np.where(support[None,:,:],masses,-np.inf)
    lp=O.normalize_log(O.logsumexp(masses,0))
    rates=np.exp(O.normalize_log(O.logsumexp(masses,2).T))
    return dict(inferred=lp,by_rate={str(r):O.normalize_log(masses[i]) for i,r in enumerate(RATES)},
                posterior_rates=rates,no_history=model.predict(x[:,:13],support))


def observations(original,rows,world,tier='complete',observed=True):
    """Replace noncontext history fields; retain current cases and every context."""
    histories=original[:,13:].reshape(len(original),-1,13)
    context=histories[:,:,3:6]@np.array([4,2,1])
    rng=np.random.default_rng(W.seed('v21-b03-learned-replacement',world))
    uniform=rng.random(context.shape);foreign=rng.integers(32,size=context.shape)
    bundle=rng.integers(32,size=len(original));draw=rng.random(context.shape)
    groups={}
    for maker in range(32):
        for cx in range(8):
            ids=np.flatnonzero((rows[:,0]==maker)&(rows[:,1]==cx));p=rows[ids,-1]
            groups[maker,cx]=(ids,np.cumsum(p/p.sum()))
    result={}
    for condition in (*RATES,'shuffled-author'):
        xx=original.copy();hh=xx[:,13:].reshape(histories.shape)
        keep=uniform<float(condition) if condition!='shuffled-author' else np.zeros(context.shape,bool)
        owners=foreign if condition!='shuffled-author' else np.broadcast_to(bundle[:,None],context.shape)
        selected=np.full(context.shape,-1,int)
        for i,step in np.argwhere(~keep):
            ids,cdf=groups[int(owners[i,step]),int(context[i,step])]
            selected[i,step]=ids[min(int(np.searchsorted(cdf,draw[i,step],side='right')),len(ids)-1)]
            hh[i,step]=W.project(rows[selected[i,step]:selected[i,step]+1],tier,observed)[0]
        assert np.array_equal(hh[:,:,3:6],histories[:,:,3:6])
        result[str(condition)]=dict(evidence=xx,retained_original=keep,foreign_maker=owners,foreign_row=selected)
    assert np.array_equal(result['1.0']['evidence'],original)
    return result


def controls():
    from .learned_persistence_check import reference
    a=np.zeros(13,dtype=np.int8);b=a.copy();b[0]=1
    m=SharedMaker.__new__(SharedMaker);m.current={a.tobytes():0};m.history={a.tobytes():0,b.tobytes():1}
    m.prior=np.array([.5,.5]);m.emission=np.array([[.89,.10,.01],[.10,.89,.01]])
    m.joint=np.full((2,2,128),.001);m.joint[0,0,0]=.8;m.joint[1,0,1]=.8
    m.joint/=m.joint.sum((1,2),keepdims=True)
    x=np.array([np.concatenate([a,a,a]),np.concatenate([a,a,b])]);support=np.ones((2,128),bool)
    p=predict(m,x,support);ref=reference(m,x,support)
    identical=SharedMaker.__new__(SharedMaker);identical.__dict__.update(m.__dict__);identical.emission=np.tile(m.emission.mean(0),(2,1))
    null=predict(identical,x,support)
    return {k:bool(v) for k,v in dict(live_scalar_reference=np.allclose(np.exp(p['inferred']),ref['inferred'],atol=1e-12),
        positive_incompatible_history_lowers_persistence=float(p['posterior_rates'][0]@np.array(RATES))>float(p['posterior_rates'][1]@np.array(RATES)),
        placebo_identical_emissions=np.allclose(null['by_rate']['1.0'],null['by_rate']['0.0'],atol=1e-12),
        rigid_recovers_original=np.allclose(p['by_rate']['1.0'],m.predict(x,support),atol=1e-12),
        context_only_ignores_noncontext=np.allclose(p['by_rate']['0.0'][0],p['by_rate']['0.0'][1],atol=1e-12)).items()}


def run(root,d,pulse):
    from .learned_persistence_check import reference
    src=Path(d['forecast_root']);c=read(src/'COMPLETE.json');sd=read(src/'PLAN.json')['design'];s=read(src/'SUMMARY.json')
    assert file_digest(src/'COMPLETE.json')==d['forecast_complete_sha256']
    assert sd['law_count']=='exact' and s['headroom']['lower']>.02 and all(f['early_stopped'] for f in s['fit_records'])
    assert sd['history_tier'] in ('context','complete')
    population=[];fits=[];maximum=0.;rate_rows=[]
    for fit in d['fit_seeds']:
        name=f'training/{fit}_points.npz';assert file_digest(src/name)==c['files'][name]
        with np.load(src/name) as z:x=z['evidence'];y=z['labels'];vx=z['validation_evidence'];vy=z['validation_labels']
        assert x.dtype==np.int8
        model=SharedMaker(x,y,fit_seed=fit,states=d.get('states',32),iterations=d.get('iterations',40),pulse=pulse)
        assert model.metadata['effective_states']>=2
        groups=EpisodeGroups(x,y);power,curve=choose_power(groups,vx,vy)
        pooled=GroupedTable(x,y);table=ExactTable(x,y);nb=counts(x,y)
        direct=Direct.__new__(Direct)
        with np.load(src/'training'/f'{fit}-direct-model_points.npz') as z:direct.params=[z[f'parameter_{i}'] for i in range(4)]
        fits.append(dict(fit=fit,latent=model.metadata,selected_power=power,validation_power_curve=curve))
        for world in d['lineages']:
            with np.load(src/'reader'/f'{world}_points.npz') as z:original=z['evidence']
            with np.load(src/'evaluator'/f'{world}_points.npz') as z:truth=z['truth'];support=np.isfinite(z['supplied_logp'])
            rows=W.enumerate_world(world,sd.get('history_shift',sd.get('shift','native')),sd.get('regime','original'))
            for condition,obs in observations(original,rows,world,sd['history_tier'],sd.get('history_observed',False)).items():
                xx=obs['evidence'];prediction=predict(model,xx,support);ref=reference(model,xx,support)
                error=float(np.max(np.abs(np.exp(prediction['inferred'])-ref['inferred'])));maximum=max(maximum,error);assert error<1e-10
                assert np.allclose(prediction['posterior_rates'],ref['posterior_rates'],atol=1e-10)
                for rate in RATES:assert np.allclose(np.exp(prediction['by_rate'][str(rate)]),ref['by_rate'][str(rate)],atol=1e-10)
                if condition=='1.0':assert np.allclose(prediction['by_rate']['1.0'],model.predict(original,support),atol=1e-10)
                informed='0.0' if condition=='shuffled-author' else condition
                arms={'inferred-persistence':prediction['inferred'],'supplied-rate':prediction['by_rate'][informed],
                    'rigid-same-maker':prediction['by_rate']['1.0'],'context-only':prediction['by_rate']['0.0'],
                    'no-history':prediction['no_history'],'converged-direct':direct.predict(xx,support),
                    'pooled-empirical':pooled.predict(xx,support),'direct-table':table.predict(xx,support),
                    'episode-calibrated':groups.predict(xx,support,power),'naive-bayes':forecast(nb,xx,support=support),
                    'naive-bayes-without-evidence':forecast(nb,xx,(1,),support=support)}
                for name,lp in arms.items():population.append(dict(fit=fit,lineage=world,persistence=condition,reader=name,metrics=retain(root,f'{fit}-{world}-{condition}-{name}',lp,truth)))
                p=root/'reader'/f'{world}-{condition}_points.npz';p.parent.mkdir(parents=True,exist_ok=True)
                if not p.exists():np.savez_compressed(p,evidence=xx)
                p=root/'evaluator'/f'{fit}-{world}-{condition}_points.npz';p.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(p,truth=truth,retained_original=obs['retained_original'],foreign_maker=obs['foreign_maker'],foreign_row=obs['foreign_row'],posterior_rates=prediction['posterior_rates'])
                rate_rows.append(dict(fit=fit,lineage=world,persistence=condition,mean_rate=float((prediction['posterior_rates']@np.array(RATES)).mean()),mean_probabilities=prediction['posterior_rates'].mean(0).tolist()))
                pulse(phase='learned-persistence',fit=fit,lineage=world,persistence=condition)
    write(root/'FIT_METADATA.json',fits)
    return dict(card='G21-B03',population=population,fit_records=fits,persistence_inference=rate_rows,controls=controls(),independent_max_probability_error=maximum,
        scope='Candidate-specific learned diagnostic, frozen clean-history fits; no contaminated-cell superiority screen.',
        limitations=['Same current cases and historical contexts; context itself can retain maker information even with wholly foreign episode content.',
            'Learned background conditional on context; latent context likelihood retained. Unknown episode mass divided equally across eight contexts.',
            'Supplied persistence rate and common current target support are privileges; generator maker IDs are evaluator-only.',
            'Direct parameters and calibration powers remain frozen from clean training/validation. Contamination-specific refitting and exact-law headroom are absent.',
            'Coherent foreign-author bundles are outside the independent-episode contamination model; zero-rate arm is a diagnostic there.',
            'Selected reused development worlds, two fits nested; no fresh confirmation or human authorship claim.'])
