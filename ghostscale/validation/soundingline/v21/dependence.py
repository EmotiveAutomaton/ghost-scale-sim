"""B02 finite learned dependence diagnostic with explicit copying provenance.

The full Bayesian arm sums the learned latent states, not the generator laws.
Copied cells do not inherit an exact-law superiority gate from independent data.
"""
from pathlib import Path
import numpy as np
from ..v18_3.io import read, write, file_digest
from . import oracle as O, world as W
from .readers import Direct, SharedMaker, GroupedTable
from .audit import counts, forecast
from .studies import retain

RATES=(0., .5, .9)
POWERS=(.25, .5, 1.)


def fixtures(root,sl):
    import importlib,json,sys,types
    from ..v20 import transport as T
    from ..v20.integration import load
    module=load('v21_dependence_process',sl/'soundingline/process_record.py')
    pkg=types.ModuleType('v21_dependence_stage13');pkg.__path__=[str(sl/'runners/stage13')];sys.modules[pkg.__name__]=pkg
    exporter=importlib.import_module(pkg.__name__+'.export_tompathy');common=importlib.import_module(pkg.__name__+'.common')
    names=('soundingline/process_record.py','runners/stage13/export_tompathy.py','runners/stage13/common.py','runners/stage13/scoring.py')
    before={n:file_digest(sl/n) for n in names};rows=W.enumerate_world(71000402);cases=[]
    for i in range(12):
        row=rows[(241*i+71)%len(rows)];packet=T.packet(row,'context',('v21-copying',i),False)
        count=(1,2,4)[i%3];excerpt=json.dumps(packet['evidence'],sort_keys=True,separators=(',',':'))
        reports=[dict(report=j,root='observed-root-0',content=excerpt,copied=j>0) for j in range(count)]
        case=module.ProcessCase(case_id=packet['case_id'],lineage_id=packet['source_lineage_id'],domain='constructed',medium='binary-linked-artifact',brief_id='synthetic',
            declared_context=dict(observed=packet['evidence'],reports=reports),participants={'human':'constructed','tool':'constructed'},route_family=packet['workflow_family'],events=[],artifact_final=packet['artifact'])
        case.validate();T.validate(packet)
        artifact,reading,outer=T.offline_input(packet,np.full(128,1/128))
        context=[]
        for j in range(count):
            source=f'Constructed report {j}; declared shared root observed-root-0'
            context.append(dict(id=common.digest([source,excerpt]),source=source,excerpt=excerpt,selected=True))
        exported,sidecar=exporter.export(artifact,reading,context=context,outer=outer)
        base=root/'reader'/packet['case_id'];write(base/'observed.json',packet);write(base/'reports.json',reports)
        write(base/'offline-packet.json',exported);write(base/'sidecar.json',sidecar)
        write(base/'candidates.json',dict(target='seven-bit constructed process and aims',labels=list(range(128))))
        write(root/'evaluator'/(packet['case_id']+'.json'),dict(truth=int(row[10]),independent_roots=1,report_count=count,scope='Copying fixture; duplicate reports add no independent observation.'))
        cases.append(dict(case_id=packet['case_id'],report_count=count,canonical_process_case=True,offline_export=True))
    assert before=={n:file_digest(sl/n) for n in names}
    write(root/'CONSUMER.json',dict(passed=True,sources=before,cases=cases,browser_tested=False,scope='Known copying provenance; unknown or corrupt root identities remain D03.'))
    return cases


def copied(x, rate, identity):
    out=x.copy();h=out[:,13:].reshape(len(x),-1,13)
    keep=np.ones(h.shape[:2],bool)
    rng=np.random.default_rng(W.seed('v21-b02-copy',identity))
    for step in range(1,h.shape[1]):
        mask=rng.random(len(x))<rate
        h[mask,step]=h[mask,0];keep[mask,step]=False
    return out,keep


def latent_predict(model,x,support,power=1.,keep=None):
    ci=model._indices(x[:,:13],model.current)
    h=x[:,13:].reshape(len(x),-1,13)
    hi=model._indices(h.reshape(-1,13),model.history).reshape(len(x),-1)
    belief=np.tile(np.log(model.prior),(len(x),1))
    for step in range(hi.shape[1]):
        weight=1 if keep is None else keep[:,step,None]
        belief+=power*weight*np.log(model.emission[:,hi[:,step]].T)
    lp=O.logsumexp(np.log(model.joint[:,ci,:].transpose(1,0,2))+belief[:,:,None],1)
    return O.normalize_log(np.where(support,lp,-np.inf))


class EpisodeGroups:
    """Pool complete episodes conditional on target; no learned maker state."""
    def __init__(self,x,y):
        self.current={v:i for i,v in enumerate(sorted({r.tobytes() for r in x[:,:13]}))}
        h=x[:,13:].reshape(-1,13)
        self.history={v:i for i,v in enumerate(sorted({r.tobytes() for r in h}))}
        self.prior=np.bincount(y,minlength=128)+.5;self.prior/=self.prior.sum()
        self.c=np.full((128,len(self.current)+1),.5)
        self.h=np.full((128,len(self.history)+1),.5)
        for row,target in zip(x,y):
            self.c[target,self.current[row[:13].tobytes()]]+=1
            for episode in row[13:].reshape(-1,13):self.h[target,self.history[episode.tobytes()]]+=1
        self.c/=self.c.sum(1,keepdims=True);self.h/=self.h.sum(1,keepdims=True)

    def predict(self,x,support,power=1.):
        ci=SharedMaker._indices(x[:,:13],self.current)
        lp=np.log(self.prior)[None,:]+np.log(self.c[:,ci].T)
        for step in range(13,x.shape[1],13):
            hi=SharedMaker._indices(x[:,step:step+13],self.history)
            lp+=power*np.log(self.h[:,hi].T)
        return O.normalize_log(np.where(support,lp,-np.inf))


def choose_power(model,vx,vy):
    losses=[]
    for power in POWERS:
        values=[]
        for rate in RATES:
            xx,_=copied(vx,rate,'validation')
            lp=model.predict(xx,np.ones((len(xx),128),bool),power)
            values.append(float(-lp[np.arange(len(vy)),vy].mean()))
        losses.append(dict(power=power,validation_mean_loss=float(np.mean(values))))
    return min(losses,key=lambda r:r['validation_mean_loss'])['power'],losses


def controls():
    x=np.arange(4*52,dtype=np.int16).reshape(4,52)
    a,k=copied(x,0,'control');b,m=copied(x,1,'control')
    return dict(placebo_no_copy=np.array_equal(a,x) and bool(k.all()),
        positive_all_copy=bool(np.all(b[:,26:39]==b[:,13:26]) and np.all(b[:,39:52]==b[:,13:26])),
        live_known_roots=bool(m[:,0].all() and not m[:,1:].any()))


def run(root,d,pulse):
    from .dependence_check import latent_reference, grouped_reference
    src=Path(d['forecast_root']);complete=read(src/'COMPLETE.json');plan=read(src/'PLAN.json')['design'];gate=read(src/'SUMMARY.json')
    assert file_digest(src/'COMPLETE.json')==d['forecast_complete_sha256']
    assert plan['law_count']=='exact' and gate['headroom']['lower']>.02 and all(f['early_stopped'] for f in gate['fit_records'])
    population=[];fits=[];maximum=0.
    for fit in d['fit_seeds']:
        name=f'training/{fit}_points.npz';assert file_digest(src/name)==complete['files'][name]
        with np.load(src/name) as z:x=z['evidence'];y=z['labels'];vx=z['validation_evidence'];vy=z['validation_labels']
        latent=SharedMaker(x,y,fit_seed=fit,states=d.get('states',32),iterations=d.get('iterations',40),pulse=pulse)
        assert latent.metadata['effective_states']>=2
        groups=EpisodeGroups(x,y);pooled=GroupedTable(x,y);nb=counts(x,y)
        selected,curve=choose_power(groups,vx,vy)
        direct=Direct.__new__(Direct)
        with np.load(src/'training'/f'{fit}-direct-model_points.npz') as z:direct.params=[z[f'parameter_{i}'] for i in range(4)]
        fits.append(dict(fit=fit,latent=latent.metadata,selected_power=selected,validation_power_curve=curve,
            calibration='Equal mean across three registered copying rates on disjoint validation laws, without supplied support.'))
        for world in d['lineages']:
            with np.load(src/'reader'/f'{world}_points.npz') as z:original=z['evidence']
            with np.load(src/'evaluator'/f'{world}_points.npz') as z:truth=z['truth'];support=np.isfinite(z['supplied_logp'])
            for rate in RATES:
                xx,keep=copied(original,rate,world)
                path=root/'reader'/f'{world}-{rate}_points.npz';path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():np.savez_compressed(path,evidence=xx)
                path=root/'evaluator'/f'{world}-{rate}_points.npz';path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():np.savez_compressed(path,truth=truth,independent_root_mask=keep)
                arms={'latent-full':latent_predict(latent,xx,support),
                    'latent-known-roots':latent_predict(latent,xx,support,keep=keep),
                    'pooled-empirical':pooled.predict(xx,support),
                    'converged-direct':direct.predict(xx,support),
                    'naive-bayes':forecast(nb,xx,support=support),
                    'naive-bayes-without-evidence':forecast(nb,xx,(1,),support=support)}
                for power in POWERS:arms[f'episode-power-{power}']=groups.predict(xx,support,power)
                arms['episode-calibrated']=arms[f'episode-power-{selected}']
                for name,lp in arms.items():
                    if name.startswith('latent-'):
                        ref=latent_reference(latent,xx,support,keep if name=='latent-known-roots' else None)
                    elif name.startswith('episode-'):
                        power=selected if name=='episode-calibrated' else float(name.removeprefix('episode-power-'))
                        ref=grouped_reference(groups,xx,support,power)
                    else:ref=np.exp(lp)
                    error=float(np.max(np.abs(np.exp(lp)-ref)));maximum=max(maximum,error);assert error<1e-10
                    population.append(dict(fit=fit,lineage=world,duplication_rate=rate,reader=name,metrics=retain(root,f'{fit}-{world}-{rate}-{name}',lp,truth)))
                pulse(phase='copied-episode-diagnostic',fit=fit,lineage=world,duplication_rate=rate)
    write(root/'FIT_METADATA.json',fits)
    cases=fixtures(root/'fixtures',Path(d['soundingline'])) if d.get('export_fixtures') else []
    return dict(card='G21-B02',population=population,fit_records=fits,controls=controls(),independent_max_probability_error=maximum,
        fixtures=cases,
        scope='Finite learned dependence diagnostic. Full latent-state summation is not an exact generator-law posterior.',
        limitations=['Known copying roots are supplied only to the named oracle ablation.',
            'Independent-data exact-law headroom does not admit superiority in new copied cells.',
            'Native-trained direct parameters are frozen; new copying-specific fit convergence is untested.',
            'Common exact current support is supplied; no inferred provenance or human mechanism claim.',
            'Initial original-law native cell; shifted, wide-law and calibration-transfer extensions remain conditional.'])
