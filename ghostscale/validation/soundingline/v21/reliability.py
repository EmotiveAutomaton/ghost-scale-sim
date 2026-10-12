"""C04: matched noisy inspection records applied to one retained predictor.

The report channel is stipulated and its reliability is supplied to every arm.
This is a constraint-handling diagnostic, not a newly trained architecture.
"""
import importlib
import json
import sys
import types
from pathlib import Path
import numpy as np
from ..v18_3.io import read, write, file_digest
from . import world as W, oracle as O
from .studies import retain

RATES=(1., .95, .8, .5)


def update(logp, report, reliability, hard=False):
    matching=W.LABELS[None,:,4]==np.asarray(report)[:,None]
    probability=np.where(matching, 1. if hard else reliability, 0. if hard else 1-reliability)
    with np.errstate(divide='ignore'):
        weighted=logp+np.log(probability)
    empty=~np.isfinite(weighted).any(1)
    # An impossible report is an explicit conflict; retain the prior, never NaNs.
    weighted[empty]=logp[empty]
    return O.normalize_log(weighted), empty


def controls():
    from .reliability_check import reference
    p=np.array([[.1,.2,.3,.4]])
    expanded=np.tile(p, (2,32))/32
    lp=np.log(expanded);report=np.array([0,1])
    soft,_=update(lp,report,.8);independent=reference(expanded,report,.8)
    indifferent,_=update(lp,report,.5)
    certain,_=update(lp,report,1.);hard,_=update(lp,report,.8,True)
    return dict(live=np.allclose(np.exp(soft),independent,atol=1e-12),
        placebo=np.allclose(indifferent,lp,atol=1e-12),
        positive=np.allclose(certain,hard,atol=1e-12),
        false_record_soft_retains_truth=bool(np.isfinite(soft).all() and np.isneginf(hard).any()))


def fixtures(root, sl):
    """False-context subset of F03, consumed read-only by canonical Stage 13."""
    from ..v20 import transport as T
    from ..v20.integration import load
    module=load('v21_reliability_process',sl/'soundingline/process_record.py')
    pkg=types.ModuleType('v21_reliability_stage13');pkg.__path__=[str(sl/'runners/stage13')];sys.modules[pkg.__name__]=pkg
    exporter=importlib.import_module(pkg.__name__+'.export_tompathy')
    common=importlib.import_module(pkg.__name__+'.common')
    source_names=('soundingline/process_record.py','runners/stage13/export_tompathy.py','runners/stage13/common.py','runners/stage13/scoring.py')
    before={name:file_digest(sl/name) for name in source_names}
    rows=W.enumerate_world(71000401);cases=[]
    meanlaw=dict(route=4,select=4,revise=4,goal=4,inspect=4,skill=6.5,shift='native')
    for i in range(12):
        row=rows[(239*i+47)%len(rows)];rate=(.95,.8,.5)[(i//2)%3]
        report=int(row[5])^(i%2)
        packet=T.packet(row,'context',('v21-noisy-record',i),False)
        claim=dict(reported_inspection=report,declared_reliability=rate,
                   source='constructed inspection witness',status='fallible reported context')
        case=module.ProcessCase(case_id=packet['case_id'],lineage_id=packet['source_lineage_id'],domain='constructed',
            medium='binary-linked-artifact',brief_id='synthetic',declared_context=dict(observed=packet['evidence'],reported=claim),
            participants={'human':'constructed','tool':'constructed'},route_family=packet['workflow_family'],events=[],artifact_final=packet['artifact'])
        case.validate();T.validate(packet)
        x=W.project(np.asarray([row]),'context',False)
        prior,_=O.posterior(x,'context',[meanlaw]);posterior,_=update(prior,[report],rate)
        artifact,reading,outer=T.offline_input(packet,np.full(128,1/128))
        excerpt=json.dumps(dict(observed=packet['evidence'],reported=claim),sort_keys=True,separators=(',',':'))
        source='Ghost V21 constructed fallible inspection context'
        context=[dict(id=common.digest([source,excerpt]),source=source,excerpt=excerpt,selected=True)]
        exported,sidecar=exporter.export(artifact,reading,context=context,outer=outer)
        base=root/'reader'/packet['case_id']
        write(base/'observed.json',packet);write(base/'reported-context.json',claim)
        write(base/'offline-packet.json',exported);write(base/'sidecar.json',sidecar)
        write(base/'candidates.json',dict(target='seven-bit constructed process and aims',labels=list(range(128))))
        write(root/'evaluator'/(packet['case_id']+'.json'),dict(truth=int(row[10]),report_correct=report==int(row[5]),
            law_marginal=np.exp(posterior[0]).tolist(),scope='H=0 exact law-prior plus stipulated binary noisy report; evaluator only'))
        cases.append(dict(case_id=packet['case_id'],declared_reliability=rate,canonical_process_case=True,offline_export=True))
    assert before=={name:file_digest(sl/name) for name in source_names}
    write(root/'CONSUMER.json',dict(passed=True,sources=before,cases=cases,
        coverage='12 fallible inspection-context fixtures: six true reports and six false reports, evaluator categories kept separate',browser_tested=False))
    return cases


def run(root,d,pulse):
    from .reliability_check import reference
    src=Path(d['forecast_root']);complete=read(src/'COMPLETE.json');gate=read(src/'SUMMARY.json')
    assert file_digest(src/'COMPLETE.json')==d['forecast_complete_sha256']
    sd=read(src/'PLAN.json')['design']
    assert gate['screen_admitted'] and sd['law_count']=='exact' and gate['headroom']['lower']>.02
    assert all(f['early_stopped'] for f in gate['fit_records'])
    population=[];max_error=0.
    for world in d['lineages']:
        with np.load(src/'evaluator'/f'{world}_points.npz') as z:truth=z['truth']
        u=np.random.default_rng(W.seed('v21-c04-noisy-record',world)).random(len(truth))
        for fit in d['fit_seeds']:
            name=f'raw/{fit}-{world}-converged-direct_points.npz'
            assert file_digest(src/name)==complete['files'][name]
            with np.load(src/name) as z:prior=z['logp'];assert np.array_equal(z['truth'],truth)
            for rate in RATES:
                report=W.LABELS[truth,4]^(u>=rate).astype(np.int8)
                rname=root/'reports'/f'{fit}-{world}-{rate}_points.npz';rname.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(rname,report=report,reliability=np.full(len(truth),rate))
                methods={'no-record':(prior,np.zeros(len(truth),bool)),
                    'soft-record':update(prior,report,rate),'hard-record':update(prior,report,rate,True)}
                for reader,(lp,conflicts) in methods.items():
                    independent=reference(np.exp(prior),report,rate,hard=reader=='hard-record') if reader!='no-record' else np.exp(prior)
                    error=float(np.max(np.abs(np.exp(lp)-independent)));max_error=max(max_error,error);assert error<1e-10
                    metrics=retain(root,f'{fit}-{world}-{rate}-{reader}',lp,truth)
                    population.append(dict(fit=fit,lineage=world,reliability=rate,reader=reader,metrics=metrics,
                        truth_exclusion=float(np.isneginf(lp[np.arange(len(truth)),truth]).mean()),
                        conflict_fraction=float(conflicts.mean()),false_report_fraction=float(np.mean(report!=W.LABELS[truth,4]))))
                pulse(phase='noisy-record',fit=fit,lineage=world,reliability=rate)
    cases=fixtures(root/'fixtures',Path(d['soundingline'])) if d.get('export_fixtures') else []
    return dict(card='G21-C04',population=population,controls=controls(),independent_max_probability_error=max_error,fixtures=cases,
        scope='Matched noisy-record constraint diagnostic from a retained converged direct forecast; no new trained architecture.',
        limitations=['Source reliability is supplied and correctly specified; inferred reliability is untested.',
            'Supplied target support is inherited equally by all arms.',
            'A new binary inspection witness is independent conditional on the true inspection; it is not ordinary semantic context.',
            'Finite development worlds reused; no confirmation or human mechanism claim.',
            'A hard contradiction falls back to the prior only when it empties all support; those conflicts are counted.'])
