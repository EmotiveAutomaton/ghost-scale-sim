"""V27 full raw blocks, fresh process replay and independent reconstruction."""
from pathlib import Path
import gzip,importlib,json,math
import numpy as np
from . import registry
from .common import read,write,digest,canonical,close,rng
from ..v26.day_study import aggregate as ordinary_aggregate,extra_environment,verify_extra_environment


def aggregate(rows,independent=False):
    regular=[];extended={}
    for row in rows:
        keep=[]
        for m in row['data']['measures']:
            if m['statistic']!='extended_mean':keep.append(m);continue
            key=(row['setting'],m['metric'],json.dumps(m['labels'],sort_keys=True))
            extended.setdefault(key,[]).append(m['values'])
        regular.append(row|dict(data=dict(measures=keep)))
    out=ordinary_aggregate(regular,independent)
    for (setting,metric,labels),blocks in extended.items():
        # None explicitly denotes minus infinity for a contradicted reader,
        # never missing data, clipping, or an arbitrary finite log-score floor.
        means=[-math.inf if any(v is None for v in b) else math.fsum(b)/len(b) for b in blocks]
        center=-math.inf if -math.inf in means else math.fsum(means)/len(means)
        draws=[];n=len(means)
        if n>1:
            for pick in rng('extended-bootstrap',setting,metric,labels).integers(0,n,(2000,n)):
                values=[means[i] for i in pick];draws.append(-math.inf if -math.inf in values else math.fsum(values)/n)
            draws.sort();lo,hi=draws[49],draws[1949]
        else:lo=hi=math.nan
        finite=lambda x:float(x) if math.isfinite(x) else None
        out.append(dict(setting=setting,metric=metric,labels=json.loads(labels),statistic='extended_mean',estimate=finite(center),
            estimate_negative_infinity=center==-math.inf,estimate_censored=False,interval95=[finite(lo),finite(hi)],
            lower_negative_infinity=lo==-math.inf,upper_negative_infinity=hi==-math.inf,lower_censored=False,upper_censored=False,
            unavailable=math.isnan(center),interval_unavailable=math.isnan(lo) or math.isnan(hi),standard_error=None,
            blocks=n,observations=sum(map(len,blocks)),method='Pointwise whole-block percentile bootstrap, 2000 resamples; exact zero model support retains minus-infinite log score'))
    return sorted(out,key=lambda r:(r['metric'],json.dumps(r['labels'],sort_keys=True),r['setting']))


def module(family,analysis_module=None):
    if analysis_module not in (None,'closure_reread'):raise ValueError('unregistered analysis module')
    return importlib.import_module(__package__+'.'+(analysis_module or registry.MODULES[family]))


def execute(root,plan,pulse,independent=False):
    root=Path(root);impl=module(plan['family'],plan.get('analysis_module'));controls=impl.controls()
    if not controls['passed']:raise ValueError('known-answer failure')
    if not independent:root.mkdir(exist_ok=False);write(root/'CONTROLS.json',controls)
    rows=[]
    for setting,config in enumerate(plan['settings']):
        for block in range(plan['blocks']):
            identity=[plan['science_identity'],plan['family'],setting,block]
            if plan.get('analysis_module')=='closure_reread':
                data=impl.retained_block(config,plan,setting,block,independent)
            else:data=impl.block(config,identity,independent)
            row=dict(setting=setting,configuration=config,block=block,data=data)
            path=root/f'world-{setting:04d}-{block:04d}_points.json.gz'
            if independent:
                with gzip.open(path,'rt',encoding='utf-8') as stream:saved=json.load(stream)
                close(row,saved,tol=1e-9)
            else:
                with path.open('xb') as f:
                    with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as stream:stream.write(canonical(row)+b'\n')
            rows.append(dict(setting=setting,block=block,data=dict(measures=data['measures'])))
            pulse(setting=setting,block=block,phase='world_complete')
    summary=dict(family=plan['family'],card=plan['card'],kind=plan['scientific_kind'],settings=plan['settings'],
        blocks=plan['blocks'],cells=aggregate(rows,independent),confirmation=False,
        architecture='miniature, architecture untested',inference='Pointwise descriptive whole-block intervals; nested observations do not increase independent units.',
        limitations='Shared declared generators and some kernels; independent inference paths and complete reaggregation validate finite calculations, not realism.')
    if independent:
        close(summary,read(root/'SUMMARY.json'),tol=1e-9)
        return dict(state='INDEPENDENT_RECONSTRUCTION_PASSED',worlds=len(rows),all_declared_cells=True,
            independent_inference=True,independent_decimal_mean_aggregation=True,complete_block_bootstrap_recomputed=True,
            shared_kernels_disclosed=True,summary_sha256=digest(root/'SUMMARY.json'))
    write(root/'SUMMARY.json',summary)
    write(root/'RAW_INDEX.json',{p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(root.glob('*_points.json.gz'))})


def reconstruct(root,plan,pulse):
    for name,bound in read(Path(root)/'RAW_INDEX.json').items():
        if digest(Path(root)/name)!=bound['sha256']:raise ValueError('raw evidence changed')
    return execute(root,plan,pulse,True)
