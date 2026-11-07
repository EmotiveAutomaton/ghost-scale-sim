"""V27 full raw blocks, fresh process replay and independent reconstruction."""
from pathlib import Path
import gzip,importlib,json
from . import registry
from .common import read,write,digest,canonical,close
from ..v26.day_study import aggregate,extra_environment,verify_extra_environment


def module(family):return importlib.import_module(__package__+'.'+registry.MODULES[family])


def execute(root,plan,pulse,independent=False):
    root=Path(root);impl=module(plan['family']);controls=impl.controls()
    if not controls['passed']:raise ValueError('known-answer failure')
    if not independent:root.mkdir(exist_ok=False);write(root/'CONTROLS.json',controls)
    rows=[]
    for setting,config in enumerate(plan['settings']):
        for block in range(plan['blocks']):
            identity=[plan['science_identity'],plan['family'],setting,block]
            data=impl.block(config,identity,independent)
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
