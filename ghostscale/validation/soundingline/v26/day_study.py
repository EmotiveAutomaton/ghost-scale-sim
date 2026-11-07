"""Whole-block data, complete replay, independent reconstruction and intervals."""
from pathlib import Path
from collections import defaultdict
import gzip,importlib,json,math
from decimal import Decimal,localcontext
from . import day
from .common import canonical,read,write,digest,close
from .day_common import summarize

def module(family):return importlib.import_module(__package__+'.'+day.MODULES[family])

def extra_environment():
    import numba,llvmlite
    result={}
    for m in (numba,llvmlite):
        root=Path(m.__file__).parent
        result[m.__name__]=dict(version=m.__version__,files={p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py','.pyd','.dll')})
    return result

def verify_extra_environment(campaign,plan):
    p=campaign/'NUMERICAL_ENVIRONMENT.json'
    if digest(p)!=plan['extra_environment_sha256'] or extra_environment()!=read(p):raise ValueError('day numerical dependencies changed')

def aggregate(rows,independent=False):
    grouped=defaultdict(dict)
    for row in rows:
        for m in row['data']['measures']:
            key=canonical(dict(setting=row['setting'],metric=m['metric'],labels=m['labels'],statistic=m['statistic'])).decode()
            grouped[key][row['block']]=m['values']
    cells=[]
    for key,blocks in sorted(grouped.items()):
        values=[blocks[b] for b in sorted(blocks)];info=json.loads(key);summary=summarize(values,info['statistic'],key,independent)
        if independent and info['statistic']=='mean':
            with localcontext() as ctx:
                ctx.prec=50;means=[sum(map(lambda x:Decimal(str(x)),v))/len(v) for v in values if v];decimal=float(sum(means)/len(means))
            if not math.isclose(decimal,summary['estimate'],rel_tol=1e-10,abs_tol=1e-10):raise ValueError('decimal block aggregate mismatch')
        cells.append(info|summary)
    return cells

def execute(root,plan,pulse,independent=False):
    root=Path(root);impl=module(plan['family']);controls=impl.controls()
    if not controls['passed']:raise ValueError('known-answer failure')
    if not independent:root.mkdir(exist_ok=False);write(root/'CONTROLS.json',controls)
    rows=[]
    for setting_id,setting in enumerate(plan['settings']):
        for block in range(plan['blocks']):
            identity=[plan['science_identity'],plan['family'],setting_id,block]
            data=impl.block(setting,identity,independent)
            row=dict(setting=setting_id,configuration=setting,block=block,data=data);path=root/f'world-{setting_id:04d}-{block:04d}_points.json.gz'
            if independent:
                with gzip.open(path,'rt',encoding='utf-8') as f:saved=json.load(f)
                close(row,saved,tol=1e-9)
                # Aggregate the independently reconstructed measurements, not saved scores.
            else:
                with path.open('xb') as handle:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0) as stream:stream.write(canonical(row)+b'\n')
            rows.append(dict(setting=setting_id,block=block,data=dict(measures=data['measures'])))
            pulse(setting=setting_id,block=block,phase='world_complete')
    summary=dict(family=plan['family'],card=plan['card'],kind=plan['scientific_kind'],settings=plan['settings'],blocks=plan['blocks'],
        cells=aggregate(rows,independent),confirmation=False,architecture='miniature, architecture untested',
        inference='Sixteen independent blocks in full studies; makers and sources are nested, not extra independent blocks. Censored medians retain unresolved crossings.',
        limitations='Pointwise descriptive intervals, not simultaneous or anytime-valid tests. Shared generators and some likelihood kernels; independent numerical inference and decimal mean aggregation.')
    if independent:
        close(summary,read(root/'SUMMARY.json'),tol=1e-9)
        return dict(state='INDEPENDENT_RECONSTRUCTION_PASSED',worlds=len(rows),all_declared_cells=True,
            independent_inference=True,independent_decimal_mean_aggregation=True,complete_block_bootstrap_recomputed=True,
            limitation='Shared declared generators and some likelihood kernels; exact filters separately checked on brute-force fixtures.',summary_sha256=digest(root/'SUMMARY.json'))
    write(root/'SUMMARY.json',summary)
    write(root/'RAW_INDEX.json',{p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(root.glob('*_points.json.gz'))})

def reconstruct(root,plan,pulse):
    for name,bound in read(Path(root)/'RAW_INDEX.json').items():
        if digest(Path(root)/name)!=bound['sha256']:raise ValueError('raw evidence changed')
    return execute(root,plan,pulse,True)
