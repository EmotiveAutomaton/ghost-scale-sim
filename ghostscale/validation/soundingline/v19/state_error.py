"""A1 diagnostic: exact, retained learned and error-matched noisy banks."""
from collections import defaultdict
import gzip
import numpy as np
from ..v18_3 import world as W
from ..v18_3.io import write,canonical
from ..v18_4 import feasible_bank as F
from ..v18_4 import neural_data as D
from ..v18_4.bank_data import repair
from .algebra import audit,controls
from .retained import ARMS,raw_blocks,class_name


def noisy_bank(exact,learned,random):
    shape=np.asarray(exact).reshape(5,16).shape
    noise=random.normal(size=shape);noise-=noise.mean(axis=1,keepdims=True)
    norm=np.linalg.norm(np.asarray(learned).reshape(-1)-exact)
    noise*=norm/max(np.linalg.norm(noise),1e-30)
    raw=np.asarray(exact).reshape(shape)+noise
    valid=repair(raw).reshape(-1)
    return valid,dict(requested_squared_error=float(norm**2),pre_projection_squared_error=float(np.sum(noise**2)),
        realized_squared_error=float(np.sum((valid-exact)**2)),
        note='variance matched before shared nonnegative normalization; realized error reported separately')


def run(root,plan,pulse):
    cfg=plan['design'];rows=[];valid=True;units=0
    for arm in ARMS:
        for block in raw_blocks(cfg['retained_root'],arm,cfg['retained_pins']):
            for unit in block:
                if unit['index'] not in cfg['indices']:continue
                pulse(phase='state-error',arm=arm,completed_histories=units)
                p=unit['payload'];world=p['world'];posterior=W.posterior(W.packet(world,p['history']))
                atoms=np.concatenate([W.artifact_matrix(world,c) for c in D.TRAIN_QUERIES],axis=1)
                target=np.concatenate([W.artifact_matrix(world,c) for c in F.QUERIES],axis=1)
                _,mapping=audit(atoms,target);exact=posterior@atoms
                truth=np.stack([W.artifact_matrix(world,c)[p['state']] for c in F.QUERIES])
                banks=[('exact','reference',0,exact,{})]
                for reader,bank in sorted(p['banks'].items()):
                    kind,seed=reader.rsplit('-seed',1)
                    if int(seed) not in cfg['retained_fit_seeds']:continue
                    learned=np.asarray(bank).reshape(-1)
                    noise,metadata=noisy_bank(exact,learned,W.rng('v19-A1-noise',unit['cell'],unit['index'],unit['support'],reader))
                    banks += [('learned',kind,int(seed),learned,{}),('matched-noise',kind,int(seed),noise,metadata)]
                for bank_kind,reader,seed,bank,metadata in banks:
                    weights,certificate=F.project(atoms,bank);valid=valid and certificate['valid']
                    linear=(np.r_[1.,bank]@mapping).reshape(5,16)
                    forecasts={'hull':(weights@target).reshape(5,16),'tight-linear-fixed-repair':repair(linear)}
                    invalid=bool(np.min(linear)<-1e-8 or np.max(linear)>1+1e-8 or np.max(abs(linear.sum(1)-1))>1e-6)
                    for decoder,forecast in forecasts.items():
                        loss,brier=F.score(truth[3:],forecast[3:])
                        rows.append(dict(arm=arm,cell=unit['cell'],lineage=unit['index'],support=unit['support'],world_class=class_name(world),
                            bank=bank_kind,reader=reader,fit_seed=seed,decoder=decoder,farther_loss=loss,farther_brier=brier,
                            bank_squared_error=float(np.sum((bank-exact)**2)),raw_invalid=invalid if decoder.startswith('tight') else False,
                            projection_gap=certificate['gap'] if decoder=='hull' else None,noise_matching=metadata))
                units+=1
    if units!=len(cfg['indices'])*16*2*2:raise ValueError('A1 population incomplete')
    path=root/'raw/state-error_points.json.gz';path.parent.mkdir(exist_ok=True)
    payload=gzip.compress(canonical(rows),mtime=0)
    if path.exists() and path.read_bytes()!=payload:raise ValueError('A1 evidence differs')
    if not path.exists():path.write_bytes(payload)
    groups=defaultdict(list)
    for row in rows:
        key=tuple(row[k] for k in ('arm','world_class','bank','reader','decoder'))
        groups[key].append(row)
    cells=[dict(zip(('arm','world_class','bank','reader','decoder'),key),
        observations=len(values),lineages=len({r['lineage'] for r in values}),
        mean_loss=float(np.mean([r['farther_loss'] for r in values])),mean_bank_error=float(np.mean([r['bank_squared_error'] for r in values])),
        raw_invalid_rate=float(np.mean([r['raw_invalid'] for r in values]))) for key,values in sorted(groups.items())]
    return dict(histories=units,scored_rows=len(rows),classes=cells,controls=dict(controls(),live_projection_certificates=bool(valid)),
        access='supplied law; retained program history; matched noise uses evaluator posterior for diagnosis only',
        uncertainty='8 reused coefficient lineages, two retained initialization seeds; no new fits or independent training draws',
        pursuit='A2 conditioning/portfolio diagnostic or B1 recursive update after classwise review',
        warrant='exploratory state-error diagnostic; repaired noise is only matched before simplex normalization; miniature — architecture untested')
