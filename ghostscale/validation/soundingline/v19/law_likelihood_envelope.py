"""Uniform likelihood-ratio bounds for rounding a supplied observation law."""
import gzip
import numpy as np
from ..v18_3.io import read,write,canonical,file_digest

DTYPES=('float64','float32','float16')
LENGTHS=(8,32,128)


def posterior_bound(log_range, length):
    if length < 1 or not isinstance(length,int) or np.isnan(log_range) or log_range < 0:
        raise ValueError('range/length')
    return float(np.tanh(length*log_range/4))


def evaluate(law,dtype):
    original=np.asarray(law,dtype=np.float64)
    if (original.shape!=(16,4,8) or not np.isfinite(original).all() or
        (original<0).any() or np.max(abs(original.sum(-1)-1))>1e-12 or dtype not in DTYPES):
        raise ValueError('law/design')
    reference=original/original.sum(-1,keepdims=True)
    stored=original.astype(dtype);cast=stored.astype(float);mass=cast.sum(-1,keepdims=True)
    if (mass==0).any():raise ValueError('empty positive law row;no floor')
    rounded=cast/mass;positive=reference>0;lost=positive&(rounded==0)
    ratios=np.full(original.shape,np.nan);logs=np.full(original.shape,np.nan)
    np.divide(rounded,reference,out=ratios,where=positive)
    with np.errstate(divide='ignore'):np.log(ratios,out=logs,where=positive)
    lower=np.full((4,8),np.nan);upper=lower.copy();ranges=lower.copy()
    supported=positive.any(0);unbounded=lost.any(0)
    for ctx in range(4):
        for endpoint in range(8):
            if not supported[ctx,endpoint]:continue
            values=logs[:,ctx,endpoint][positive[:,ctx,endpoint]]
            lower[ctx,endpoint]=values.min();upper[ctx,endpoint]=values.max()
            ranges[ctx,endpoint]=np.inf if unbounded[ctx,endpoint] else values.max()-values.min()
    maximum=float(np.max(ranges[supported]))
    return dict(original=original,reference=reference,stored=stored,rounded=rounded,
        reference_normalization_delta=reference-original,rounded_normalization_delta=rounded-cast,
        positive_reference=positive,lost_support=lost,ratios=ratios,log_ratios=logs,
        supported_observations=supported,unbounded_observations=unbounded,
        lower=lower,upper=upper,log_ranges=ranges,maximum_log_range=np.array(maximum),
        lengths=np.array(LENGTHS),posterior_tv_bounds=np.array([posterior_bound(maximum,n) for n in LENGTHS]))


def fixture():
    law=np.full((16,4,8),1/8)
    for s in range(16):
        law[s,:,0]+=(s-7)*.0000031;law[s,:,1]-=(s-7)*.0000031
    return law


def controls():
    noisy=evaluate(fixture(),'float16');exact=evaluate(np.full((16,4,8),1/8),'float16')
    # Constant ratios leave the posterior unchanged; a nonzero span permits change.
    return {'live:relative_rounding_detected':bool(noisy['maximum_log_range']>0),
        'placebo:exact_cast':bool(np.all(exact['posterior_tv_bounds']==0)),
        'positive:constant_ratio':posterior_bound(0.,128)==0,
        'negative:undersized_bound':posterior_bound(float(np.log(9)),1)>.49}


def run(root,plan,pulse):
    cfg=plan['design']
    if cfg['storage_dtypes']!=list(DTYPES) or cfg['lengths']!=list(LENGTHS):raise ValueError('frozen variants')
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    rows=[];(root/'raw').mkdir(exist_ok=True)
    for lineage in cfg['lineages']:
        pulse(phase='law-likelihood-envelope',lineage=lineage)
        law=read(root/'inputs'/f'{lineage}-law.json')
        for dtype in DTYPES:
            raw=evaluate(law,dtype)
            np.savez_compressed(root/'raw'/f'{lineage}-{dtype}_points.npz',**raw)
            maximum=float(raw['maximum_log_range'])
            for length,bound in zip(LENGTHS,raw['posterior_tv_bounds']):
                rows.append(dict(lineage=lineage,storage_dtype=dtype,length=length,
                    maximum_log_ratio_range=None if np.isinf(maximum) else maximum,
                    accumulated_log_ratio_range=None if np.isinf(maximum) else length*maximum,
                    unbounded=bool(np.isinf(maximum)),posterior_tv_bound=float(bound),
                    lost_support_count=int(raw['lost_support'].sum()),
                    supported_observations=int(raw['supported_observations'].sum()),
                    stored_law_bytes=int(raw['stored'].nbytes),
                    maximum_reference_normalization_change=float(abs(raw['reference_normalization_delta']).max())))
    (root/'raw/likelihood_envelope_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='supplied original/rounded laws;all statewise likelihood ratios,extrema and uniform posterior bounds',scope='algebraic worst-case bound over latent trajectories sharing prior and transition law;not realized inference error,learned access,process correspondence or human intent'))
    return dict(controls=checks,lineages=len(cfg['lineages']),variants=3,rows=len(rows),numerical_acceptance=False,
        scope='same-prior same-transition supplied-law likelihood bound;support loss explicit;no reachability or attained-error claim')
