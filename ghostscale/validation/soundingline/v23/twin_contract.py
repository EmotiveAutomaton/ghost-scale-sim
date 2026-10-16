"""Real-shape admission and explicit unresolved fields for synthetic twins.

No approximate number in the commissioning prose becomes a measured marginal.
An engineering fixture can exercise a generator without satisfying this gate.
"""
import math
from .common import canonical

REQUIRED={
    'T1':('makers','sessions','decisions','records_per_maker','sessions_per_maker',
          'options_per_decision','latency_quantiles_s','keystroke_quantiles','handling_marginals'),
    'T2':('players','games_per_player','move_clock_quantiles_s','moves_per_game'),
    'T3':('editors','edits_per_editor','editors_per_article','summary_coverage','revert_fraction'),
    'T4':('raters','items','ratings_per_rater','raters_per_item','rating_semantics','rating_marginals')}


def realism(family,reference,constructed):
    if family not in REQUIRED:raise ValueError('unknown twin')
    source=reference.get('source',{})
    authenticated=(len(source.get('revision',''))==40 and len(source.get('sha256',''))==64 and bool(source.get('citation')))
    empirical=reference.get('aggregates',{});rows=[];missing=[]
    for key in REQUIRED[family]:
        observed=empirical.get(key);synthetic=constructed.get(key)
        available=observed is not None and synthetic is not None
        if not available:missing.append(key)
        rows.append(dict(quantity=key,empirical=observed,constructed=synthetic,
            status='COMPARED' if available else 'MISSING',exact_match=available and canonical(observed)==canonical(synthetic)))
    semantics=(family!='T4' or empirical.get('rating_semantics')==constructed.get('rating_semantics'))
    return dict(family=family,source=source,table=rows,missing=missing,authenticated_source=authenticated,
        semantics_match=semantics,real_shape_citation_allowed=authenticated and not missing and semantics,
        note='Differences remain explicit; a comparison table is not evidence of human mechanism validity.')


def require_realism(record):
    if not record.get('real_shape_citation_allowed') or record.get('missing') or not record.get('semantics_match'):
        raise ValueError('realism evidence incomplete or target semantics incompatible')
    if not record.get('authenticated_source'):raise ValueError('empirical source is not pinned')


def empirical_quantiles(probabilities,values,rng,size):
    """Sample a declared empirical inverse CDF, without inventing tail timing."""
    import numpy as np
    probabilities=np.asarray(probabilities,float);values=np.asarray(values,float)
    if probabilities.shape!=values.shape or len(values)<2 or probabilities[0]!=0 or probabilities[-1]!=1:
        raise ValueError('complete empirical quantile support required')
    if not np.isfinite(values).all() or np.any(values<0) or np.any(np.diff(values)<0) or np.any(np.diff(probabilities)<=0):
        raise ValueError('invalid empirical timing distribution')
    return np.interp(rng.random(size),probabilities,values)
