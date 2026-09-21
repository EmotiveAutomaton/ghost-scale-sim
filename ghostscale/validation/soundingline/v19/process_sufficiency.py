"""Reachable history aliases for a bank depending only on persistent makers."""
from collections import defaultdict
import gzip
import numpy as np
from ..v18_3.io import canonical,digest,file_digest,write
from . import local_world as L
from .readout_data import local_targets,local_cache


def synthetic_controls():
    # Duplicate persistent-state columns, disjoint preceding operation answers.
    bank=np.array([[.2,.2],[.8,.8]]);target=np.eye(2)
    near=np.array([[.5,.5+1e-10],[.5,.5-1e-10]])
    return dict(live_exact_alias=bool(np.array_equal(bank[:,0],bank[:,1]) and not np.array_equal(target[:,0],target[:,1])),
        placebo_near_singular_not_exact=bool(np.linalg.det(near)!=0 and not np.array_equal(near[:,0],near[:,1])),
        positive_normalization=bool(np.allclose(bank.sum(0),1)))


def groups(records,tier):
    grouped=defaultdict(list)
    for r in records:grouped[digest(L.project(r,tier))].append(r)
    out=[]
    for key,rr in sorted(grouped.items()):
        weights=np.array([r['probability'] for r in rr]);mass=float(weights.sum());weights/=mass
        posterior=np.bincount([r['maker_index'] for r in rr],weights=weights,minlength=16)
        targets=weights@np.array([local_targets(r) for r in rr])
        out.append(dict(key=key,packet=L.project(rr[0],tier),mass=mass,posterior=posterior,targets=targets,records=len(rr)))
    return out


def common_path_likelihood(world,maker,context):
    # Paths P,I,I and I,P,I share this expression for EACH maker.
    # Presentation edits leave claim/evidence unchanged; first two inspections
    # sum all local goals, while final inspection excludes dependency (undo).
    options=list(L.choices(world,maker,context['initial'],0,context))
    presentation=sum(p for g,op,p in options if op=='replace-presentation')
    final_inspect=sum(p for g,op,p in L.choices(world,maker,context['initial'],2,context) if op=='inspect')
    return presentation*(1-world['action_rate'])*final_inspect/64


def run(root,plan,pulse):
    cfg=plan['design'];public=[];witnesses=[];inventory=[];candidates=[]
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('retained trajectories differ')
    for lineage in cfg['lineages']:
        pulse(phase='sufficiency-grouping',lineage=lineage)
        records=__import__('json').loads(gzip.decompress((root/'inputs'/f'lineage-{lineage}_points.json.gz').read_bytes()))
        assert len(records)==13824 and abs(sum(r['probability'] for r in records)-1)<1e-10
        old=local_cache(records);by_tier={}
        for tier in ('E0','E1','E2-sparse','E2-full'):
            pulse(phase='sufficiency-projection',lineage=lineage,tier=tier)
            gg=groups(records,tier);by_tier[tier]=gg
            inventory.append(dict(lineage=lineage,tier=tier,packets=len(gg),mass=sum(g['mass'] for g in gg)))
            # Deterministic bin screen only; a floating equality is never a proof.
            for digits in (8,10,12):
                buckets={};best=None
                for g in gg:
                    key=tuple(np.round(g['posterior'],digits))
                    if key not in buckets:buckets[key]=g;continue
                    a=buckets[key];gap=float(np.max(abs(a['targets']-g['targets'])))
                    residual=float(np.max(abs(a['posterior']-g['posterior'])))
                    if gap>0 and (best is None or gap>best['target_gap']):best=dict(a=a['key'],b=g['key'],target_gap=gap,posterior_residual=residual)
                candidates.append(dict(lineage=lineage,tier=tier,rounding_digits=digits,candidate=best,classification='numerical screen only; no exact certificate'))
            write(root/'evaluator'/f'{lineage}-{tier}.json',dict(groups=[dict(g,posterior=g['posterior'].tolist(),targets=g['targets'].tolist()) for g in gg]))
        for context_index,context in enumerate(L.CONTEXTS):
            paths=[('replace-presentation','inspect','inspect'),('inspect','replace-presentation','inspect')]
            selected=[]
            for path in paths:
                r=next(r for r in records if r['context_index']==context_index and tuple(e['operation'] for e in r['steps'])==path)
                key=digest(L.project(r,'E2-full'));selected.append(next(g for g in by_tier['E2-full'] if g['key']==key))
            a,b=selected;expected=np.array([common_path_likelihood(L.law(lineage),m,context) for m in L.MAKERS]);posterior=expected/expected.sum()
            residual=max(float(np.max(abs(g['posterior']-posterior))) for g in selected)
            bank_residual=float(np.max(abs(a['posterior']@old-b['posterior']@old)))
            assert residual<1e-12 and bank_residual<1e-12
            assert a['packet']!=b['packet'] and np.max(abs(a['targets'][9:]-b['targets'][9:]))>1-1e-12
            case_ids=[]
            for j,g in enumerate(selected):
                cid=f'pair-{len(witnesses):03d}-{j}';case_ids.append(cid)
                L.validate_public(g['packet']);public.append(dict(case_id=cid,packet=g['packet'],input_sha256=digest(g['packet'])))
            witnesses.append(dict(lineage=lineage,context_index=context_index,cases=case_ids,paths=[list(p) for p in paths],
                probability_expression='presentation action probability * (1-action_rate) * final inspection probability / 64 for each maker; same factors and same artifact claim/evidence in both paths',
                analytic_reason='presentation changes only display; goal strengths never depend on display or undo buffer; first/second inspect sum all goals, final inspect sums meaning and presentation; multiplication commutes',
                common_maker_likelihood=expected.tolist(),evidence_masses=[a['mass'],b['mass']],posterior_residual=residual,bank_residual=bank_residual,
                operation_target_gap=float(np.max(abs(a['targets'][9:]-b['targets'][9:]))),classification='analytic reachable equality under this exact executor, numerically verified for the retained lineage'))
    write(root/'PUBLIC_PACKET.json',dict(role='blind witnessed evidence only; proof/hidden posterior/targets withheld',cases=public))
    write(root/'EVALUATOR_ONLY.json',dict(witnesses=witnesses,numerical_candidates=candidates))
    return dict(controls=dict(synthetic_controls(),live_reachable_witnesses=len(witnesses)==4*len(cfg['lineages'])),
        enumeration=inventory,witness_pairs=len(witnesses),lineages=len(cfg['lineages']),
        scope='any exact fresh-episode bank that factors only through persistent-maker posterior loses the order distinction between these two reachable complete-witness histories; no claim about noisy learned banks leaking other history features',
        warrant='finite analytic method counterexample with numerical executable verification; miniature — architecture untested',
        pursuit='retain transient process information separately from persistent maker predictions; no learned-reader or human conclusion')
