"""Reconstruct the completed B07 controls with fits nested inside worlds."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from ghostscale.validation.soundingline.v21.metrics import scores, mean_scores, bootstrap
from runners.report_v21_support import safe

PRIMARY = ('modular', 'direct-finetuned', 'fresh-law-finetuned')
READERS = PRIMARY + ('naive-bayes', 'naive-bayes-without-evidence')
METRICS = ('log_loss', 'brier', 'correct', 'confident_error')


def interval(values):
    if not np.isfinite(np.asarray(values, float)).all():
        return dict(mean=None, lower=None, upper=None, state='infinite; no finite interval')
    return dict(bootstrap(values), state='descriptive paired world bootstrap')


def nested(rows):
    """Reject missing/duplicate fits instead of treating rows as independent worlds."""
    grouped = {}
    for row in rows:
        key = (row['lineage'], row['condition'], row['reader'], row['k'])
        grouped.setdefault(key, []).append(row)
    result = []
    for (world, condition, reader, k), subset in grouped.items():
        assert sorted(r['fit'] for r in subset) == [0, 1], 'two distinct nested fits required'
        metrics = {}
        for metric in subset[0]['metrics']:
            values = [r['metrics'][metric] for r in subset]
            metrics[metric] = None if any(v is None for v in values) else safe(np.mean(values, axis=0))
        result.append(dict(world=world, condition=condition, reader=reader, k=k,
                           fits_nested=2, metrics=metrics))
    return result


def build(campaign, review, out):
    validity = read(review/'VALIDITY.json'); assert validity['passed']
    bindings = {Path(b['root']).name: b for b in read(review/'INPUTS.json')['packets']}
    cells = []
    for laws in (8, 64):
        for labels in (512, 2048, 8192):
            original = campaign/'packets'/f'v21-b07-02-l{laws}-n{labels}'
            supplement = campaign/'packets'/f'v21-b07-controls-l{laws}-n{labels}'
            d = read(supplement/'PLAN.json')['design']; rows = []
            for root in (original, supplement):
                assert file_digest(root/'COMPLETE.json') == bindings[root.name]['complete_sha256']
                summary = read(root/'SUMMARY.json')
                for row in summary['population']:
                    condition = row.get('condition', 'shifted')
                    pieces = [str(row[k]) for k in ('fit', 'k', 'lineage')]
                    if root == supplement: pieces.append(condition)
                    pieces.append(row['reader'])
                    with np.load(root/'raw'/('-'.join(pieces)+'_points.npz')) as z:
                        metrics = mean_scores({k: z[k] for k in z.files if k not in ('logp', 'truth')})
                    for metric, value in metrics.items():
                        old = row['metrics'][metric]
                        assert old == value if value is None else np.allclose(old, value, atol=1e-12)
                    rows.append(dict(row, condition=condition, metrics=metrics))
            worlds = nested(rows); assert len(worlds) == 8*2*5*4
            lookup = {(r['world'],r['condition'],r['reader'],r['k']):r['metrics'] for r in worlds}
            ceilings = {}; gates = {}
            for condition, key in (('native','native_headroom_root'), ('shifted','headroom_root')):
                root = Path(d[key]); gate = read(root/'SUMMARY.json'); gd = read(root/'PLAN.json')['design']
                assert file_digest(root/'COMPLETE.json') == bindings[root.name]['complete_sha256']
                gates[condition] = dict(headroom=gate['headroom'], disposition=gate['screen_disposition'],
                    exact=gd['law_count']=='exact', both_validation_stopped=all(f['early_stopped'] for f in gate['fit_records']))
                if condition == 'shifted':
                    assert gates[condition]['exact'] and gates[condition]['both_validation_stopped'] and gate['headroom']['lower']>.02
                for world in d['lineages']:
                    with np.load(root/'evaluator'/f'{world}_points.npz') as z:
                        ceilings[world,condition] = {kind:mean_scores(scores(z[key],z['truth']))
                            for kind,key in (('marginal','marginal_logp'),('supplied','supplied_logp'))}
            curves=[]; shifts=[]; contrasts=[]; decomposition=[]
            for reader in READERS:
                for k in (0,16,64,256):
                    for condition in ('native','shifted'):
                        means={metric:interval([lookup[w,condition,reader,k][metric] for w in d['lineages']]) for metric in METRICS}
                        excess={metric:interval([lookup[w,condition,reader,k][metric]-ceilings[w,condition]['marginal'][metric] for w in d['lineages']]) for metric in ('log_loss','brier')}
                        curves.append(dict(reader=reader,k=k,condition=condition,metrics=means,excess=excess,
                            mean_metrics=safe({m:np.mean([lookup[w,condition,reader,k][m] for w in d['lineages']],axis=0) for m in worlds[0]['metrics']})))
                        for metric in ('log_loss','brier'):
                            decomposition.append(dict(reader=reader,k=k,condition=condition,metric=metric,
                                supplied_uncertainty=interval([ceilings[w,condition]['supplied'][metric] for w in d['lineages']]),
                                law_uncertainty=interval([ceilings[w,condition]['marginal'][metric]-ceilings[w,condition]['supplied'][metric] for w in d['lineages']]),
                                reader_excess=excess[metric]))
                    shifts.append(dict(reader=reader,k=k,changes={m:interval([lookup[w,'shifted',reader,k][m]-lookup[w,'native',reader,k][m] for w in d['lineages']]) for m in METRICS}))
                    for reference in ('direct-finetuned','fresh-law-finetuned','naive-bayes-without-evidence'):
                        if reader==reference:continue
                        contrasts.append(dict(reader=reader,reference=reference,k=k,metrics={m:interval([lookup[w,'shifted',reader,k][m]-lookup[w,'shifted',reference,k][m] for w in d['lineages']]) for m in METRICS}))
            target={reader:next((k for k in (0,16,64,256) if next(c for c in curves if c['reader']==reader and c['condition']=='shifted' and c['k']==k)['excess']['log_loss']['mean']<=.1),None) for reader in READERS}
            fits=[dict(fit=f,**read(original/'fits'/f'{f}.json')) for f in (0,1)]
            cells.append(dict(packet=original.name,supplement=supplement.name,training_laws=laws,native_labels=labels,
                worlds=8,fits_per_world=2,cases_per_world=256,world_rows=worlds,curves=curves,
                native_to_shift=shifts,shifted_matched_contrasts=contrasts,uncertainty_decomposition=decomposition,
                first_k_reaching_mean_target=target,quarter_label_criterion_met=target['modular'] is not None and target['direct-finetuned'] is not None and target['modular']<=target['direct-finetuned']/4,
                gates=gates,fit_metadata=fits,adaptation_metadata=read(original/'ADAPTATION_METADATA.json'),
                bindings={p.name:dict(complete_sha256=file_digest(p/'COMPLETE.json'),plan_sha256=file_digest(p/'PLAN.json'),archive_sha256=read(p/'PLAN.json')['archive_sha256']) for p in (original,supplement)}))
    result=dict(cells=cells,fixed_target_excess_nats=.1,confirmation_worlds_opened=0,
        multiplicity=dict(development_cells=6,worlds_reused_across_cells=8,formal_promotions=0),
        limitations=['Constructed method; miniature - architecture untested.',
            'Supplied two-kernel execution library and exact target support are distinct privileges.',
            'Fresh-law training consumes additional generated labels; it is not a matched native-label rival.',
            'Finite native fits validation-stop; fresh-law convergence and modular policy convergence remain separately reported.',
            'Shifted tuning uses 60 fixed updates and no shifted validation labels.',
            'Intervals pair worlds after nesting fits; individual native and shifted test cases differ.',
            'Descriptive unadjusted intervals and reused development worlds are not confirmation.'])
    write(out,result)
    return result
