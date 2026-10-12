"""World-paired interpretation of immutable V21 score/regroup receipts.

No fits, samples, thresholds, or producer files are changed. Selection of the
strongest observed matched rival is descriptive, not a confirmation procedure.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from runners.report_v21_support import safe, interval

METRICS = ('log_loss', 'brier', 'correct', 'confident_error')
RIVALS = ('converged-direct', 'fixed-schedule-mlp', 'direct-table',
          'grouped-empirical', 'naive-bayes', 'naive-bayes-without-evidence')


def contrast(rows, arm, reference, metric, k=-1):
    worlds = sorted({r['world'] for r in rows})
    values = []
    for w in worlds:
        a = next(r['metrics'][metric] for r in rows if r['world'] == w and r['reader'] == arm and r['k'] == k)
        b = next(r['metrics'][metric] for r in rows if r['world'] == w and r['reader'] == reference and r['k'] == k)
        values.append(np.nan if a is None or b is None else a-b)
    return dict(arm=arm, reference=reference, metric=metric, **interval(values))


def fit_digest(record):
    if isinstance(record, dict):
        return {k: fit_digest(v) for k, v in record.items() if k != 'fit_cpu_seconds'}
    if isinstance(record, list):
        return [fit_digest(v) for v in record]
    return record


def fit_summary(f):
    return dict(fit=f['fit'], **{name: {k: f[name].get(k) for k in
                ('early_stopped', 'convergence_disposition', 'parameters', 'effective_states',
                 'unique_finite_labels', 'fresh_generated_labels', 'fit_cpu_seconds')}
                for name in ('direct', 'fresh', 'shared')})


def build(campaign, review, out):
    assert read(review/'VALIDITY.json')['passed']
    inputs = read(review/'INPUTS.json')['packets']
    checked = {Path(b['root']).name: b for b in inputs}
    root = lambda name: campaign/'packets'/name
    learned = {}; supports = {}; gates = []; bindings = {}
    for name, bound in checked.items():
        packet = root(name)
        assert file_digest(packet/'COMPLETE.json') == bound['complete_sha256']
        plan = read(packet/'PLAN.json'); d = plan['design']; s = read(packet/'SUMMARY.json')
        bindings[name] = dict(complete_sha256=bound['complete_sha256'], plan_sha256=bound['plan_sha256'],
                             archive_sha256=plan['archive_sha256'],
                             score_receipt_sha256=file_digest(review/'scores'/(name+'.json')))
        if name.startswith('v21-headroom-'):
            grouped = read(review/'regroup'/(name+'.json'))['world_rows']
            values=[]
            for world in d['lineages']:
                direct=next(r['metrics']['log_loss'] for r in grouped if r['world']==world and r['reader']=='converged-direct')
                marginal=next(r['metrics']['log_loss'] for r in grouped if r['world']==world and r['reader']=='law-marginal')
                values.append(direct-marginal)
            rebuilt=interval(values)
            assert all(np.isclose(rebuilt[k],s['headroom'][k],atol=1e-12) for k in ('mean','lower','upper'))
            assert s['screen_admitted'] == (rebuilt['lower']>.02 and d['law_count']=='exact' and all(f['early_stopped'] for f in s['fit_records']))
            gates.append(dict(packet=name, headroom=s['headroom'], admitted=s['screen_admitted'],
                disposition=s['screen_disposition'], both_validation_stopped=all(f['early_stopped'] for f in s['fit_records']),
                exact=d['law_count']=='exact'))
        if name.startswith('v21-b01-'):
            grouped = read(review/'regroup'/(name+'.json'))
            rows = grouped['world_rows']; arms = sorted({r['reader'] for r in rows})
            means = {arm: safe({m: np.mean([r['metrics'][m] for r in rows if r['reader']==arm], axis=0)
                               for m in rows[0]['metrics']}) for arm in arms}
            best = min(RIVALS, key=lambda a: means[a]['log_loss'])
            pairs = [contrast(rows, 'shared-maker', rival, m) for rival in RIVALS for m in METRICS]
            primary = next(p for p in pairs if p['reference']==best and p['metric']=='log_loss')
            gate = read(Path(d['headroom_root'])/'SUMMARY.json')
            assert gate['screen_admitted'] and all(f['early_stopped'] for f in gate['fit_records'])
            learned[name] = dict(packet=name, design=grouped['design'], world_rows=rows, arm_means=means,
                strongest_matched_rival=best, shared_minus_strongest=primary, all_matched_contrasts=pairs,
                descriptive_margin_screen=primary['upper'] < -.02, headroom=gate['headroom'],
                fit_metadata=[fit_summary(f) for f in s['fit_records']],
                selection='Strongest mean log-loss rival selected within this development cell; multiplicity unadjusted.')
        if name.startswith('v21-c03-'):
            if s.get('population'):
                supports[name] = read(review/'support'/(name+'.json'))
            else:
                prerequisite=read(Path(d['forecast_root'])/'SUMMARY.json')
                supports[name] = dict(packet=name, disposition=s['state'], forecast_rows=0,
                    headroom=prerequisite['headroom'],
                    both_validation_stopped=all(f['early_stopped'] for f in prerequisite['fit_records']),
                    reason='No support comparison admitted; this is a gate disposition, not a null result.')
    shifts = []
    for native_cell, shifted_cell in ((5,6),(11,12),(17,18)):
        for labels in (512,2048,8192):
            an=f'v21-b01-{native_cell:02}-l8-n{labels}'; bn=f'v21-b01-{shifted_cell:02}-l8-n{labels}'
            if an not in learned or bn not in learned: continue
            a,b=learned[an],learned[bn]
            sa,sb=read(root(an)/'SUMMARY.json'),read(root(bn)/'SUMMARY.json')
            assert fit_digest(sa['fit_records']) == fit_digest(sb['fit_records'])
            da,db=read(root(an)/'PLAN.json')['design'],read(root(bn)/'PLAN.json')['design']
            for fit in da['fit_seeds']:
                assert file_digest(Path(da['headroom_root'])/'training'/f'{fit}_points.npz') == file_digest(Path(db['headroom_root'])/'training'/f'{fit}_points.npz')
            changes=[]
            for reader in a['arm_means']:
                for m in METRICS:
                    av={r['world']: r['metrics'][m] for r in a['world_rows'] if r['reader']==reader}
                    bv={r['world']: r['metrics'][m] for r in b['world_rows'] if r['reader']==reader}
                    assert av.keys()==bv.keys()
                    changes.append(dict(reader=reader,metric=m,**interval([bv[w]-av[w] for w in av])))
            shifts.append(dict(native=an,shifted=bn,training_bytes_identical=True,fit_metadata_identical_except_timing=True,
                changes=changes,scope='Paired worlds; test cases differ. This change affects current and historical execution.'))
    support_shifts=[]
    for ac,bc in ((1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16),(17,18)):
        an=f'v21-c03-{ac:02}';bn=f'v21-c03-{bc:02}';a,b=supports.get(an),supports.get(bn)
        if not a or not b or not a.get('arm_means') or not b.get('arm_means'):continue
        changes=[]
        for arm in a['arm_means']:
            for m in (*METRICS,'truth_exclusion','infinite_log_loss_fraction'):
                av={r['world']:r['metrics'][m] for r in a['paired_world_rows'] if r['support']==arm}
                bv={r['world']:r['metrics'][m] for r in b['paired_world_rows'] if r['support']==arm}
                changes.append(dict(arm=arm,metric=m,**interval([np.nan if av[w] is None or bv[w] is None else bv[w]-av[w] for w in av])))
        support_shifts.append(dict(native=an,shifted=bn,changes=changes,
            scope='Same retained direct method under support interventions; not a method-by-support interaction.'))
    adaptations=[]
    for name in checked:
        if not name.startswith('v21-b07-'):continue
        s=read(root(name)/'SUMMARY.json');d=read(root(name)/'PLAN.json')['design'];rows=s.get('population',[])
        curves=[]
        for reader in ('modular','direct-finetuned','fresh-law-finetuned'):
            for k in (0,16,64,256):
                vals=[np.mean([r['excess_over_shift_aware_ceiling'] for r in rows if r['lineage']==w and r['reader']==reader and r['k']==k]) for w in d['lineages']]
                curves.append(dict(reader=reader,k=k,excess=interval(vals)))
        adaptations.append(dict(packet=name,curves=curves,any_arm_mean_reaches_target=any(c['excess']['mean']<=.1 for c in curves),
            acceptance='Incomplete instrumentation; no transfer claim accepted.',
            missing=['native modular forecasts','duplicate-removal adaptation controls']))
    write(out/'LEARNED.json',dict(cells=list(learned.values()),native_to_shift=shifts,
        multiplicity=dict(completed_cells=len(learned),formal_promotions=0,confirmation_worlds_opened=0),
        limits=['Eight development worlds reused across cells; fits nested within worlds, not independent repetitions.',
                'All compared readers receive supplied exact target support.',
                'Fresh-law direct uses extra generated labels and can reach its epoch cap; separate diagnostic.',
                'Shared-state capacity differs substantially from the direct network; no capacity-matched architecture claim.',
                'Descriptive unadjusted intervals do not establish a confirmed mechanism.']))
    write(out/'SUPPORT.json',dict(cells=list(supports.values()),native_to_shift=support_shifts))
    write(out/'HEADROOM.json',dict(cells=gates,scope='Exact-law headroom, not a learned comparison or confirmation.'))
    write(out/'ADAPTATION_PENDING.json',dict(cells=adaptations))
    write(out/'BINDINGS.json',bindings)
    return dict(learned_cells=len(learned),support_cells=len(supports),
        headroom_cells=len(gates),descriptive_candidates=[n for n,r in learned.items() if r['descriptive_margin_screen']])
