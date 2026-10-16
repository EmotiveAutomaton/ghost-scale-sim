"""Reserved repair-family execution; confirmation hypotheses never change.

Order is frozen from all discovery dispositions before any reserved exposure.
One world per law forms an independent balanced block, so 128/256 worlds are
8/16 betting units. The separately refitted direct readers do not replace the
registered primary pair. All fits, raw gains, ledgers and diagnostics replay.
"""
import math
from pathlib import Path
import numpy as np
from . import registry, support, evidence
from .common import canonical, digest, read, seed, write


DISPOSITIONS = {'NOMINATED','STOPPED_BELOW_MARGIN','INCONCLUSIVE_AT_CAP',
    'INSTRUMENT_FAILED','RESOURCE_DEFERRED','UNIMPLEMENTED_AT_FREEZE','DEPENDENCY_BLOCKED_AT_FREEZE','PRIMARY_UNAVAILABLE'}


def validate_order(order):
    if order.get('state') != 'DISCOVERY_FROZEN': raise ValueError('discovery order is not frozen')
    dispositions = order.get('dispositions', {})
    if set(dispositions) != set(registry.ORDER) or any(x not in DISPOSITIONS for x in dispositions.values()):
        raise ValueError('every family requires an explicit discovery disposition')
    entries = order.get('nominations', [])
    if len({row['family'] for row in entries}) != len(entries): raise ValueError('repeated nominated family')
    if {row['family'] for row in entries} != {f for f, state in dispositions.items() if state == 'NOMINATED'}:
        raise ValueError('nomination coverage differs')
    for row in entries:
        if not math.isfinite(row['shrunk_mean']): raise ValueError('nonfinite discovery estimate')
        if row['variant'] not in {v['id'] for v in registry.variants(row['family'])}:
            raise ValueError('unknown nomination')
        if len(row.get('machine_card_sha256','')) != 64: raise ValueError('unbound nomination')
    if entries != sorted(entries, key=lambda r: (-r['shrunk_mean'], registry.ORDER.index(r['family']))):
        raise ValueError('confirmation order must be strongest first with fixed family tie-break')
    return entries


def verify_definition(plan):
    if plan['family'].startswith('N'):
        from .support_new import verify
        verify(plan)
    else:support.verify(plan)
    order_path = Path(plan['nomination_order'])
    if digest(order_path) != plan['nomination_order_sha256']: raise ValueError('nomination order changed')
    entries = validate_order(read(order_path)); index = plan['confirmation_index']
    if not isinstance(index, int) or not 1 <= index <= len(entries): raise ValueError('confirmation order index invalid')
    entry = entries[index-1]
    card_path = Path(plan['nomination_card'])
    if digest(card_path) != entry['machine_card_sha256']: raise ValueError('nominated card changed')
    card = read(card_path)
    if not plan.get('nomination_review'): raise ValueError('discovery numerical review required')
    if digest(plan['nomination_review'])!=plan['nomination_review_sha256']:
        raise ValueError('discovery numerical review changed')
    reviewed=read(plan['nomination_review'])
    if (reviewed.get('state')!='verified_discovery' or reviewed.get('decision')!='NOMINATED'
            or reviewed.get('family')!=entry['family'] or reviewed.get('nominated')!=entry['variant']
            or not reviewed.get('full_fresh_process_replay')
            or reviewed['receipts'][-1]['machine_card_sha256']!=entry['machine_card_sha256']):
        raise ValueError('discovery numerical review does not accept this nomination')
    selected = next(r for r in card['curve'] if r['variant']['id'] == entry['variant'])
    if card['state'] != 'NOMINATED' or card['family'] != entry['family'] or card['best_id'] != entry['variant']:
        raise ValueError('different discovery hypothesis')
    if selected['shrunk_mean'] != entry['shrunk_mean']: raise ValueError('discovery strength changed')
    worlds = 256 if abs(entry['shrunk_mean']-.02) <= .02 else 128
    if plan['family'] != entry['family'] or plan['variants'] != [entry['variant']]:
        raise ValueError('confirmation cannot replace nominated variant')
    cases=next(v['settings']['makers'] for v in registry.variants('N1') if v['id']==entry['variant']) if plan['family']=='N1' else 256
    if plan['worlds'] != worlds or plan['cases'] != cases or plan['batch'] != 'confirmation':
        raise ValueError('reserved-world allocation changed')
    if plan.get('fit_directory'):
        if not plan.get('confirmation_fit_packet'):raise ValueError('variant-only confirmation refitting required')
        from .family import verify_packet
        fp,fs=verify_packet(plan['confirmation_fit_packet'])
        if (fp.get('task')!='confirmation-fits' or fs.get('state')!='FITS_VERIFIED' or
                any(fp[k]!=plan[k] for k in ('family','variants','nomination_order_sha256','confirmation_index')) or
                digest(Path(plan['confirmation_fit_packet'])/'COMPLETE.json')!=plan['confirmation_fit_complete_sha256'] or
                Path(plan['fit_directory']).resolve()!=(Path(plan['confirmation_fit_packet'])/'original/fits').resolve()):
            raise ValueError('confirmation fits do not belong to the frozen reserved hypothesis')
        for name,expected in plan['fit_hashes'].items():
            if digest(Path(plan['fit_directory'])/name)!=expected:raise ValueError('reserved fit changed')
    return entry


def seed_audit(variant, order_sha256, worlds, cases=256):
    """Explicit disjoint full world/case draw identities, including parked looks."""
    from .adaptive import RUNGS
    reserved = {seed(variant, ['confirmation', order_sha256], w, c) for w in range(worlds) for c in range(cases)}
    if len(reserved) != worlds*cases: raise ValueError('confirmation seed collision')
    for batch, count in enumerate(RUNGS, 1):
        discovery = {seed(variant, ['discovery', batch], w, c) for w in range(count) for c in range(64)}
        if reserved & discovery: raise ValueError('discovery/confirmation world collision')
    return dict(state='passed', reserved_case_identities=len(reserved), discovery_batches_checked=7,
        independent_blocks=worlds//16, laws_per_block=16,
        sampling='Each block has one fresh independently keyed world per fixed law; all complete blocks have the same mixture expectation.')


def reconstruct_evidence(root, plan):
    if plan['family']=='N1':
        from .population_confirmation import reconstruct
        return reconstruct(root,plan)
    from .family import rows
    identity = plan['variants'][0]; folder = Path(root)/identity
    rival, mechanism = registry.PRIMARY[plan['family']][:2]
    gains = [[] for _ in range(plan['worlds'])]; seen=set()
    for pred, truth in zip(rows(folder/'prediction_points.jsonl.gz'), rows(folder/'truth_points.jsonl.gz'), strict=True):
        world, case = pred['world'], pred['case']; key=(world,case)
        if key in seen or not 0<=world<plan['worlds'] or not 0<=case<plan['cases']: raise ValueError('reserved coverage mismatch')
        if truth['identity'] != [plan['family'],identity,'confirmation',world,case] or truth['truth']['law'] != world%16:
            raise ValueError('independent balanced law block mismatch')
        seen.add(key); target=truth['truth']['y']; delta=[]
        if plan['family']=='N6' and (truth['truth']['warmth'] or truth['truth']['competence']):continue
        for head,y in enumerate(target):
            a=float(pred['probabilities'][mechanism][head][y]);b=float(pred['probabilities'][rival][head][y])
            # No smoothing or flooring: undefined/nonfinite primary is unavailable.
            delta.append(math.log(a)-math.log(b) if a>0 and b>0 else float('nan'))
        gains[world].append(float(np.mean(delta)))
    if len(seen)!=plan['worlds']*plan['cases']: raise ValueError('incomplete reserved sample')
    values=[math.fsum(row)/len(row) if row and all(math.isfinite(x) for x in row) else float('nan') for row in gains]
    result=evidence.betting_evalue(values,plan['confirmation_support']['lower_bound'],independent_balanced_blocks=True)
    # JSON remains strict; raw sidecars preserve the probabilities producing any nonfinite loss.
    return dict(result,world_gains=[x if math.isfinite(x) else None for x in values],
        primary_readers=[rival,mechanism],direct_comparison='separate variant-only fitted diagnostic',
        raw_predictions_sha256=digest(folder/'prediction_points.jsonl.gz'),raw_truth_sha256=digest(folder/'truth_points.jsonl.gz'))


def run(root, plan, pulse):
    if plan['family']=='N1':
        from .population_confirmation import run as confirm_population
        return confirm_population(root,plan,pulse)
    if plan['family'].startswith('N'):
        from .pipeline_new import run_batch
    else:from .pipeline import run_batch
    entry=verify_definition(plan)
    run_batch(root,plan,pulse)
    if plan.get('world_indices') is None:
        audit=seed_audit(entry['variant'],plan['nomination_order_sha256'],plan['worlds'],plan['cases'])
        write(Path(root)/'SEED_AUDIT.json',audit)
        write(Path(root)/'EVIDENCE.json',reconstruct_evidence(root,plan))


def run_fits(root,plan,pulse):
    """Separate complete replay of variant-only reserved training, before parts."""
    from . import pipeline,pipeline_new,checker
    from .common import Direct
    entry=verify_definition(plan);root=Path(root);root.mkdir(parents=True,exist_ok=False)
    variant=next(v for v in registry.variants(plan['family']) if v['id']==entry['variant'])
    builder=pipeline_new.fit_family if plan['family'].startswith('N') else pipeline.fit_family
    if plan['family']=='N1':
        from .population_pipeline import fit_family
        builder=lambda root,family,pulse,variant,namespace:fit_family(root,pulse,variant,namespace)
    fitted,meta=builder(root/'fits',plan['family'],pulse,variant=variant,namespace=['confirmation',plan['nomination_order_sha256']])
    checked={}
    for name,model in fitted.items():
        suffix='-masked' if name.endswith('-masked') else ''
        with np.load(root/'fits'/('validation'+suffix+'.npz')) as raw:x,y=raw['x'],raw['y']
        prediction=model.predict(x);independent=checker.scores(prediction.reshape(-1,prediction.shape[-1]),y.ravel())
        score=float(np.mean(independent['log_loss']))
        if not math.isclose(score,meta[name]['validation_log_loss'],abs_tol=1e-8,rel_tol=0):raise ValueError('reserved fit score differs')
        if not np.array_equal(Direct.load(root/'fits'/(name+'.npz')).predict(x),prediction):raise ValueError('reserved fit serialization differs')
        checked[name]=dict(validation_log_loss=score,early_stopped=meta[name]['early_stopped'])
    write(root/'SUMMARY.json',dict(state='FITS_VERIFIED',family=plan['family'],variants=plan['variants'],fits=checked,
        nomination_order_sha256=plan['nomination_order_sha256'],reserved_test_exposure=False,
        training_namespace=['confirmation',plan['nomination_order_sha256']]))


def ordered_decisions(packets, order_path, order_sha256):
    """Only earlier completed, fully replayed confirmations count as discoveries."""
    from .family import verify_packet
    if digest(order_path)!=order_sha256: raise ValueError('frozen order changed')
    entries=validate_order(read(order_path));logs=[]
    for index, root in enumerate(packets,1):
        plan,_=verify_packet(root);verify_definition(plan)
        if plan['nomination_order_sha256']!=order_sha256 or plan['confirmation_index']!=index:
            raise ValueError('earlier completed confirmations must form the frozen prefix')
        result=reconstruct_evidence(Path(root)/'original',plan)
        if canonical(result)!=canonical(read(Path(root)/'original/EVIDENCE.json')):
            raise ValueError('confirmation evidence reconstruction differs')
        logs.append(result['log_e'])
    if len(logs)>len(entries): raise ValueError('too many confirmations')
    return evidence.elond(logs)
