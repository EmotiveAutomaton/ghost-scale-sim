"""Retained single-maker N-family execution; unavailable predictions stay explicit.

Population and twin worlds have separate contracts. Production admission requires
a bound plan, family controls, a complete independent checker and cost pilot.
"""
import hashlib
import hmac
import math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from . import registry, dispatch, ledger, checker, checker_new, diagnosticity, projection, statements
from .common import Direct, canonical, digest, interval, losses, read, rng, write
from .pipeline import Stream, headroom_groups

BUILT = ('N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8')
CLASSES = {'N2': 2, 'N3': 4, 'N4':4, 'N5': 4, 'N6': 4, 'N7': 3, 'N8': 2}


def variants(family):
    # N4b is a separate conditional probe, with a binary target and no declared
    # drive-weights contrast. It must not silently enter the four-class fit.
    return [v for v in registry.variants(family) if family!='N4' or not v['settings']['extension']]


def features(v, public):
    return np.r_[dispatch.features(v, public), registry.variant_features(v)]


def training(family, phase, count, pulse=lambda **kw: None, law=None,variant=None):
    population = variants(family) if variant is None else [variant]; x, y = [], []
    if variant is not None and variant not in variants(family):raise ValueError('unknown variant-only training target')
    sampler=dispatch.sample
    if family=='N2':
        from .diagnosticity_training import sample as sampler
    for i in range(count):
        v = population[i % len(population)]
        public, truth = sampler(v, phase, i//len(population), i,
            law=(i//len(population)) % 16 if law is None else law)
        x.append(features(v, public)); y.append(truth['y'])
        if i % 256 == 0: pulse(phase='training-generation', labels=i, target_labels=count)
    return np.array(x), np.array(y)


def fit_family(root, family, pulse,variant=None,namespace=None):
    root.mkdir(parents=True, exist_ok=False); models, metadata = {}, {}
    phase=lambda label:label if namespace is None else [namespace,label]
    vx, vy = training(family, phase('validation'), 4096, pulse,variant=variant)
    np.savez_compressed(root/'validation.npz', x=vx, y=vy)
    for labels in (2048, 8192):
        x, y = training(family, phase('training'), labels, pulse,variant=variant)
        np.savez_compressed(root/f'training-{labels}.npz', x=x, y=y)
        for fit in (0, 1):
            for fresh in ((False, True) if labels == 8192 else (False,)):
                name = f'direct-{labels}-fit{fit}'+('-fresh' if fresh else '')
                draws = []
                def provider(epoch, offset, size):
                    law = int(rng('fresh-law', family, fit, epoch, offset).integers(16))
                    draw_phase = phase(f'fresh-{fit}-{epoch}-{offset}')
                    xx, yy = training(family, draw_phase, size, pulse, law,variant)
                    draws.append(dict(phase=draw_phase, count=size, law=law,
                                      x_sha256=hashlib.sha256(xx.tobytes()).hexdigest(),
                                      y_sha256=hashlib.sha256(yy.tobytes()).hexdigest()))
                    return xx, yy
                fit_identity=family if namespace is None else canonical([namespace,family]).decode()
                model = Direct(x, y, vx, vy, CLASSES[family], fit_identity, fit,
                               fresh=provider if fresh else None, pulse=pulse)
                model.save(root/(name+'.npz')); models[name] = model
                metadata[name] = model.metadata
                write(root/(name+'-cost.json'), {'cpu_seconds': model.cpu_seconds})
                if fresh: write(root/(name+'-draws.json'), draws)
    write(root/'FITS.json', metadata)
    return models, metadata


def run_fits(root, plan, pulse,fit_builder=None):
    """Freeze rival training before discovery; all training work is replayed."""
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    models,metadata=(fit_builder or fit_family)(root/'fits',plan['family'],pulse)
    with np.load(root/'fits/validation.npz') as data:
        x,y=data['x'],data['y']
    checked={}
    for name,model in models.items():
        prediction=model.predict(x)
        rebuilt=checker.scores(prediction.reshape(-1,prediction.shape[-1]),y.ravel())
        observed=float(np.mean(rebuilt['log_loss']))
        if not math.isclose(observed,metadata[name]['validation_log_loss'],abs_tol=1e-8,rel_tol=0):
            raise ValueError('saved fit validation score mismatch')
        loaded=Direct.load(root/'fits'/(name+'.npz'))
        if not np.array_equal(loaded.predict(x),prediction):raise ValueError('fit serialization changed predictions')
        checked[name]=dict(validation_log_loss=observed,early_stopped=metadata[name]['early_stopped'])
    write(root/'SUMMARY.json',dict(state='FITS_VERIFIED',family=plan['family'],fits=checked,
        discovery_exposure=False,confirmation_exposure=False,
        convergence_note='validation stopping alone does not establish direct-ceiling parity'))


def retain_auxiliary(root, v):
    """Retain fitted heuristic evidence separately from test makers and public data."""
    if v['family'] == 'N2':
        s = v['settings']
        weights, counts, rows = diagnosticity.weight_training(v['id'], s['mark_class'],
            s['nondiagnostic'], s['rt_slope'], s['beta'])
        totals = np.zeros(5); rebuilt_counts = np.zeros(5)
        stream = Stream(root/'timing_training_points.jsonl.gz')
        try:
            for record in rows:
                row, hidden = record['public'], record['evaluator']
                # Independent likelihood-ratio arithmetic for supervised labels.
                beta = s['beta']; sigmoid = lambda x: 1/(1+math.exp(-x))
                probs = []
                for sign in (-1, 1):
                    probs.append((sigmoid(beta*sign), sigmoid(.05*beta*sign),
                        .15+.7*hidden['law']/15, .5,
                        .6*row['previous']+.4*sigmoid(beta*sign),
                        sigmoid(beta*(sign+2*row['context'])))[hidden['kind']])
                d = abs(math.log(probs[1] if row['choice'] else 1-probs[1])-
                        math.log(probs[0] if row['choice'] else 1-probs[0]))
                if not math.isclose(d, hidden['diagnosticity'], abs_tol=1e-12):
                    raise ValueError('timing-training diagnosticity changed')
                totals[row['latency_bin']] += d; rebuilt_counts[row['latency_bin']] += 1
                stream.write(record)
        finally: stream.close()
        reconstructed = np.minimum(1, np.maximum(0, totals/np.maximum(rebuilt_counts, 1)/s['beta']))
        if not np.array_equal(counts, rebuilt_counts) or not np.allclose(weights, reconstructed, atol=1e-12):
            raise ValueError('timing weight reconstruction mismatch')
        write(root/'TIMING_FIT.json', dict(weights=weights, counts=counts, makers=len(rows),
              training_namespace='diagnosticity-rt-training', supervised_diagnosticity_labels=True,
              test_labels_used=False, independent_reconstruction=True))
    if v['family'] == 'N5':
        nodes = projection.prior_nodes(v['settings']['concentration'], 10)
        np.savez_compressed(root/'quadrature.npz', nodes=nodes)
    if v['family']=='N4':
        from .values import pooling_training
        from .checker_recipes import verify_pooling_training
        s=v['settings'];prior,rows=pooling_training(v['id'],s['drives'],s['recipes'],s['meaning'],s['seeking'])
        verify_pooling_training(s['drives'],s['recipes'],s['meaning'],s['seeking'])
        stream=Stream(root/'pooling_training_points.jsonl.gz')
        try:
            for row in rows:
                stream.write(row)
        finally:stream.close()
        write(root/'POOLING_FIT.json',dict(prior=prior,makers=32,records=8,prior_pseudocount=.5,
            hidden_labels_used=False,training_namespace='recipe-pooling-training',exact_ceiling=False,
            independent_reconstruction=True,shared_across_history_depth_and_menu_design=True))
    if v['family']=='N6':
        from .payoff import policy,verify_policy
        s=v['settings'];args=(s['lowlow_share'],s['stakes'],s['records'],s['coupled'])
        table=policy(*args);validity=verify_policy(*args)
        np.savez_compressed(root/'payoff-policy.npz',**{f'value_{k}':x for k,x in table['values'].items()},
                            **{f'acquire_{k}':x for k,x in table['acquire'].items()})
        write(root/'PAYOFF_POLICY.json',dict(validity=validity,expected_initial_payoff=table['expected_initial_payoff'],
              scope=table['scope'],prediction_weight='stakes for low-low makers; one otherwise',
              separate_purpose_benefit=0,compute_fee=.001))
    if v['family'] == 'N7':
        from .checker_recipes import verify_shift_training
        s=v['settings'];identity=statements.shift_training_identity(s)
        args=(identity,s['statement_sd'],s['shift'],s['shift_kind'],s['aspiration'])
        shift,rows=statements.shift_training(*args);verify_shift_training(*args)
        stream=Stream(root/'shift_training_points.jsonl.gz')
        try:
            for row in rows:
                stream.write(row)
        finally:stream.close()
        write(root/'SHIFT_FIT.json',dict(shift=shift,makers=32,records=128,training_namespace='shift-training',
              shared_across_personal_history_depth=True,training_identity=identity,
              hidden_labels_used=False,independent_reconstruction=True))


def verify_case(v, p, truth, prediction, detail):
    family = v['family']
    if family=='N4':
        from .checker_recipes import verify_n4
        return verify_n4(v,p,truth,prediction,detail)
    if family == 'N7':
        from .checker_recipes import verify_n7
        return verify_n7(v,p,truth,prediction,detail)
    if family == 'N5':
        from .checker_projection import verify
        return verify(v, p, truth, prediction, detail, projection.offsets(v),
                      projection.prior_nodes(v['settings']['concentration'], 10))
    error = getattr(checker_new, 'verify_'+family.lower())(v, p, truth, prediction, detail)
    if family == 'N6':
        from .payoff import verify_case as verify_payoff
        error=max(error,verify_payoff(v,p,truth,prediction['payoff-optimal'],detail['payoff-optimal']))
    if family == 'N2' and 'oracle-weighted' in prediction:
        # Diagnostic oracle is evaluated only on the evaluator side of the boundary.
        value = math.fsum((2*row['choice']-1)*hidden['diagnosticity']
                         for row, hidden in zip(p['history'], truth['marks'], strict=True))
        q = 1/(1+math.exp(-value)) if value >= 0 else math.exp(value)/(1+math.exp(value))
        if not np.allclose(prediction['oracle-weighted'], [[1-q, q]], atol=1e-11, rtol=0):
            raise ValueError('oracle diagnostic reconstruction mismatch')
    return error


def score(prediction, truth):
    if prediction is None:
        return dict(available=False, reason='EMPTY_HARD_CONSTRAINT_SUPPORT',
                    log_loss=None, half_brier=None, accuracy=None)
    a, b = losses(prediction, truth['y']), checker.scores(prediction, truth['y'])
    for metric in a:
        if not np.allclose(a[metric], b[metric], atol=1e-12, rtol=0, equal_nan=False):
            raise ValueError('independent proper score mismatch')
    values = {name: float(np.mean(value)) for name, value in a.items()}
    return dict(available=all(math.isfinite(x) for x in values.values()),
        **{name: value if math.isfinite(value) else None for name, value in values.items()},
        reason=None if math.isfinite(values['log_loss']) else 'INFINITE_RAW_LOG_LOSS')


def secondary(v, p, truth, prediction, detail):
    family = v['family']; out = {}
    if family == 'N2':
        actual = np.array([row['diagnosticity'] for row in truth['marks']])
        estimated = np.array(detail['estimated_diagnosticity'])
        out['diagnosticity:rank_correlation'] = (float(spearmanr(actual, estimated).statistic)
            if np.ptp(actual) > 0 and np.ptp(estimated) > 0 else None)
        for name, prob in prediction.items():
            beta = v['settings']['beta']; q = 1/(1+math.exp(-beta))
            # A fresh preference-choice forecast, scored against its planted conditional law.
            forecast = (1-q)*prob[0, 0]+q*prob[0, 1]
            target = q if truth['y'][0] else 1-q
            out[name+':heldout_preference_choice_expected_loss'] = float(-target*np.log(forecast)-(1-target)*np.log1p(-forecast))
            out[name+':signed_preference_error'] = float(prob[0, 1]-truth['y'][0])
            out[name+':calibration_residual'] = float(truth['y'][0]-prob[0, 1])
    elif family == 'N3':
        from .corrections import directions
        directions_ = directions(v['settings']['dimensions'])
        for name, item in detail.items():
            if name == 'half-space' and item.get('state') == 'EMPTY_HARD_CONSTRAINT_SUPPORT':
                # Keep the diagnostic schema when the literal reader is undefined.
                # A missing value invalidates the whole-world mean; it is not zero
                # and must not disappear depending on which case comes first.
                out[name+':direction_cosine'] = None
                continue
            if 'direction_posterior' in item: direction = np.array(item['direction_posterior'])@directions_
            elif name == 'coactive': direction = np.array(item['direction'])
            else: continue
            length = np.linalg.norm(direction)
            out[name+':direction_cosine'] = float(np.dot(direction, truth['w'])/length) if length else 0.
        out['half-space:available'] = float(prediction['half-space'] is not None)
    elif family=='N4':
        from .makers import recipe_library
        s=v['settings'];true_recipes=recipe_library(s['drives'],exactly=s['recipes'],seeking=s['seeking'])[0]
        true_presence=dict(zip(true_recipes,truth['presence']))
        for name,row in detail.items():
            recipes_=recipe_library(s['drives'],exactly=s['recipes'],seeking=row['seeking_model'])[0]
            target=np.array([true_presence.get(r,0) for r in recipes_]);presence=np.array(row['recipe_presence'])
            with np.errstate(divide='ignore'):loss=-np.log(np.where(target,presence,1-presence))
            out[name+':recipe_presence_loss']=float(loss.mean()) if np.isfinite(loss).all() else None
            probability=row['act_probability'] if truth['acted'] else 1-row['act_probability']
            out[name+':act_or_abstain_loss']=-math.log(probability) if probability else None
            probability=row['vigor_posterior'][truth['vigor']]
            out[name+':seeking_level_loss']=-math.log(probability) if probability else None
            out[name+':profile_domination']=float(row['seeking_profile_mass']>.5)
    elif family == 'N5':
        out['approximate:quadrature_max_difference'] = detail['quadrature_max_difference']
        for name, prob in prediction.items():
            for k in range(4):
                out[f'{name}:class-{k}:bias'] = float(prob[0, k]-(truth['y'][0] == k))
                out[f'{name}:class-{k}:named'] = float(prob[0].argmax() == k)
                out[f'class-{k}:observed'] = float(truth['y'][0] == k)
    elif family == 'N6':
        for mode in ('model', 'withdraw', 'absorb-everything', 'payoff-optimal'):
            for key in ('updates', 'adopted_purposes', 'adopted_techniques', 'unwanted_uptake',
                        'useful_technique', 'failed_technique', 'nonprediction_payoff'):
                out[mode+':'+key] = float(detail[mode][key])
            name='model-and-gate' if mode=='model' else mode
            weight=v['settings']['stakes'] if truth['warmth']==truth['competence']==0 else 1.
            out[mode+':total_payoff']=detail[mode]['nonprediction_payoff']-weight*float(losses(prediction[name],truth['y'])['log_loss'].mean())
        out['withdraw:false_withdrawal'] = float(any(row['stopped'] for row in detail['withdraw']['path'])
                                                and (truth['warmth'] or truth['competence']))
    elif family == 'N7':
        for name,row in detail.items():
            if name=='training':continue
            presence=np.array(row['recipe_presence']);target=np.array(truth['presence'])
            with np.errstate(divide='ignore'):
                loss=-np.log(np.where(target,presence,1-presence))
            out[name+':recipe_presence_loss']=float(loss.mean()) if np.isfinite(loss).all() else None
            out[name+':desirable_bias']=float(np.mean((presence-target)*statements.shift_direction()))
        out['aspirational_maker']=float(truth['aspiration'])
    return out


def aggregate(values, identity):
    """No deletion of unavailable or nonfinite observations to manufacture a mean."""
    if any(value is None or not math.isfinite(value) for value in values):
        return dict(finite=False, mean=None, lower=None, upper=None, se=None,
                    worlds=len(values), unavailable=sum(value is None for value in values))
    return interval(values, identity)


def run_batch(root, plan, pulse):
    family = plan['family']
    if family not in BUILT: raise ValueError('N-family pipeline/checker not built: '+family)
    confirmation=plan.get('task')=='confirmation'
    if confirmation:
        from .confirmation import verify_definition
        verify_definition(plan)
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    schema, _ = ledger.canonical_schema(plan['schema_directory'], plan['schema_pin'])
    population = [v for v in variants(family) if v['id'] in plan['variants']]
    if len(population) != len(plan['variants']) or not population: raise ValueError('variant membership changed')
    batch = plan['batch']; sample_phase = ['confirmation',plan['nomination_order_sha256']] if confirmation else ['discovery', batch]
    from .partition import world_indices
    worlds=world_indices(plan)
    if plan.get('fit_directory'):
        fitroot = Path(plan['fit_directory']); fitmeta = read(fitroot/'FITS.json')
        for name, expected in plan['fit_hashes'].items():
            if digest(fitroot/name) != expected: raise ValueError('frozen fit changed')
        fitted = {name: Direct.load(fitroot/(name+'.npz')) for name in fitmeta}
    else:
        if batch != 1 and not confirmation: raise ValueError('only first batch may fit')
        fitted, fitmeta = fit_family(root/'fits', family, pulse,variant=population[0] if confirmation else None,
                                    namespace=sample_phase if confirmation else None)
    summaries = []; index = {}
    write(root/'DESIGN.json', {k: plan[k] for k in ('family', 'batch', 'variants', 'worlds', 'cases', 'schema_pin')})
    for v in population:
        folder = root/v['id']; folder.mkdir(); retain_auxiliary(folder, v)
        streams = {kind: Stream(folder/(kind+'_points.jsonl.gz')) for kind in ('public','truth','prediction','posterior','ledger')}
        reader_worlds = {}; gains = []; headroom = {}; secondary_worlds = {}; strata = {}; target_counts = []
        max_check = 0.; max_roundtrip = 0.; unavailable = {}
        try:
            for world in worlds:
                publics = []; truths = []; predictions = []; details = []
                for case in range(plan['cases']):
                    public, truth = dispatch.sample(v, sample_phase, world, case)
                    prediction, detail = dispatch.evaluate(v, public, truth)
                    max_check = max(max_check, verify_case(v, public, truth, prediction, detail))
                    identity = [family, v['id'], batch, world, case]
                    canary = hmac.new(bytes.fromhex(plan['canary_key']), canonical(identity), hashlib.sha256).hexdigest()
                    bundle = ledger.emit(v, public, schema, identity)
                    if canary.encode() in canonical(bundle):raise ValueError('truth canary leaked')
                    # decode performs the complete schema/reference/content validation.
                    # Do not perform the identical full validation twice per case.
                    decoded = ledger.decode(v, bundle, schema)
                    decoded_prediction = dispatch.evaluate(v, decoded)[0]
                    for name, expected in prediction.items():
                        if name.startswith('oracle-'): continue
                        actual = decoded_prediction[name]
                        if expected is None:
                            if actual is not None: raise ValueError('ledger obscured unavailable prediction')
                        else:
                            max_roundtrip = max(max_roundtrip, float(np.max(np.abs(expected-actual))))
                    if max_roundtrip > 1e-12: raise ValueError('ledger/native prediction mismatch')
                    streams['public'].write(dict(identity=identity, public=public))
                    streams['truth'].write(dict(identity=identity, truth=truth, canary=canary))
                    streams['ledger'].write(dict(identity=identity, ledger=bundle))
                    publics.append(public); truths.append(truth); predictions.append(prediction); details.append(detail)
                for name, model in fitted.items():
                    probs = model.predict(np.array([features(v, public) for public in publics]))
                    for prediction, prob in zip(predictions, probs): prediction[name] = prob
                scores = [{name: score(prob, truth) for name, prob in prediction.items()}
                          for prediction, truth in zip(predictions, truths)]
                for name in predictions[0]:
                    unavailable[name] = unavailable.get(name, 0)+sum(not row[name]['available'] for row in scores)
                    values = {metric: (float(np.mean([row[name][metric] for row in scores]))
                        if all(row[name][metric] is not None for row in scores) else None)
                        for metric in ('log_loss','half_brier','accuracy')}
                    reader_worlds.setdefault(name, []).append(values)
                eligible = [i for i, truth in enumerate(truths) if family != 'N6' or truth['warmth'] == truth['competence'] == 0]
                target_counts.append(len(eligible)); rival, mechanism = registry.PRIMARY[family][:2]
                values = [(scores[i][rival]['log_loss'], scores[i][mechanism]['log_loss']) for i in eligible]
                gains.append(float(np.mean([a-b for a,b in values])) if values and all(a is not None and b is not None for a,b in values) else None)
                ceiling = 'approximate-hierarchical' if family == 'N5' else 'law-marginal'
                for name in fitted:
                    a, b = reader_worlds[name][-1]['log_loss'], reader_worlds[ceiling][-1]['log_loss']
                    headroom.setdefault(name, []).append(a-b if a is not None and b is not None else None)
                if family == 'N6':
                    for warmth in (0, 1):
                        for competence in (0, 1):
                            selected = [i for i,t in enumerate(truths) if (t['warmth'],t['competence']) == (warmth,competence)]
                            for name in predictions[0]:
                                strata.setdefault(f'{warmth}-{competence}:{name}', []).append(
                                    float(np.mean([scores[i][name]['log_loss'] for i in selected])) if selected else None)
                for case, (prediction, detail) in enumerate(zip(predictions, details)):
                    streams['prediction'].write(dict(world=world, case=case, probabilities=prediction, scores=scores[case], primary_eligible=case in eligible))
                    streams['posterior'].write(dict(world=world, case=case, detail=detail))
                seconds = [secondary(v,p,t,pr,d) for p,t,pr,d in zip(publics,truths,predictions,details)]
                for key in seconds[0]:
                    vals = [row[key] for row in seconds]
                    secondary_worlds.setdefault(key, []).append(float(np.mean(vals)) if all(x is not None and math.isfinite(x) for x in vals) else None)
                pulse(phase='family-batch', family=family, batch=batch, variant=v['id'], world=world+1, worlds=plan['worlds'])
        finally:
            for stream in streams.values(): stream.close()
        if plan.get('world_indices') is not None:
            write(folder/'WORLD_AGGREGATES.json',dict(worlds=worlds,reader_worlds=reader_worlds,
                gains=gains,headroom=headroom,secondary_worlds=secondary_worlds,strata=strata,
                target_counts=target_counts,unavailable=unavailable,max_check=max_check,max_roundtrip=max_roundtrip))
        primary = aggregate(gains, [v['id'], batch, 'primary'])
        structural = family == 'N5' and v['settings']['offset'] == 0 or family == 'N8' and v['settings']['play_share'] == 0 or family == 'N2' and v['settings']['nondiagnostic'] == 0 or family=='N4' and v['settings']['meaning']=='additive'
        if structural and any(x is None or abs(x)>1e-11 for x in gains): raise ValueError('structural null failed')
        groups = headroom_groups(headroom, fitmeta, v, batch) if family != 'N5' and all(all(x is not None for x in vals) for vals in headroom.values()) else {}
        summary = dict(variant=v, primary=primary, world_gains=gains, primary_counts=target_counts,
            primary_status='AVAILABLE' if primary['finite'] else 'UNAVAILABLE_NO_CASE_DELETION',
            readers={name:{metric:aggregate([r[metric] for r in rows],[v['id'],batch,name,metric]) for metric in rows[0]} for name,rows in reader_worlds.items()},
            secondary={name:aggregate(vals,[v['id'],batch,'secondary',name]) for name,vals in secondary_worlds.items()},
            strata={name:aggregate(vals,[v['id'],batch,'stratum',name]) for name,vals in strata.items()},
            unavailable_predictions=unavailable, matched_headroom_groups=groups,
            approximate_ceiling=family == 'N5', structural_zero=structural,
            headroom_eligible=family != 'N5', laws=len({w%16 for w in worlds}), independent_worlds=len(worlds),
            law_counts={str(i):sum(w%16==i for w in worlds) for i in range(16)},
            validity=dict(independent_posterior_max_error=max_check,ledger_forecast_max_error=max_roundtrip))
        write(folder/'SUMMARY.json', summary); summaries.append(summary)
        index[v['id']] = {p.name:digest(p) for p in folder.iterdir() if p.is_file()}
    write(root/'RAW_INDEX.json', index)
    write(root/'SUMMARY.json', dict(family=family,batch=batch,variants=summaries,fits=fitmeta,
        status='independent-check-passed; complete replay still required',
        result_kind='constructed mechanism' if family == 'N6' else 'method',
        scope='finite constructed generator; conditional on frozen training; unavailable primaries retained'))
