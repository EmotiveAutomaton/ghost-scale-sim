"""A03 full, partial or withheld answers from unchanged joint forecasts."""
from pathlib import Path
import numpy as np
from ..v18_3.io import read, write, file_digest

RATIOS = (1., 2., 5., 10.)


def forecast_entries(summary, source_format):
    """Select a preregistered forecast population without refitting or relabeling."""
    if source_format == 'learned':
        return [(entry, f"raw/{entry['fit']}-{entry['lineage']}-{entry['reader']}_points.npz")
                for entry in summary['population']]
    if source_format != 'dependence-calibrated':
        raise ValueError('unknown frozen forecast format')
    entries = [entry for entry in summary['population']
               if entry['reader'] == 'episode-calibrated' and entry['duplication_rate'] == 0.]
    identities = [(entry['fit'], entry['lineage']) for entry in entries]
    if not entries or len(set(identities)) != len(identities):
        raise ValueError('missing or duplicate calibrated forecast identity')
    return [(entry, f"raw/{entry['fit']}-{entry['lineage']}-0.0-episode-calibrated_points.npz")
            for entry in entries]


def frontier(probability):
    """Best supported conjunction for each number of asserted binary fields.

    Ties use the lowest integer mask/assignment within 1e-12. The integer bit
    convention is the target label's binary encoding, not a new target.
    """
    n = len(probability)
    confidence = np.zeros((n, 8)); masks = np.zeros((n, 8), int)
    values = np.zeros((n, 8), int); confidence[:, 0] = 1.
    tensor = probability.reshape((n,) + (2,) * 7)
    for mask in range(1, 128):
        bits = [b for b in range(7) if mask & (1 << b)]
        axes = tuple(7-b for b in range(7) if b not in bits)
        marginal = tensor.sum(axis=axes).reshape(n, -1) if axes else probability
        best = marginal.max(1)
        choice = np.argmax(marginal >= best[:, None]-1e-12, axis=1)
        assignments = np.array([sum(((v >> j) & 1) << b for j, b in enumerate(bits)) for v in range(2**len(bits))])
        k = len(bits); improve = best > confidence[:, k]+1e-12
        # Save the chosen event's actual mass, including any declared near-tie.
        confidence[improve, k] = marginal[np.arange(n), choice][improve]
        masks[improve, k] = mask; values[improve, k] = assignments[choice][improve]
    return masks, values, confidence


def decide(masks, values, confidence, ratio):
    risk = (7-np.arange(8))/7 + ratio*(1-confidence)
    best = risk.min(1)
    # Fewer assertions wins a numerical tie across cardinalities.
    k = np.argmax(risk <= best[:, None]+1e-12, axis=1)
    at = np.arange(len(k)), k
    return dict(mask=masks[at], value=values[at], confidence=confidence[at],
                asserted=k, expected_loss=risk[at])


def assess(answer, truth, ratio):
    wrong = (truth & answer['mask']) != answer['value']
    return dict(answer, false_specificity=wrong, loss=(7-answer['asserted'])/7+ratio*wrong,
                alternatives=2**(7-answer['asserted']))


def controls():
    certain = np.eye(128)[[19, 100]]
    m, v, p = frontier(certain); a = decide(m, v, p, 10.)
    mm, vv, pp = frontier(np.full((1, 128), 1/128)); b = decide(mm, vv, pp, 2.)
    return dict(positive_certain_full=bool(np.all(a['asserted']==7) and np.array_equal(a['value'], [19, 100])),
                placebo_uniform_abstains=bool(b['asserted'][0]==0),
                live_abstention_retains_universe=bool(assess(b, np.array([31]), 2.)['alternatives'][0]==128))


def run(root, design, pulse):
    from .answer_policy_check import reference_frontier, check
    rows = []; maximum = 0.; original_forecasts = 0
    for binding in design['forecast_packets']:
        source = Path(binding['root']); complete = read(source/'COMPLETE.json')
        assert file_digest(source/'COMPLETE.json') == binding['complete_sha256']
        source_design = read(source/'PLAN.json')['design']
        assert source_design['lineages'] == design['lineages']
        assert not source_design.get('split') == 'confirmation'
        entries = forecast_entries(read(source/'SUMMARY.json'), design.get('forecast_format', 'learned'))
        if design.get('forecast_format') == 'dependence-calibrated':
            assert {(entry['fit'], entry['lineage']) for entry, _ in entries} == {
                (fit, world) for fit in design['fits'] for world in design['lineages']}
        for entry, name in entries:
            fit, world, reader = entry['fit'], entry['lineage'], entry['reader']
            assert file_digest(source/name) == complete['files'][name]
            with np.load(source/name) as z: probability = np.exp(z['logp']); truth = z['truth']
            m, v, p = frontier(probability); original_forecasts += len(truth)
            # Separate scalar enumeration checks the complete first case of every block.
            rm, rv, rp = reference_frontier(probability[0])
            error = float(np.max(abs(p[0]-rp))); maximum = max(maximum, error)
            assert error < 1e-12 and np.array_equal(m[0], rm) and np.array_equal(v[0], rv)
            for ratio in RATIOS:
                answer = assess(decide(m, v, p, ratio), truth, ratio)
                check(probability, truth, answer, ratio)
                for k in range(8):
                    fixed = assess(dict(mask=m[:,k], value=v[:,k], confidence=p[:,k], asserted=np.full(len(truth), k), expected_loss=(7-k)/7+ratio*(1-p[:,k])), truth, ratio)
                    key = f"{binding['cell']}-{fit}-{world}-{reader}-{ratio}-fixed-{k}"
                    (root/'raw').mkdir(parents=True,exist_ok=True);np.savez_compressed(root/'raw'/f'{key}_points.npz', **fixed)
                    rows.append(dict(cell=binding['cell'],fit=fit,lineage=world,reader=reader,ratio=ratio,policy='fixed-specificity',fields=k,metrics={key:float(np.mean(val)) for key,val in fixed.items() if key not in ('mask','value')}))
                key = f"{binding['cell']}-{fit}-{world}-{reader}-{ratio}-adaptive"
                np.savez_compressed(root/'raw'/f'{key}_points.npz', **answer)
                rows.append(dict(cell=binding['cell'],fit=fit,lineage=world,reader=reader,ratio=ratio,policy='adaptive',fields=-1,metrics={key:float(np.mean(val)) for key,val in answer.items() if key not in ('mask','value')}))
            pulse(phase='answer-policy',cell=binding['cell'],fit=fit,world=world,reader=reader)
    return dict(card='G21-A03',controls=controls(),population=rows,original_forecasts=original_forecasts,independent_max_probability_error=maximum,
                scope='Decision policies on unchanged selected-development forecasts; no refitting or new independent worlds.',
                loss='Fraction of seven fields withheld plus ratio times any false asserted field. All-withheld costs one; a false full answer costs ratio.',
                limits=['Same action library for every reader; fresh-law is a separate extra-label diagnostic.',
                        'Expected policy risk is not observed risk. Fixed-cardinality rows match specificity, not case coverage.',
                        'All 128 finite binary target types remain available; partial answers do not identify exact history.',
                        'Inherited supplied current support, capacity and convergence limitations remain. No confirmation.'])
