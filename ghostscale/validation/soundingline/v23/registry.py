"""Complete declared design grids, priority and finite program limits."""
from itertools import product
from .common import canonical

ORDER = ('R1', 'R2', 'R3', 'R4', 'N1', 'N2', 'N5', 'N3', 'N4', 'N7', 'N6',
         'T1', 'T2', 'T4', 'T3', 'N8')
EXPECTED = dict(R1=36, R2=16, R3=72, R4=90, N1=240, N2=124, N3=96,
                N4=172, N5=128, N6=108, N7=128, N8=24)
PRIMARY = {
    'R1': ('masked', 'law-marginal', 'content', 'information_gain'),
    'R2': ('withdraw-all-0.05-0.7-none', 'no-gate', 'purpose', 'recognition_cost'),
    'R3': ('unit', 'effort-weighted', 'future-choice', 'heuristic_gain'),
    'R4': ('bottom-up', 'law-marginal', 'root', 'information_gain'),
    'N1': ('individual-only', 'law-marginal', 'out-of-context-choice', 'pooling_gain'),
    'N2': ('everything-is-preference', 'response-time-weighted', 'preference', 'weighting_gain'),
    'N3': ('final-choice-only', 'half-space', 'future-choice', 'correction_gain'),
    'N4': ('drive-weights-only', 'law-marginal', 'separating-choice', 'interaction_gain'),
    'N5': ('projection', 'calibrated-projection', 'handling', 'calibration_gain'),
    'N6': ('withdraw', 'model-and-gate', 'low-low-next-action', 'modeling_gain'),
    'N7': ('choices-only', 'pooled-correction', 'future-choice', 'statement_gain'),
    'N8': ('mode-blind', 'law-marginal', 'preference', 'mode_gain'),
}


def grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, values)) for values in product(*(axes[k] for k in keys))]


def variants(family):
    if family == 'R1':
        rows = grid(prevalence=[.05, .2, .5], strength=[.5, .6, .7, .9], bits=[2, 8, 32])
    elif family == 'R2':
        rows = grid(coupled=[False, True], prefix=[4, 8, 16, 32], benign_share=[.5, .8])
    elif family == 'R3':
        rows = grid(cost_role=['relevant', 'irrelevant', 'half'], records=[2, 8, 32, 128],
                    dimensions=[2, 3, 4], shift=[False, True])
    elif family == 'R4':
        rows = grid(cue_reliability=[.5, .7, .9, .99], coverage=[0, .1, .25, .5, 1],
                    branching=[2, 3], cue_position=['root', 'function'])
        rows = [dict(row, misleading=False) for row in rows]
        rows += [dict(row, cue_reliability=.3, misleading=True, cue_position='root')
                 for row in grid(coverage=[0, .1, .25, .5, 1], branching=[2, 3])]
    elif family == 'N1':
        rows = grid(makers=[10, 100, 1000, 10000], records=[2, 8, 32, 128, 512],
                    concentration=[.05, .5, 5], structure=['id', 'alias', 'drift-.01', 'drift-.05'])
    elif family == 'N2':
        rows = grid(mark_class=['near-tie', 'mixed-strategy', 'spontaneous', 'prime', 'context'],
                    nondiagnostic=[.25, .75], rt_slope=[0, 1, 2], records=[16, 128], beta=[1, 4])
        rows += [dict(row, mark_class='preference', nondiagnostic=0, rt_slope=0)
                 for row in grid(records=[16, 128], beta=[1, 4])]
    elif family == 'N3':
        rows = grid(dimensions=[3, 5], beta=[1, 4], local_share=[0, .25, .5, .75],
                    edits=[4, 16], records=[2, 8, 32])
    elif family == 'N4':
        rows = []
        for drives, counts in [(6, [1, 2, 3]), (9, [1, 2])]:
            rows += [dict(row, drives=drives, extension=False)
                     for row in grid(meaning=['additive', 'conjunctive'], recipes=counts,
                                     seeking=['vigor', 'dimension'], records=[8, 32, 128, 512],
                                     menus=['random', 'separating'])]
        rows += [dict(row, extension=True) for row in grid(
            mechanism=['transition', 'habit', 'intrinsic'], records=[8, 32], changed=[False, True])]
    elif family == 'N5':
        rows = grid(offset=[0, .5, 1, 2], shape=['anchor', 'random'], records=[0, 2, 8, 32],
                    concentration=[2, 20], quality=['correct', 'absent'])
    elif family == 'N6':
        rows = grid(lowlow_share=[.1, .25, .5], stakes=[0, .5, 1], records=[2, 8, 32],
                    coupled=[False, True], threshold=[.7, .9])
    elif family == 'N7':
        rows = grid(statement_sd=[.25, 1.], shift=[0, .5, 1, 2], shift_kind=['shared', 'individual'],
                    records=[0, 2, 8, 32], aspiration=[0, .3])
    elif family == 'N8':
        rows = grid(play_share=[0, .3, .6], signal_reliability=[.6, .9], threat_rate=[0, .2], records=[8, 32])
    elif family == 'T1':
        rows = grid(makers=[61], personal_effect=[0, .1, .3, .6], dirty=[False, True], sessions=[24])
    elif family == 'T2':
        rows = grid(players=[200, 2000], games=[20, 200, 5000], depth=[6, 10], branching=[3, 5], dirty=[False, True])
    elif family == 'T3':
        rows = grid(makers=[500, 5000], max_records=[100, 1000, 10000], summary_honesty=[.5, .9], dirty=[False, True])
    elif family == 'T4':
        rows = grid(makers=[100, 1000], items=[32, 128], flood=['none', 'individual', 'group'], dirty=[False, True])
    else:
        raise ValueError(family)
    if family in EXPECTED and len(rows) != EXPECTED[family]:
        raise ValueError('specification grid count changed: '+family)
    return [dict(id=f'{family}-{i:04d}', family=family, settings=row,
                 late_effect=(family == 'N1' and row['records'] == 512) or
                 (family == 'N3' and row.get('records') == 32) or
                 (family == 'N4' and row.get('records') == 512) or
                 (family == 'N7' and row.get('records') == 32))
            for i, row in enumerate(rows)]


def selection_eligible(variant):
    family, s = variant['family'], variant['settings']
    if family == 'N1':
        return s['records'] == 512 and s['structure'] != 'id'
    if family == 'R3':
        return s['cost_role'] == 'irrelevant'
    if family == 'N7':
        return s['records'] <= 8
    if family == 'N4':
        return not s['extension']
    return family in PRIMARY


def variant_features(variant):
    """One frozen encoding of settings; no result-dependent category order."""
    family = variant['family']
    population = variants(family)
    # Design labels are not automatically observed evidence. These knobs are
    # marginalized/inferred by the exact reference, so direct fits must infer
    # them from the same public history rather than receiving the planted value.
    hidden = {'N4': {'mechanism'}}.get(family, set())
    keys = sorted({k for v in population for k in v['settings']} - hidden)
    out = []
    for key in keys:
        values = sorted({canonical(v['settings'].get(key)).decode() for v in population})
        value = canonical(variant['settings'].get(key)).decode()
        out.extend(float(value == item) for item in values)
    return out


def manifest():
    return dict(version='v23', total_cpu_seconds=90*3600, protected_cpu_seconds=18*3600,
                max_workers=3, thread_count=1, gpu=False, elapsed_days=7,
                priority=list(ORDER), variants={f: variants(f) for f in ORDER},
                primary_contrasts={f: dict(rival=p[0], mechanism=p[1], target=p[2], meaning=p[3])
                                   for f, p in PRIMARY.items()})
