"""Pre-flight for Ghost V31 family S (theory card G10): why a maker stays.
A waiting task. With odds p a reward R is coming; otherwise it never comes. Each step of waiting costs c.
Two arrival laws: 'constant' (each step, a coming reward arrives with chance g) and 'rising' (arrival uniform on 1..H).
The maker chooses softly between staying and quitting from the value of staying (dynamic programme, exact).
Four ways to stay longer than baseline:
  cost     the cost weighs less (kappa < 1)
  shift    the prior that the reward is coming is shifted up by a constant in log-odds
  sharpen  the prior's log-odds are multiplied by gamma > 1 (more confident either way)
  slow     evidence that nothing has arrived yet is discounted (power lam < 1 on the survival term)
Each is matched to the same lift at the reference odds. Exact computation, no sampling.
Usage: python3 -I s_stay.py out.json"""
import sys, json, numpy as np
from scipy.optimize import minimize_scalar, brentq
TAU = 0.5
def stay_probs(law, p_true, c=1.0, kappa=1.0, shift=0.0, gamma=1.0, lam=1.0):
    H, R, g = law['H'], law['R'], law.get('g')
    lo = gamma * np.log(p_true / (1 - p_true)) + shift; ph = 1 / (1 + np.exp(-lo))
    V = np.zeros(H + 1); q = np.zeros(H)
    for t in range(H - 1, -1, -1):
        if law['kind'] == 'constant': surv, h = (1 - g) ** t, g
        else: surv, h = (H - t) / H, 1.0 / (H - t)
        s = surv ** lam; b = ph * s / (ph * s + 1 - ph)
        cont = -kappa * c + b * h * R + (1 - b * h) * V[t + 1]
        q[t] = cont; V[t] = TAU * np.logaddexp(0, cont / TAU)
    return 1 / (1 + np.exp(-q / TAU))
def surv_true(law, t):
    return (1 - law['g']) ** t if law['kind'] == 'constant' else (law['H'] - t) / law['H']
def reach(law, p, stay):
    H = law['H']; w = np.ones(H); alive = 1.0
    for t in range(H):
        w[t] = alive * (1 - p + p * surv_true(law, t)); alive *= stay[t]
    return w
def kl_bern(a, b):
    a = np.clip(a, 1e-12, 1 - 1e-12); b = np.clip(b, 1e-12, 1 - 1e-12)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
def wait(law, stay):
    alive, m = 1.0, 0.0
    for t in range(law['H']):
        m += alive * (1 - stay[t]) * t; alive *= stay[t]
    return m + alive * law['H']
LAWS = {'constant hazard': dict(kind='constant', H=40, R=20.0, g=0.15), 'rising hazard': dict(kind='rising', H=20, R=14.0)}
RANGE = {'kappa': (0.05, 1.0), 'shift': (0.0, 8.0), 'gamma': (1.0, 8.0), 'lam': (0.0, 1.0)}
NAME = {'kappa': 'cost', 'shift': 'shift', 'gamma': 'sharpen', 'lam': 'slow'}
ODDS = [0.25, 0.5, 0.75, 0.9]; REF, LIFT = 0.75, 4.0
res = {'temperature': TAU, 'reference_odds': REF, 'target_lift_steps': LIFT, 'laws': {}}
for lname, law in LAWS.items():
    base = {p: wait(law, stay_probs(law, p)) for p in ODDS}
    P = {}
    for par, (lo, hi) in RANGE.items():
        f = lambda x: wait(law, stay_probs(law, REF, **{par: x})) - base[REF] - LIFT
        try: P[par] = {par: float(brentq(f, lo, hi, xtol=1e-10))}
        except ValueError: P[par] = None
    out = {'task': law, 'baseline_wait_on_omission': {str(p): round(base[p], 2) for p in ODDS},
           'matched_parameters': {NAME[k]: (None if v is None else round(v[k], 4)) for k, v in P.items()}}
    out['lift_by_odds'] = {NAME[k]: {str(p): round(wait(law, stay_probs(law, p, **v)) - base[p], 2) for p in ODDS} for k, v in P.items() if v}
    out['lift_when_cost_doubles_at_reference'] = {NAME[k]: round(wait(law, stay_probs(law, REF, c=2.0, **v)) - wait(law, stay_probs(law, REF, c=2.0)), 2) for k, v in P.items() if v}
    def trials(truth, fam, conds):
        f = lambda x: np.mean([float((reach(law, p, stay_probs(law, p, c=c, **truth)) * kl_bern(stay_probs(law, p, c=c, **truth), stay_probs(law, p, c=c, **{fam: x}))).sum()) for p, c in conds])
        lo, hi = (0.02, 3.0) if fam == 'kappa' else (-6, 8) if fam == 'shift' else (0.2, 8) if fam == 'gamma' else (0.0, 2.0)
        grid = np.linspace(lo, hi, 80); x0 = grid[int(np.argmin([f(x) for x in grid]))]; st = (hi - lo) / 79
        r = minimize_scalar(f, bounds=(max(lo, x0 - st), min(hi, x0 + st)), method='bounded', options={'xatol': 1e-8})
        return None if r.fun < 1e-7 else int(min(np.ceil(3 / r.fun), 10 ** 7))
    sets = {'odds 0.75 only': [(0.75, 1.0)], 'odds 0.9 and 0.75': [(0.9, 1.0), (0.75, 1.0)], 'odds 0.75 and 0.5': [(0.75, 1.0), (0.5, 1.0)],
            'odds 0.75 and 0.25': [(0.75, 1.0), (0.25, 1.0)], 'all four odds': [(p, 1.0) for p in ODDS], 'odds 0.75, cost 1 and 2': [(0.75, 1.0), (0.75, 2.0)]}
    out['trials_to_3_log_odds'] = {}
    for sname, conds in sets.items():
        out['trials_to_3_log_odds'][sname] = {f'{NAME[t]} told from {NAME[f_]}': trials(P[t], f_, conds) for t in P for f_ in P if t != f_ and P[t] and P[f_]}
    res['laws'][lname] = out
json.dump(res, open(sys.argv[1], 'w'), indent=1)
for lname, o in res['laws'].items():
    print('==', lname, o['task']); print('  baseline wait', o['baseline_wait_on_omission']); print('  matched', o['matched_parameters'])
    for k, v in o['lift_by_odds'].items(): print('  lift', k, v, '| cost doubled:', o['lift_when_cost_doubles_at_reference'][k])
    for s, row in o['trials_to_3_log_odds'].items(): print('  ', s, {k: v for k, v in row.items()})
