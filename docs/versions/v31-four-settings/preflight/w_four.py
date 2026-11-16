"""Pre-flight for Ghost V31 family W (theory card G9): four settings at four points of the loop.
A maker in a world with a hidden context (one of K), a cue each step, and drops on a need.
  a      how far the incoming cue outweighs the held picture (0.5 is exact Bayes)
  ts     the temperature of the swap between held schemas
  d      how hard the maker works for a given level of confidence in the whole chain
  s      how hard unresolved Stress presses on the plan in hand (abandoning it)
The reader here sees everything the maker sees (cues, drops) and everything it does (held schema through its
action family, work or idle, response time, plan kept or dropped), so the likelihood is exact.
For each pair of settings: one maker has only the first moved, the rival has only the second moved.
Reported: log-odds per 1,000 steps in favour of the truth, by kind of observation, and steps to 3 log-odds.
Usage: python3 -I w_four.py out.json"""
import sys, json, zlib, itertools, numpy as np
K = 4
BASE = dict(a=0.5, ts=0.35, d=2.0, s=1.0)
MOVE = dict(a=0.75, ts=0.7, d=1.0, s=2.0)
STICK, RT_SD, PLAN_LEN, LEAK, DROP = 1.0, 0.3, 4, 0.05, 0.10
def sig(x): return 1 / (1 + np.exp(-x))
class Inner:
    """A maker's deterministic inner state under one parameter set, driven by the shared observed history."""
    def __init__(self, par, v_assumed):
        self.p = par; self.pi = np.full(K, 1 / K); self.cbar = 0.5; self.v = v_assumed
    def read(self, cue, r_assumed):
        prior = self.pi * (1 - self.v) + (1 - self.pi) * self.v / (K - 1)
        like = np.where(np.arange(K) == cue, r_assumed, (1 - r_assumed) / (K - 1))
        lp = 2 * (1 - self.p['a']) * np.log(prior) + 2 * self.p['a'] * np.log(like)
        self.pi = np.exp(lp - lp.max()); self.pi /= self.pi.sum()
    def swap_probs(self, held):
        z = (np.log(self.pi + 1e-12) + STICK * (np.arange(K) == held)) / self.p['ts']
        z = np.exp(z - z.max()); return z / z.sum()
    def after_swap(self, held):
        c = self.pi[held]; self.cbar = 0.8 * self.cbar + 0.2 * c
    def p_work(self): return sig(self.p['d'] * (self.cbar - 0.5) + 1.0)
    def rt_mu(self): return 0.0 - 0.5 * self.p['d'] * (self.cbar - 0.5)
    def p_drop_plan(self, U): return sig(self.p['s'] * U - 3.0)
def run(seed, truth, rival, vol, rel, actionable, T=20000):
    r = np.random.default_rng(seed)
    A, B = Inner(truth, vol), Inner(rival, vol)
    x = int(r.integers(K)); held = int(r.integers(K)); U = 0.0; plan = 0
    llr = dict(swap=0.0, work=0.0, time=0.0, plan=0.0); stats = dict(switch=0, idle=0, plans_done=0, plans_dropped=0, rt=0.0, n_rt=0)
    for t in range(T):
        if r.random() < vol: x = int((x + 1 + r.integers(K - 1)) % K)
        cue = x if r.random() < rel else int((x + 1 + r.integers(K - 1)) % K)
        if r.random() < DROP:
            U += 1.0
            if plan == 0 and r.random() < actionable: plan = PLAN_LEN
        A.read(cue, rel); B.read(cue, rel)
        pa, pb = A.swap_probs(held), B.swap_probs(held)
        new = int(r.choice(K, p=pa)); llr['swap'] += np.log(pa[new] / pb[new]); stats['switch'] += new != held; held = new
        A.after_swap(held); B.after_swap(held)
        wa, wb = A.p_work(), B.p_work(); work = r.random() < wa
        llr['work'] += np.log((wa if work else 1 - wa) / (wb if work else 1 - wb)); stats['idle'] += (not work)
        if work:
            ma, mb = A.rt_mu(), B.rt_mu(); lt = ma + RT_SD * r.standard_normal()
            llr['time'] += (-(lt - ma) ** 2 + (lt - mb) ** 2) / (2 * RT_SD ** 2); stats['rt'] += np.exp(lt); stats['n_rt'] += 1
            if plan > 0:
                da, db = A.p_drop_plan(U), B.p_drop_plan(U); drop = r.random() < da
                llr['plan'] += np.log((da if drop else 1 - da) / (db if drop else 1 - db))
                if drop: plan = 0; stats['plans_dropped'] += 1
                else:
                    plan -= 1
                    if plan == 0: U = max(0.0, U - 1.0); stats['plans_done'] += 1
        U *= (1 - LEAK)
    per1000 = {k: 1000 * v / T for k, v in llr.items()}; tot = sum(per1000.values())
    head = dict(switches_per_100=100 * stats['switch'] / T, idle_share=stats['idle'] / T, plans_done_per_1000=1000 * stats['plans_done'] / T,
                plans_dropped_per_1000=1000 * stats['plans_dropped'] / T, mean_response_time=stats['rt'] / max(1, stats['n_rt']))
    return per1000, tot, head
WORLDS = {'calm, clear cues, drops fixable': (0.02, 0.9, 0.8), 'calm, clear cues, drops not fixable': (0.02, 0.9, 0.2),
          'volatile, clear cues, drops fixable': (0.10, 0.9, 0.8), 'volatile, murky cues, drops fixable': (0.10, 0.6, 0.8),
          'calm, murky cues, drops not fixable': (0.02, 0.6, 0.2)}
res = {'baseline': BASE, 'moved': MOVE, 'worlds': {}, 'headline': {}}
for wname, (vol, rel, act) in WORLDS.items():
    res['worlds'][wname] = {}
    for X, Y in itertools.permutations(BASE, 2):
        truth = dict(BASE, **{X: MOVE[X]}); rival = dict(BASE, **{Y: MOVE[Y]})
        per, tot, head = run(zlib.crc32(f'{wname}|{X}|{Y}'.encode()), truth, rival, vol, rel, act)
        top = max(per, key=per.get)
        res['worlds'][wname][f'{X} moved, rival {Y} moved'] = dict(per_1000={k: round(v, 1) for k, v in per.items()}, steps_to_3=int(np.ceil(3000 / tot)) if tot > 0 else None, most_from=top)
    for X in list(BASE) + ['none']:
        m = dict(BASE, **({X: MOVE[X]} if X != 'none' else {}))
        _, _, head = run(zlib.crc32(f'{wname}|head|{X}'.encode()), m, m, vol, rel, act)
        res['headline'].setdefault(wname, {})[X] = {k: round(v, 2) for k, v in head.items()}
json.dump(res, open(sys.argv[1], 'w'), indent=1)
for w, rows in res['worlds'].items():
    print('==', w)
    for k, v in rows.items(): print('   %-28s steps %-6s most from %-5s %s' % (k, v['steps_to_3'], v['most_from'], v['per_1000']))
    print('   headline:', {k: (v['switches_per_100'], v['idle_share'], v['plans_done_per_1000'], v['plans_dropped_per_1000'], v['mean_response_time']) for k, v in res['headline'][w].items()})
