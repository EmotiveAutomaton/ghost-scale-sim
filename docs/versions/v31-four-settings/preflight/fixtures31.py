"""Computed fixture values for the Sounding Line Week 6 extension cards (15 November 2026).
Every planted value quoted in the packet is produced here. Usage: python3 -I fixtures31.py out.json"""
import sys, json, zlib, numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
R = lambda name: np.random.default_rng(zlib.crc32(name.encode()))
out = {}
# ---------- KP1 (E13): pause remainder and a deletion soon after
r = R('KP1'); n, W = 400000, 40
writer = r.integers(W, size=n); boundary = r.random(n) < 0.10
resid = r.standard_normal(n)                       # remainder of log pause after the joint fit, within-writer SD 1
raw = resid + 0.8 * boundary + 0.3 * r.standard_normal(W)[writer]     # raw log pause: boundaries are slower, writers differ
def plant(beta, bmult):
    lo = np.log(0.08 / 0.92) + beta * resid + np.log(bmult) * boundary
    return r.random(n) < 1 / (1 + np.exp(-lo))
y = plant(0.30, 2.0)
X = np.column_stack([resid, boundary]); m = LogisticRegression(C=1e6, max_iter=500).fit(X, y)
m_raw = LogisticRegression(C=1e6, max_iter=500).fit(raw.reshape(-1, 1), y)
y0 = plant(0.0, 2.0); m0 = LogisticRegression(C=1e6, max_iter=500).fit(X, y0); m0_raw = LogisticRegression(C=1e6, max_iter=500).fit(raw.reshape(-1, 1), y0)
out['KP1'] = dict(planted_odds_ratio_per_sd=round(float(np.exp(0.30)), 3), recovered_odds_ratio=round(float(np.exp(m.coef_[0][0])), 3),
    area_from_remainder_alone=round(float(roc_auc_score(y, resid)), 3), base_rate=round(float(y.mean()), 4),
    null_world_recovered_odds_ratio=round(float(np.exp(m0.coef_[0][0])), 3),
    null_world_trap_raw_pause_no_boundary_term=round(float(np.exp(m0_raw.coef_[0][0])), 3))
# ---------- KP2 (E12a): own against donor history as held-out prediction
r = R('KP2'); n = 200000
own = r.standard_normal(n); donor = 0.6 * own + 0.8 * r.standard_normal(n); base = r.standard_normal(n)
def gain(b_own, b_donor):
    y = 0.3 * base + b_own * own + b_donor * donor + 0.7 * r.standard_normal(n)
    tr, te = slice(0, n // 2), slice(n // 2, n)
    mse = lambda cols: float(np.mean((y[te] - LinearRegression().fit(np.column_stack(cols)[tr], y[tr]).predict(np.column_stack(cols)[te])) ** 2))
    b, o, d = mse([base]), mse([base, own]), mse([base, donor])
    return dict(base_mse=round(b, 5), own_gain_pct=round(100 * (b - o) / b, 3), donor_gain_pct=round(100 * (b - d) / b, 3), own_minus_donor_pct=round(100 * (d - o) / b, 3))
out['KP2'] = {'own carries it (0.05, 0)': gain(0.05, 0.0), 'shared part only (0.03, 0.03)': gain(0.03, 0.03), 'donor carries it (0, 0.05)': gain(0.0, 0.05)}
# ---------- CH13 (S20): the lifetime of a loss
r = R('CH13'); P, G = 3000, 120
sens = np.abs(0.03 * (1 + 0.5 * r.standard_normal(P)))        # each player's after-loss shift at zero wait
def chess(chosen):
    rows = []
    for p in range(P):
        prev_loss = r.random(G) < 0.5
        gap = r.lognormal(np.log(4), 1.2, G)                      # minutes, set by the pairing system
        if chosen: gap = np.where(prev_loss, gap * np.exp(-8 * (sens[p] - 0.03) / 0.03 * 0.25), gap)   # touchier players come back sooner after a loss
        shift = -sens[p] * np.exp(-gap / 15) * prev_loss
        resid = shift + 0.45 * r.standard_normal(G)
        rows.append(np.column_stack([np.full(G, p), prev_loss, gap, resid]))
    return np.vstack(rows)
def curve(D):
    bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 60), (60, 1e9)]; res = {}
    for lo, hi in bins:
        m = (D[:, 2] >= lo) & (D[:, 2] < hi)
        res[f'{lo}-{int(hi) if hi < 1e8 else "up"} min'] = round(float(D[m & (D[:, 1] == 1), 3].mean() - D[m & (D[:, 1] == 0), 3].mean()), 4)
    return res
D = chess(False)
exp_bins = {}
for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 60), (60, 1e9)]:
    g = D[(D[:, 2] >= lo) & (D[:, 2] < hi) & (D[:, 1] == 1), 2]; exp_bins[f'{lo}-{int(hi) if hi < 1e8 else "up"} min'] = round(float(-0.03 * np.exp(-g / 15).mean()), 4)
out['CH13'] = dict(planted='shift = -0.03 * exp(-minutes / 15), score points', expected_by_bin=exp_bins, recovered_system_set_waits=curve(D),
                   recovered_when_touchier_players_return_sooner=curve(chess(True)), games=int(len(D)))
# ---------- CH14 (S21): the readiness curve, conditional logit over lines already played
r = R('CH14')
def readiness(b_count, b_rec, players=400, games=300, lines=6):
    Xs, ys = [], []
    for p in range(players):
        count = np.ones(lines); last = np.zeros(lines); t = 0.0; pref = 0.5 * r.standard_normal(lines)
        for g in range(games):
            t += r.exponential(6.0)                                  # hours between games
            u = pref + b_count * np.log(count) + b_rec * np.log1p(t - last)
            pr = np.exp(u - u.max()); pr /= pr.sum(); c = r.choice(lines, p=pr)
            if g >= 30: Xs.append(np.column_stack([np.log(count), np.log1p(t - last)])); ys.append(c)
            count[c] += 1; last[c] = t
    X = np.array(Xs); y = np.array(ys)                              # (choices, lines, 2)
    beta = np.zeros(2)
    for _ in range(60):                                             # Newton steps on the conditional logit (no line constants)
        u = X @ beta; u -= u.max(1, keepdims=True); pr = np.exp(u); pr /= pr.sum(1, keepdims=True)
        gsum = (X[np.arange(len(y)), y] - (pr[:, :, None] * X).sum(1)).sum(0)
        mean = (pr[:, :, None] * X).sum(1); Hs = np.einsum('nl,nli,nlj->ij', pr, X, X) - mean.T @ mean
        beta = beta + np.linalg.solve(Hs, gsum)
    return [round(float(b), 3) for b in beta]
out['CH14'] = {'planted count 1.0, recency -0.30 (no standing preference term in the fit)': readiness(1.0, -0.30),
               'planted count 1.0, recency 0': readiness(1.0, 0.0)}
# ---------- ES1: which status event fires which feeling (within person, six events entered together)
r = R('ES1'); Pn, T = 1500, 40
S = r.standard_normal((Pn, T, 6)); S[..., 1] += 0.3 * S[..., 2]     # 0 admired 1 criticised 2 ignored 3 respected 4 upstaged 5 sidelined; criticised and ignored correlate
Bm = np.zeros((4, 6))                                               # rows: ashamed, lonely, angry, proud
Bm[0, 1], Bm[0, 2] = 0.25, 0.10; Bm[1, 2], Bm[1, 1] = 0.25, 0.05; Bm[2, 1], Bm[2, 2], Bm[2, 5] = 0.30, 0.10, 0.10; Bm[3, 0] = 0.30
Fm = S @ Bm.T + r.standard_normal((Pn, 1, 4)) * 0.5 + r.standard_normal((Pn, T, 4))
Sw = (S - S.mean(1, keepdims=True)).reshape(-1, 6); Fw = (Fm - Fm.mean(1, keepdims=True)).reshape(-1, 4)
Bh = np.linalg.lstsq(Sw, Fw, rcond=None)[0].T
out['ES1'] = dict(planted=dict(a_ashamed_criticised_minus_ignored=0.15, b_lonely_ignored_minus_criticised=0.20, c_angry_criticised_minus_mean_ignored_sidelined=0.20),
    recovered=dict(a=round(float(Bh[0, 1] - Bh[0, 2]), 3), b=round(float(Bh[1, 2] - Bh[1, 1]), 3), c=round(float(Bh[2, 1] - 0.5 * (Bh[2, 2] + Bh[2, 5])), 3)),
    trap_one_event_at_a_time=dict(a=round(float(np.polyfit(Sw[:, 1], Fw[:, 0], 1)[0] - np.polyfit(Sw[:, 2], Fw[:, 0], 1)[0]), 3)))
# ---------- ES3 (S18): carry-over into the next non-social activity, by minutes
r = R('ES3'); n = 400000
gap = r.lognormal(np.log(120), 0.8, n); day = 0.35 * r.standard_normal(n)       # a shared day effect
a0 = day + r.standard_normal(n); a1 = day + 0.4 * np.exp(-gap / 90) * a0 + r.standard_normal(n)
bins = [(0, 90), (90, 240), (240, 1e9)]; rec = {}; exp_ = {}
for lo, hi in bins:
    m = (gap >= lo) & (gap < hi); rec[f'{lo}-{int(hi) if hi < 1e8 else "up"}'] = round(float(np.polyfit(a0[m], a1[m], 1)[0]), 3)
    exp_[f'{lo}-{int(hi) if hi < 1e8 else "up"}'] = round(float(0.4 * np.exp(-gap[m] / 90).mean()), 3)
out['ES3'] = dict(planted='0.4 * exp(-minutes / 90) plus a shared day effect', planted_mean_by_bin=exp_, recovered_slope_by_bin=rec,
    primary_short_minus_long=round(rec['0-90'] - rec['240-up'], 3), day_effect_alone_slope=round(float(0.35 ** 2 / (0.35 ** 2 + 1)), 3))
# ---------- EN1 (S11a): colored remainder between a sender's messages
r = R('EN1'); senders, per = 600, 400; pairs = {k: [] for k in ('0-15', '15-60', '60-240', '240-up')}; pairs_tod = {k: [] for k in pairs}
def binname(g): return '0-15' if g < 15 else '15-60' if g < 60 else '60-240' if g < 240 else '240-up'
for s in range(senders):
    gaps = r.lognormal(np.log(25), 1.5, per); t = np.cumsum(gaps); state = np.zeros(per); state[0] = r.standard_normal()
    for i in range(1, per):
        rho = np.exp(-gaps[i] / 60); state[i] = rho * state[i - 1] + np.sqrt(1 - rho ** 2) * r.standard_normal()
    hour = (t / 60) % 24; tod = 0.5 * np.cos(2 * np.pi * hour / 24)
    rem = np.sqrt(0.2) * state + np.sqrt(0.8) * r.standard_normal(per)             # colored share 0.2
    rem_tod = tod + r.standard_normal(per)                                         # no carry-over, only a time-of-day rhythm
    for i in range(1, per):
        b = binname(gaps[i]); pairs[b].append((rem[i - 1], rem[i])); pairs_tod[b].append((rem_tod[i - 1], rem_tod[i]))
c = lambda L: round(float(np.corrcoef(np.array(L).T)[0, 1]), 3)
out['EN1'] = dict(planted='colored share 0.2, time constant 60 minutes', recovered_correlation_by_gap={k: c(v) for k, v in pairs.items()},
                  rhythm_only_world_before_removing_time_of_day={k: c(v) for k, v in pairs_tod.items()}, pairs={k: len(v) for k, v in pairs.items()})
# ---------- MI1 (S16): which area at risk, which feeling (difference of within-person slopes)
r = R('MI1'); Pn, Dn = 2000, 8
risk = (r.random((Pn, Dn, 3)) < 0.15).astype(float)      # 0 how others feel about you, 1 daily routine, 2 plans or health
ash = 0.30 * risk[..., 0] + 0.05 * risk[..., 1] + r.standard_normal((Pn, Dn)); ang = 0.10 * risk[..., 0] + 0.30 * risk[..., 1] + 0.05 * risk[..., 2] + r.standard_normal((Pn, Dn))
afr = 0.05 * risk[..., 1] + 0.30 * risk[..., 2] + r.standard_normal((Pn, Dn))
w = lambda A: (A - A.mean(1, keepdims=True)).reshape(-1, *A.shape[2:])
Bh = np.linalg.lstsq(w(risk), np.column_stack([w(ash), w(ang), w(afr)]), rcond=None)[0]      # rows risk, cols feeling
out['MI1'] = dict(planted=dict(c1_others_ashamed_minus_angry=0.20, c2_routine_angry_minus_afraid=0.25, c3_plans_afraid_minus_angry=0.25),
                  recovered=dict(c1=round(float(Bh[0, 0] - Bh[0, 1]), 3), c2=round(float(Bh[1, 1] - Bh[1, 2]), 3), c3=round(float(Bh[2, 2] - Bh[2, 1]), 3)))
# ---------- IP1 (S22): anger's two homes
r = R('IP1'); n = 300000
N = r.standard_normal(n); A = -0.25 * N + np.sqrt(1 - 0.25 ** 2) * r.standard_normal(n)
fac = lambda ln, la, u=0.6: ln * N + la * A + u * r.standard_normal(n)
anger, anx = fac(0.5, -0.4), fac(0.7, 0.0); otherN = np.mean([fac(0.7, 0.0) for _ in range(4)], 0); Acomp = np.mean([fac(0.0, 0.7) for _ in range(6)], 0)
z = lambda v: (v - v.mean()) / v.std()
def betas(y): return [round(float(b), 3) for b in LinearRegression().fit(np.column_stack([z(otherN), z(Acomp)]), z(y)).coef_]
out['IP1'] = dict(planted='Anger loads 0.5 on Neuroticism and -0.4 on Agreeableness; Anxiety 0.7 and 0', anger_betas_on_N_rest_and_A=betas(anger), anxiety_betas_on_N_rest_and_A=betas(anx))
json.dump(out, open(sys.argv[1], 'w'), indent=1); print(json.dumps(out, indent=1))
