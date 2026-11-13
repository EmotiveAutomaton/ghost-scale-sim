"""V29 pre-flight Q3 (queue or sum) and Q2 (a bill with and without control). numpy only, fixed seeds."""
import numpy as np
sig = lambda x: 1 / (1 + np.exp(-x))
def lse(a, ax): m = a.max(ax, keepdims=True); return np.squeeze(m, ax) + np.log(np.exp(a - m).sum(ax))
# ---------------- Q3: three needs, one act per step ----------------
A = np.array([1.0, 1.5, 3.0]); QARR = np.array([0.20, 0.15, 0.10])
def q3_logp(kind, lam, beta, d, cost):
    """log P(action) over [rest, act0, act1, act2]; d (nw,3)."""
    if kind == 'SUM':  u = A * (d - 0.5) - lam * cost          # marginal fall in pooled quadratic error, net of cost
    else:              u = A * d                                 # QUEUE: the loudest need, cost ignored
    u = np.where(d > 0, u, -np.inf); rest = np.zeros((len(d), 1)) if kind == 'SUM' else np.full((len(d), 1), 0.5)
    U = beta * np.concatenate([rest, u], 1); return U - lse(U, 1)[:, None]
Q3H = [('SUM', l, b) for l in (0.5, 1, 2) for b in (1, 3)] + [('QUEUE', 0, b) for b in (1, 3)]
def q3(truth, cost, T=512, nw=200, seed=1):
    rng = np.random.default_rng(seed); d = np.zeros((nw, 3)); LL = np.zeros((nw, len(Q3H))); sep = np.full(nw, np.inf); conflict = 0
    isq = np.array([h[0] == 'QUEUE' for h in Q3H]); tq = truth[0] == 'QUEUE'
    for t in range(T):
        d = np.minimum(5, d + (rng.random((nw, 3)) < QARR))
        lp = q3_logp(*truth, d, cost); a = (rng.random(nw)[:, None] > np.cumsum(np.exp(lp), 1)).sum(1).clip(0, 3)
        any_d = (d > 0).sum(1) >= 2
        s_best = np.argmax(np.where(d > 0, A * (d - 0.5) - 1.0 * cost, -np.inf), 1); q_best = np.argmax(np.where(d > 0, A * d, -np.inf), 1)
        conflict += np.mean(any_d & (s_best != q_best))
        for j, h in enumerate(Q3H): LL[:, j] += q3_logp(*h, d, cost)[np.arange(nw), a]
        lo = (lse(LL[:, isq], 1) - np.log(isq.sum())) - (lse(LL[:, ~isq], 1) - np.log((~isq).sum())); lo = lo if tq else -lo
        sep = np.where((lo >= 3) & ~np.isfinite(sep), t + 1, np.where(lo < 3, np.inf, sep))
        d[np.arange(nw), np.maximum(a - 1, 0)] -= (a > 0)
    return np.median(sep) if np.mean(np.isfinite(sep)) > .5 else np.inf, np.mean(~np.isfinite(sep)), conflict / T
print('Q3 queue or sum: truth, costs -> median steps to log-odds 3 | unresolved | share of steps in conflict')
for cost in (np.array([1., 1., 1.]), np.array([1., 2., 3.]), np.array([3., 2., 1.])):
    for tr in (('SUM', 1, 3), ('QUEUE', 0, 3), ('SUM', 1, 1), ('QUEUE', 0, 1)):
        print('  %-16s cost %s -> %6s | %.2f | %.3f' % (tr, cost.astype(int), *q3(tr, cost)))
# ---------------- Q2: one deficit, a push that works with probability pi ----------------
C, PIS = 1.0, {'U': 0.05, 'C': 0.7}
Q2H = [(pr, k) for pr in ('PASSIVE', 'ACTIVE') for k in (0.25, 0.5, 1.0, 2.0)]     # prior on control x deficit sensitivity
PRI = {'PASSIVE': (1., 4.), 'ACTIVE': (4., 1.)}
def q2(truth, order, L=150, nw=300, seed=2, beta=2.0):
    rng = np.random.default_rng(seed); D = np.full(nw, 2.0); s = np.zeros(nw); f = np.zeros(nw)   # own successes, failures
    LL = np.zeros((nw, len(Q2H))); out = []; push_rate = []
    for ep in order:
        pr_ep = 0
        for t in range(L):
            D = np.minimum(6, D + (rng.random(nw) < 0.3))
            a0, b0 = PRI[truth[0]]; ph = (a0 + s) / (a0 + b0 + s + f)
            pp = np.clip(sig(beta * (ph * truth[1] * D - C)), 0.01, 0.99); push = rng.random(nw) < pp
            for j, (pr, k) in enumerate(Q2H):
                a1, b1 = PRI[pr]; phj = (a1 + s) / (a1 + b1 + s + f); pj = np.clip(sig(beta * (phj * k * D - C)), 0.01, 0.99)
                LL[:, j] += np.log(np.where(push, pj, 1 - pj))
            ok = push & (rng.random(nw) < PIS[ep]); s += ok; f += push & ~ok; D = np.maximum(0, D - 2 * ok); pr_ep += push.mean()
        push_rate.append(pr_ep / L)
        isP = np.array([h[0] == 'PASSIVE' for h in Q2H]); lo = lse(LL[:, isP], 1) - lse(LL[:, ~isP], 1); lo = lo if truth[0] == 'PASSIVE' else -lo
        kpost = np.array([lse(LL[:, [j for j, h in enumerate(Q2H) if h[1] == k]], 1) for k in (0.25, 0.5, 1.0, 2.0)]).T
        out.append((np.mean(lo >= 3), np.mean(lo <= -3), np.mean(np.argmax(kpost, 1) == (0.25, 0.5, 1.0, 2.0).index(truth[1]))))
    return out, push_rate
print('\nQ2 bill and control: truth (prior on control, deficit sensitivity), episode order -> per episode: push rate; prior read right / read wrong (log-odds 3); sensitivity read right')
for tr in (('PASSIVE', 1.0), ('ACTIVE', 1.0), ('ACTIVE', 0.25), ('PASSIVE', 2.0)):
    for order in ('U', 'C', 'UC', 'CU', 'CUC'):
        o, pr = q2(tr, order)
        print('  %-18s %-4s ' % (tr, order) + ' ; '.join('%s: push %.2f, prior %.2f/%.2f, sens %.2f' % (e, p, *x) for e, p, x in zip(order, pr, o)))
