"""V29 pre-flight X: a refusal as a live sum, a frozen rule, or a hub. numpy only, fixed seeds (crc32 of the cell).
World as in the packet: the kept good starts at 8, an empty stock draws no sell offer, and a freeze is absorbing."""
import numpy as np, sys, zlib
EPS, BETA, BASE, MU = 0.02, 1.5, 0.5, 1.0
REG = {'LOW': ([1, 2], [.5, .5]), 'MIXED': ([1, 2, 4, 8], [.4, .3, .2, .1])}
ENV = {'plenty': (0.8, 0.15), 'scarce': (0.3, 0.3)}          # (money inflow, need arrival per need)
# hypotheses: (class, param)
HY = [('LIVE', w) for w in (1, 2, 4, 8, 16)] + [('RULE', 0)] + [('HUB', h) for h in (2, 4)] + \
     [('FROZE', (n, w)) for n in (5, 20) for w in (2, 4)]
CLS = ['LIVE', 'RULE', 'HUB', 'FROZE']
sig = lambda x: 1 / (1 + np.exp(-x))
def money_gain(M, d, p):      # value of p extra units of money given stock M and total deficit d
    return BASE * p + MU * np.minimum(p, np.maximum(0, d - M))
def money_loss(M, d, p):      # value of the p units given up
    return BASE * p + MU * np.maximum(0, np.minimum(p, d - (M - p)))
def probs(h, M, d, p, kind, nref):
    """P(accept offer), P(meet a need | deficit, M>0) for hypothesis h. kind: 0 sell-X offer, 1 buy-X offer."""
    c, par = h
    if c == 'LIVE': w, keep = par, BASE
    elif c == 'RULE': w, keep = (par or 2), BASE
    elif c == 'HUB': w, keep = 2, par
    else: w, keep = par[1], BASE
    frozen = (c in ('RULE', 'HUB')) | ((c == 'FROZE') & (nref >= (par[0] if c == 'FROZE' else 0)))   # nref here is the longest refusal run so far: the freeze is absorbing
    sell = np.where(frozen, EPS, np.clip(sig(BETA * (money_gain(M, d, p) - w)), EPS, 1 - EPS))
    buy = np.where(M >= p, np.clip(sig(BETA * (w - money_loss(M, d, p))), EPS, 1 - EPS), 0.0) if c != 'HUB' \
        else np.where(M >= p, 1 - EPS, 0.0)
    acc = np.where(kind == 0, sell, buy)
    meet = np.clip(sig(BETA * (MU * np.minimum(d, 3) - keep)), EPS, 1 - EPS)
    return acc, meet
def run(truth, reg, env, T=1024, nw=200, seed=0, qoff=0.25, qbuy=0.5):
    rng = np.random.default_rng(seed); qin, qneed = ENV[env]; pv, pp = REG[reg]
    M = np.full(nw, 5); d = np.zeros(nw, int); X = np.full(nw, 8); nref = np.zeros(nw, int); ever = np.zeros(nw, int)
    LL = np.zeros((nw, len(HY))); out = {}
    cidx = np.array([CLS.index(h[0]) for h in HY]); present = [c for c in range(4) if (cidx == c).any()]; tc = present.index(CLS.index(truth[0]))
    sep = np.full(nw, np.inf); desperate_refusal = np.zeros(nw, int); nsell = np.zeros(nw, int)
    for t in range(T):
        M = np.minimum(30, M + (rng.random(nw) < qin)); d = np.minimum(6, d + rng.binomial(2, qneed, nw))
        off = rng.random(nw) < qoff; kind = (rng.random(nw) < qbuy).astype(int); p = rng.choice(pv, nw, p=pp)
        off = off & ~((kind == 0) & (X == 0))            # an empty stock draws no sell offer (never converted to a buy offer)
        acc_t, meet_t = probs(truth, M, d, p, kind, ever)
        a = off & (rng.random(nw) < acc_t)
        for j, h in enumerate(HY):
            ah, mh = probs(h, M, d, p, kind, ever)
            LL[:, j] += np.where(off, np.log(np.where(a, ah, 1 - ah) + 1e-300), 0.0)
        sellref = off & (kind == 0) & ~a
        desperate_refusal += sellref & (d > M); nsell += off & (kind == 0)
        nref = np.where(off & (kind == 0), np.where(a, 0, nref + 1), nref); ever = np.maximum(ever, nref)
        X = X + np.where(a, np.where(kind == 0, -1, 1), 0); X = np.clip(X, 0, 10)
        M = M + np.where(a, np.where(kind == 0, p, -p), 0); M = np.clip(M, 0, 30)
        can = (d > 0) & (M > 0); m = can & (rng.random(nw) < meet_t)
        for j, h in enumerate(HY):
            ah, mh = probs(h, M, d, p, kind, ever)
            LL[:, j] += np.where(can, np.log(np.where(m, mh, 1 - mh)), 0.0)
        M = M - m; d = d - m
        cl = np.stack([np.logaddexp.reduce(np.where(cidx == c, LL, -np.inf), 1) - np.log((cidx == c).sum()) for c in present], 1)
        best_wrong = np.max(np.delete(cl, tc, 1), 1); lo = cl[:, tc] - best_wrong
        sep = np.where((lo >= 3) & ~np.isfinite(sep), t + 1, np.where(lo < 3, np.inf, sep))
        if t + 1 in (64, 256, 1024): out[t + 1] = (np.mean(lo >= 3), np.bincount(np.array(present)[np.argmax(cl, 1)], minlength=4) / nw)
    return out, sep, desperate_refusal.mean(), nsell.mean()
if __name__ == '__main__':
    truths = [('LIVE', 4), ('LIVE', 16), ('RULE', 0), ('HUB', 4), ('FROZE', (20, 4))]
    print('truth      regime env     | sep>=3 at 64/256/1024 | median steps | unresolved | read-as at 1024 [LIVE RULE HUB FROZE] | desperate refusals')
    for tr in truths:
        for reg in ('LOW', 'MIXED'):
            for env in ('plenty', 'scarce'):
                o, sep, dr, ns = run(tr, reg, env, seed=zlib.crc32(repr((tr, reg, env)).encode()))
                med = np.median(sep) if np.mean(np.isfinite(sep)) > .5 else float('inf')
                print('%-14s %-5s %-6s | %.2f %.2f %.2f | %6s | %.2f | %s | %.1f of %.0f sell offers' % (
                    str(tr), reg, env, o[64][0], o[256][0], o[1024][0], med, np.mean(~np.isfinite(sep)),
                    np.round(o[1024][1], 2), dr, ns))
