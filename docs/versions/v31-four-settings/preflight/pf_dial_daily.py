"""Pre-flight for the daily-diary form of 'one dial or two' (MIDUS card MI2): eight or sixteen diary days,
one report a day, stressor days at a rate of about 0.4. Same estimator as pf_dial.py.
Usage: python3 -I pf_dial_daily.py out.json"""
import sys, json, zlib, numpy as np
def world(seed, n, days, rho, rate=0.4):
    r = np.random.default_rng(seed)
    z = r.standard_normal((n, 2)); L = np.linalg.cholesky(np.array([[1, rho], [rho, 1.0]]) + 1e-12 * np.eye(2)); th = z @ L.T
    slope = np.exp(0.5 * th); base = 0.6 + 0.5 * r.standard_normal(n)
    p = np.clip(r.beta(4, 6, n) * rate / 0.4, 0.05, 0.95); ev = r.random((n, days)) < p[:, None]
    shared = 0.5 * r.standard_normal((n, days)); out = []
    for c in range(2):
        items = [np.clip(np.rint(base[:, None] + slope[:, c][:, None] * ev + shared + 0.8 * r.standard_normal((n, days)) ), 0, 4) for _ in range(3)]
        out.append(np.mean(items, 0))
    return out[0], out[1], ev
def est(A, F, ev):
    days = A.shape[1]; odd = (np.arange(days) % 2 == 1)[None, :] & np.ones_like(ev); even = ~odd
    def sl(Y, m):
        e1 = (ev & m).sum(1); e0 = (~ev & m).sum(1); ok = (e1 >= 1) & (e0 >= 1)
        return np.where(ev & m, Y, 0).sum(1) / np.maximum(e1, 1) - np.where(~ev & m, Y, 0).sum(1) / np.maximum(e0, 1), ok
    ao, k1 = sl(A, odd); ae, k2 = sl(A, even); fo, k3 = sl(F, odd); fe, k4 = sl(F, even); ok = k1 & k2 & k3 & k4
    c = lambda x, y: float(np.corrcoef(x[ok], y[ok])[0, 1])
    ra, rf = c(ao, ae), c(fo, fe); cross = 0.5 * (c(ao, fe) + c(ae, fo))
    return cross / np.sqrt(max(ra * rf, 1e-9)) if ra > 0 and rf > 0 else np.nan, ra, int(ok.sum())
res = {}
for n, days in ((2000, 8), (2000, 16), (1200, 16)):
    for rho in (1.0, 0.8, 0.5):
        E = [est(*world(zlib.crc32(f'{n}|{days}|{rho}|{k}'.encode()), n, days, rho)) for k in range(40)]
        lat = np.array([e[0] for e in E]); lat = lat[np.isfinite(lat)]
        res[f'people {n}, days {days}, true {rho}'] = dict(latent_mean=round(float(lat.mean()), 3), sd_over_worlds=round(float(lat.std()), 3),
            half_reliability=round(float(np.mean([e[1] for e in E])), 3), people_used=int(np.mean([e[2] for e in E])), share_of_worlds_above_0_9=round(float((lat > 0.9).mean()), 2))
json.dump(res, open(sys.argv[1], 'w'), indent=1)
for k, v in res.items(): print(k, v)
