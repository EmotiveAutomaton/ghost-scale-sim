"""Pre-flight for 'one dial or two' (Sounding Line ES2 / MIDUS M2; Ghost V31 family D).
Constructed diaries. Asks: if anger-type and fear-type reactivity share one dial (true correlation 1)
or not (0.8, 0.5, 0), what does a split-half-corrected correlation of per-person slopes return,
with a rating scale that has a floor?  Usage: python3 -I pf_dial.py out.json"""
import sys, json, zlib, numpy as np
def world(seed, n=2000, prompts=60, rho=1.0, floor=True, event_rate=0.2, shared_prompt_noise=0.5, base_sd=0.5, base_link=0.0, rival=None):
    r = np.random.default_rng(seed)
    z = r.standard_normal((n, 2)); L = np.linalg.cholesky(np.array([[1, rho], [rho, 1.0]]) + 1e-12 * np.eye(2)); th = z @ L.T
    slope = np.exp(0.0 + 0.5 * th)                       # true reactivity, two constructs, corr of logs = rho
    if rival is not None:                                 # the rival: one dial, and a second input into the first family only
        slope = np.column_stack([np.exp(0.5 * z[:, 0] + rival * z[:, 1]), np.exp(0.5 * z[:, 0])])
    base = 0.6 + base_sd * r.standard_normal(n) + base_link * th.mean(1)   # person's usual level (latent units)
    p = np.clip(r.beta(2, 8, n) * event_rate / 0.2, 0.02, 0.9)
    ev = r.random((n, prompts)) < p[:, None]
    day = np.arange(prompts) // 6                          # six prompts a day
    shared = shared_prompt_noise * r.standard_normal((n, prompts))
    out = []
    for c in range(2):
        for item in range(3):                              # three items per construct
            lat = base[:, None] + slope[:, c][:, None] * ev + shared + 0.8 * r.standard_normal((n, prompts))
            out.append(np.clip(np.rint(lat + 1), 1, 6) if floor else lat + 1)
    A = np.mean(out[:3], 0); F = np.mean(out[3:], 0)
    return A, F, ev, day, slope
def slopes(Y, ev, mask):
    e1 = (ev & mask).sum(1); e0 = (~ev & mask).sum(1)
    ok = (e1 >= 3) & (e0 >= 3)
    m1 = np.where(ev & mask, Y, 0).sum(1) / np.maximum(e1, 1); m0 = np.where(~ev & mask, Y, 0).sum(1) / np.maximum(e0, 1)
    return m1 - m0, ok
def estimate(A, F, ev, day):
    odd = (day % 2 == 1)[None, :] & np.ones_like(ev); even = ~odd
    ao, k1 = slopes(A, ev, odd); ae, k2 = slopes(A, ev, even); fo, k3 = slopes(F, ev, odd); fe, k4 = slopes(F, ev, even)
    ok = k1 & k2 & k3 & k4
    c = lambda x, y: float(np.corrcoef(x[ok], y[ok])[0, 1])
    rel_a, rel_f = c(ao, ae), c(fo, fe)
    cross = 0.5 * (c(ao, fe) + c(ae, fo)); same = 0.5 * (c(ao, fo) + c(ae, fe))
    lat = cross / np.sqrt(max(rel_a * rel_f, 1e-9))
    return dict(people=int(ok.sum()), half_reliability_anger=rel_a, half_reliability_fear=rel_f, cross_half_raw=cross, same_half_raw=same,
                latent=lat, latent_from_same_half=same / np.sqrt(max(rel_a * rel_f, 1e-9)))
def between_reactivities(rho, a2=0.25):
    """Correlation between the two reactivities themselves when their logs correlate rho (each log has variance a2)."""
    return float((np.exp(a2 * rho) - 1) / (np.exp(a2) - 1))
def rival_truth(c):
    cov = np.exp((1.0 + c * c) / 2) - np.exp((0.25 + c * c) / 2 + 0.125)
    return float(cov / np.sqrt((np.exp(2 * (0.25 + c * c)) - np.exp(0.25 + c * c)) * (np.exp(0.5) - np.exp(0.25))))
res = {}
for name, kw in {
    'no floor, people 2000, prompts 60': dict(floor=False),
    'floor, people 2000, prompts 60': dict(floor=True),
    'floor, people 2000, prompts 84': dict(floor=True, prompts=84),
    'floor, people 300, prompts 60': dict(floor=True, n=300),
    'floor, usual level tied to the dial': dict(floor=True, base_link=0.4),
    'floor, rare events (rate 0.08)': dict(floor=True, event_rate=0.08, prompts=84),
}.items():
    res[name] = {}
    for rho in (1.0, 0.8, 0.5, 0.0):
        E = [estimate(*world(zlib.crc32(f'{name}|{rho}|{k}'.encode()), rho=rho, **kw)[:4]) for k in range(20)]
        lat = np.array([e['latent'] for e in E]); ls = np.array([e['latent_from_same_half'] for e in E])
        res[name][f'true {rho}'] = dict(latent_mean=round(float(lat.mean()), 3), latent_sd_over_worlds=round(float(lat.std()), 3),
            same_half_version_mean=round(float(ls.mean()), 3), half_reliability=round(float(np.mean([e['half_reliability_anger'] for e in E])), 3),
            people_used=int(np.mean([e['people'] for e in E])))
res['truth on the estimator scale'] = {f'logs correlate {rho}': round(between_reactivities(rho), 3) for rho in (1.0, 0.8, 0.5, 0.0)}
res['rival'] = {}
for c in (0.3, 0.6):
    for label, kw in (('floor, people 2000, prompts 60', dict()), ('floor, people 2000, prompts 84', dict(prompts=84))):
        E = [estimate(*world(zlib.crc32(f'rival|{c}|{label}|{k}'.encode()), rival=c, **kw)[:4]) for k in range(20)]
        lat = np.array([e['latent'] for e in E]); ls = np.array([e['latent_from_same_half'] for e in E])
        res['rival'][f'second input spread {c}, {label}'] = dict(truth_between_logs=round(0.5 / float(np.sqrt(0.25 + c * c)), 3), truth_between_reactivities=round(rival_truth(c), 3),
            latent_mean=round(float(lat.mean()), 3), latent_sd_over_worlds=round(float(lat.std()), 3), same_half_version_mean=round(float(ls.mean()), 3),
            share_of_worlds_with_estimate_above_0_85=round(float((lat > 0.85).mean()), 2))
json.dump(res, open(sys.argv[1], 'w'), indent=1)
for k, v in res.items():
    print(k)
    for kk, vv in v.items(): print('   ', kk, vv)
