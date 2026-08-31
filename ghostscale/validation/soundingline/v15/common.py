"""Shared machinery for V15 cards: seeds and lineages, proper scores, calibration, selective
prediction, bootstrap and equivalence tests over independent units, exact partial-information
decomposition and Shapley rulers, evaluation budgets, equivalence-class coverage, checkpoints,
process accounting, verdict writing and the committed completion ledger.

Seeds derive from ``zlib.crc32`` of a named string (house rule; never ``hash()``, whose string
hashing is randomised per process -- T-3 returned 2.29 on one run and 2.05 on the next from
identical code). A lineage name always contains its lane, so no object generated in one lane can
share an ancestor with another lane.

Two V15-specific pieces live here because every trunk needs them:

``criterion``
    builds the criterion block. A criterion carries a magnitude; a *gate* never does. Passing a
    criterion bar into ``GateReport.live`` is what turned three small real V14 effects into
    INSTRUMENT_FAILED at scale, and ``tests/test_v15_gates.py`` fails the suite if any gate's
    expected bar equals its card's SESOI.
``Budget``
    counts likelihood evaluations and proposals. Spec §3.5 requires every architecture comparison
    to be reported twice: information-matched (same observations) and budget-matched (same
    evaluations). An architecture that buys more observations or searches a larger model space
    must debit them, and an unbudgeted comparison is refused at reduce time.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
import zlib
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import numpy as np

from ....methods import gates as G
from ....methods import provenance as PROVENANCE
from . import REPO, SEED_OFFSET_V15, v15_dir, verdict_dir
from .schemas import TIERS, completion_entry

_EPS = 1e-12
_TINY = 1e-300

# --------------------------------------------------------------------------- #
# Lineages. World ids are disjoint by lane AND every seed string carries the lane name.
# --------------------------------------------------------------------------- #
LANE_BASE = {"discovery": 0, "confirmation": 10_000, "transfer": 20_000, "attack": 20_000,
             "coverage": 40_000, "pilot": 90_000}
LANE_CAP = {"discovery": 10_000, "confirmation": 10_000, "transfer": 10_000, "attack": 10_000,
            "coverage": 50_000, "pilot": 1_000}


def seed(name: str) -> int:
    return SEED_OFFSET_V15 + (zlib.crc32(("v15|" + name).encode("utf-8")) % 1_000_000)


def lane_of(wid: int) -> str:
    if wid >= 90_000:
        return "pilot"
    if wid >= 40_000:
        return "coverage"
    if wid >= 20_000:
        return "transfer"
    if wid >= 10_000:
        return "confirmation"
    return "discovery"


def world_seed(lane: str, wid: int) -> int:
    lane = "transfer" if lane == "attack" else lane
    assert lane_of(wid) == lane, (lane, wid)
    return seed(f"world|{lane}|{wid}")


def lane_ids(lane: str, tier: dict | str, limit: int | None = None) -> list:
    t = TIERS[tier] if isinstance(tier, str) else tier
    key = {"discovery": "discovery_worlds", "transfer": "transfer_worlds",
           "attack": "transfer_worlds", "confirmation": "confirmation_worlds",
           "coverage": "transfer_worlds", "pilot": "discovery_worlds"}[lane]
    n = int(t.get(key, 4))
    if lane == "pilot":
        n = min(n, 8)
    base = LANE_BASE[lane]
    ids = list(range(base, base + n))
    return ids[:limit] if limit is not None else ids


def rng_for(lane: str, card: str, wid: int, rep: int, tag: str = "") -> np.random.Generator:
    return np.random.default_rng(seed(f"{lane}|{card}|w{wid}|r{rep}|{tag}"))


def lineage_disjoint(ids_by_lane: dict) -> bool:
    seen = {}
    for lane, ids in ids_by_lane.items():
        for i in ids:
            if i in seen and seen[i] != lane:
                return False
            seen[i] = lane
    return True


# --------------------------------------------------------------------------- #
# Small numerics.
# --------------------------------------------------------------------------- #
def softmax(x, axis=-1):
    x = np.asarray(x, float)
    m = x.max(axis=axis, keepdims=True)
    z = np.exp(x - m)
    return z / z.sum(axis=axis, keepdims=True)


def logsumexp(x, axis=None):
    x = np.asarray(x, float)
    if axis is None:
        m = float(np.max(x))
        return float(m + np.log(np.sum(np.exp(x - m))))
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def entropy(p) -> float:
    p = np.asarray(p, float).ravel()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def kl(p, q) -> float:
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    s = p > 0
    return float((p[s] * np.log(p[s] / np.maximum(q[s], _TINY))).sum())


def js(p, q) -> float:
    p = np.asarray(p, float) + _EPS
    q = np.asarray(q, float) + _EPS
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def tv(p, q) -> float:
    p, q = normalize(p), normalize(q)
    return float(0.5 * np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def normalize(v):
    v = np.asarray(v, float)
    s = v.sum()
    return v / s if s > 0 else np.full(v.shape, 1.0 / max(v.size, 1))


def temper(p, tau: float):
    """Raise a distribution to 1/tau and renormalize. tau -> 0 is argmax, tau -> inf uniform."""
    p = np.asarray(p, float)
    if tau <= 1e-9:
        out = np.zeros_like(p)
        out[int(np.argmax(p))] = 1.0
        return out
    with np.errstate(divide="ignore"):
        lg = np.log(np.maximum(p, _TINY)) / float(tau)
    return softmax(lg)


# --------------------------------------------------------------------------- #
# Proper scores, calibration, selective prediction (spec §8.1).
# --------------------------------------------------------------------------- #
def log_score(post, truth, floor: float = 1e-12) -> float:
    p = post[truth] if isinstance(post, dict) else post[int(truth)]
    return float(np.log(max(float(p), floor)))


def brier(post, truth) -> float:
    if isinstance(post, dict):
        return float(sum((p - (1.0 if k == truth else 0.0)) ** 2 for k, p in post.items()))
    v = np.asarray(post, float)
    y = np.zeros_like(v)
    y[int(truth)] = 1.0
    return float(((v - y) ** 2).sum())


def top1(post):
    if isinstance(post, dict):
        return max(post, key=post.get)
    return int(np.argmax(post))


def confidence(post) -> float:
    return float(max(post.values())) if isinstance(post, dict) else float(np.max(post))


def ece(confidences, correct, bins: int = 10) -> float:
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if c.size == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c >= lo) & (c < hi) if hi < 1 else (c >= lo) & (c <= hi)
        if m.any():
            out += m.mean() * abs(c[m].mean() - y[m].mean())
    return float(out)


def calibration_slope(confidences, correct) -> float:
    c = np.asarray(confidences, float)
    y = np.asarray(correct, float)
    if c.size < 3 or c.std() < 1e-9:
        return float("nan")
    return float(np.cov(c, y, bias=True)[0, 1] / c.var())


def risk_coverage(confidences, correct, grid=(0.2, 0.4, 0.6, 0.8, 1.0)) -> dict:
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    order = np.argsort(-c)
    out = {}
    for cov in grid:
        k = max(1, int(round(cov * c.size))) if c.size else 0
        idx = order[:k]
        out[str(cov)] = {"coverage": float(k / max(c.size, 1)),
                         "risk": float(1.0 - y[idx].mean()) if k else float("nan")}
    return out


def selective_error(confidences, correct, threshold: float) -> dict:
    c = np.asarray(confidences, float)
    y = np.asarray(correct, float)
    m = c >= threshold
    return {"answered": float(m.mean()) if c.size else float("nan"),
            "error": float(1.0 - y[m].mean()) if m.any() else float("nan")}


def calibration_block(confidences, correct) -> dict:
    return {"ece": ece(confidences, correct), "slope": calibration_slope(confidences, correct),
            "risk_coverage": risk_coverage(confidences, correct), "n": int(len(confidences))}


# --------------------------------------------------------------------------- #
# Inference over independent units (spec §8.1, §8.2).
# Rows, actions, tokens, posterior doses and repeated edits are NOT independent makers.
# --------------------------------------------------------------------------- #
def hboot(values_by_unit: dict, rng, draws: int = 800, stat=np.mean) -> dict:
    units = list(values_by_unit)
    if not units:
        return {"mean": float("nan"), "interval": [float("nan"), float("nan")],
                "n_units": 0, "n_rows": 0}
    arrays = [np.asarray(values_by_unit[u], dtype=float).ravel() for u in units]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        return {"mean": float("nan"), "interval": [float("nan"), float("nan")],
                "n_units": 0, "n_rows": 0}
    est = []
    for _ in range(int(draws)):
        pick = rng.choice(len(arrays), len(arrays), replace=True)
        rows = [rng.choice(arrays[i], arrays[i].size, replace=True) for i in pick]
        est.append(float(stat(np.concatenate(rows))))
    est = np.asarray(est)
    allrows = np.concatenate(arrays)
    return {"mean": float(stat(allrows)),
            "interval": [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))],
            "n_units": len(arrays), "n_rows": int(allrows.size)}


def paired_hboot(a_by_unit: dict, b_by_unit: dict, rng, draws: int = 800) -> dict:
    keys = [k for k in a_by_unit if k in b_by_unit]
    diffs = {}
    for k in keys:
        a, b = np.asarray(a_by_unit[k], float), np.asarray(b_by_unit[k], float)
        n = min(a.size, b.size)
        if n:
            diffs[k] = a[:n] - b[:n]
    r = hboot(diffs, rng, draws)
    r["excludes_zero"] = bool(r["interval"][0] > 0 or r["interval"][1] < 0)
    return r


def equivalence(a_by_unit: dict, b_by_unit: dict, rng, margin: float, draws: int = 800) -> dict:
    """Two one-sided bootstrap test for a declared null (spec §8.2).

    A null is *claimed* only when the paired difference's whole interval sits inside
    ``[-margin, +margin]``. "Failed to reject" is not equivalence and is reported as
    ``inconclusive`` so the distinction survives into the packet.
    """
    r = paired_hboot(a_by_unit, b_by_unit, rng, draws)
    lo, hi = r["interval"]
    inside = bool(math.isfinite(lo) and math.isfinite(hi) and lo > -margin and hi < margin)
    r.update({"margin": float(margin), "equivalent": inside,
              "verdict": "equivalent" if inside else ("different" if r["excludes_zero"] else "inconclusive")})
    return r


def cluster_permutation_p(a_by_unit: dict, b_by_unit: dict, rng, draws: int = 400) -> float:
    keys = [k for k in a_by_unit if k in b_by_unit]
    d = np.array([np.mean(a_by_unit[k]) - np.mean(b_by_unit[k]) for k in keys])
    if d.size == 0:
        return float("nan")
    obs = abs(d.mean())
    hits = 0
    for _ in range(int(draws)):
        s = rng.choice([-1.0, 1.0], d.size)
        hits += abs((d * s).mean()) >= obs
    return float((hits + 1) / (draws + 1))


def bh_fdr(pvals: dict, q: float = 0.10) -> dict:
    items = [(k, v) for k, v in pvals.items() if v == v]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out = {k: False for k, _ in items}
    thresh = 0
    for i, (k, p) in enumerate(items, 1):
        if p <= q * i / m:
            thresh = i
    for i, (k, _) in enumerate(items, 1):
        out[k] = i <= thresh
    return out


def by_unit(rows: list, key: str, unit: str = "wid", where=None) -> dict:
    out = {}
    for r in rows:
        if where is not None and not where(r):
            continue
        v = r.get(key)
        if v is None or (isinstance(v, float) and v != v):
            continue
        out.setdefault(r[unit], []).append(float(v))
    return out


def by_unit_pair(rows: list, a: str, b: str, unit: str = "wid", where=None):
    return by_unit(rows, a, unit, where), by_unit(rows, b, unit, where)


def compare(rows: list, a: str, b: str, rng, unit: str = "wid", where=None,
            draws: int = 800) -> dict:
    """The standard paired comparison: ``a`` minus ``b`` bootstrapped over independent units."""
    ab, bb = by_unit_pair(rows, a, b, unit, where)
    out = paired_hboot(ab, bb, rng, draws)
    out["a"], out["b"] = a, b
    out["a_mean"] = hboot(ab, rng, draws)["mean"]
    out["b_mean"] = hboot(bb, rng, draws)["mean"]
    return out


# --------------------------------------------------------------------------- #
# Criteria. A criterion carries the magnitude; a gate never does (spec §4.2, §8.2).
# --------------------------------------------------------------------------- #
def criterion(name: str, observed: float, bar: float, direction: str = "greater",
              basis: str = "", interval=None, detail: str = "") -> dict:
    """Evaluate one pre-registered criterion and say plainly whether it held.

    ``direction``: ``greater`` (observed must clear ``bar``), ``less`` (must stay under it),
    ``equivalent`` (must sit inside ``[-bar, +bar]``).
    """
    o = float(observed) if observed is not None else float("nan")
    ok = False
    if math.isfinite(o):
        if direction == "greater":
            ok = o >= bar
        elif direction == "less":
            ok = o <= bar
        elif direction == "equivalent":
            ok = abs(o) <= bar
        else:                                                   # pragma: no cover - guarded below
            raise ValueError(direction)
    return {"name": name, "observed": None if not math.isfinite(o) else o, "bar": float(bar),
            "direction": direction, "held": bool(ok), "basis": basis,
            "interval": list(interval) if interval is not None else None, "detail": detail}


def criterion_status(criteria: list) -> str:
    """HELD only if every declared criterion held; FAILED if any evaluated one did not."""
    if not criteria:
        return "NOT_APPLICABLE"
    evaluated = [c for c in criteria if c.get("observed") is not None]
    if not evaluated:
        return "UNEVALUATED"
    return "HELD" if all(c["held"] for c in evaluated) else "FAILED"


# --------------------------------------------------------------------------- #
# Exact partial information decomposition and Shapley rulers (spec §6 C05, P04).
#
# Two sources: the Williams-Beer redundancy lattice with the I_min redundancy function, computed
# exactly from a joint table. Three or more: I_min does not have an agreed unique extension, so
# the ruler reported for three components is instead the exact Shapley decomposition of the
# held-out predictive log score over the 2^k subsets, which is unique, additive and computable.
# Both are declared in the preregistration; neither is estimated.
# --------------------------------------------------------------------------- #
def specific_information(joint, src_axis: int, tgt_axis: int = -1):
    """I(S=s ; T) for each target value t, per Williams & Beer: the average reduction in surprise
    about ``s`` provided by ``t``."""
    p = np.asarray(joint, float)
    p = p / p.sum()
    axes = list(range(p.ndim))
    tgt = axes[tgt_axis]
    keep = (src_axis, tgt)
    drop = tuple(a for a in axes if a not in keep)
    pst = p.sum(axis=drop) if drop else p
    if src_axis > tgt:
        pst = pst.T
    ps = pst.sum(axis=1)
    pt = pst.sum(axis=0)
    out = np.zeros(pt.size)
    for j in range(pt.size):
        if pt[j] <= 0:
            continue
        ps_t = pst[:, j] / pt[j]
        m = (ps_t > 0) & (ps > 0)
        out[j] = float((ps_t[m] * (np.log(1.0 / ps[m]) - np.log(1.0 / ps_t[m]))).sum())
    return out, pt


def pid_two_source(joint) -> dict:
    """Exact Williams-Beer PID for two sources and one target from a 3-d joint table
    ``joint[s1, s2, t]``. Returns redundancy, the two uniques, synergy and the mutual
    informations, all in nats."""
    p = np.asarray(joint, float)
    assert p.ndim == 3, p.shape
    p = p / p.sum()
    i1, pt = specific_information(p, 0, 2)
    i2, _ = specific_information(p, 1, 2)
    red = float((pt * np.minimum(i1, i2)).sum())
    mi1 = float((pt * i1).sum())
    mi2 = float((pt * i2).sum())
    # I({1,2};T) from the joint source alphabet
    s1, s2, nt = p.shape
    pj = p.reshape(s1 * s2, nt)
    pjs = pj.sum(axis=1)
    mi12 = 0.0
    for j in range(nt):
        if pt[j] <= 0:
            continue
        col = pj[:, j] / pt[j]
        m = (col > 0) & (pjs > 0)
        mi12 += float(pt[j] * (col[m] * np.log(col[m] / pjs[m])).sum())
    u1, u2 = mi1 - red, mi2 - red
    syn = mi12 - red - u1 - u2
    return {"redundancy": red, "unique_1": u1, "unique_2": u2, "synergy": syn,
            "mi_1": mi1, "mi_2": mi2, "mi_joint": mi12,
            "definition": "williams_beer_imin_exact"}


def shapley_decomposition(value_of_subset) -> dict:
    """Exact Shapley values of a set function over named components.

    ``value_of_subset`` maps a frozenset of component names to a scalar (here: held-out predictive
    log score relative to the empty set). Exact over all 2^k subsets; k is 3 or 4 in this program.
    """
    names = sorted(value_of_subset.keys(), key=lambda s: (len(s), tuple(sorted(s))))
    comps = sorted({c for s in value_of_subset for c in s})
    k = len(comps)
    fact = [math.factorial(i) for i in range(k + 1)]
    phi = {c: 0.0 for c in comps}
    for c in comps:
        others = [x for x in comps if x != c]
        for r in range(len(others) + 1):
            for sub in combinations(others, r):
                s = frozenset(sub)
                w = fact[r] * fact[k - r - 1] / fact[k]
                phi[c] += w * (value_of_subset[s | {c}] - value_of_subset[s])
    total = value_of_subset[frozenset(comps)] - value_of_subset[frozenset()]
    return {"shapley": phi, "total": float(total),
            "sums_to_total": bool(abs(sum(phi.values()) - total) < 1e-9),
            "definition": "exact_shapley_over_subsets", "n_subsets": len(names)}


# --------------------------------------------------------------------------- #
# Equivalence classes (spec §8.1). An equivalence class is reported, never collapsed.
# --------------------------------------------------------------------------- #
def class_mass(posterior: dict, classes: dict) -> dict:
    """``classes`` maps class name -> members. Returns per-class mass, the largest single-member
    mass inside the true class, and the coverage of the true class."""
    out = {}
    for cname, members in classes.items():
        out[cname] = float(sum(posterior.get(m, 0.0) for m in members))
    return out


def class_receipt(posterior: dict, classes: dict, truth) -> dict:
    tc = next((c for c, ms in classes.items() if truth in ms), None)
    masses = class_mass(posterior, classes)
    members = classes.get(tc, [])
    inside = {m: float(posterior.get(m, 0.0)) for m in members}
    return {"true_class": tc, "class_mass": masses.get(tc, float("nan")),
            "max_member_mass": max(inside.values()) if inside else float("nan"),
            "n_members": len(members), "all_class_mass": masses,
            "unjustified_member_mass": (max(inside.values()) - 1.0 / max(len(members), 1))
            if inside else float("nan")}


# --------------------------------------------------------------------------- #
# Evaluation budgets (spec §3.5). Information-matched AND budget-matched, or it is not reported.
# --------------------------------------------------------------------------- #
class Budget:
    """Counts what an architecture spent: likelihood evaluations, proposals, observations bought.

    A reader that buys an extra observation debits ``observations``; one that searches a larger
    model space debits ``proposals``. The reduce step refuses an architecture comparison whose
    budgets were never touched, because an unmeasured budget is indistinguishable from an
    unlimited one.
    """

    __slots__ = ("likelihood", "proposals", "observations", "cpu_s", "_t0")

    def __init__(self):
        self.likelihood = 0
        self.proposals = 0
        self.observations = 0
        self.cpu_s = 0.0
        self._t0 = None

    def lik(self, n: int = 1):
        self.likelihood += int(n)
        return self

    def prop(self, n: int = 1):
        self.proposals += int(n)
        return self

    def obs(self, n: int = 1):
        self.observations += int(n)
        return self

    @contextmanager
    def timing(self):
        t0 = time.process_time()
        try:
            yield self
        finally:
            self.cpu_s += time.process_time() - t0

    def to_dict(self) -> dict:
        return {"likelihood_evaluations": int(self.likelihood), "proposals": int(self.proposals),
                "observations": int(self.observations), "cpu_s": round(float(self.cpu_s), 6)}

    def touched(self) -> bool:
        return bool(self.likelihood or self.proposals or self.observations or self.cpu_s > 0)


def budget_receipt(budgets: dict, tolerance: float = 0.25) -> dict:
    """Are these architectures compute-matched? ``budgets`` maps architecture -> Budget dict.

    Matched means every non-oracle architecture's likelihood-evaluation count sits within
    ``tolerance`` (relative) of the median. Oracles are excluded from the match but reported.
    """
    from .schemas import NON_PROMOTABLE
    live = {k: v for k, v in budgets.items() if not k.startswith("oracle")
            and k not in NON_PROMOTABLE}
    ev = {k: float(v.get("likelihood_evaluations", 0)) for k, v in live.items()}
    vals = [v for v in ev.values() if v > 0]
    med = float(np.median(vals)) if vals else 0.0
    spread = {k: (abs(v - med) / med if med > 0 else float("nan")) for k, v in ev.items()}
    untouched = [k for k, v in budgets.items() if not any(
        float(v.get(f, 0)) > 0 for f in ("likelihood_evaluations", "proposals", "observations", "cpu_s"))]
    return {"median_likelihood_evaluations": med, "relative_spread": spread,
            "tolerance": float(tolerance),
            "compute_matched": bool(vals and all(s <= tolerance for s in spread.values())),
            "untouched_budgets": untouched, "all_budgets": budgets}


# --------------------------------------------------------------------------- #
# Cells (spec §6). Every declared factor level realized per independent unit.
# --------------------------------------------------------------------------- #
def realized_cells(rows: list, factors: dict, unit: str = "wid") -> dict:
    keys = list(factors)
    counts = {}
    for r in rows:
        cell = tuple(str(r.get(k, "?")) for k in keys)
        uk = f"{r.get(unit, 0)}|{r.get('rep', 0)}"
        counts.setdefault(cell, {}).setdefault(uk, 0)
        counts[cell][uk] += int(r.get("n", 1))
    return {"|".join(c): {"units": len(u), "rows": int(sum(u.values())),
                          "min_rows_in_a_unit": int(min(u.values()))}
            for c, u in counts.items()}


def cell_receipt(realized: dict, expected: dict) -> dict:
    n_cells = len(realized)
    units = min((c["units"] for c in realized.values()), default=0)
    min_rows = min((c["min_rows_in_a_unit"] for c in realized.values()), default=0)
    need_rows = expected.get("min_rows_per_unit", 0)
    ok = (n_cells >= expected.get("levels", 0)) \
        and (units >= expected.get("units_required", 0)) \
        and (min_rows >= need_rows or need_rows == 0)
    return {"expected_levels": expected.get("levels"), "realized_levels": n_cells,
            "expected_units": expected.get("units_required"), "realized_units": units,
            "expected_min_rows_per_unit": need_rows, "realized_min_rows_per_unit": min_rows,
            "ok": bool(ok)}


def receipt_for(card, ctx, rows: list) -> dict:
    """The receipt every card writes. Applies the sparsest-cell rule: a ``list`` card's cells each
    live in exactly one unit, so requiring every cell in every unit is arithmetically impossible.
    """
    expected = dict(ctx.get("expected") or {})
    if card.unit_kind != "world":
        expected["units_required"] = 1 if card.unit_kind == "list" else int(ctx.get("n_units") or 1)
    else:
        expected.setdefault("units_required", int(ctx.get("n_units") or 1))
    return cell_receipt(realized_cells(rows, card.factors), expected)


# --------------------------------------------------------------------------- #
# Environment, priority, accelerators, process accounting (spec §9.4, §9.6).
# --------------------------------------------------------------------------- #
def set_numeric_threads(n: int = 1) -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(n)


def hide_accelerators() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"


def lower_priority() -> str:
    try:
        import psutil
        p = psutil.Process()
        if os.name == "nt":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
        return "below_normal"
    except Exception as exc:                                    # noqa: BLE001
        return f"unchanged ({exc!r})"


def process_accounting() -> dict:
    out = {"pid": os.getpid(), "process_time": time.process_time()}
    try:
        import psutil
        p = psutil.Process()
        t = p.cpu_times()
        out.update({"user": float(t.user), "system": float(t.system),
                    "children_user": float(getattr(t, "children_user", 0.0)),
                    "children_system": float(getattr(t, "children_system", 0.0))})
        mi = p.memory_info()
        out["rss"] = int(mi.rss)
        out["peak_rss"] = int(getattr(mi, "peak_wset", mi.rss))
    except Exception as exc:                                    # noqa: BLE001
        out["psutil"] = repr(exc)
    return out


def tree_cpu_seconds() -> dict:
    """Parent plus *all live descendants* (spec §9.4 requires descendant CPU, not wall time)."""
    out = {"parent": 0.0, "descendants": 0.0, "n_descendants": 0}
    try:
        import psutil
        p = psutil.Process()
        t = p.cpu_times()
        out["parent"] = float(t.user + t.system)
        tot, n = 0.0, 0
        for ch in p.children(recursive=True):
            try:
                ct = ch.cpu_times()
                tot += float(ct.user + ct.system)
                n += 1
            except Exception:                                   # noqa: BLE001, S112
                continue
        out["descendants"], out["n_descendants"] = tot, n
    except Exception as exc:                                    # noqa: BLE001
        out["psutil"] = repr(exc)
    out["total"] = out["parent"] + out["descendants"]
    return out


def environment() -> dict:
    import numpy
    import scipy
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jax_platforms": os.environ.get("JAX_PLATFORMS")}


@contextmanager
def timed(record: dict, key: str = "runtime_seconds"):
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        yield
    finally:
        record[key] = round(time.perf_counter() - t0, 3)
        record.setdefault("runtime", {})["reduce_process_time_s"] = round(time.process_time() - c0, 3)


# --------------------------------------------------------------------------- #
# JSON, hashing, checkpoints.
# --------------------------------------------------------------------------- #
def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if f != f else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, frozenset):
        return sorted(str(x) for x in obj)
    return obj


def dumps(obj) -> str:
    return json.dumps(to_jsonable(obj), indent=2, sort_keys=False, default=str)


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def obj_sha(obj) -> str:
    return text_sha(json.dumps(to_jsonable(obj), sort_keys=True, default=str))


def ckpt_path(lane: str, card: str, wid: int, rep: int) -> Path:
    d = v15_dir("checkpoints") / lane / card
    d.mkdir(parents=True, exist_ok=True)
    return d / f"w{wid}_r{rep}.json"


def load_ckpt(lane: str, card: str, wid: int, rep: int, source_sha: str):
    p = ckpt_path(lane, card, wid, rep)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if d.get("source_sha256") != source_sha:
        return None                                  # hash-aware resume: stale checkpoints recompute
    return d.get("unit")


def save_ckpt(lane: str, card: str, wid: int, rep: int, source_sha: str, unit: dict,
              runtime: dict) -> Path:
    from .atomicio import write_text_atomic
    return write_text_atomic(ckpt_path(lane, card, wid, rep),
                             dumps({"card": card, "lane": lane, "wid": wid, "rep": rep,
                                    "source_sha256": source_sha, "runtime": runtime,
                                    "unit": unit}))


# --------------------------------------------------------------------------- #
# Verdicts and the committed completion ledger (spec §10.2).
# --------------------------------------------------------------------------- #
COMPLETION = v15_dir() / "COMPLETION.json"


def write_verdict(card_id: str, verdict: dict, gr: G.GateReport, module_file: str, lane: str,
                  out_dir: Path | None = None, ledger: bool = True) -> Path:
    from .atomicio import write_text_atomic
    verdict["environment"] = environment()
    verdict.setdefault("runtime", {})["reduce_accounting"] = process_accounting()
    PROVENANCE.stamp(verdict, module_file, gr)
    d = out_dir if out_dir is not None else verdict_dir(lane)
    out = write_text_atomic(d / f"{card_id}.json", dumps(verdict))
    write_text_atomic(d / f"{card_id}.produced",
                      json.dumps({"card": card_id, "sha256": file_sha(out),
                                  "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    if ledger and lane in ("discovery", "transfer", "attack", "confirmation"):
        record_completion(card_id, lane, out, verdict)
    return out


def record_completion(card_id: str, lane: str, out: Path, verdict: dict) -> None:
    from .atomicio import write_json_atomic
    doc = json.loads(COMPLETION.read_text(encoding="utf-8")) if COMPLETION.exists() \
        else {"program": "v15", "entries": {}}
    try:
        rel = out.resolve().relative_to(REPO).as_posix()
    except ValueError:
        rel = out.as_posix()
    doc["entries"][f"{lane}:{card_id}"] = completion_entry(
        card_id, lane, rel, file_sha(out),
        verdict.get("produced_by", {}).get("sha256", ""),
        verdict.get("expected_cell_receipt", {}), verdict.get("state", ""),
        verdict.get("criterion_status", "UNEVALUATED"), time.strftime("%Y-%m-%dT%H:%M:%S"))
    doc["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json_atomic(COMPLETION, doc, sort_keys=True)


def load_verdict(card_id: str, lane: str = "discovery") -> dict | None:
    p = verdict_dir(lane) / f"{card_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def gate_state(gr: G.GateReport) -> str:
    return "INSTRUMENT_VALID" if gr.to_dict()["all_passed"] else "INSTRUMENT_FAILED"


def finish(verdict: dict, card, gr: G.GateReport, criteria: list, rows: list, ctx: dict) -> dict:
    """Close a verdict: receipt, criterion status, state. The one place both axes are set, so
    they cannot drift apart."""
    verdict["expected_cell_receipt"] = receipt_for(card, ctx, rows)
    verdict["criteria"] = criteria
    verdict["criterion_status"] = criterion_status(criteria)
    st = gate_state(gr)
    if st == "INSTRUMENT_VALID" and not verdict["expected_cell_receipt"]["ok"]:
        st = "RESOURCE_BLOCKED"
    verdict["state"] = "LANDED" if st == "INSTRUMENT_VALID" else st
    verdict["effective_n"] = {"units": len({(r.get("wid"), r.get("rep")) for r in rows}),
                              "rows": len(rows)}
    return verdict
