"""Shared machinery for V13 cards: seeds and lineages, proper scores, calibration, bootstrap,
checkpoints, process accounting, verdict writing, and the committed completion ledger.

Seeds derive from crc32 of named strings (house rule; never ``hash()``). A lineage name always
contains its lane, so no object generated in one lane can share an ancestor with another lane;
card I12 asserts it and ``lineage_disjoint`` is the test.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ....methods import gates as G
from ....methods import provenance as PROVENANCE
from . import REPO, SEED_OFFSET_V13, v13_dir, verdict_dir
from .schemas import TIERS, completion_entry

_EPS = 1e-12
_TINY = 1e-300

# --------------------------------------------------------------------------- #
# Lineages. World ids are disjoint by lane AND every seed string carries the lane name.
# --------------------------------------------------------------------------- #
LANE_BASE = {"discovery": 0, "confirmation": 1000, "transfer": 2000, "attack": 2000, "pilot": 9000}
LANE_CAP = {"discovery": 1000, "confirmation": 1000, "transfer": 1000, "attack": 1000, "pilot": 100}


def seed(name: str) -> int:
    return SEED_OFFSET_V13 + (zlib.crc32(("v13|" + name).encode("utf-8")) % 1_000_000)


def lane_of(wid: int) -> str:
    if wid >= 9000:
        return "pilot"
    if wid >= 2000:
        return "transfer"
    if wid >= 1000:
        return "confirmation"
    return "discovery"


def world_seed(lane: str, wid: int) -> int:
    lane = "transfer" if lane == "attack" else lane
    assert lane_of(wid) == lane, (lane, wid)
    return seed(f"world|{lane}|{wid}")


def lane_ids(lane: str, tier: dict | str, limit: int | None = None) -> list:
    t = TIERS[tier] if isinstance(tier, str) else tier
    key = {"discovery": "discovery_worlds", "transfer": "transfer_worlds", "attack": "transfer_worlds",
           "confirmation": "confirmation_worlds", "pilot": "pilot_worlds"}[lane]
    n = int(t.get(key, 4))
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


def js(p, q) -> float:
    p = np.asarray(p, float) + _EPS
    q = np.asarray(q, float) + _EPS
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def kl(p, q) -> float:
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    s = p > 0
    return float((p[s] * np.log(p[s] / np.maximum(q[s], _TINY))).sum())


def normalize(v):
    v = np.asarray(v, float)
    s = v.sum()
    return v / s if s > 0 else np.full(v.shape, 1.0 / v.size)


# --------------------------------------------------------------------------- #
# Proper scores, calibration, selective prediction (spec §6.1).
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
    """Slope of accuracy on confidence (1 = calibrated, <1 = overconfident, >1 = underconfident)."""
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
        k = max(1, int(round(cov * c.size)))
        idx = order[:k]
        out[str(cov)] = {"coverage": float(k / max(c.size, 1)),
                         "risk": float(1.0 - y[idx].mean()) if k else float("nan")}
    return out


def selective_error(confidences, correct, threshold: float) -> dict:
    """Error among answers whose confidence clears a threshold, and the fraction abstained."""
    c = np.asarray(confidences, float)
    y = np.asarray(correct, float)
    m = c >= threshold
    return {"answered": float(m.mean()) if c.size else float("nan"),
            "error": float(1.0 - y[m].mean()) if m.any() else float("nan")}


def calibration_block(confidences, correct) -> dict:
    return {"ece": ece(confidences, correct), "slope": calibration_slope(confidences, correct),
            "risk_coverage": risk_coverage(confidences, correct),
            "n": int(len(confidences))}


# --------------------------------------------------------------------------- #
# Inference over independent units (spec §6.2).
# --------------------------------------------------------------------------- #
def hboot(values_by_unit: dict, rng, draws: int = 1000, stat=np.mean) -> dict:
    units = list(values_by_unit)
    if not units:
        return {"mean": float("nan"), "interval": [float("nan"), float("nan")], "n_units": 0, "n_rows": 0}
    arrays = [np.asarray(values_by_unit[u], dtype=float).ravel() for u in units]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        return {"mean": float("nan"), "interval": [float("nan"), float("nan")], "n_units": 0, "n_rows": 0}
    est = []
    for _ in range(int(draws)):
        pick = rng.choice(len(arrays), len(arrays), replace=True)
        rows = [rng.choice(arrays[i], arrays[i].size, replace=True) for i in pick]
        est.append(float(stat(np.concatenate(rows))))
    est = np.asarray(est)
    allrows = np.concatenate(arrays)
    return {"mean": float(stat(allrows)), "interval": [float(np.percentile(est, 2.5)),
                                                       float(np.percentile(est, 97.5))],
            "n_units": len(arrays), "n_rows": int(allrows.size)}


def paired_hboot(a_by_unit: dict, b_by_unit: dict, rng, draws: int = 1000) -> dict:
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


def cluster_permutation_p(a_by_unit: dict, b_by_unit: dict, rng, draws: int = 500) -> float:
    """Two-sided p for a paired difference by sign-flipping whole units."""
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
    """Benjamini–Hochberg within a trunk: which named tests survive at FDR q."""
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


def quantile_bins(values, n_bins: int = 4):
    v = np.asarray(values, float)
    qs = np.quantile(v, np.linspace(0, 1, n_bins + 1))
    return np.clip(np.searchsorted(qs[1:-1], v, side="right"), 0, n_bins - 1), qs


# --------------------------------------------------------------------------- #
# Cells (spec §7.5): every declared factor level must be realized per unit.
# --------------------------------------------------------------------------- #
def realized_cells(rows: list, factors: dict, unit: str = "wid") -> dict:
    """Count rows per (unit, cell) where a cell is the tuple of declared factor levels."""
    keys = list(factors)
    counts = {}
    for r in rows:
        cell = tuple(str(r.get(k, "?")) for k in keys)
        counts.setdefault(cell, {}).setdefault(str(r.get(unit, 0)), 0)
        counts[cell][str(r.get(unit, 0))] += int(r.get("n", 1))
    return {"|".join(c): {"units": len(u), "rows": int(sum(u.values())), "min_rows_in_a_unit": int(min(u.values()))}
            for c, u in counts.items()}


def cell_receipt(realized: dict, expected: dict) -> dict:
    n_cells = len(realized)
    units = min((c["units"] for c in realized.values()), default=0)
    min_rows = min((c["min_rows_in_a_unit"] for c in realized.values()), default=0)
    ok = (n_cells >= expected.get("levels", 0)) and (units >= expected.get("units_required", 0)) \
        and (min_rows >= expected.get("min_rows_per_unit", 0) or expected.get("min_rows_per_unit", 0) == 0)
    return {"expected_levels": expected.get("levels"), "realized_levels": n_cells,
            "expected_units": expected.get("units_required"), "realized_units": units,
            "expected_min_rows_per_unit": expected.get("min_rows_per_unit"), "realized_min_rows_per_unit": min_rows,
            "ok": bool(ok)}


# --------------------------------------------------------------------------- #
# Environment, priority, accelerators, process accounting (spec §4.2, §7.4, §20.3).
# --------------------------------------------------------------------------- #
def set_numeric_threads(n: int = 1) -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
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
    """User+system CPU of this process and its finished children, and peak resident set."""
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


def environment() -> dict:
    import numpy
    import scipy
    try:
        import pymdp
        pv = getattr(pymdp, "__version__", "1.0.2 (pinned)")
    except Exception:                                           # noqa: BLE001
        pv = None
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "pymdp": pv, "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
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
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and obj != obj:
        return None
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
    d = v13_dir("checkpoints") / lane / card
    d.mkdir(parents=True, exist_ok=True)
    return d / f"w{wid}_r{rep}.json"


def load_ckpt(lane: str, card: str, wid: int, rep: int, source_sha: str):
    p = ckpt_path(lane, card, wid, rep)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if d.get("source_sha256") != source_sha:
        return None                                              # hash-aware resume: stale checkpoints are recomputed
    return d.get("unit")


def save_ckpt(lane: str, card: str, wid: int, rep: int, source_sha: str, unit: dict, runtime: dict) -> Path:
    p = ckpt_path(lane, card, wid, rep)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(dumps({"card": card, "lane": lane, "wid": wid, "rep": rep, "source_sha256": source_sha,
                          "runtime": runtime, "unit": unit}), encoding="utf-8")
    os.replace(tmp, p)
    return p


# --------------------------------------------------------------------------- #
# Verdicts and the committed completion ledger (spec §20.2).
# --------------------------------------------------------------------------- #
COMPLETION = v13_dir() / "COMPLETION.json"


def write_verdict(card_id: str, verdict: dict, gr: G.GateReport, module_file: str, lane: str,
                  out_dir: Path | None = None, ledger: bool = True) -> Path:
    verdict["environment"] = environment()
    verdict.setdefault("runtime", {})["reduce_accounting"] = process_accounting()
    PROVENANCE.stamp(verdict, module_file, gr)
    d = out_dir if out_dir is not None else verdict_dir(lane)
    out = d / f"{card_id}.json"
    out.write_text(dumps(verdict), encoding="utf-8")
    marker = d / f"{card_id}.produced"
    marker.write_text(json.dumps({"card": card_id, "sha256": file_sha(out),
                                  "written": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
    if ledger and lane in ("discovery", "transfer", "attack", "confirmation"):
        record_completion(card_id, lane, out, verdict)
    return out


def record_completion(card_id: str, lane: str, out: Path, verdict: dict) -> None:
    doc = json.loads(COMPLETION.read_text(encoding="utf-8")) if COMPLETION.exists() else {"program": "v13", "entries": {}}
    try:
        rel = out.resolve().relative_to(REPO).as_posix()
    except ValueError:
        rel = out.as_posix()
    key = f"{lane}:{card_id}"
    doc["entries"][key] = completion_entry(card_id, lane, rel, file_sha(out),
                                           verdict.get("produced_by", {}).get("sha256", ""),
                                           verdict.get("expected_cell_receipt", {}), verdict.get("state", ""),
                                           time.strftime("%Y-%m-%dT%H:%M:%S"))
    doc["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    COMPLETION.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")


def load_verdict(card_id: str, lane: str = "discovery") -> dict | None:
    p = verdict_dir(lane) / f"{card_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def gate_state(gr: G.GateReport) -> str:
    d = gr.to_dict()
    return "INSTRUMENT_VALID" if d["all_passed"] else "INSTRUMENT_FAILED"
