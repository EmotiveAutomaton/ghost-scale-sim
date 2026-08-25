"""Shared machinery for V12 cards: seeds, lineages, scoring, bootstrap, verdict writing.

Seeds derive from crc32 of named strings (house rule). Discovery and confirmation world lineages
are disjoint by construction and asserted by card I06.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ....methods import gates as G
from ....methods import provenance as PROVENANCE
from . import REPO, SEED_OFFSET_V12, v12_dir, verdict_dir

N_DISCOVERY = 12
N_CONFIRMATION = 12
DISCOVERY_IDS = list(range(0, N_DISCOVERY))
CONFIRMATION_IDS = list(range(100, 100 + N_CONFIRMATION))


def seed(name: str) -> int:
    return SEED_OFFSET_V12 + (zlib.crc32(name.encode("utf-8")) % 1_000_000)


def world_seed(world_id: int) -> int:
    lane = "conf" if world_id >= 100 else "disc"
    return seed(f"world:{lane}:{world_id}")


def rng_for(card: str, world_id: int, s: int, tag: str = "") -> np.random.Generator:
    return np.random.default_rng(seed(f"{card}:w{world_id}:s{s}:{tag}"))


# --------------------------------------------------------------------------- #
# Proper scores and calibration.
# --------------------------------------------------------------------------- #
def log_score(post: dict, truth: str, floor: float = 1e-12) -> float:
    return float(np.log(max(float(post.get(truth, 0.0)), floor)))


def brier(post: dict, truth: str) -> float:
    return float(sum((p - (1.0 if k == truth else 0.0)) ** 2 for k, p in post.items()))


def top1(post: dict) -> str:
    return max(post, key=post.get)


def ece(confidences, correct, bins: int = 10) -> float:
    """Expected calibration error over equal-width confidence bins."""
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


def risk_coverage(confidences, correct, grid=(0.2, 0.4, 0.6, 0.8, 1.0)) -> dict:
    """Error rate among the most-confident fraction of answers, at several coverages."""
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


def hboot(values_by_unit: dict, rng, draws: int = 2000, stat=np.mean) -> dict:
    """Hierarchical bootstrap over independent units (worlds or makers): resample units with
    replacement, then rows within each resampled unit."""
    units = list(values_by_unit)
    if not units:
        return {"mean": float("nan"), "interval": [float("nan"), float("nan")], "n_units": 0}
    est = []
    for _ in range(int(draws)):
        pick = rng.choice(len(units), len(units), replace=True)
        rows = []
        for i in pick:
            v = np.asarray(values_by_unit[units[i]], dtype=float)
            if v.size:
                rows.append(rng.choice(v, v.size, replace=True))
        if rows:
            est.append(float(stat(np.concatenate(rows))))
    est = np.asarray(est)
    allrows = np.concatenate([np.asarray(v, dtype=float) for v in values_by_unit.values()])
    return {"mean": float(stat(allrows)), "interval": [float(np.percentile(est, 2.5)),
                                                       float(np.percentile(est, 97.5))],
            "n_units": len(units), "n_rows": int(allrows.size)}


def paired_hboot(a_by_unit: dict, b_by_unit: dict, rng, draws: int = 2000) -> dict:
    """Hierarchical bootstrap of a paired difference a - b, matched by unit key."""
    keys = [k for k in a_by_unit if k in b_by_unit]
    diffs = {k: np.asarray(a_by_unit[k], dtype=float) - np.asarray(b_by_unit[k], dtype=float)
             for k in keys}
    r = hboot(diffs, rng, draws)
    r["excludes_zero"] = bool(r["interval"][0] > 0 or r["interval"][1] < 0)
    return r


def overlap_with_null(effect_draws, null_draws, bins: int = 40) -> float:
    """Shared area of two empirical densities, in [0, 1]. 1 = indistinguishable."""
    a = np.asarray(effect_draws, dtype=float)
    b = np.asarray(null_draws, dtype=float)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    w = (hi - lo) / bins
    return float(np.minimum(ha, hb).sum() * w)


# --------------------------------------------------------------------------- #
# Environment, timing, verdicts.
# --------------------------------------------------------------------------- #
def environment() -> dict:
    import numpy
    import scipy
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "omp_threads": os.environ.get("OMP_NUM_THREADS")}


@contextmanager
def timed(record: dict, key: str = "runtime_seconds"):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record[key] = round(time.perf_counter() - t0, 3)


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_verdict(card_id: str, verdict: dict, gr: G.GateReport, module_file: str) -> Path:
    verdict["environment"] = environment()
    PROVENANCE.stamp(verdict, module_file, gr)
    out = verdict_dir() / f"{card_id}.json"
    out.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    marker = verdict_dir() / f"{card_id}.produced"
    marker.write_text(json.dumps({"card": card_id, "sha256": file_sha(out),
                                  "written": time.strftime("%Y-%m-%dT%H:%M:%S")}),
                      encoding="utf-8")
    return out


def load_verdict(card_id: str) -> dict | None:
    p = verdict_dir() / f"{card_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def gate_state(gr: G.GateReport) -> str:
    """The card state a gate report implies before science is read."""
    d = gr.to_dict()
    return "INSTRUMENT_VALID" if d["all_passed"] else "INSTRUMENT_FAILED"
