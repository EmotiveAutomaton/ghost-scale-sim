"""Shared plumbing for the experiment modules: CLI parsing, deterministic per-observer
seeding, and output paths. Kept deliberately small — each experiment owns its own design.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..config import Config, load_config


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def run_parallel(chunk_payloads: list, worker_fn, workers: int | None = None) -> list:
    """Run ``worker_fn(payload) -> list[record]`` over ``chunk_payloads`` and concatenate.

    ``worker_fn`` MUST be a module-level function (picklable). Observers are independent,
    so each payload typically covers one (cell, seed) chunk and builds its own model. With
    ``workers==1`` runs sequentially (used by --quick / tests / reproducibility)."""
    workers = default_workers() if workers is None else int(workers)
    if workers <= 1 or len(chunk_payloads) <= 1:
        out = []
        for payload in chunk_payloads:
            out.extend(worker_fn(payload))
        return out
    records: list = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for recs in ex.map(worker_fn, chunk_payloads):
            records.extend(recs)
    return records

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"


def standard_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to YAML config (default: config/default.yaml)")
    p.add_argument("--quick", action="store_true", help="fast smoke-scale run (dev)")
    p.add_argument("--seed", type=int, default=None, help="override base seed")
    p.add_argument("--out", default=None, help="override results output dir")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel worker processes (default: cpu-2; 1 = sequential)")
    return p


def resolve_config(args) -> Config:
    cfg = load_config(path=args.config, quick=args.quick)
    if args.seed is not None:
        cfg.set("run.base_seed", int(args.seed))
    return cfg


def observer_seed(base_seed: int, cell_index: int, seed_rep: int, observer_i: int) -> int:
    """Deterministic, collision-resistant per-observer seed. Identical inputs -> identical
    output (reproducibility invariant, Spec §10)."""
    return int((base_seed * 1_000_003
                + cell_index * 100_003
                + seed_rep * 10_007
                + observer_i) % (2**63 - 1))


def observer_rng(base_seed: int, cell_index: int, seed_rep: int, observer_i: int) -> np.random.Generator:
    return np.random.default_rng(observer_seed(base_seed, cell_index, seed_rep, observer_i))


def ensure_dirs(out_dir: Path | None = None) -> tuple[Path, Path]:
    """Resolve (results_dir, figures_dir).

    An explicit ``--out`` redirects the FIGURES too, into ``<out>/figures``. Without this a
    run directed away from ``results/`` would still overwrite the committed figures — which is
    the failure it was invoked to avoid. (Learned the hard way: a ``--quick`` run without
    ``--out`` destroyed V2's E8 outputs; see DECISIONS_V3.md, "An incident worth recording".)
    """
    if out_dir is not None:
        res = Path(out_dir)
        figs = res / "figures"
    else:
        res, figs = RESULTS_DIR, FIGURES_DIR
    res.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    return res, figs
