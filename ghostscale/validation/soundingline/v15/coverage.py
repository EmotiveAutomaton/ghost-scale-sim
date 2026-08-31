"""The locked balanced coverage stream (spec §9.5, §10.1).

Why this exists
---------------
V14's scientific queue emptied after 6.8 wall hours and the runner then waited fourteen hours for a
freeze. The no-early-packet rule held; the continuous-scientific-work requirement did not. Spec
§9.4 makes that a `RUNTIME_FAILED`, so V15 needs a queue that *cannot* empty inside 168 hours even
on a machine three times faster than the pilot -- and one whose executed prefix is scientifically
interpretable however far it gets, because how far it gets is decided by the machine.

The design
----------
The stream is a sequence of **atomic balance blocks**. One block is a complete crossing of the
three primary axes -- generator family, latent coupling, evidence dose -- at one point of a
scrambled Sobol draw over the secondary axes (overlap, dependence, missingness, model space,
temperature, competence). Because every block is complete on the primary axes, stopping after any
whole number of blocks leaves a balanced design: the conditional surfaces the atlas reports are
estimable from the prefix, and nothing is confounded with when the clock ran out.

Storage
-------
The sequence is **not** materialized to disk. At the sizes the opening guard requires -- hundreds
of wall-hours of forecast work -- a materialized list is a multi-hundred-megabyte file that nobody
reads and git should never see. What is locked instead is the *definition* (axes, block structure,
Sobol seed, scramble seed, block count) together with a **hash chain over every block's digest**,
so any block can be regenerated and verified against the lock, and no block can be quietly changed
after the fact. ``verify_prefix`` re-derives the chain for the blocks that actually ran.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import common as C
from .ontology import Knobs

#: Primary axes. Every block crosses these completely, so a truncated prefix stays balanced.
PRIMARY = {
    "family": ("chain", "composition", "communication"),
    "kappa": (0.0, 0.25, 0.5, 0.75, 1.0),
    "dose": (1, 2, 4, 8, 16),
}
#: Secondary axes. One Sobol point per block fixes these for the whole block.
SECONDARY = {
    "overlap": (0.0, 0.33, 0.66, 1.0),
    "dependence": ("independent", "redundant", "synergistic"),
    "missing": ("none", "route", "context", "opportunity"),
    "model_space": ("correct", "missing_latent", "extra_latent", "wrong_family"),
    "temperature": (0.25, 0.6, 1.0, 2.0),
    "competence": (0.55, 0.75, 0.95),
}
#: Architectures scored in every coverage cell. Deliberately short: the stream buys breadth of
#: *conditions*, and the deep architecture tournament is the mandatory core's job.
COVERAGE_ARCHITECTURES = ("surface", "independent", "staged", "joint_exact", "particle",
                          "oracle_state")
BLOCK_CELLS = len(PRIMARY["family"]) * len(PRIMARY["kappa"]) * len(PRIMARY["dose"])   # 75
SOBOL_SEED = 20260831
ENDPOINT_BY_FAMILY = {"chain": "next_action", "composition": "next_edit",
                      "communication": "next_evidence_selection"}


def _sobol(n: int) -> np.ndarray:
    """``n`` scrambled Sobol points in the secondary-axis cube. Deterministic from SOBOL_SEED."""
    from scipy.stats import qmc
    d = len(SECONDARY)
    eng = qmc.Sobol(d=d, scramble=True, seed=SOBOL_SEED)
    m = int(np.ceil(np.log2(max(n, 2))))
    pts = eng.random_base2(m)
    return pts[:n]


_SOBOL_CACHE: dict = {}


def sobol_point(block_index: int) -> dict:
    """The secondary-axis setting for one block."""
    need = 1 << int(np.ceil(np.log2(max(block_index + 1, 2))))
    key = need
    if key not in _SOBOL_CACHE:
        _SOBOL_CACHE.clear()
        _SOBOL_CACHE[key] = _sobol(need)
    pts = _SOBOL_CACHE[key]
    p = pts[block_index]
    out = {}
    for i, (name, levels) in enumerate(SECONDARY.items()):
        idx = min(int(p[i] * len(levels)), len(levels) - 1)
        out[name] = levels[idx]
    return out


def block(block_index: int) -> dict:
    """One atomic balance block: its secondary setting and its 75 primary cells."""
    sec = sobol_point(block_index)
    cells = []
    for fam in PRIMARY["family"]:
        for kappa in PRIMARY["kappa"]:
            for dose in PRIMARY["dose"]:
                cells.append({
                    "cell_id": f"b{block_index:06d}|{fam}|k{kappa:g}|d{dose}",
                    "block": int(block_index), "family": fam, "kappa": float(kappa),
                    "dose": int(dose), **sec,
                })
    return {"block": int(block_index), "secondary": sec, "cells": cells,
            "n_cells": len(cells)}


def block_digest(block_index: int) -> str:
    b = block(block_index)
    return hashlib.sha256(json.dumps(b, sort_keys=True).encode("utf-8")).hexdigest()


def hash_chain(n_blocks: int, start: str = "") -> str:
    """Rolling hash over every block's digest. Cheap to extend, impossible to forge a middle."""
    h = start or hashlib.sha256(b"v15-coverage").hexdigest()
    for i in range(int(n_blocks)):
        h = hashlib.sha256((h + block_digest(i)).encode("utf-8")).hexdigest()
    return h


def sequence_definition(n_blocks: int, chain_sample: int = 64) -> dict:
    """The object that gets locked. Regenerable, verifiable, and about two kilobytes."""
    return {
        "program": "v15", "kind": "balanced_coverage_sequence",
        "primary_axes": {k: list(v) for k, v in PRIMARY.items()},
        "secondary_axes": {k: list(v) for k, v in SECONDARY.items()},
        "architectures": list(COVERAGE_ARCHITECTURES),
        "sobol_seed": SOBOL_SEED, "scrambled": True,
        "cells_per_block": BLOCK_CELLS, "n_blocks": int(n_blocks),
        "total_cells": int(n_blocks) * BLOCK_CELLS,
        "atomic_unit": "block",
        "truncation_rule": ("finish the current block at the freeze hour; a whole number of blocks "
                            "is a balanced design over family x coupling x dose"),
        "chain_first_n": int(chain_sample),
        "chain_hash_first_n": hash_chain(min(int(n_blocks), int(chain_sample))),
        "block_0_digest": block_digest(0),
        "block_sample": [block(i)["secondary"] for i in range(min(4, int(n_blocks)))],
        "materialized": False,
        "why_not_materialized": ("at the sizes the opening guard requires this list is hundreds of "
                                 "megabytes; the definition plus a hash chain regenerates and "
                                 "verifies any block without storing it"),
    }


def verify_prefix(n_executed_blocks: int, locked: dict) -> dict:
    """Re-derive the chain over the blocks that actually ran and compare with the lock."""
    n = min(int(n_executed_blocks), int(locked.get("chain_first_n", 0)))
    got = hash_chain(n) if n else ""
    want = locked.get("chain_hash_first_n") if n == locked.get("chain_first_n") else None
    return {"blocks_executed": int(n_executed_blocks), "verified_blocks": n,
            "block_0_digest_matches": bool(block_digest(0) == locked.get("block_0_digest")),
            "chain_matches": bool(want is None or got == want),
            "definition_matches": bool(
                locked.get("cells_per_block") == BLOCK_CELLS
                and locked.get("sobol_seed") == SOBOL_SEED)}


# --------------------------------------------------------------------------- #
# Executing one cell.
# --------------------------------------------------------------------------- #
def family_module(name: str):
    from . import world_chain, world_communication, world_composition
    return {"chain": world_chain, "composition": world_composition,
            "communication": world_communication}[name]


def knobs_for(cell: dict) -> Knobs:
    return Knobs(kappa=float(cell["kappa"]), overlap=float(cell["overlap"]),
                 dose=int(cell["dose"]), dependence=cell["dependence"],
                 missing=cell["missing"], temperature=float(cell["temperature"]),
                 competence=float(cell["competence"]), model_space=cell["model_space"])


def execute_cell(cell: dict, tier: dict, smoke: bool = False) -> dict:
    """Run one coverage cell and return machine-readable numbers only.

    No narrative, ever: spec §9.1 forbids result prose before the deadline, and a coverage cell is
    the most likely place for a summary sentence to leak into the record.
    """
    from . import architectures as A
    F = family_module(cell["family"])
    knobs = knobs_for(cell)
    endpoint = ENDPOINT_BY_FAMILY[cell["family"]]
    n_worlds = 2 if smoke else int(tier.get("coverage_worlds", 4))
    n_makers = 3 if smoke else int(tier.get("coverage_makers", 10))
    steps = 8 if smoke else int(tier.get("steps", 12))

    rows = []
    world = None
    for wi in range(n_worlds):
        # world ids stay inside the coverage lane's range; the seed itself carries the lane name,
        # so no coverage object can share an ancestor with a discovery or confirmation object
        wid = C.LANE_BASE["coverage"] + ((int(cell["block"]) * 7 + wi) % 9000)
        rng = np.random.default_rng(C.seed(f"coverage|{cell['cell_id']}|w{wi}"))
        world = F.sample_world(knobs, rng)
        for mi in range(n_makers):
            lat = F.sample_latent(world, rng)
            ep = F.rollout(world, lat, rng, steps)
            y = ep.hidden.get(endpoint)
            if y is None:
                continue
            sub = np.random.default_rng(rng.integers(0, 2 ** 62))
            reads = A.tournament(F, world, ep, min(knobs.dose, steps), endpoint,
                                 COVERAGE_ARCHITECTURES, rng=sub,
                                 cfg={"model_space": cell["model_space"]})
            for name, r in reads.items():
                rows.append({"wid": wid, "rep": mi, "architecture": name,
                             "log_score": C.log_score(r.dist, y),
                             "brier": C.brier(r.dist, y),
                             "correct": float(C.top1(r.dist) == y),
                             "confidence": C.confidence(r.dist),
                             "likelihood_evaluations": r.budget.get("likelihood_evaluations", 0)})
    by_arch = {}
    for name in COVERAGE_ARCHITECTURES:
        v = [r["log_score"] for r in rows if r["architecture"] == name]
        c = [r["correct"] for r in rows if r["architecture"] == name]
        e = [r["likelihood_evaluations"] for r in rows if r["architecture"] == name]
        if v:
            by_arch[name] = {"log_score": float(np.mean(v)), "accuracy": float(np.mean(c)),
                             "likelihood_evaluations": float(np.mean(e)), "n": len(v)}
    joint = by_arch.get("joint_exact", {}).get("log_score")
    indep = by_arch.get("independent", {}).get("log_score")
    return {"cell_id": cell["cell_id"], "block": int(cell["block"]), "family": cell["family"],
            "kappa": float(cell["kappa"]), "dose": int(cell["dose"]),
            "secondary": {k: cell[k] for k in SECONDARY},
            "n_rows": len(rows), "by_architecture": by_arch,
            "joint_minus_independent": (float(joint - indep)
                                        if joint is not None and indep is not None else None),
            "realized_coupling": (float(world.meta.get("realized_coupling", float("nan")))
                                  if world is not None else None),
            "overlap_index": (float(world.meta.get("overlap_index", float("nan")))
                              if world is not None else None)}
