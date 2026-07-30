"""V-1 — does the inference approximation distort the headline results?

THE HIGHEST-PRIORITY ITEM IN THE PASS, because the failure is confirmed rather than
hypothetical. pymdp legacy's mean-field solver could not see hierarchical structure at all and
reported near-certainty while being wrong; V5's N21 measured it and worked around it by merging
the coupled factors. That fixed one case. It did not establish that anything earlier is safe.

Nothing before V5 had hierarchical depth, so there was nothing latent to penalise — but "content
whose structure the observer cannot express" sits uncomfortably close to "hypothesis carrying
latent structure", and that is exactly what the foreign-content experiments are made of.

-----------------------------------------------------------------------------------------
HOW THE COMPARISON IS MADE, AND WHY IT IS MADE THIS WAY.

Each target experiment is re-run through ITS OWN UNMODIFIED CODE, twice, with `inference.exact`
false and then true. That is the entire difference. No experiment is reimplemented here, because
a reimplementation would introduce a second thing that could have moved the number, and then the
comparison would answer a different question — which is the failure mode this whole pass exists
to catch.

Both arms run at the same reduced scale, so the scale reduction cannot be mistaken for a solver
effect. The committed full-scale approximate number is carried alongside as a third column, so a
reader can see what the scale reduction on its own did.

**Where the observation tapes stop being identical, and why that is not a defect.** The spec asks
for identical observation tapes. The artifact, the creator, the environment RNG and the observer's
prior draw are identical between the two arms by construction — the same seeds are consumed in the
same order. The tape then diverges at the first FREE step where the two solvers choose different
attention, because what the observer sees depends on how hard it looked. That divergence is not
noise to be eliminated: choosing differently IS one of the ways the approximation can distort a
result, and the forced-DEEP prefix every one of these designs already carries means the arms see
the same observations over the window where inference is being tested.

-----------------------------------------------------------------------------------------
THE SPECIFIC THING TO WATCH, quoted from the spec because it decides what happens next: the
readability axis has an interior peak where invention is highest and the collapse signature
fires, and a great deal is anchored to that location. If the peak moves under exact inference,
every claim referencing it moves with it — including the prediction card built for a human study.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from . import criteria as CR
from . import validation_dir


# --------------------------------------------------------------------------- #
# What "the headline result" means for each target, written down before the run.
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    key: str
    label: str
    # Scalars compared with a tolerance, and booleans that must match exactly.
    scalars: tuple
    booleans: tuple
    note: str


TARGETS = (
    Target("e20", "the collapse-and-invention sweep across the readability axis",
           scalars=("fabrication_peak_omega", "fabrication_peak_value",
                    "engagement_crossing_omega"),
           booleans=("fabrication_peak_is_interior", "any_cell_crashes"),
           note=("the interior peak is the load-bearing location in the project. If it moves, "
                 "every claim anchored to it moves with it.")),
    Target("e19", "sustained attention on unreadable content",
           scalars=("foreign_engaged_fraction", "foreign_final_entropy",
                    "foreign_explore_mass"),
           booleans=("positive_control_passed", "foreign_absorbed"),
           note=("the result that inverted the framework's own metabolic prediction. It is "
                 "measured on content the observer cannot express, which is the case where a "
                 "solver that penalises latent structure would be expected to misbehave.")),
    Target("e31", "the two-gates result",
           scalars=("update_tracks_mu_rho", "exploit_mu_gap", "fabrication_gap"),
           booleans=("update_tracks_recovered_depth", "dishonest_label_inflates_depth",
                     "mu_theta_dissociate_on_behaviour"),
           note=("the strongest and most legible thing in the project, and the one that becomes "
                 "the public headline. It runs on the V5 geometry, which is the geometry N21 "
                 "found the defect in.")),
    Target("e32", "the silent-versus-loud failure comparison",
           scalars=("foreign_within_at_matched", "inexpert_within_at_matched",
                    "foreign_engaged_at_matched", "inexpert_engaged_at_matched"),
           booleans=("two_dimensions",),
           note=("the most counterintuitive measured result, and it rests on a contrast between "
                 "two failures rather than on either one alone.")),
    Target("e2", "the label-induced disagreement result",
           scalars=("ghost_as_creator_within", "ghost_as_creator_between",
                    "ghost_as_ghost_within", "creator_as_creator_within"),
           booleans=("label_induces_confidence",),
           note=("the original result, from V1, on the original 8-feature geometry. It is here "
                 "because it is the oldest thing still being quoted.")),
)


# --------------------------------------------------------------------------- #
# Running one target under one solver.
# --------------------------------------------------------------------------- #
def _scaled_config(loader, cfg_keys: dict, n_obs: int, n_seeds: int, exact: bool,
                   base_cfg: Config) -> Config:
    cfg = loader()
    cfg.set("inference.exact", bool(exact))
    cfg.set("run.base_seed", int(base_cfg.run.base_seed))
    for key, value in cfg_keys.items():
        cfg.set(key, value)
    return cfg


def _run_e20(out: Path, exact: bool, cfg: Config, workers: int) -> dict:
    from ..experiments import e20_omega_sweep as E20
    from ..v4_model import load_v4_config
    c = _scaled_config(lambda: load_v4_config(include_explore=False),
                       {"experiments.e20.n_observers": _n_obs(cfg),
                        "experiments.e20.n_seeds": _n_seeds(cfg)},
                       _n_obs(cfg), _n_seeds(cfg), exact, cfg)
    E20.run(c, out_dir=out, workers=workers, make_fig=False)
    v = json.loads((out / "e20_verdict.json").read_text(encoding="utf-8"))
    stats = pd.read_csv(out / "e20_omega_sweep.csv")
    return {
        "fabrication_peak_omega": float(v["fabrication_peak_omega"]),
        "fabrication_peak_value": float(v["fabrication_peak_value"]),
        "engagement_crossing_omega": (float(v["engagement_crossing_omega"])
                                      if v.get("engagement_crossing_omega") is not None
                                      else float("nan")),
        "fabrication_peak_is_interior": bool(v["fabrication_peak_is_interior"]),
        "any_cell_crashes": bool(v["any_cell_crashes"]),
        "_grid": stats.omega.tolist(),
        "_sem": {"engagement_crossing_omega": float(np.nanmean(stats.engaged_sem_across_seeds)),
                 "fabrication_peak_value": _sem_of(stats.fabrication_index),
                 "fabrication_peak_omega": _grid_step(stats.omega.tolist())},
        "_outcome": v["outcome"],
    }


def _run_e19(out: Path, exact: bool, cfg: Config, workers: int) -> dict:
    from ..experiments import e19_explore as E19
    from ..v4_model import load_v4_config
    c = _scaled_config(lambda: load_v4_config(include_explore=True),
                       {"experiments.e19.n_observers": _n_obs(cfg),
                        "experiments.e19.n_seeds": _n_seeds(cfg)},
                       _n_obs(cfg), _n_seeds(cfg), exact, cfg)
    E19.run(c, out_dir=out, workers=workers, make_fig=False)
    v = json.loads((out / "e19_verdict.json").read_text(encoding="utf-8"))
    stats = pd.read_csv(out / "e19_cell_stats.csv")
    on_foreign = stats[(stats.arm == "explore_on") & (stats.content == "foreign")]
    row = on_foreign.iloc[0] if len(on_foreign) else stats.iloc[0]
    return {
        "foreign_engaged_fraction": float(row["engaged_fraction"]),
        "foreign_final_entropy": float(row["final_entropy"]),
        "foreign_explore_mass": float(row["explore_mass"]),
        "positive_control_passed": bool(v["positive_control_passed"]),
        "foreign_absorbed": bool(v["foreign_absorbed"]),
        "_sem": {"foreign_engaged_fraction": _sem_from_sd(row["engaged_fraction_sd"], row["n"]),
                 "foreign_final_entropy": _sem_from_sd(row["final_entropy_sd"], row["n"]),
                 "foreign_explore_mass": _sem_from_sd(row["explore_mass_sd"], row["n"])},
        "_outcome": v["verdict"],
    }


def _run_e31(out: Path, exact: bool, cfg: Config, workers: int) -> dict:
    from ..experiments import e31_two_gates as E31
    from ..v5_model import load_v5_config
    c = _scaled_config(load_v5_config,
                       {"experiments.e31.n_observers": max(20, _n_obs(cfg) // 3),
                        "experiments.e31.n_seeds": _n_seeds(cfg)},
                       _n_obs(cfg), _n_seeds(cfg), exact, cfg)
    E31.run(c, out_dir=out, workers=workers, make_fig=False)
    v = json.loads((out / "e31_verdict.json").read_text(encoding="utf-8"))
    stats = pd.read_csv(out / "e31_cell_stats.csv")
    return {
        "update_tracks_mu_rho": float(v["update_tracks_mu_rho"]),
        "exploit_mu_gap": float(v["exploit_mu_gap"]),
        "fabrication_gap": float(v["fabrication_gap"]),
        "update_tracks_recovered_depth": bool(v["update_tracks_recovered_depth"]),
        "dishonest_label_inflates_depth": bool(v["dishonest_label_inflates_depth"]),
        "mu_theta_dissociate_on_behaviour": bool(v["mu_theta_dissociate_on_behaviour"]),
        "_sem": {"exploit_mu_gap": _sem_of(stats.recovered_mu),
                 "fabrication_gap": _sem_of(stats.fabrication),
                 "update_tracks_mu_rho": float("nan")},
        "_outcome": v["verdict"],
    }


def _run_e32(out: Path, exact: bool, cfg: Config, workers: int) -> dict:
    from ..experiments import e32_omega_d as E32
    from ..v4_model import load_v4_config
    c = _scaled_config(lambda: load_v4_config(include_explore=False),
                       {"experiments.e32.n_observers": _n_obs(cfg),
                        "experiments.e32.n_seeds": _n_seeds(cfg)},
                       _n_obs(cfg), _n_seeds(cfg), exact, cfg)
    E32.run(c, out_dir=out, workers=workers, make_fig=False)
    v = json.loads((out / "e32_verdict.json").read_text(encoding="utf-8"))
    stats = pd.read_csv(out / "e32_cell_stats.csv")
    # The comparison cell: the most foreign content, and the observer matched to it. That is
    # where "fails loudly" and "fails silently" are furthest apart, so it is where a solver
    # effect on the contrast would show up first.
    om = float(stats.omega.min())
    fo = stats[(stats.arm == "foreign_content") & (stats.omega == om)].iloc[0]
    ix = stats[(stats.arm == "unskilled_reader") & (stats.omega == om)].iloc[0]
    return {
        "foreign_within_at_matched": float(fo["within_observer"]),
        "inexpert_within_at_matched": float(ix["within_observer"]),
        "foreign_engaged_at_matched": float(fo["engaged_fraction"]),
        "inexpert_engaged_at_matched": float(ix["engaged_fraction"]),
        "two_dimensions": bool(v["verdict"] == "TWO_DIMENSIONS"),
        "_sem": {k: _sem_of(stats.within_observer) for k in
                 ("foreign_within_at_matched", "inexpert_within_at_matched")}
        | {k: _sem_of(stats.engaged_fraction) for k in
           ("foreign_engaged_at_matched", "inexpert_engaged_at_matched")},
        "_outcome": v["verdict"],
    }


def _run_e2(out: Path, exact: bool, cfg: Config, workers: int) -> dict:
    from ..config import load_config
    from ..experiments import e2_variance as E2
    c = _scaled_config(load_config,
                       {"run.n_observers": _n_obs(cfg), "run.n_seeds": _n_seeds(cfg)},
                       _n_obs(cfg), _n_seeds(cfg), exact, cfg)
    E2.run(c, out_dir=out, workers=workers, make_fig=False)
    stats = pd.read_csv(out / "e2_cell_stats.csv")

    def cell(prov, sig):
        s = stats[(stats.true_provenance == prov) & (stats.declared_signal == sig)]
        return s.iloc[0]

    lie = cell("GHOST", "SIG_CREATOR")
    honest = cell("GHOST", "SIG_GHOST")
    human = cell("CREATOR", "SIG_CREATOR")
    n_cells = int(len(stats))
    return {
        "ghost_as_creator_within": float(lie["within"]),
        "ghost_as_creator_between": float(lie["between"]),
        "ghost_as_ghost_within": float(honest["within"]),
        "creator_as_creator_within": float(human["within"]),
        # The result in one boolean: a false human label leaves the reader far MORE certain
        # about machine content than an honest label does, on the same content.
        "label_induces_confidence": bool(float(lie["within"]) < float(honest["within"])),
        "_sem": {"ghost_as_creator_within": _sem_from_sd(lie["within_sd"], n_cells),
                 "ghost_as_creator_between": _sem_from_sd(lie["between_sd"], n_cells),
                 "ghost_as_ghost_within": _sem_from_sd(honest["within_sd"], n_cells),
                 "creator_as_creator_within": _sem_from_sd(human["within_sd"], n_cells)},
        "_outcome": "LABEL_INDUCES_CONFIDENCE" if float(lie["within"]) < float(honest["within"])
                    else "NO_LABEL_EFFECT",
    }


RUNNERS = {"e20": _run_e20, "e19": _run_e19, "e31": _run_e31,
           "e32": _run_e32, "e2": _run_e2}


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _n_obs(cfg: Config) -> int:
    return int(cfg.get("validation.n_observers", 60))


def _n_seeds(cfg: Config) -> int:
    return int(cfg.get("validation.n_seeds", 12))


def _sem_of(series) -> float:
    a = np.asarray(series, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan")


def _sem_from_sd(sd, n) -> float:
    try:
        n = float(n)
        return float(sd) / float(np.sqrt(n)) if n > 0 and np.isfinite(float(sd)) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _grid_step(grid) -> float:
    g = sorted(float(x) for x in grid)
    return float(np.min(np.diff(g))) if len(g) > 1 else float("nan")


def _committed(key: str) -> dict:
    """The full-scale approximate numbers already in results/, for the third column."""
    from ..experiments._common import RESULTS_DIR
    p = RESULTS_DIR / f"{key}_verdict.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


# --------------------------------------------------------------------------- #
# The check.
# --------------------------------------------------------------------------- #
def run(cfg: Config, workers: int = 1, only: tuple = ()) -> dict:
    from ..exact import assert_agrees_on_factorised_case

    out_root = validation_dir() / "v1"
    out_root.mkdir(parents=True, exist_ok=True)

    # Before anything is compared: the exact solver must agree with mean field on a construction
    # where mean field is provably exact. If it does not, every number below is meaningless and
    # the failure is in exact.py rather than in the model.
    agreement = assert_agrees_on_factorised_case()

    rows, per_target = [], {}
    targets = [t for t in TARGETS if not only or t.key in only]
    for t in targets:
        got = {}
        for arm, exact in (("approx", False), ("exact", True)):
            out = out_root / arm
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            got[arm] = RUNNERS[t.key](out, exact, cfg, workers)
            got[arm]["_seconds"] = round(time.time() - t0, 1)

        committed = _committed(t.key)
        target_rows = []
        for name in t.scalars:
            a, e = float(got["approx"][name]), float(got["exact"][name])
            sem = float(got["approx"].get("_sem", {}).get(name, float("nan")))
            if t.key == "e20" and name == "fabrication_peak_omega":
                # Judged in GRID STEPS, per the pre-registered tolerance, not in standard errors.
                step = got["approx"]["_sem"]["fabrication_peak_omega"]
                agrees = bool(abs(a - e) <= CR.V1_PEAK_TOLERANCE_STEPS * step + 1e-12)
            else:
                agrees = CR.within_spread(e, a, sem)
            target_rows.append({
                "target": t.key, "quantity": name, "kind": "scalar",
                "approx_reduced": a, "exact_reduced": e,
                "committed_full_scale": _pluck(committed, name),
                "sem_across_cells": sem, "agrees": agrees,
            })
        for name in t.booleans:
            a, e = bool(got["approx"][name]), bool(got["exact"][name])
            target_rows.append({
                "target": t.key, "quantity": name, "kind": "boolean",
                "approx_reduced": a, "exact_reduced": e,
                "committed_full_scale": _pluck(committed, name),
                "sem_across_cells": float("nan"), "agrees": bool(a == e),
            })
        rows.extend(target_rows)
        per_target[t.key] = {
            "label": t.label,
            "note": t.note,
            "outcome_approx": got["approx"].get("_outcome"),
            "outcome_exact": got["exact"].get("_outcome"),
            "outcome_survives": got["approx"].get("_outcome") == got["exact"].get("_outcome"),
            "seconds_approx": got["approx"]["_seconds"],
            "seconds_exact": got["exact"]["_seconds"],
            "quantities": target_rows,
        }

    table = pd.DataFrame(rows)
    table.to_csv(out_root / "v1_side_by_side.csv", index=False)

    booleans_match = bool(table[table.kind == "boolean"].agrees.all()) if len(table) else False
    scalars_within = bool(table[table.kind == "scalar"].agrees.all()) if len(table) else False
    peak = table[(table.target == "e20") & (table.quantity == "fabrication_peak_omega")]
    peak_holds = bool(peak.agrees.all()) if len(peak) else None

    verdict = {
        "check": "V-1",
        "question": "Does the inference approximation distort the headline results?",
        "plain_language": (
            "Every experiment before version 5 used a fast approximate way of updating the "
            "reader's beliefs. That shortcut is known to have been badly wrong once, in a case "
            "version 5 caught. This check re-runs the headline experiments with the shortcut "
            "removed, using exact arithmetic over every combination of possibilities, and puts "
            "the two answers next to each other."),
        "solver_sanity_check": agreement,
        "scale": {"n_observers": _n_obs(cfg), "n_seeds": _n_seeds(cfg),
                  "note": ("both arms at the same reduced scale, so the reduction cannot be "
                           "mistaken for a solver effect; the committed full-scale number is "
                           "carried as a third column")},
        "criteria": {"peak_tolerance_steps": CR.V1_PEAK_TOLERANCE_STEPS,
                     "spread_standard_errors": CR.V1_SPREAD_SE,
                     "booleans_must_match_exactly": True},
        "interior_peak_holds": peak_holds,
        "all_booleans_match": booleans_match,
        "all_scalars_within_spread": scalars_within,
        "targets": per_target,
        "table": rows,
    }
    verdict["verdict"] = _v1_verdict(peak_holds, booleans_match, scalars_within)
    verdict["statement"] = _v1_statement(verdict)
    (validation_dir() / "v1_solver.json").write_text(
        json.dumps(verdict, indent=2, default=_jsonable), encoding="utf-8")
    return verdict


def _pluck(committed: dict, name: str):
    if name in committed:
        return committed[name]
    return None


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return float(o) if isinstance(o, (int, float)) else str(o)


def _v1_verdict(peak_holds, booleans_match, scalars_within) -> str:
    """Branches written before the run, per the spec's §6 requirement."""
    if peak_holds is False:
        return "PEAK_MOVED_RE_ANCHOR_EVERYTHING"
    if not booleans_match:
        return "A_VERDICT_FLIPPED_UNDER_EXACT_INFERENCE"
    if not scalars_within:
        return "VERDICTS_SURVIVE_MAGNITUDES_MOVE"
    return "APPROXIMATION_IS_NOT_DRIVING_THE_RESULTS"


def _v1_statement(v: dict) -> str:
    if v["verdict"] == "PEAK_MOVED_RE_ANCHOR_EVERYTHING":
        return ("The interior peak on the readability axis is in a different place under exact "
                "inference. Every claim anchored to that location, including the prediction "
                "card written for a human study, has to be re-anchored, and the README says so "
                "above the table.")
    if v["verdict"] == "A_VERDICT_FLIPPED_UNDER_EXACT_INFERENCE":
        flipped = [r["target"] + "." + r["quantity"] for r in v["table"]
                   if r["kind"] == "boolean" and not r["agrees"]]
        return ("At least one verdict is a property of the approximation rather than of the "
                f"model: {', '.join(flipped)}. Those claims are reported with the failure "
                "attached, in the same cell as the claim.")
    if v["verdict"] == "VERDICTS_SURVIVE_MAGNITUDES_MOVE":
        moved = [r["target"] + "." + r["quantity"] for r in v["table"]
                 if r["kind"] == "scalar" and not r["agrees"]]
        return ("Every verdict survives exact inference, but some magnitudes move by more than "
                f"their own run-to-run spread: {', '.join(moved)}. The shapes are the claims; "
                "the specific numbers are reported as solver-dependent.")
    return ("Every headline verdict and every headline magnitude survives the removal of the "
            "approximation. The interior peak on the readability axis does not move. The "
            "mean-field shortcut is not what produced these results.")
