"""V-3 — are the headline results knife-edge or robust?

**THIS IS A SCOPING EXERCISE, NOT A PASS/FAIL**, and that framing is the whole point. A result
that holds across an entire parameter range and a result that holds in a narrow window are both
real. They are DIFFERENT CLAIMS, and the README has to say which is which. What V-3 produces is
the information needed to say it.

One matrix: result by parameter, each cell recording whether the verdict holds, weakens or flips.

-----------------------------------------------------------------------------------------
THE ONE PASS/FAIL CLAUSE, and it is about honesty rather than about robustness. Any result that
survives only in a window narrower than the range actually explored during development is reported
as TUNED. That is a real risk here: several defaults in this project were recalibrated on contact
with the implementation, each logged as a deviation, and "we tried values until it worked" and "the
effect holds over a wide range" look identical from outside if nobody sweeps.

-----------------------------------------------------------------------------------------
WHAT IS SWEPT, AND THE COST DECISION BEHIND WHICH HEADLINES GET WHICH PARAMETERS.

Two headlines are swept, chosen because they are the two the public-facing material leans on and
because they sit on different geometries:

* **the label effect** (E2's design, 8 features): the same machine-made content read as certain or
  uncertain depending only on the declared label. Cheap, so it gets every parameter.
* **the interior peak** (E20's design, 16 features): where on the readability axis confident
  invention is highest. Roughly an order of magnitude more expensive per run, because it sweeps
  eight overlap levels, so it gets the parameters that plausibly move a peak LOCATION and not the
  ones that only scale a magnitude.

Cells that were not run are recorded as ``not_run`` with the reason, rather than left out. A blank
in a robustness matrix reads as "fine" to every reader who does not check, which is the same
failure this pass exists to catch.

ONE PARAMETER AT A TIME, and the limitation stated: this finds knife edges along the axes, not
knife edges along diagonals. A full factorial over nine parameters at three levels is 19,683 runs
of each headline and is not affordable; the axis-aligned sweep is what fits, and interactions
remain unmeasured. That is a real gap and it is named here rather than in a footnote.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from . import criteria as CR
from . import validation_dir
from .v2_nulls import _label_effect_from_e2


# --------------------------------------------------------------------------- #
# The parameters, their ranges, and why each range is the defensible one.
# --------------------------------------------------------------------------- #
@dataclass
class Sweep:
    name: str
    plain: str
    levels: tuple            # (label, setter-payload) pairs; the middle one is the default
    apply: str = "scalar"    # "scalar" -> cfg.set(key, value); "custom" -> a named handler
    key: str = ""
    headlines: tuple = ("label_effect", "interior_peak")
    why: str = ""


SWEEPS = (
    Sweep("effort_cost", "how much it costs the reader to look closely",
          key="preferences.c_effort",
          levels=(("low", 0.05), ("default", None), ("high", 0.75)),
          why=("the low and high ends are the ends of E3's own committed sweep, so the range is "
               "the one the project already explored rather than one chosen here")),
    Sweep("label_precision", "how much the reader trusts the label",
          key="signal_model.kappa",
          levels=(("low", 0.30), ("default", None), ("high", 0.99)),
          why=("E4 swept kappa from 0 to 0.99 and located a threshold near 0.20; the low end sits "
               "above that threshold deliberately, so this measures robustness inside the regime "
               "the claim is made in rather than the already-published collapse below it")),
    Sweep("planning_horizon", "how far ahead the reader plans",
          key="agent.policy_len",
          levels=(("myopic", 1), ("default", None), ("longer", 3)),
          why=("policy_len = 1 collapses the attention dynamics by the config's own note, so it "
               "is included precisely because it is the adversarial end")),
    Sweep("prior_strength", "how strong the reader's starting hunch is",
          key="priors.eps",
          levels=(("weak", 0.01), ("default", None), ("strong", 0.50)),
          why=("reader heterogeneity is what the between-reader measure is made of, so this is "
               "the parameter most likely to move a disagreement number")),
    Sweep("separation_floor", "how distinguishable the maker's possible purposes are",
          key="artifact_model.sig_peak_mass",
          levels=(("shallow", 0.60), ("default", None), ("sharp", 0.98)),
          why=("the peak mass is what the pairwise Jensen-Shannon floor is asserted against; "
               "0.60 is close to where that assertion starts to bind")),
    Sweep("opacity_ramp", "the Ghost Scale opacity values read as recoverable intent",
          apply="custom", key="alpha_ramp",
          levels=(("compressed", 0.5), ("default", None), ("stretched", 1.0)),
          why=("this is the framework's one empirical commitment, so a result that depends on the "
               "exact published values rather than on their ordering is a much weaker claim. "
               "Compressed pulls all four tiers halfway to their mean; stretched pushes them to "
               "the full 0 - 1 range")),
    Sweep("observer_count", "how many readers are simulated",
          apply="custom", key="observers",
          levels=(("half", 0.5), ("default", None), ("double", 2.0)),
          headlines=("label_effect",),
          why=("included because the between-reader measures are the ones whose estimates depend "
               "on the number of readers; it is a power check wearing a robustness hat, and V-7 "
               "does the same job at a different seed block")),
    Sweep("goal_count", "how many purposes the reader considers",
          apply="custom", key="num_goals",
          levels=(("two", 2), ("default", None), ("default_again", 4)),
          headlines=("label_effect",),
          why=("the disagreement ceiling is ln(goals), so this changes the yardstick as well as "
               "the measurement, and the verdict has to survive that. Only the downward "
               "direction is available: the feature partition tiles eight features into four "
               "pairs, so more goals needs more features, which is the next sweep")),
    Sweep("feature_count", "how much detail the work carries",
          apply="custom", key="num_features",
          levels=(("eight", 8), ("default", None), ("sixteen", 16)),
          headlines=("label_effect",),
          why=("the whole V4 reframe needed the feature space doubled to get a disjoint "
               "partition, which makes the feature count a parameter with a documented history "
               "of mattering")),
)

# What V-3 does NOT sweep, said explicitly. The number of encounters is E31's parameter and E31 is
# not one of V-3's two headlines; sweeping it here would produce a column of not-run cells that
# looks like coverage.
NOT_SWEPT = {
    "number_of_encounters": ("belongs to the sequential designs (E29, E31), which are not among "
                             "V-3's two swept headlines; V-1 covers E31 under the solver check "
                             "and its own pre-registration covers the encounter count"),
    "interactions_between_parameters": ("the sweep is axis-aligned. A full factorial is not "
                                        "affordable and diagonal knife edges are therefore "
                                        "unmeasured"),
}


# --------------------------------------------------------------------------- #
# Applying one level.
# --------------------------------------------------------------------------- #
def _pairs_for(num_goals: int, num_features: int) -> list:
    """Goal/feature pairs that tile the space, for the cardinality sweeps."""
    return [[2 * g, 2 * g + 1] for g in range(num_goals)]


def _apply(cfg: Config, sweep: Sweep, value, n_obs: int, n_seeds: int) -> tuple:
    """Mutate ``cfg`` for one level and return the (n_observers, n_seeds) it should run at."""
    if value is None:
        return n_obs, n_seeds
    if sweep.apply == "scalar":
        cfg.set(sweep.key, value)
        return n_obs, n_seeds
    if sweep.key == "alpha_ramp":
        from .. import constants as K
        raw = dict(cfg.artifact_model.alpha.raw)
        vals = np.array([float(raw[n]) for n in K.PROVENANCE_NAMES])
        if value == 0.5:                      # compressed: halfway to the mean
            new = vals.mean() + 0.5 * (vals - vals.mean())
        else:                                 # stretched: rescaled to the full [0, 1] range
            lo, hi = vals.min(), vals.max()
            new = (vals - lo) / (hi - lo)
        for name, v in zip(K.PROVENANCE_NAMES, new):
            cfg.set(f"artifact_model.alpha.{name}", float(np.clip(v, 0.0, 1.0)))
        return n_obs, n_seeds
    if sweep.key == "observers":
        return max(4, int(round(n_obs * float(value)))), n_seeds
    if sweep.key == "num_goals":
        g = int(value)
        cfg.set("cardinalities.num_goals", g)
        cfg.set("artifact_model.goal_feature_pairs",
                _pairs_for(g, int(cfg.cardinalities.num_features)))
        cfg.set("experiments.e2.true_goal", min(1, g - 1))
        return n_obs, n_seeds
    if sweep.key == "num_features":
        f = int(value)
        cfg.set("cardinalities.num_features", f)
        cfg.set("artifact_model.goal_feature_pairs",
                _pairs_for(int(cfg.cardinalities.num_goals), f))
        # The structured-content ceiling is stated relative to uniform entropy, so it has to move
        # with the feature count or the assertion means something different at each level.
        cfg.set("artifact_model.structured_ceiling", float(0.87 * np.log(f)))
        return n_obs, n_seeds
    raise ValueError(f"no handler for sweep {sweep.name}")


# --------------------------------------------------------------------------- #
# Scoring one headline at one level.
# --------------------------------------------------------------------------- #
def _score_label_effect(out: Path, cfg: Config, workers: int, n_obs: int, n_seeds: int) -> dict:
    from ..experiments import e2_variance as E2
    cfg.set("run.n_observers", int(n_obs))
    cfg.set("run.n_seeds", int(n_seeds))
    E2.run(cfg, out_dir=out, workers=workers, make_fig=False)
    eff = _label_effect_from_e2(out)
    return {"primary": float(eff["honest_doubt_multiple"]),
            "primary_name": "how many times more doubt an honest label leaves",
            "reportable": bool(eff["reportable"]),
            "detail": eff}


def _score_interior_peak(out: Path, cfg: Config, workers: int, n_obs: int, n_seeds: int) -> dict:
    from ..experiments import e20_omega_sweep as E20
    cfg.set("experiments.e20.n_observers", int(n_obs))
    cfg.set("experiments.e20.n_seeds", int(n_seeds))
    E20.run(cfg, out_dir=out, workers=workers, make_fig=False)
    v = json.loads((out / "e20_verdict.json").read_text(encoding="utf-8"))
    # The primary quantity is the PEAK HEIGHT, because a robustness label needs something
    # continuous to compare. The peak LOCATION is carried alongside and is what the tuned-window
    # judgement is made on, since that is the quantity the project's claims are anchored to.
    return {"primary": float(v["fabrication_peak_value"]),
            "primary_name": "how much confident invention at the peak",
            "peak_omega": float(v["fabrication_peak_omega"]),
            "peak_is_interior": bool(v["fabrication_peak_is_interior"]),
            "reportable": bool(v["fabrication_peak_is_interior"]),
            "detail": {"outcome": v["outcome"]}}


HEADLINES = {
    "label_effect": {
        "plain": ("the same machine-made content read as certain or uncertain depending only on "
                  "what the label says"),
        "loader": "base",
        "scorer": _score_label_effect,
        "seed_divisor": 1,
    },
    "interior_peak": {
        "plain": ("confident invention peaks in the middle of the readability axis, not at the "
                  "unreadable end"),
        "loader": "v4",
        "scorer": _score_interior_peak,
        # Half the seeds: E20 sweeps eight overlap levels, so a full-seed sweep of nine parameters
        # at three levels does not fit. Recorded rather than silently applied.
        "seed_divisor": 2,
    },
}


def _load(kind: str) -> Config:
    from ..config import load_config
    from ..v4_model import load_v4_config
    cfg = load_config() if kind == "base" else load_v4_config(include_explore=False)
    cfg.set("inference.exact", True)
    return cfg


# --------------------------------------------------------------------------- #
# The matrix.
# --------------------------------------------------------------------------- #
def run(cfg: Config, workers: int = 1) -> dict:
    out_root = validation_dir() / "v3"
    out_root.mkdir(parents=True, exist_ok=True)
    n_obs = int(cfg.get("validation.n_observers", 60))
    n_seeds = int(cfg.get("validation.n_seeds", 12))

    rows, baselines = [], {}
    for hname, spec in HEADLINES.items():
        seeds = max(2, n_seeds // spec["seed_divisor"])
        base_out = out_root / hname / "default"
        base_out.mkdir(parents=True, exist_ok=True)
        baselines[hname] = spec["scorer"](base_out, _load(spec["loader"]), workers, n_obs, seeds)

    for sweep in SWEEPS:
        for hname, spec in HEADLINES.items():
            if hname not in sweep.headlines:
                rows.append({
                    "headline": hname, "parameter": sweep.name, "level": "-",
                    "value": None, "primary": None, "label": "not_run",
                    "reason": (f"{sweep.name} is swept for "
                               f"{', '.join(sweep.headlines)} only; see the sweep's own note"),
                })
                continue
            seeds = max(2, n_seeds // spec["seed_divisor"])
            for level_name, value in sweep.levels:
                if value is None:
                    rows.append({
                        "headline": hname, "parameter": sweep.name, "level": level_name,
                        "value": "default", "primary": baselines[hname]["primary"],
                        "label": "holds", "reason": "the default; the reference cell",
                    })
                    continue
                out = out_root / hname / f"{sweep.name}__{level_name}"
                out.mkdir(parents=True, exist_ok=True)
                c = _load(spec["loader"])
                obs_here, seeds_here = _apply(c, sweep, value, n_obs, seeds)
                try:
                    got = spec["scorer"](out, c, workers, obs_here, seeds_here)
                    label = CR.robustness_label(baselines[hname]["primary"], got["primary"])
                    # A cell whose verdict-level boolean fails is a FLIP whatever the magnitude
                    # did: a matrix that scores only magnitudes can report "holds" for a cell
                    # where the claim has stopped being true.
                    #
                    # THE GATE IS CONDITIONAL ON THE BASELINE PASSING IT, and that is not a
                    # loophole. The boolean clause is a threshold on an absolute quantity, so at a
                    # reduced scale it can fail in the DEFAULT cell too — the disagreement clause
                    # needs enough readers for the between-reader entropy to reach its ceiling.
                    # Applying an absolute gate the reference cell also fails would mark every
                    # cell as a flip and the matrix would carry no information at all. When the
                    # baseline fails its own gate the matrix says so, in ``boolean_gate_applied``,
                    # rather than quietly scoring magnitudes only.
                    if baselines[hname].get("reportable", True) and not got.get("reportable", True):
                        label = "flips"
                    row = {
                        "headline": hname, "parameter": sweep.name, "level": level_name,
                        "value": value, "primary": got["primary"], "label": label,
                        "reason": "",
                    }
                    if "peak_omega" in got:
                        row["peak_omega"] = got["peak_omega"]
                    rows.append(row)
                except (AssertionError, ValueError, KeyError, IndexError) as exc:
                    # A construction assertion firing IS a result: it says the parameter cannot
                    # take that value without violating something the model asserts about itself.
                    rows.append({
                        "headline": hname, "parameter": sweep.name, "level": level_name,
                        "value": value, "primary": None, "label": "unreachable",
                        "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
                    })

    table = pd.DataFrame(rows)
    table.to_csv(out_root / "v3_matrix.csv", index=False)

    summary = {}
    for hname in HEADLINES:
        sub = table[(table.headline == hname) & (table.label.isin(
            ["holds", "weakens", "flips", "unreachable"]))]
        run_cells = sub[sub.label != "unreachable"]
        n_run = int(len(run_cells))
        n_holds = int((run_cells.label == "holds").sum())
        # THE TUNED JUDGEMENT. A result is tuned if it holds in fewer than this fraction of the
        # cells actually reachable — which is the operational version of "survives only in a
        # window narrower than the range that was explored during development".
        tuned = bool(n_run > 0 and (n_holds / n_run) < CR.V3_TUNED_WINDOW_FRACTION)
        summary[hname] = {
            "plain_language": HEADLINES[hname]["plain"],
            "baseline_primary": baselines[hname]["primary"],
            "baseline_primary_name": baselines[hname]["primary_name"],
            "cells_run": n_run,
            "holds": n_holds,
            "weakens": int((run_cells.label == "weakens").sum()),
            "flips": int((run_cells.label == "flips").sum()),
            "unreachable": int((sub.label == "unreachable").sum()),
            "flipped_by": run_cells[run_cells.label == "flips"].parameter.tolist(),
            "weakened_by": run_cells[run_cells.label == "weakens"].parameter.tolist(),
            "reported_as_tuned": tuned,
            "boolean_gate_applied": bool(baselines[hname].get("reportable", True)),
            "boolean_gate_note": (
                "the verdict-level boolean was checked in every cell"
                if baselines[hname].get("reportable", True) else
                "the verdict-level boolean fails in the DEFAULT cell at this reduced scale, so "
                "it was not used to score the other cells; the magnitudes below are the whole of "
                "what this row measures and the row is weaker for it"),
            "scope_sentence": _scope_sentence(hname, n_holds, n_run, run_cells, tuned),
        }
        if hname == "interior_peak":
            peaks = run_cells.get("peak_omega")
            if peaks is not None:
                vals = sorted(set(float(x) for x in peaks.dropna().tolist()))
                summary[hname]["peak_locations_observed"] = vals
                summary[hname]["peak_location_is_stable"] = bool(len(vals) <= 1)

    verdict = {
        "check": "V-3",
        "question": "Are the headline results knife-edge or robust?",
        "framing": ("A scoping exercise, not a pass/fail. A result that holds across the whole "
                    "range and a result that holds in a narrow window are both real; they are "
                    "different claims, and the point of this matrix is to make the README able "
                    "to say which is which."),
        "plain_language": (
            "Every setting in the model was chosen by somebody. This check moves each one to a "
            "low and a high value in turn and asks whether the finding is still there. A finding "
            "that only appears at one setting is a much smaller claim than a finding that "
            "survives the whole range, and both get described accurately."),
        "scale": {"n_observers": n_obs, "n_seeds": n_seeds,
                  "note": "the readability-axis headline runs at half the seeds; see HEADLINES"},
        "criteria": {"weaken_fraction": CR.V3_WEAKEN_FRACTION,
                     "tuned_window_fraction": CR.V3_TUNED_WINDOW_FRACTION},
        "sweeps": [{"parameter": s.name, "plain_language": s.plain,
                    "levels": [lv for lv, _ in s.levels], "why_this_range": s.why,
                    "headlines": list(s.headlines)} for s in SWEEPS],
        "not_swept": NOT_SWEPT,
        "headlines": summary,
        "matrix": rows,
    }
    verdict["verdict"] = ("SOME_RESULTS_ARE_TUNED"
                          if any(v["reported_as_tuned"] for v in summary.values())
                          else "BOTH_HEADLINES_HOLD_ACROSS_THE_SWEPT_RANGE")
    verdict["statement"] = " ".join(v["scope_sentence"] for v in summary.values())
    (validation_dir() / "v3_robustness.json").write_text(
        json.dumps(verdict, indent=2, default=float), encoding="utf-8")
    return verdict


def _scope_sentence(hname: str, holds: int, run: int, cells: pd.DataFrame, tuned: bool) -> str:
    plain = HEADLINES[hname]["plain"]
    flipped = sorted(set(cells[cells.label == "flips"].parameter.tolist()))
    weakened = sorted(set(cells[cells.label == "weakens"].parameter.tolist()))
    lead = f"{plain.capitalize()} holds in {holds} of {run} swept cells"
    if tuned:
        return (lead + ", which is narrow enough that it is reported as TUNED: it depends on "
                f"where the settings were left. It is lost when "
                f"{' or '.join(flipped) or 'several parameters move'} changes.")
    if flipped:
        return (lead + f", and is lost only when {' or '.join(flipped)} changes. That is the "
                "boundary of the claim and the README states it.")
    if weakened:
        return (lead + f", weakening but keeping its direction when "
                f"{' or '.join(weakened)} moves. No parameter reverses it.")
    return lead + " and no swept parameter reverses or halves it."
