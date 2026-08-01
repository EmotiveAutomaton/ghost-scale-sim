"""S-1 — how often does a randomly parameterised model of this shape produce the finding?

WHY THIS RUNS FIRST AND GATES EVERYTHING ELSE.

Two rounds came back almost entirely positive: roughly twelve of sixteen new results confirmed
their hypothesis. For one person's framework written down as code, that is not a comfortable hit
rate, and nobody has measured how much of the agreement is the theory and how much is the
apparatus -- except once.

The validation pass asked it of the label effect and got **64%**: nearly two thirds of randomly
parameterised readers of the same shape reproduce it. That number is why the label result is now
reported as architecture-dependent, and it is the single most useful thing in the repository.

**It has never been asked of anything else.** Not the interior peak, not the two-dimensions result,
and not one of the sixteen findings from versions 6 and 7.

-----------------------------------------------------------------------------------------
WHAT A HIGH RATE MEANS, STATED BEFORE THE NUMBERS ARRIVE.

If a finding reproduces in most random models, it is a property of building a reader THIS SHAPE at
all, and the theory is not entitled to it. That is not the same as the finding being wrong -- the
label effect is real and reproducible; it is just architectural. What a high rate removes is the
claim that the finding is evidence FOR the theory as against evidence about the architecture.

A low rate is the opposite and is worth a great deal: it says the specific commitments the theory
makes are what produce the result.

**Every rate is reported whatever it is, including the ones that come back badly.**

-----------------------------------------------------------------------------------------
THE SECOND HALF: THE FORKING-PATHS LEDGER.

Across versions 6 and 7 a design or a criterion was changed AFTER seeing a null in seven places.
Each change is individually defensible and each is documented where it happened. The aggregate has
a shape -- flat result, find a reason, change the design, positive result -- and that is a garden of
forking paths whether or not each fork was justified.

The count exists in the commit history and the module docstrings and has never been collected.
Collecting it is free and it belongs beside the false-positive rates, because the two are the same
question asked from different ends.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import metrics
from ..config import Config
from ..v5_model import make_v5_observer
from ..v6 import harness as H
from ..validation.v2_nulls import draw_random_within_partition
from . import SEED_OFFSET, v8_dir

# Each entry: the finding, and a callable that returns True if a model reproduces it.
# Kept as CONTRASTS rather than absolute magnitudes, because a randomly parameterised model has a
# different natural scale and an absolute bar would measure the scale rather than the finding.


def _world_from_draw(cfg: Config, sig, synth):
    """A V5/V6 world built over a randomly drawn likelihood family."""
    return H.build_alt_world(cfg, np.asarray(sig, dtype=float))


def _depth_moves_method(world, cfg_r, n_mu, n_sub, ng, n_obs, n_timesteps) -> dict:
    """E36: depth moves what the reader takes of the METHOD, and cannot move the PURPOSE."""
    kappa = float(world.cfg.signal_model.kappa)
    out = {}
    for mu in (1, 3):
        proc, goal = [], []
        for i in range(int(n_obs)):
            rng = np.random.default_rng(SEED_OFFSET + 80_000 + i)
            g = int(rng.integers(ng))
            creator, art, env = H.make_artifact_and_env(world, cfg_r, g, mu, 1.0,
                                                        n_timesteps, rng)
            enc = H.run_encounter(world, cfg_r, art, env,
                                  H.make_alt_observer(world, rng, ng), creator, rng,
                                  n_timesteps, n_timesteps, n_sub, n_mu, ng, kappa)
            proc.append(enc.process["process_error_reduction"])
            goal.append(enc.error_reduction)
        out[mu] = (float(np.mean(proc)), float(np.mean(goal)))
    d_proc = out[3][0] - out[1][0]
    d_goal = out[3][1] - out[1][1]
    # SCORED AS A RATIO, AND THE FIRST VERSION OF THIS CRITERION WAS NEARLY TAUTOLOGICAL. Depth is
    # CONSTRUCTED so the goal is equally recoverable at every level, so the goal contrast is about
    # zero in any model of this shape. Requiring only "process contrast bigger than goal contrast"
    # is therefore satisfied by almost any positive process contrast, which is not a test.
    # Requiring the process contrast to dominate by a factor makes it scale-free and fair.
    ratio = abs(d_proc) / max(abs(d_goal), 1e-6)
    return {"process_contrast": d_proc, "goal_contrast": d_goal, "dominance_ratio": ratio,
            "reproduces": bool(d_proc > 0 and ratio >= 3.0)}


def _wall_is_distinct(world, cfg_r, n_mu, n_sub, ng, n_obs, n_timesteps) -> dict:
    """E37: content that is LEGIBLE AND EMPTY is a distinct failure from content you cannot parse.

    THE FIRST VERSION OF THIS PROBE DID NOT TEST E37 AND IS RECORDED BECAUSE IT IS THE SAME MISTAKE
    THE PROJECT KEEPS MAKING. It compared ordinary human content against ordinary machine content,
    which is E2's contrast, not E37's -- and then scored a threshold that never fired, so it read
    as a clean zero when it was measuring the wrong thing. A false-positive rate on the wrong
    quantity is worse than no rate at all, because it looks like evidence.

    The wall proper needs the non-invertible family: content on FAMILIAR features whose maker
    cannot be inverted from the surface, because several maker states emit the same thing. Built
    from whatever signature family this draw produced, so the probe travels with the random model
    rather than smuggling in the designed one.
    """
    from .. import v6_model as V6

    kappa = float(world.cfg.signal_model.kappa)
    sig_true = np.asarray(world.gm.sig_true, dtype=float)
    try:
        family = V6.build_noninvertible_family(sig_true, n_states=ng, collapse_to=2)
    except AssertionError:
        return {"separation": float("nan"), "reproduces": False, "buildable": False}

    machine_sigs = V6.build_machine_matched_signatures(family, ng)
    try:
        nw, nb, nr, _, _, _ = H.build_alt_world(world.cfg, machine_sigs)
    except AssertionError:
        return {"separation": float("nan"), "reproduces": False, "buildable": False}

    cells = {}
    for arm in ("foreign", "noninvertible"):
        ents, engs, accs = [], [], []
        for i in range(int(n_obs)):
            rng = np.random.default_rng(SEED_OFFSET + 81_000 + i)
            g = int(rng.integers(ng))
            if arm == "noninvertible":
                creator, art, env = H.make_artifact_and_env(nw, nr, g, 2, 1.0, n_timesteps,
                                                            rng, provenance=K.CREATOR)
            else:
                creator, art, env = H.make_artifact_and_env(world, cfg_r, g, 2, 0.0,
                                                            n_timesteps, rng,
                                                            provenance=K.GHOST)
            enc = H.run_encounter(world, cfg_r, art, env,
                                  H.make_alt_observer(world, rng, ng), creator, rng,
                                  n_timesteps, 10, n_sub, n_mu, ng, kappa)
            ents.append(enc.final_entropy)
            engs.append(enc.engaged_fraction)
            accs.append(enc.correct)
        cells[arm] = (float(np.mean(ents)), float(np.mean(engs)), float(np.mean(accs)))

    sep = (abs(cells["foreign"][0] - cells["noninvertible"][0])
           + abs(cells["foreign"][1] - cells["noninvertible"][1]))
    legible_and_empty = bool(cells["noninvertible"][0] < cells["foreign"][0]
                             and cells["noninvertible"][2] <= 0.40)
    return {"separation": sep, "legible_and_empty": legible_and_empty, "buildable": True,
            "reproduces": bool(sep >= 0.30 and legible_and_empty)}


def _label_moves_you_wrong(world, cfg_r, n_mu, n_sub, ng, n_obs, n_timesteps) -> dict:
    """E2/R-5: honest human work and a false label have OPPOSITE-SIGNED uptake."""
    kappa = float(world.cfg.signal_model.kappa)
    scores = {}
    for arm, prov, sig in (("human_honest", K.CREATOR, K.SIG_CREATOR),
                           ("machine_as_human", K.GHOST, K.SIG_CREATOR)):
        vals = []
        for i in range(int(n_obs)):
            rng = np.random.default_rng(SEED_OFFSET + 82_000 + i)
            g = int(rng.integers(ng))
            creator, art, env = H.make_artifact_and_env(
                world, cfg_r, g, 2, 1.0 if prov == K.CREATOR else 0.0, n_timesteps, rng,
                provenance=prov, declared_signal=sig, signing_rate=1.0)
            enc = H.run_encounter(world, cfg_r, art, env,
                                  H.make_alt_observer(world, rng, ng), creator, rng,
                                  n_timesteps, n_timesteps, n_sub, n_mu, ng, kappa)
            vals.append(enc.error_reduction)
        scores[arm] = float(np.mean(vals))
    return {"human_honest": scores["human_honest"],
            "machine_as_human": scores["machine_as_human"],
            "reproduces": bool(scores["human_honest"] > 0 > scores["machine_as_human"])}


FINDINGS = [
    ("E36 depth moves the method, not the purpose", _depth_moves_method),
    ("E37 the wall is a distinct failure", _wall_is_distinct),
    ("E2/R-5 a false label moves you the wrong way", _label_moves_you_wrong),
]

# The forking-paths ledger, collected from the module docstrings and the commit record.
FORKING_PATHS = [
    {"experiment": "E35 depletion", "designs_tried": 2, "criteria_tried": 2,
     "what_changed": "content type: low-legibility hierarchical -> foreign; absolute -> relative"},
    {"experiment": "E42 vulnerability", "designs_tried": 2, "criteria_tried": 1,
     "what_changed": "regime changed to the one known to sustain attention"},
    {"experiment": "E45 efficiency", "designs_tried": 2, "criteria_tried": 1,
     "what_changed": "task made harder; held-out goals 1 -> 2"},
    {"experiment": "E36 process", "designs_tried": 1, "criteria_tried": 3,
     "what_changed": "null statistic accuracy -> information; a second temporal ordering test added"},
    {"experiment": "E46 leak", "designs_tried": 1, "criteria_tried": 2,
     "what_changed": "absolute magnitude -> ratio; the null restated"},
    {"experiment": "E38 expertise", "designs_tried": 2, "criteria_tried": 1,
     "what_changed": "machine family degenerate -> well-formed"},
    {"experiment": "C-3 co-location", "designs_tried": 2, "criteria_tried": 1,
     "what_changed": "one artifact per cell rather than per observer"},
]


def run(cfg: Config, n_draws: int = 60, n_obs: int = 12, n_timesteps: int = 12,
        workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", 16)))

    js = float(cfg.artifact_model.js_threshold)
    ceiling = float(cfg.artifact_model.structured_ceiling)
    floor = float(cfg.artifact_model.get("synth_floor", 1.0e-3))

    rows = []
    usable = 0
    for d in range(int(n_draws)):
        rng = np.random.default_rng(SEED_OFFSET + 90_000 + d)
        drawn = draw_random_within_partition(rng, cfg, js_threshold=js,
                                             entropy_ceiling=ceiling, synth_floor=floor)
        if drawn is None:
            continue
        sig, synth = drawn
        usable += 1
        try:
            world, cb, cr, n_mu, n_sub, ng = _world_from_draw(cfg, sig, synth)
        except AssertionError:
            continue
        for name, fn in FINDINGS:
            try:
                out = fn(world, cr, n_mu, n_sub, ng, n_obs, n_timesteps)
            except Exception:                            # noqa: BLE001
                continue
            rows.append({"draw": d, "finding": name, **out})

    df = pd.DataFrame(rows)
    out_dir = v8_dir("s1_severity")
    if len(df):
        df.to_csv(out_dir / "s1_draws.csv", index=False)

    rates = {}
    for name, _ in FINDINGS:
        s = df[df.finding == name] if len(df) else df
        rates[name] = {
            "draws_scored": int(len(s)),
            "false_positive_rate": float(s.reproduces.mean()) if len(s) else float("nan"),
        }

    verdict = {
        "check": "S-1 severity",
        "question": ("How often does a randomly parameterised model of this shape produce the "
                     "finding, with the theory's settings thrown away?"),
        "plain_language": (
            "If a result shows up in most random models of the same shape, it is a property of "
            "having built a reader like this at all, and the theory is not entitled to it. That is "
            "not the same as the result being wrong. It is the difference between evidence FOR the "
            "theory and evidence ABOUT the architecture."),
        "reference_point": {
            "finding": "the label effect (E2), measured during the validation pass",
            "false_positive_rate": 0.64,
            "note": ("nearly two thirds of randomly parameterised readers reproduce it, which is "
                     "why it is now reported as architecture-dependent"),
        },
        "rates": rates,
        "usable_draws": usable, "draws_attempted": int(n_draws),
        "forking_paths_ledger": {
            "statement": ("across versions 6 and 7 a design or criterion was changed AFTER seeing "
                          "a null in seven places. Each change is individually defensible and is "
                          "documented where it happened. The aggregate has a shape."),
            "entries": FORKING_PATHS,
            "total_designs_tried": sum(e["designs_tried"] for e in FORKING_PATHS),
            "total_criteria_tried": sum(e["criteria_tried"] for e in FORKING_PATHS),
            "why_it_is_reported": (
                "a criterion that a false theory would also pass is not evidence, and neither is a "
                "design chosen because the previous one came back flat. Collecting the count is "
                "free and it is the same question the false-positive rates ask, from the other end."),
        },
        "how_to_read_a_rate": {
            "high": "the finding is architectural; the theory is not entitled to it",
            "low": "the theory's specific commitments are what produce it, and that is worth a lot",
        },
    }
    (v8_dir() / "s1_severity.json").write_text(json.dumps(verdict, indent=2, default=str),
                                               encoding="utf-8")
    return verdict
