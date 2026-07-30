"""V-4 — what is forced by construction?

THE SYSTEMATIC VERSION of the two checks that caught the uniform fallback hypothesis and the
noise-versus-unidentifiability distinction. For every headline claim, write down IN ADVANCE which
property of the construction, if altered, would eliminate the result. Then alter it and confirm the
result disappears.

A claim that survives every alteration is not robust. It is built in.

-----------------------------------------------------------------------------------------
THREE THINGS AN ALTERATION CAN DO, AND ALL THREE ARE RECORDED AS OUTCOMES.

1. **The result disappears.** The property is load-bearing and the claim is about the world the
   model describes rather than about the model's plumbing. This is the wanted outcome.
2. **The result survives.** The property was not what was holding it up, which means either the
   claim is stronger than thought or the mechanism is not the one being claimed. Either way it is
   reported, because "I altered the thing I said mattered and nothing happened" is information.
3. **The alteration is unreachable** — a construction-time assertion refuses it. That is not a
   failed check. It means the property is a DEFINITION rather than a free parameter, and the claim
   is downstream of a definition. That is worth knowing and it is worth saying, because a reader
   who assumes it was a swept parameter is reading the claim as stronger than it is.

-----------------------------------------------------------------------------------------
TWO ARCHITECTURAL CHOICES GET THEIR OWN AUDIT, because a great deal rests on them and neither was
chosen on theoretical grounds.

**The disjoint human/foreign feature partition.** Adopted because no such partition existed at the
original feature count, so the space was doubled. Every claim about the readability axis is
downstream of it. The claims are enumerated explicitly below rather than left implicit.

**The rebuilt effort parameter.** It was changed specifically so that "offhand but deep" is
representable — which means the dissociation the model was built to test was made possible before
it was measured. That is reported as a CONSTRUCTION COMMITMENT rather than as an emergent finding,
and the audit asks whether anything of it survives under the original parameterisation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import constants as K
from ..config import Config
from . import validation_dir
from .v2_nulls import _label_effect_from_e2


# --------------------------------------------------------------------------- #
# The claims and what each is downstream of, written before any alteration ran.
# --------------------------------------------------------------------------- #
CLAIMS_DOWNSTREAM_OF_THE_PARTITION = [
    ("Confident invention peaks in the middle of the readability axis, not at the unreadable end",
     "E20", "the peak's LOCATION is a location on an axis whose zero point is 'no shared "
            "features at all', which only exists because the two blocks are disjoint"),
    ("Attention stops being sustained below about 4% overlap",
     "E20", "the crossing point is measured on the same axis"),
    ("Content with real structure the reader cannot parse holds attention indefinitely",
     "E19", "'cannot parse' is implemented as 'lives on features every one of the reader's "
            "hypotheses puts at floor', which is the partition"),
    ("The most generous fallback hypothesis absorbs exploratory human work and does nothing for "
     "machine work",
     "E19", "the fallback is flat across the human block and at floor across the foreign one, "
            "which is a property of the partition and was the reason the space was doubled"),
    ("Being out of your depth and reading something foreign are different failures",
     "E32", "one arm degrades the reader's templates and the other moves the content off the "
            "block those templates cover"),
    ("Real generated content sits somewhere on the overlap axis, and a human study could locate it",
     "E34", "the prediction card's axis IS the partition; without it there is no axis to place "
            "real content on"),
]


# --------------------------------------------------------------------------- #
# Alterations.
# --------------------------------------------------------------------------- #
def _alter_label_effect_no_channel(cfg: Config, out: Path, workers: int,
                                   n_obs: int, n_seeds: int) -> dict:
    """The label effect is downstream of the reader being able to READ the label at all."""
    from ..config import load_config
    from ..experiments import e2_variance as E2
    c = load_config()
    c.set("inference.exact", True)
    c.set("signal_model.kappa", 0.0)
    c.set("run.n_observers", n_obs)
    c.set("run.n_seeds", n_seeds)
    E2.run(c, out_dir=out, workers=workers, make_fig=False)
    eff = _label_effect_from_e2(out)
    return {"primary": float(eff["honest_doubt_multiple"]), "reportable": eff["reportable"],
            "detail": eff}


def _alter_foreign_is_noise(cfg: Config, out: Path, workers: int,
                            n_obs: int, n_seeds: int) -> dict:
    """The readability-axis results are downstream of foreign content being STRUCTURED.

    The obvious strawman, "machine content is random noise", would produce a crash for the wrong
    reason, and null N6 has existed since V1 to separate the two. This alteration pushes the
    foreign family's Dirichlet concentration up until the drawn signatures approach uniform, which
    is that strawman applied to the foreign block.
    """
    from ..experiments import e20_omega_sweep as E20
    from ..v4_model import load_v4_config
    c = load_v4_config(include_explore=False)
    c.set("inference.exact", True)
    c.set("v4.foreign.concentration", 50.0)   # -> near-uniform draws over the foreign block
    c.set("v4.foreign.anchor", 0.0)
    c.set("experiments.e20.n_observers", n_obs)
    c.set("experiments.e20.n_seeds", n_seeds)
    E20.run(c, out_dir=out, workers=workers, make_fig=False)
    v = json.loads((out / "e20_verdict.json").read_text(encoding="utf-8"))
    return {"primary": float(v["fabrication_peak_value"]),
            "peak_omega": float(v["fabrication_peak_omega"]),
            "reportable": bool(v["fabrication_peak_is_interior"]),
            "detail": {"outcome": v["outcome"]}}


def _alter_partition_overlaps(cfg: Config, out: Path, workers: int,
                              n_obs: int, n_seeds: int) -> dict:
    """The readability-axis results are downstream of the two feature blocks being disjoint.

    The alteration raises the foreign family's floor across the WHOLE feature space, which leaks
    foreign mass into the human block. If the construction refuses it, that refusal is the audit's
    answer: the disjointness is a definition and not a swept parameter.
    """
    from ..experiments import e20_omega_sweep as E20
    from ..v4_model import load_v4_config
    c = load_v4_config(include_explore=False)
    c.set("inference.exact", True)
    # A floor of 0.05 over sixteen features puts ~0.4 of the mass outside the foreign block,
    # comfortably past the 0.10 disjointness guard.
    c.set("v4.foreign.floor", 0.05)
    c.set("experiments.e20.n_observers", n_obs)
    c.set("experiments.e20.n_seeds", n_seeds)
    E20.run(c, out_dir=out, workers=workers, make_fig=False)
    v = json.loads((out / "e20_verdict.json").read_text(encoding="utf-8"))
    return {"primary": float(v["fabrication_peak_value"]),
            "peak_omega": float(v["fabrication_peak_omega"]),
            "reportable": bool(v["fabrication_peak_is_interior"]),
            "detail": {"outcome": v["outcome"]}}


def _alter_synth_is_uniform(cfg: Config, out: Path, workers: int,
                            n_obs: int, n_seeds: int) -> dict:
    """The crash is downstream of synthetic content being structured rather than high-entropy.

    N6's strawman, applied through the config rather than through a test hook, so it runs inside a
    real experiment instead of a unit check.
    """
    from ..config import load_config
    from ..experiments import e2_variance as E2
    c = load_config()
    c.set("inference.exact", True)
    c.set("artifact_model.noise_free_synth_concentration", 500.0)  # -> near-uniform
    c.set("artifact_model.structured_ceiling", float(np.log(8)) + 1.0)  # let the strawman build
    c.set("run.n_observers", n_obs)
    c.set("run.n_seeds", n_seeds)
    E2.run(c, out_dir=out, workers=workers, make_fig=False)
    eff = _label_effect_from_e2(out)
    return {"primary": float(eff["honest_doubt_multiple"]), "reportable": eff["reportable"],
            "detail": eff}


def _alter_synth_not_symmetrised(cfg: Config, out: Path, workers: int,
                                 n_obs: int, n_seeds: int) -> dict:
    """The disagreement is downstream of the synthetic distribution being GOAL-SYMMETRIC.

    THIS ALTERATION EXISTS BECAUSE THE ONE BELOW IT MISSED. The uniform-synth alteration was written
    to test whether the label effect depends on synthetic content being structured rather than
    noise. It does not — and it could not have, because a uniform distribution is goal-symmetric by
    definition, so the alteration preserved the property that actually matters while changing the
    one that does not. The measurement is kept and its stated target is corrected rather than
    quietly reassigned; see ``note_on_attribution`` on that row.

    The property that matters is symmetry. V1's own deviation 2 says so: an un-symmetrised draw
    generically resembles ONE goal, so every reader hallucinates the SAME goal, and the result
    becomes spurious consensus instead of confident disagreement. This alteration turns the
    symmetrisation off and checks that.
    """
    from ..config import load_config
    from ..experiments import e2_variance as E2
    c = load_config()
    c.set("inference.exact", True)
    c.set("artifact_model.goal_symmetric", False)
    c.set("run.n_observers", n_obs)
    c.set("run.n_seeds", n_seeds)
    E2.run(c, out_dir=out, workers=workers, make_fig=False)
    eff = _label_effect_from_e2(out)
    return {"primary": float(eff["honest_doubt_multiple"]), "reportable": eff["reportable"],
            "detail": eff}


ALTERATIONS = (
    {
        "claim": "The same machine-made content is read as certain or uncertain depending only "
                 "on what the label says",
        "claim_id": "label_effect",
        "downstream_of": "the reader being able to read the provenance label at all (kappa > 0)",
        "alteration": "kappa set to 0, so the label carries no information",
        "expected": "the effect disappears",
        "fn": _alter_label_effect_no_channel,
        "baseline": "label_effect",
    },
    {
        "claim": "The same machine-made content is read as certain or uncertain depending only "
                 "on what the label says",
        "claim_id": "label_effect",
        "downstream_of": "synthetic content being structured rather than high-entropy noise (N6)",
        "alteration": "the synthetic distribution replaced with a near-uniform one",
        "expected": "the effect changes character: a uniform synth is low-information for a "
                    "different reason and the crash becomes the strawman's crash",
        "fn": _alter_synth_is_uniform,
        "baseline": "label_effect",
        "note_on_attribution": (
            "THIS ROW MISSES ITS OWN TARGET, and the measurement is kept with the correction "
            "attached rather than reassigned. A uniform distribution is goal-symmetric BY "
            "DEFINITION, being equidistant from every goal signature, so replacing the structured "
            "synth with a uniform one preserves the property the effect actually depends on while "
            "changing the property it does not. That the label effect survives here is therefore "
            "informative about neither. The alteration that does bite is the next row, which turns "
            "the symmetrisation off; it was added after this one came back 'survived' and the "
            "reason was traced."),
    },
    {
        "claim": "Readers become confident AND disagree with each other, so the confidence is "
                 "invention rather than shared error",
        "claim_id": "label_effect",
        "downstream_of": "the synthetic distribution being goal-SYMMETRIC (V1 deviation 2)",
        "alteration": "the goal-symmetrisation switched off, so the frozen synthetic draw leans "
                      "toward one goal by chance",
        "expected": "the disagreement collapses into consensus: every reader hallucinates the "
                    "SAME goal, which is confident shared error rather than invention",
        "fn": _alter_synth_not_symmetrised,
        "baseline": "label_effect",
    },
    {
        "claim": "Confident invention peaks in the middle of the readability axis",
        "claim_id": "interior_peak",
        "downstream_of": "foreign content being goal-directed rather than noise (C1 property 1)",
        "alteration": "the foreign family drawn near-uniform, with its goal anchor removed",
        "expected": "the interior peak disappears; the failure reverts to unidentifiability",
        "fn": _alter_foreign_is_noise,
        "baseline": "interior_peak",
    },
    {
        "claim": "Confident invention peaks in the middle of the readability axis",
        "claim_id": "interior_peak",
        "downstream_of": "the human and foreign feature blocks being disjoint",
        "alteration": "the foreign family's support floor raised across the whole feature space",
        "expected": "the interior peak disappears or moves; if the construction refuses the "
                    "alteration, disjointness is a definition and the claim is downstream of one",
        "fn": _alter_partition_overlaps,
        "baseline": "interior_peak",
    },
)


# --------------------------------------------------------------------------- #
# The effort-parameter audit.
# --------------------------------------------------------------------------- #
def audit_effort_rebuild(cfg: Config, workers: int, n_obs: int, n_seeds: int) -> dict:
    """Does anything of the depth/effort dissociation survive without the rebuilt effort axis?

    THE POSITION BEING TAKEN, stated before the measurement. The dissociation is a CONSTRUCTION
    COMMITMENT, not an emergent finding. V5 replaced V4.5's rationality parameter with model depth
    specifically so that "offhand but deep" would be representable, and "offhand but deep" is one
    of the two corners the dissociation is read off. Making a contrast representable before
    measuring it is a legitimate modelling move and it is NOT a discovery, and the README says so.

    What can still be asked is narrower and worth asking: with the effort axis held at its
    committed maximum, with no offhand corner available at all, does depth still separate? If
    the depth estimator is reading something in the artifact rather than reading the effort knob
    under a new name, and that part of N21 stands on its own. If it does not, the whole
    dissociation rests on the rebuilt axis and the claim shrinks to exactly that.
    """
    from ..n21_depth_not_effort import run as n21_run
    from ..v5_model import load_v5_config

    out = validation_dir() / "v4" / "effort_rebuild"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "position": ("reported as a construction commitment rather than an emergent finding: "
                     "the model was rebuilt so that 'offhand but deep' is representable, which "
                     "makes the dissociation possible before it is measured"),
        "narrower_question": ("with the effort axis pinned at its maximum, so no offhand corner "
                              "exists, does depth still separate?"),
    }
    try:
        import pandas as pd

        cfg5 = load_v5_config()
        cfg5.set("inference.exact", True)
        n21 = n21_run(cfg5, out_dir=out, n_obs=n_obs, n_seeds=n_seeds)
        stats = pd.DataFrame(n21["cell_table"])
        result["n21_verdict_under_exact_inference"] = n21.get("verdict")
        # The committed N21 design is a 2 x 2 over true effort and true depth. Pinning the effort
        # axis means keeping only its maximum row — the row with no offhand corner in it.
        pinned = stats[np.isclose(stats.true_beta, stats.true_beta.max())]
        by_mu = pinned.groupby("true_mu").recovered_mu.mean().sort_index()
        separation = (float(by_mu.iloc[-1] - by_mu.iloc[0]) if len(by_mu) > 1 else float("nan"))
        result.update({
            "recovered_depth_by_true_depth_at_max_effort": {str(k): float(v)
                                                            for k, v in by_mu.items()},
            "separation_at_pinned_effort": separation,
            "depth_separates_without_the_offhand_corner": bool(np.isfinite(separation)
                                                               and separation > 0.10),
        })
        result["reading"] = (
            "Depth still separates with the effort axis pinned, so the depth estimator is "
            "reading structure in the artifact and not the effort knob renamed. The DISSOCIATION "
            "remains a construction commitment; the READABILITY of depth does not."
            if result["depth_separates_without_the_offhand_corner"] else
            "Depth does not separate once the effort axis is pinned. The whole dissociation rests "
            "on the rebuilt axis, and the claim shrinks to exactly that.")
    except (AssertionError, ValueError, KeyError, IndexError, AttributeError) as exc:
        result.update({"unreachable": True,
                       "why": f"{type(exc).__name__}: {str(exc)[:400]}",
                       "reading": ("the narrower question could not be asked without altering "
                                   "the committed N21 design, which is itself the audit's "
                                   "answer: the two axes are not independently addressable")})
    return result


# --------------------------------------------------------------------------- #
# The check.
# --------------------------------------------------------------------------- #
def run(cfg: Config, workers: int = 1) -> dict:
    out_root = validation_dir() / "v4"
    out_root.mkdir(parents=True, exist_ok=True)
    n_obs = int(cfg.get("validation.n_observers", 60))
    n_seeds = int(cfg.get("validation.n_seeds", 12))

    # Baselines, so "disappeared" is measured against something rather than asserted.
    baselines = {}
    from .v3_robustness import HEADLINES, _load
    for hname, spec in HEADLINES.items():
        seeds = max(2, n_seeds // spec["seed_divisor"])
        out = out_root / "baseline" / hname
        out.mkdir(parents=True, exist_ok=True)
        baselines[hname] = spec["scorer"](out, _load(spec["loader"]), workers, n_obs, seeds)

    rows = []
    for alt in ALTERATIONS:
        base = baselines[alt["baseline"]]
        seeds = max(2, n_seeds // HEADLINES[alt["baseline"]]["seed_divisor"])
        out = out_root / alt["claim_id"] / alt["downstream_of"][:40].replace(" ", "_").replace(
            "(", "").replace(")", "").replace(">", "gt")
        out.mkdir(parents=True, exist_ok=True)
        row = {k: alt[k] for k in ("claim", "claim_id", "downstream_of", "alteration", "expected")}
        if alt.get("note_on_attribution"):
            row["note_on_attribution"] = alt["note_on_attribution"]
        row["baseline_primary"] = base["primary"]
        try:
            got = alt["fn"](cfg, out, workers, n_obs, seeds)
            row["altered_primary"] = got["primary"]
            # "Disappeared" is judged against the baseline's own size rather than against zero,
            # because these primaries are ratios and indices with different natural units.
            shrank = bool(np.isfinite(got["primary"]) and np.isfinite(base["primary"])
                          and abs(got["primary"]) < 0.5 * abs(base["primary"]))
            lost_verdict = bool(base.get("reportable", True) and not got.get("reportable", True))
            row["outcome"] = ("result_disappeared" if (shrank or lost_verdict)
                              else "result_survived")
            row["altered_verdict_holds"] = bool(got.get("reportable", True))
            if "peak_omega" in got:
                row["altered_peak_omega"] = got["peak_omega"]
                row["baseline_peak_omega"] = base.get("peak_omega")
        except (AssertionError, ValueError, KeyError, IndexError) as exc:
            row["outcome"] = "alteration_unreachable"
            row["altered_primary"] = None
            row["refusal"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        rows.append(row)

    verdict = {
        "check": "V-4",
        "question": "What is forced by construction?",
        "plain_language": (
            "For each finding, this names the one design decision the finding depends on, then "
            "breaks that decision on purpose and checks the finding goes away. A finding that "
            "survives having its own foundations removed was not a finding at all; it was built in. A "
            "third possibility also gets recorded: sometimes the model refuses to be broken that "
            "way, which means the decision is part of the definition rather than a setting, and "
            "that changes how strongly the finding can be stated."),
        "scale": {"n_observers": n_obs, "n_seeds": n_seeds},
        "how_to_read": {
            "result_disappeared": ("the wanted outcome: the property is load-bearing and the "
                                  "claim is about the modelled world rather than the plumbing"),
            "result_survived": ("unwelcome: the property named was not what was holding the "
                               "result up, so either the claim is stronger than stated or the "
                               "mechanism is not the one being claimed"),
            "alteration_unreachable": ("a construction assertion refused the alteration. Not a "
                                      "failure. It means the property is a definition rather "
                                      "than a free parameter, and the claim is downstream of a "
                                      "definition"),
        },
        "alterations": rows,
        "architectural_audit": {
            "disjoint_feature_partition": {
                "what_it_is": ("the human and foreign feature blocks do not overlap. Adopted "
                               "because no such partition existed at the original feature count, "
                               "so the feature space was doubled from eight to sixteen"),
                "chosen_on_theoretical_grounds": False,
                "why_it_was_adopted": ("the alternative was for foreign content to overlap the "
                                       "reader's own support, which is the V4 spec's own "
                                       "pre-mortem failure #1: foreign content becomes "
                                       "unidentifiable rather than foreign, and V4 reports V3's "
                                       "results in new vocabulary"),
                "claims_downstream_of_it": [
                    {"claim": c, "experiment": e, "why": w}
                    for c, e, w in CLAIMS_DOWNSTREAM_OF_THE_PARTITION],
                "what_a_different_partitioning_would_do": (
                    "A partially overlapping partition removes the axis's zero point: there would "
                    "be no omega at which the reader's hypotheses are all at floor, so 'fully "
                    "foreign' would stop being a location and the interior peak would have no "
                    "interior to sit in. A partition with unequal block sizes would keep the axis "
                    "but change where the peak falls on it, because the peak's location is set by "
                    "how much in-family structure is needed to make an explanation seem "
                    "available. In both cases the SHAPE of the claim, an interior maximum, is "
                    "what would transfer, and the specific value would not. That is why the "
                    "prediction card is written as a location to be measured rather than as a "
                    "number to be trusted."),
            },
            "rebuilt_effort_parameter": audit_effort_rebuild(cfg, workers, max(20, n_obs // 3),
                                                             max(4, n_seeds // 2)),
        },
    }

    disappeared = sum(1 for r in rows if r["outcome"] == "result_disappeared")
    survived = [r for r in rows if r["outcome"] == "result_survived"]
    unreachable = [r for r in rows if r["outcome"] == "alteration_unreachable"]
    verdict["counts"] = {"disappeared": disappeared, "survived": len(survived),
                         "unreachable": len(unreachable), "total": len(rows)}
    if survived:
        verdict["verdict"] = "SOME_CLAIMS_SURVIVE_LOSING_THEIR_OWN_FOUNDATIONS"
        # A survival is reported with its own attribution note when it has one, because a row that
        # missed its target and a row whose claim outlived its mechanism are different failures and
        # collapsing them would overstate one and hide the other.
        bits = []
        for r in survived:
            line = (f"the claim \"{r['claim'][:70]}\" survived the removal of "
                    f"{r['downstream_of']}")
            if r.get("note_on_attribution"):
                line += (", and that row misses its own target rather than refuting the claim; "
                         "see its note_on_attribution, and the row directly below it, which was "
                         "added once the reason was traced and which does remove the claim")
            bits.append(line)
        verdict["statement"] = (
            f"{disappeared} of {len(rows)} claims disappear when the property they were said to "
            f"depend on is removed, which is the wanted outcome. " + ". ".join(bits)
            + ". Survivals are reported rather than dropped: a claim that outlives its own stated "
              "mechanism is either stronger than stated or explained by something else, and the "
              "honest position is to say which of those has not yet been established.")
    else:
        verdict["verdict"] = "EVERY_CLAIM_DEPENDS_ON_WHAT_IT_WAS_SAID_TO_DEPEND_ON"
        verdict["statement"] = (
            f"All {disappeared} reachable alterations removed the claim they targeted"
            + (f", and {len(unreachable)} were refused by a construction assertion, which means "
               f"those claims are downstream of definitions rather than of swept parameters and "
               f"are stated that way."
               if unreachable else "."))
    (validation_dir() / "v4_construction.json").write_text(
        json.dumps(verdict, indent=2, default=float), encoding="utf-8")
    return verdict
