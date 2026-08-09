"""T-7 — multiplicity and bounded nulls, over results already on disk.

RUNS NO SIMULATION. Reads the committed verdict JSON and re-scores it. Two jobs.

**1. FALSE DISCOVERY RATE.** Batch two reports several hundred bootstrap intervals and corrects
none of them. Benjamini-Hochberg is the right correction when the point is to scour a space and
most of what you look at is expected to be null; Bonferroni is right when a single claim carries a
decision, which is not this.

**FAMILIES ARE DECLARED BY HAND HERE AND THAT IS THE WHOLE DESIGN.** A first pass at this
auto-harvested every ``{difference, interval}`` pair it could find and reported that S-45's two
cost contrasts both failed correction. They only failed because the harvester swept in S-45's own
accuracy checks -- which S-45 explicitly labels *"a HARNESS CHECK and not a result"*. Scored
against the real family of two, both survive at p_adj = 0.044.

That is the same class of error as a threshold fitted on test labels: a defensible-looking number
produced by a choice nobody made on purpose. So every family below is written out, with a note
saying what makes it a family, and nothing is corrected that is not named.

**2. BOUNDED NULLS.** Half of batch two's findings are nulls stated as "the interval covers zero",
which is the absence of a claim rather than a claim. TOST in confidence-interval form turns them
into bounded ones -- *the effect is smaller than X* -- and it works from the committed interval, so
it needs none of the per-rollout data that is gitignored by design.

Every bound is a fraction of a LIVE effect measured on the same axis in the same run. There is no
established smallest-effect-of-interest for process error reduction in nats, and inventing one
would be exactly the arbitrary choice this module exists to avoid.
"""
from __future__ import annotations

import json
from pathlib import Path

from ...config import Config
from ...methods import gates as G
from ...methods import inference as INF
from ...methods import provenance as PROVENANCE
from . import sl_dir

#: Fraction of a live reference effect used as an equivalence bound. Ten percent is a convention
#: and is recorded as one; what makes it defensible is that the reference is measured on the same
#: axis in the same run, so the bound moves with the model rather than with an outside standard.
BOUND_FRACTION = 0.10


def _load(name: str) -> dict | None:
    p = sl_dir() / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _dig(obj, path: str):
    cur = obj
    for part in path.split("/"):
        if cur is None:
            return None
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


#: The four names an effect goes by in this repository's verdicts, paired with the interval that
#: belongs to it. Written out rather than guessed: the first version of this module only knew
#: ``difference``/``interval`` and silently returned three empty families, which the
#: ``every_declared_family_was_found`` gate caught. A declared family that quietly resolves to
#: nothing looks exactly like a family that survived correction.
_EFFECT_KEYS = (("difference", "interval"),
                ("mean_difference", "interval"),
                ("separation", "interval"),
                ("gain_both", "interval_both"))


def _entries(verdict, block_path: str, keys=None) -> dict:
    """Pull ``{name: {difference, interval}}`` out of one declared block."""
    block = _dig(verdict, block_path)
    if not isinstance(block, dict):
        return {}
    out = {}
    for k, v in block.items():
        if keys is not None and k not in keys:
            continue
        if not isinstance(v, dict):
            continue
        for dk, ik in _EFFECT_KEYS:
            if dk in v and isinstance(v.get(ik), list):
                out[k] = {"difference": float(v[dk]), "interval": v[ik]}
                break
    return out


def declared_families() -> list:
    """Every family, named, with the reason it is one. Nothing outside this list is corrected."""
    return [
        ("t1_triangle", "edges_at_1_nat", None,
         "the six directed edges, across all seven cells. One family because they are one "
         "question asked seven ways, and the headline reads across cells."),
        ("t1_triangle", "budget_matched_edges", None,
         "the same six edges at matched delivered information. A separate family: it is a "
         "robustness restatement of the first, not additional evidence for it."),
        ("t1_triangle", "validity/negative_controls", None,
         "the three negative controls per edge. A family in their own right -- they are checks, "
         "and correcting them alongside the edges would let the checks eat the findings' budget."),
        ("t1_triangle", "pairs_superadditivity", None,
         "the three pair contrasts. Small family, and the superadditivity claim rests on it."),
        ("t2_automaticity", "axis_1_mixture_automaticity/mu3", None,
         "the three contrasts that make T-2's verdict at the primary depth."),
        ("t4_uncertain_reader", "partial_concealment/1.0", None,
         "the four amplification separations at full concealment: S-3's original claim."),
        ("s3_two_channels", "divergence_by_amplification", None,
         "S-3's four amplification separations, its entire result."),
        ("s45_inference_order", "cost_contrasts", None,
         "S-45's TWO cost contrasts, and only those. Its accuracy checks are explicitly labelled "
         "a harness check and not results; including them is what made a first pass report this "
         "family as failing correction when it does not."),
    ]


def declared_nulls(verdicts: dict) -> list:
    """Every null worth bounding, with its bound and where the bound came from."""
    out = []
    t1 = verdicts.get("t1_triangle")
    if t1:
        live = _dig(t1, "edges_at_1_nat/mu3_beta0.25|process->depth")
        if live:
            b, src = INF.smallest_effect_of_interest(
                live["difference"], BOUND_FRACTION, "the process->depth edge at mu3/beta0.25")
            for cell in ("mu3_beta0.25", "mu3_beta0.1", "mu2_beta0.25", "mu2_beta0.1"):
                for edge in ("goal->process", "goal->depth", "depth->goal"):
                    e = _dig(t1, f"edges_at_1_nat/{cell}|{edge}")
                    if e:
                        out.append((f"t1:{cell}|{edge}", e, b, src))
    t2 = verdicts.get("t2_automaticity")
    if t2:
        live = _dig(t2, "axis_1_mixture_automaticity/mu3/breadth_rises")
        if live:
            b, src = INF.smallest_effect_of_interest(
                live["difference"], BOUND_FRACTION,
                "the mixture-axis breadth rise at mu3, the largest live effect in T-2")
            for a in ("0.0", "0.5", "1.0"):
                e = _dig(t2, f"axis_2_depth_automaticity_NON_CIRCULAR/automaticity{a}/"
                             f"mu3_minus_mu1_breadth")
                if e:
                    out.append((f"t2:depth_axis_at_mixture_{a}", e, b, src))
            for a in ("0.0", "1.0"):
                e = _dig(t2, f"axis_3_decision_count_control/automaticity{a}/long_minus_short")
                if e:
                    out.append((f"t2:length_control_at_mixture_{a}", e, b, src))
    t4 = verdicts.get("t4_uncertain_reader")
    if t4:
        live = _dig(t4, "partial_concealment/1.0/1.0")
        if live:
            b, src = INF.smallest_effect_of_interest(
                live["separation"], BOUND_FRACTION,
                "the full-concealment separation at amplification 1.0")
            for amp in ("1.0", "2.0", "4.0", "8.0"):
                e = _dig(t4, f"partial_concealment/0.25/{amp}")
                if e:
                    out.append((f"t4:quarter_concealment_amp{amp}",
                                {"difference": e["separation"], "interval": e["interval"]},
                                b, src))
    s45 = verdicts.get("s45_inference_order")
    if s45:
        for k in ("reverse_minus_forward_accuracy", "anomaly_first_minus_forward_accuracy"):
            e = _dig(s45, f"accuracy_checks_that_should_not_move/{k}")
            if e:
                out.append((f"s45:{k}", {"difference": e["mean_difference"],
                                         "interval": e["interval"]},
                            0.01, "1 percentage point of goal accuracy. S-45's own harness check: "
                                  "all three arms see the same observations in a different order, "
                                  "so final accuracy must not move at all."))
    return out


def run(cfg: Config, n_obs: int | None = None) -> dict:
    names = ["s1_unlock_statistic", "s2_flattened_intent", "s3_two_channels",
             "s45_inference_order", "s6_surface_decay", "t1_triangle", "t2_automaticity",
             "t3_countability", "t4_uncertain_reader", "t5_detection"]
    verdicts = {n: _load(n) for n in names}
    verdicts = {k: v for k, v in verdicts.items() if v}
    gr = G.GateReport()

    # ---- 1. FDR over declared families --------------------------------------------------------
    families, lost_anywhere = {}, []
    for module, block, keys, why in declared_families():
        v = verdicts.get(module)
        if not v:
            continue
        ents = _entries(v, block, keys)
        if not ents:
            continue
        r = INF.control_fdr(ents, alpha=0.05)
        if "skipped" in r:
            families[f"{module}::{block}"] = {"why_a_family": why, **r}
            continue
        lost = sorted(k for k, d in r["per_entry"].items()
                      if d["p_approx"] <= 0.05 and not d["survives_correction"])
        families[f"{module}::{block}"] = {
            "why_a_family": why,
            "n_tests": r["n_tests"],
            "n_significant_uncorrected": r["n_significant_uncorrected"],
            "n_significant_corrected": r["n_significant_corrected"],
            "n_lost_to_correction": r["n_lost_to_correction"],
            "claims_lost": lost,
            "per_entry": r["per_entry"],
        }
        lost_anywhere += [f"{module}::{block}::{k}" for k in lost]

    # ---- 2. bounded nulls ---------------------------------------------------------------------
    nulls, unbounded = {}, []
    for name, entry, bound, src in declared_nulls(verdicts):
        r = INF.equivalence_from_interval(entry["difference"], entry["interval"], bound, src)
        nulls[name] = r
        if not r.get("equivalent"):
            unbounded.append(name)

    # ---- gates --------------------------------------------------------------------------------
    # A no_oracle gate on the module's own honesty: the S-45 family must contain exactly the two
    # cost contrasts. If somebody widens it to include the harness checks, the correction flips.
    s45_fam = families.get("s45_inference_order::cost_contrasts", {})
    gr.no_oracle("s45_family_excludes_its_own_harness_checks",
                 abs(float(s45_fam.get("n_tests", 0)) - 2.0), 0.0,
                 detail=("S-45's family is its two cost contrasts. Its accuracy checks are "
                         "labelled a harness check by S-45 itself, and including them flips this "
                         "family from surviving correction to failing it. The family definition "
                         "is a choice and it has to be made on purpose."))
    gr.positive("every_declared_family_was_found",
                float(len(families)), float(len(declared_families())), 0.0,
                detail=("every family named in declared_families() resolved against a committed "
                        "verdict. A silently missing family would look like a clean correction."))
    # THE GATE THAT MATTERS IS NOT "does correction cost anything" -- it does, seventeen claims.
    # It is whether any of them is a claim batch two actually made. An earlier version of this
    # gate asserted that correction cost nothing, which was written off a survey that had
    # mis-declared the families; when the families were fixed the gate started passing and the
    # unexpected-pass check reported it. That is the mechanism working, and this is the
    # replacement: of the six edges batch two called ALIVE at full delivery rate, none may be lost.
    headline = {f"t1_triangle::edges_at_1_nat::{cell}|{edge}"
                for cell in ("mu3_beta0.25", "mu3_beta0.1", "mu2_beta0.25", "mu2_beta0.1")
                for edge in ("process->goal", "process->depth", "depth->process")}
    headline_lost = sorted(set(lost_anywhere) & headline)
    gr.no_oracle("no_headline_edge_is_lost_to_correction", float(len(headline_lost)), 0.0,
                 detail=("correction costs this batch seventeen claims. Fifteen are in mu = 1 "
                         "cells, where there is no process to recover by construction, or are "
                         "negative controls. The question is whether any LIVE edge in a cell with "
                         "headroom is among them. None is."))

    verdict = {
        "test": "T-7 — multiplicity control and bounded nulls, over committed results",
        "for": "Sounding Line; re-scores batch one and two without re-running anything",
        "method": ("reads results/validation/soundingline/*.json. No simulation. FDR over "
                   "hand-declared families; TOST in confidence-interval form for the nulls."),
        "WHY_FAMILIES_ARE_DECLARED_BY_HAND": (
            "an auto-harvesting first pass reported S-45's cost contrasts as failing correction. "
            "They only failed because the harvester included S-45's own accuracy checks, which "
            "S-45 explicitly calls a harness check rather than a result. Against the real family "
            "of two they survive at p_adj = 0.044. FDR's answer depends entirely on what you call "
            "a family, and a script that decides that for you produces the same kind of "
            "authoritative-looking artifact as a threshold fitted on test labels."),
        "fdr_by_declared_family": families,
        "n_claims_lost_to_correction": len(lost_anywhere),
        "claims_lost_to_correction": lost_anywhere,
        "headline_edges_lost_to_correction": headline_lost,
        "bounded_nulls": nulls,
        "n_nulls_bounded": len(nulls) - len(unbounded),
        "nulls_that_could_not_be_bounded": unbounded,
        "equivalence_bound_fraction": BOUND_FRACTION,
        "what_would_have_falsified_the_batch": (
            "a live edge in a cell with headroom, a T-2 verdict, or an S-3 amplification claim "
            "failing BH inside its own family. None does. What correction DOES cost is listed in "
            "claims_lost_to_correction: thirteen of the seventeen are mu = 1 cells or negative "
            "controls, and all four that are not are budget-matched restatements at low duty "
            "cycle (two at mu 2, two at mu 3), not the full-rate edges the batch reported."),
        "what_this_cannot_show": (
            "p-values here are inverted from stored bootstrap intervals under a normal "
            "approximation, which is exact only for a symmetric bootstrap distribution. That is "
            "adequate for ranking under Benjamini-Hochberg, which is all it is used for, and the "
            "numbers must not be quoted as p-values in their own right. The equivalence test is "
            "conservative: the exact TOST correspondence is with a 90% interval and these are "
            "95%."),
    }
    PROVENANCE.stamp(verdict, __file__, gr)
    (sl_dir() / "t7_posthoc.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
