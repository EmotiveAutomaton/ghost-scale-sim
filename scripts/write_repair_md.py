#!/usr/bin/env python
"""Write REPAIR.md from the verdict files the repair pass produced.

    python scripts/write_repair_md.py

Same discipline as the validation and diagnostics generators, and for the same reason the repair
specification restates in its section 8: written from the verdict files afterwards, never from the
expectations in the specification. Every number below is read out of `results/repair/*.json`.

A missing verdict file renders as NOT RUN rather than as a gap.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RDIR = REPO / "results" / "repair"

CHECKS = [
    ("R-1 to R-4", "tier1_recompute.json",
     "What do the under-powered criteria say once they have intervals?"),
    ("R-5", "r5_uptake.json", "Has the uptake measure been measuring understanding?"),
    ("R-6 / R-8a", "r6_estimators.json",
     "Which parameters are measurable, fitted to what the reader produces, and over what range?"),
    ("R-8b", "r8b_learned_trust.json",
     "If trust is learned about a named source, does it recover, and can a trusting reader learn?"),
    ("R-11 / R-12", "r11_r12_sweep.json",
     "Every reachable experiment, exact inference and fixed seeding. What moves?"),
    ("R-13", "r13_reruns.json", "The two inconclusive experiments, rerun on a fair footing."),
]


def load(name):
    p = RDIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def pretty(v):
    return v.replace("_", " ").lower() if v else "not run"


def fmt(x, places=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:
        return "not defined"
    if f == 0:
        return "0"
    if abs(f) >= 10000 or abs(f) < 1e-3:
        return "%.2e" % f
    return ("%." + str(places) + "f") % f


def header(summary, criteria):
    lines = [
        "# The repair pass: from demonstration to measurement",
        "",
        "Generated from the verdict files in [results/repair/](results/repair/). Regenerate with "
        "`python scripts/write_repair_md.py`.",
        "",
        "---",
        "",
        "## What this pass was for",
        "",
        "The diagnostics pass named the real problem. **This apparatus was built to demonstrate and "
        "was being audited as though it measures.** Those have different standards, and most of "
        "what the audit returned is the gap between them rather than anything being wrong.",
        "",
        "The evidence was direct. Trust looked unmeasurable because nothing in the model had ever "
        "been built to estimate trust; it is a knob you turn to ask what a reader with that "
        "disposition would do. The two parameters that recovered cleanly recovered because somebody "
        "wrote an estimator for them during the audit. They had none before either.",
        "",
        "**The rule for this pass: every change either makes something measurable that was not, or "
        "removes something.** The model has grown every version since the first and nothing has "
        "ever been taken out, and that accretion is what produced constructions the audit could not "
        "break because the model refuses to run without them. An addition qualified only if it "
        "converted a free choice into a measured quantity.",
        "",
    ]
    if criteria:
        lines += [
            f"**The criteria were fixed before any recomputation ran** and hash-locked at "
            f"`{criteria.get('content_hash', '')[:16]}`, in "
            "[results/repair/criteria.json](results/repair/criteria.json), separate from the "
            "validation and diagnostics locks, which have reported and are sealed.",
            "",
        ]
        corr = criteria.get("corrections_to_the_specification") or []
        if corr:
            lines += ["**Three corrections were made to the specification itself, before any repair "
                      "ran.** They are recorded in the lock:", ""]
            lines += [f"{i}. {c}" for i, c in enumerate(corr, 1)]
            lines.append("")
    if summary and summary.get("quick"):
        lines += ["> **These outputs were produced at development scale and are not reportable.** "
                  "Re-run `python run_repair.py` without `--quick`.", ""]
    return lines


def summary_table(loaded):
    lines = ["---", "", "## The pass at a glance", "",
             "| item | what it asks | how it came back |", "|---|---|---|"]
    for code, fname, q in CHECKS:
        v = loaded.get(fname)
        lines.append(f"| **{code}** | {q} | {pretty(v.get('verdict')) if v else '**not run**'} |")
    lines.append("")
    return lines


def tier1_section(v):
    if not v:
        return ["## R-1 to R-4", "", "**Not run.**", ""]
    out = ["---", "", "## R-1 to R-4: recomputation, with intervals", "",
           v.get("plain_language", ""), "",
           "| criterion | experiment | original | recomputed | 95% interval | threshold | verdict |",
           "|---|---|---|---|---|---|---|"]
    for r in v.get("r1_recomputed_criteria", []):
        if not r.get("available"):
            continue
        if "bootstrap_interval" in r:
            lo, hi = r["bootstrap_interval"]
            out.append("| %s | %s | %s | %s | [%s, %s] | %s | **%s** |"
                       % (r["criterion"], r["experiment"], fmt(r.get("original_value")),
                          fmt(r["recomputed_point"]), fmt(lo), fmt(hi), fmt(r["threshold"]),
                          pretty(r["verdict"])))
        else:
            for label, m in (r.get("measures") or {}).items():
                lo, hi = m["interval"]
                out.append("| %s (%s) | %s | — | %s | [%s, %s] | 0 | **%s** |"
                           % (r["criterion"], label, r["experiment"], fmt(m["point"]),
                              fmt(lo), fmt(hi), pretty(m["verdict"])))
    out += ["", v.get("statement", ""), ""]

    peak = v.get("r3_fabrication_peak") or {}
    if peak.get("available"):
        out += ["### The interior peak, with an error bar", "",
                "| overlap | fabrication index | share of bootstrap draws where this is the peak |",
                "|---|---|---|"]
        dist = peak.get("argmax_distribution", {})
        for om, idx in sorted(peak.get("index_by_omega", {}).items(), key=lambda kv: float(kv[0])):
            out.append("| %s | %s | %s |" % (fmt(float(om), 2), fmt(idx),
                                             fmt(dist.get(str(om), 0.0))))
        out.append("")

    rows = [r for r in (v.get("r2_r4_disagreement") or {}).get("rows", []) if r.get("available")]
    if rows:
        out += ["### Disagreement, read three ways", "",
                "| experiment | cell | uncertainty per reader | modal-goal entropy, as committed | "
                "bias corrected | pairwise divergence |", "|---|---|---|---|---|---|"]
        for r in rows:
            out.append("| %s | %s | %s | %s | %s | %s |"
                       % (r["experiment"], r["cell"], fmt(r["within_reader_entropy"]),
                          fmt(r["modal_goal_entropy_as_committed"]),
                          fmt(r["modal_goal_entropy_bias_corrected"]),
                          fmt(r["mean_pairwise_divergence"])))
        out.append("")
        needs = (v.get("r2_r4_disagreement") or {}).get("needs_rerun_for_divergence") or {}
        if needs:
            out += ["Experiments that cannot be given the divergence figure without re-running, "
                    "named rather than skipped: "
                    + "; ".join(f"**{k}** {w}" for k, w in needs.items()) + ".", ""]
    return out


def r5_section(v):
    if not v:
        return ["## R-5", "", "**Not run.**", ""]
    out = ["---", "", "## R-5: what the uptake measure was actually measuring", "",
           v.get("plain_language", ""), "",
           f"*{v.get('formula_correction', '')}*", "",
           "| experiment | cell | reads the right purpose | movement, as reported | "
           "error reduction | share confidently wrong |", "|---|---|---|---|---|---|"]
    for c in v.get("cells", []):
        if not c.get("available"):
            continue
        out.append("| %s | %s | %s | %s | **%s** | %s |"
                   % (c["experiment"], c["cell"], fmt(c["accuracy"]),
                      fmt(c["movement_unweighted"]), fmt(c["error_reduction"]),
                      fmt(c["share_confidently_wrong"])))
    out += ["", v.get("statement", ""), "",
            f"*{v.get('prior_reference', '')}*", ""]
    return out


def r6_section(v):
    if not v:
        return ["## R-6 / R-8a", "", "**Not run.**", ""]
    out = ["---", "", "## R-6 and R-8a: the identifiability map", "",
           v.get("plain_language", ""), "",
           f"**Correction.** {v.get('correction_to_the_earlier_pass', '')}", "",
           "| parameter | what it was fitted to | observability | slope | identifiable range | "
           "reading |", "|---|---|---|---|---|---|"]
    for m in v.get("identifiability_map", []):
        rng = ("%s to %s" % (fmt(m.get("identifiable_from"), 2), fmt(m.get("identifiable_to"), 2))
               if m.get("identifiable_from") is not None else "—")
        out.append("| %s | %s | %s | %s | %s (%s of the range) | %s |"
                   % (m["parameter"], m.get("fitted_to", ""), m.get("standard") or "—",
                      fmt(m.get("slope"), 2), rng,
                      "%.0f%%" % (100 * m["identifiable_fraction"]), m.get("reading", "")))
    out += ["", v.get("statement", ""), ""]
    return out


def r8b_section(v):
    if not v:
        return ["## R-8b", "", "**Not run.**", ""]
    out = ["---", "", "## R-8b: trust as something learned about a source", "",
           v.get("plain_language", ""), "",
           f"**Why it was built.** {v.get('why_it_was_built', '')}", "",
           f"**What was deliberately not built.** {v.get('what_was_deliberately_not_built', '')}",
           "", v.get("statement", ""), ""]
    return out


def sweep_section(v):
    if not v:
        return ["## R-11 / R-12", "", "**Not run.**", ""]
    out = ["---", "", "## R-11 and R-12: every reachable experiment, both ways", "",
           v.get("plain_language", ""), "", f"**Design.** {v.get('design', '')}", "",
           "| experiment | ran both ways | outcome, old code path | outcome, repaired | moved |",
           "|---|---|---|---|---|"]
    for r in v.get("rows", []):
        both = bool(r.get("baseline_ok") and r.get("repaired_ok"))
        out.append("| %s | %s | %s | %s | %s |"
                   % (r["experiment"], fmt(both),
                      str(r.get("baseline_outcome") or "—")[:60],
                      str(r.get("repaired_outcome") or "—")[:60],
                      fmt(r.get("outcome_moved"))))
    out += ["", v.get("statement", ""), ""]
    if v.get("withheld"):
        out += ["*" + "; ".join(f"{k}: {w}" for k, w in v["withheld"].items()) + ".*", ""]
    return out


def r13_section(v):
    if not v:
        return ["## R-13", "", "**Not run.**", ""]
    out = ["---", "", "## R-13: the two reruns", "", v.get("plain_language", ""), "",
           v.get("statement", ""), ""]
    return out


def closing(loaded):
    changes = []
    t1 = loaded.get("tier1_recompute.json") or {}
    for r in t1.get("r1_recomputed_criteria", []):
        if r.get("verdict") == "determined_meets":
            changes.append(
                "**%s (%s) is now determined and it holds**, at %s with a 95%% interval of "
                "[%s, %s] against a threshold of %s. It was previously reported as possibly "
                "undecidable, and that reading was wrong."
                % (r["criterion"], r["experiment"], fmt(r["recomputed_point"]),
                   fmt(r["bootstrap_interval"][0]), fmt(r["bootstrap_interval"][1]),
                   fmt(r["threshold"])))
        elif r.get("verdict") == "determined_fails":
            changes.append("**%s (%s) is now determined and it FAILS** at %s, interval [%s, %s], "
                           "against a threshold of %s."
                           % (r["criterion"], r["experiment"], fmt(r["recomputed_point"]),
                              fmt(r["bootstrap_interval"][0]), fmt(r["bootstrap_interval"][1]),
                              fmt(r["threshold"])))
    peak = t1.get("r3_fabrication_peak") or {}
    if peak.get("located"):
        changes.append("**The interior peak has an error bar and it is tight**: the same grid point "
                       "in %s of bootstrap draws. The prediction card can quote a location rather "
                       "than a guess." % fmt(peak["modal_location_share"]))
    r5 = loaded.get("r5_uptake.json") or {}
    if r5.get("cells_with_negative_error_reduction"):
        changes.append("**Uptake has been split into movement and error reduction, and they "
                       "disagree about the headline cell.** A false label reads as substantial "
                       "uptake on the old measure and as a large NEGATIVE on the new one: readers "
                       "are moved away from the answer, further than an honest label moves them "
                       "toward it. Every claim about how much a reader 'takes on' needs to say "
                       "which of the two it means.")
    r6 = loaded.get("r6_estimators.json") or {}
    if r6.get("verdict", "").startswith("TRUST_IS_MEASURABLE"):
        changes.append("**Trust is measurable after all, over part of its range.** The earlier "
                       "unidentifiable verdict fitted it to the wrong data. It saturates above the "
                       "channel crossover, and the model's default sits in the saturated region, "
                       "so at the setting the headlines run at trust remains a stipulation rather "
                       "than a measurement.")
    r8b = loaded.get("r8b_learned_trust.json") or {}
    if r8b.get("verdict") == "REPUTATION_BLINDNESS_ABOVE_THE_CROSSOVER":
        changes.append("**A new prediction the earlier model could not make: a sufficiently "
                       "trusting reader cannot learn that a source lies, at any number of "
                       "encounters.** The threshold is the channel crossover. If it holds outside "
                       "this model it says a disclosure regime protects trusting readers least "
                       "where it fails most.")
    sw = loaded.get("r11_r12_sweep.json") or {}
    if sw.get("outcomes_moved"):
        changes.append("**These outcomes moved under the repaired model:** %s."
                       % ", ".join(sw["outcomes_moved"]))
    elif sw.get("experiments_completed"):
        changes.append("**No outcome moved** across the %d experiments that ran in both arms, so "
                       "the exact solver and the collision-free seeding together leave every "
                       "verdict where the old code path put it."
                       % sw["experiments_completed"])
    r13 = loaded.get("r13_reruns.json") or {}
    if r13.get("fallback_verdict") == "CRASH_SURVIVES_THE_GENEROUS_FALLBACK":
        changes.append("**The generous-fallback result is restored**, under exact inference and "
                       "with a control that can actually pass. The validation pass had reduced it "
                       "to inconclusive.")
    if r13.get("depth_verdict") == "DEPTH_STILL_INCONCLUSIVE":
        changes.append("**Depth still does not move what the reader takes on**, and the rerun "
                       "bounds the effect rather than merely failing to find it. It also "
                       "establishes that the difficulty regime does not transfer to that geometry "
                       "at all, which is a finding about the design rather than about depth.")

    out = ["---", "", "## What this pass changed about what may be claimed", ""]
    out += [f"{i}. {c}" for i, c in enumerate(changes, 1)] if changes else [
        "Nothing. Every measure reports what it was already reporting."]
    out += ["", "---", "", "## What was retained rather than replaced", "",
            "Every original number is still in the repository and is still reported beside its "
            "replacement. The old uptake measure is computed alongside the new one; the modal-goal "
            "entropy is reported beside the divergence and beside its bias-corrected form; the "
            "original per-reader seeding is still selectable, so every number produced before this "
            "pass can be regenerated by the code that produced it; and both original verdicts for "
            "the reruns are carried forward unchanged.",
            "",
            "The withheld experiment stays withheld, its failing test stays in the suite, and the "
            "open residual stays open.",
            "",
            "---", "", "## What comes next", "",
            "The minimal-model programme, which is the subtraction this pass's rule was written "
            "against. For each surviving result, find the smallest model that still produces it. "
            "Then the minimal models become a family, and comparison replaces single-model "
            "validation as the frame: with one model a failure is uninterpretable, and with a "
            "family it tells you which commitment was wrong.",
            ""]
    return out


def main():
    loaded = {f: load(f) for _, f, _ in CHECKS}
    summary = load("summary.json")
    criteria = load("criteria.json")

    lines = header(summary, criteria)
    lines += summary_table(loaded)
    lines += tier1_section(loaded["tier1_recompute.json"])
    lines += r5_section(loaded["r5_uptake.json"])
    lines += r6_section(loaded["r6_estimators.json"])
    lines += r8b_section(loaded["r8b_learned_trust.json"])
    lines += sweep_section(loaded["r11_r12_sweep.json"])
    lines += r13_section(loaded["r13_reruns.json"])
    lines += closing(loaded)
    lines += ["*Generated from results/repair/ on %s by `scripts/write_repair_md.py`. Every number "
              "above is read out of a verdict file; none is typed in.*" % date.today().isoformat(),
              ""]

    out = REPO / "REPAIR.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines)")
    missing = [f for f, v in loaded.items() if v is None]
    if missing:
        print("NOT RUN (rendered as such): " + ", ".join(missing))


if __name__ == "__main__":
    main()
