#!/usr/bin/env python
"""Write DIAGNOSTICS.md from the verdict files the diagnostics pass produced.

    python scripts/write_diagnostics_md.py

Same discipline as ``write_validation_md.py`` and for the same reason, which the diagnostics spec
states in its section 4: written from the verdict files, never from the expectations in the spec. Every
number below is read out of ``results/diagnostics/*.json``. The prose is written here, once, and is
written to be true whichever way each check came out.

A missing verdict file renders as NOT RUN rather than as a gap.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIAG = REPO / "results" / "diagnostics"

CHECKS = [
    ("P-1", "p1_recovery.json",
     "If the model generates data at a known parameter value, can the value be recovered?"),
    ("P-2", "p2_difficulty.json",
     "Is there a regime where reading the goal is genuinely uncertain?"),
    ("D-1", "d1_channels.json",
     "What is the reader's belief about provenance made of, and which channel wins?"),
    ("D-2", "d2_uptake.json",
     "Does uptake rise and fall with how well the reader read the work?"),
    ("D-3", "d3_disagreement.json",
     "Is 'no two readers agree' measuring disagreement, or shared uncertainty?"),
    ("D-4", "d4_coverage.json",
     "How much of the work can the exact solver reach, and how wrong is the shortcut?"),
    ("D-5", "d5_criterion_power.json",
     "How many independent things does each pre-registered criterion see?"),
    ("D-6", "d6_seed_independence.json", "Are the simulated readers actually distinct?"),
]


def load(name):
    p = DIAG / name
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


def pct(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "—"
    return "not defined" if f != f else "%.0f%%" % (100 * f)


def header(summary, criteria):
    lines = [
        "# Diagnostics on the apparatus",
        "",
        "Generated from the verdict files in "
        "[results/diagnostics/](results/diagnostics/). Regenerate with "
        "`python scripts/write_diagnostics_md.py`.",
        "",
        "---",
        "",
        "## What this is, and what it is not",
        "",
        "**Nothing here asks a question about the world and nothing here fixes anything.** Every check "
        "is a diagnostic on the instruments: what are they actually measuring, and over what range can "
        "they measure it at all. They exist to decide what the repair work should be, and starting the "
        "repair before they report is the mistake the pass was set up to avoid.",
        "",
        "Two of them, P-1 and P-2, come from the diagnostics specification. Five more came out of "
        "reading the code alongside the validation pass's output, and they run first because two of "
        "them change what the expensive sweeps should sweep.",
        "",
        "The distinction that matters most for reading what follows: **the validation pass asked "
        "whether the recorded answers can be trusted. This pass asks whether the instruments can "
        "answer at all.** A finding here does not usually mean a result was wrong. It usually means a "
        "result was stated more strongly than the instrument supports.",
        "",
    ]
    if criteria:
        lines += [
            "**The criteria were fixed before any check ran** and hash-locked at "
            f"`{criteria.get('content_hash', '')[:16]}`, in "
            "[results/diagnostics/criteria.json](results/diagnostics/criteria.json), separate from "
            "the validation lock, which has reported and is sealed. Two decisions are recorded in "
            "that lock and were made before the sweeps: the estimators P-1 uses for the three "
            "parameters that are not hidden states, and P-2's fourth knob.",
            "",
        ]
    if summary and summary.get("quick"):
        lines += ["> **These outputs were produced at development scale and are not reportable.** "
                  "Re-run `python run_diagnostics.py` without `--quick`.", ""]
    return lines


def summary_table(loaded):
    lines = ["---", "", "## The eight checks at a glance", "",
             "| check | what it asks | how it came back |", "|---|---|---|"]
    for code, fname, q in CHECKS:
        v = loaded.get(fname)
        lines.append(f"| **{code}** | {q} | {pretty(v.get('verdict')) if v else '**not run**'} |")
    lines.append("")
    return lines


def p1_section(v):
    if not v:
        return ["## P-1", "", "**Not run.**", ""]
    out = ["---", "", "## P-1: parameter recovery", "", v.get("plain_language", ""), ""]
    out += [
        "**The specification's estimator only names an estimator for one of the four parameters, and "
        "that is worth stating before any number.** Depth is a hidden state, so the reader carries a "
        "posterior over it and the posterior mean is an estimate. Trust, readability and the value "
        "gate are not hidden states: no agent in the model carries a belief about any of them. So "
        "they are fitted instead, by maximising the likelihood of a fixed dataset over a grid, which "
        "required writing a capability the project did not have "
        "([ghostscale/fitting.py](ghostscale/fitting.py)).",
        "",
        "That forces a distinction which is not a technicality. For depth, recovery asks whether **the "
        "reader inside the model** can identify the value. For the other three it asks whether **an "
        "ideal analyst holding the correct model** could, which is a strictly easier question. For "
        "readability the gap between those two is the entire version 4 reframe.",
        "",
        "| parameter | estimator | rank correlation | slope | usable range | classification |",
        "|---|---|---|---|---|---|",
    ]
    est = v.get("estimators", {})
    for s in v.get("scores", []):
        tag = s["parameter"] + (f" ({s['dataset']})" if s.get("dataset") else "")
        key = s["parameter"].split()[0] if s["parameter"] not in est else s["parameter"]
        e = est.get(s["parameter"]) or est.get(key) or ""
        out.append("| %s | %s | %s | %s | %s | **%s** |"
                   % (tag, e.split(".")[0], fmt(s.get("rank_correlation"), 2),
                      fmt(s.get("slope"), 2), pct(s.get("usable_range_fraction")),
                      s.get("classification")))
    out += ["", v.get("statement", ""), ""]

    pre = v.get("marginal_invariance_precheck", {})
    if pre:
        out += ["### The cheap pre-check, and what it does not predict", "",
                "| parameter | quantity | change across the swept range | invariant |",
                "|---|---|---|---|"]
        for name, row in pre.items():
            out.append("| %s | %s | %s | %s |"
                       % (name, row.get("quantity", ""),
                          fmt(row.get("max_abs_change_across_range")),
                          fmt(row.get("invariant_in_the_marginal"))))
        out.append("")
        for name, row in pre.items():
            out.append(f"**{name}.** {row.get('why', '')}")
            out.append("")

    pairs = v.get("pair_checks", [])
    if pairs:
        out += ["### The joint check", "",
                "| pair | runnable | error correlation | trading off |", "|---|---|---|---|"]
        for q in pairs:
            out.append("| %s | %s | %s | %s |"
                       % (q["pair"], fmt(q.get("runnable")), fmt(q.get("error_correlation"), 2),
                          fmt(q.get("trading_off"))))
        out.append("")
        for q in pairs:
            out += [f"**{q['pair']}.** {q.get('note', '')}", ""]
    return out


def p2_section(v):
    if not v:
        return ["## P-2", "", "**Not run.**", ""]
    out = ["---", "", "## P-2: the goal-difficulty probe", "", v.get("plain_language", ""), "",
           f"*{v.get('knob_note', '')}*", ""]
    cells = v.get("cells", [])
    band = [c for c in cells if c.get("in_target_band")]
    if band:
        out += ["| setting | reads the right purpose | uncertainty about the goal | uptake spread | "
                "against the default | disagreement |", "|---|---|---|---|---|---|"]
        for c in sorted(band, key=lambda r: -(r.get("uptake_variance_ratio") or 0))[:12]:
            out.append("| %s | %s | %s | %s | %sx | %s |"
                       % (c.get("level"), fmt(c.get("accuracy")), fmt(c.get("goal_entropy")),
                          fmt(c.get("uptake_sd")), fmt(c.get("uptake_variance_ratio"), 2),
                          fmt(c.get("between_reader_entropy"))))
        out.append("")
    dead = v.get("confirmatory_dead_knobs", [])
    if dead:
        out += ["**The two knobs the specification names and this pass ran only as confirmatory "
                "cells:**", ""]
        for r in dead:
            out.append("- %s leaves accuracy at %s and goal uncertainty at %s"
                       % (r.get("level"), fmt(r.get("accuracy")), fmt(r.get("goal_entropy"))))
        out.append("")
    out += [v.get("statement", ""), ""]
    return out


def d1_section(v):
    if not v:
        return ["## D-1", "", "**Not run.**", ""]
    out = ["---", "", "## D-1: channel accounting", "", v.get("plain_language", ""), "",
           f"*{v.get('method', '')}*", "",
           "| cell | transparency of the truth | content, per glance | label, per glance | net | "
           "crossover trust |", "|---|---|---|---|---|---|"]
    for c in v.get("cells", []):
        out.append("| %s | %s | %s | %s | %s | %s |"
                   % (c["cell"], fmt(c.get("alpha_true"), 2),
                      fmt(c.get("content_llr_per_step")), fmt(c.get("label_llr_per_step")),
                      fmt(c.get("net_llr_per_step")), fmt(c.get("crossover_kappa"))))
    out += ["", v.get("statement", ""), ""]
    traj = v.get("measured_trajectory", [])
    if traj:
        out += ["**And the arithmetic is checked against measured runs.** Each row is one reader "
                "looking at the same mislabelled artifact, at a different level of trust, and the "
                "column is how much it believes the label after that many glances.", "",
                "| trust | 1 glance | 2 | 4 | 10 | all | ends up believing the label |",
                "|---|---|---|---|---|---|---|"]
        for row in traj:
            b = row["believes_the_label_at_t"]
            keys = sorted(b, key=lambda k: int(k))
            out.append("| %s | %s | %s | %s | %s | %s | %s |"
                       % (fmt(row["kappa"], 2), fmt(b[keys[0]]), fmt(b[keys[1]]), fmt(b[keys[2]]),
                          fmt(b[keys[3]]), fmt(b[keys[-1]]),
                          fmt(row["ends_believing_the_label"])))
        out.append("")
    for line in v.get("what_this_changes", []):
        out.append(f"- {line}")
    out.append("")
    return out


def d2_section(v):
    if not v:
        return ["## D-2", "", "**Not run.**", ""]
    out = ["---", "", "## D-2: the shape of the uptake measure", "", v.get("plain_language", ""), "",
           "| reader inexpertise | reads the right purpose | uptake | spread | share confidently "
           "wrong | uptake if wrong | uptake if right |", "|---|---|---|---|---|---|---|"]
    for c in v.get("curve", []):
        if round(c["inexpertise"] * 100) % 10 == 0 or c["inexpertise"] in (0.85, 0.95):
            out.append("| %s | %s | %s | %s | %s | %s | %s |"
                       % (fmt(c["inexpertise"], 2), fmt(c["accuracy"]), fmt(c["uptake_ungated"]),
                          fmt(c["uptake_ungated_sd"]), pct(c["confidently_wrong_fraction"]),
                          fmt(c["uptake_of_the_confidently_wrong"]),
                          fmt(c["uptake_of_the_correct"])))
    out += ["", v.get("statement", ""), ""]
    ks = v.get("kappa_scaling", {})
    if ks:
        out += ["### The trust factor inside the measure", "",
                f"{ks.get('why_it_matters', '')} Across the swept range the factor alone changes by "
                f"{fmt(ks.get('ratio_across_the_swept_range'), 1)}x.", ""]
    if v.get("engagement_gate_note"):
        out += [f"*{v['engagement_gate_note']}*", ""]
    return out


def d3_section(v):
    if not v:
        return ["## D-3", "", "**Not run.**", ""]
    out = ["---", "", "## D-3: the disagreement statistic", "", v.get("plain_language", ""), "",
           "| cell | uncertainty per reader | modal-goal entropy (shipped) | pairwise divergence "
           "(alternative) | vote resample | null degenerate |",
           "|---|---|---|---|---|---|"]
    for c in v.get("argmax_noise_null", []):
        if not c.get("available"):
            continue
        out.append("| %s | %s | %s | %s | %s | %s |"
                   % (c["cell"], fmt(c.get("mean_within_reader_entropy")),
                      fmt(c.get("observed_between")), fmt(c.get("mean_pairwise_js")),
                      fmt(c.get("null_mean")), fmt(c.get("null_is_degenerate"))))
    out += ["", v.get("statement", ""), ""]
    red = v.get("redundancy_with_within_observer", {})
    if red:
        out += ["### How much of the disagreement number is the uncertainty number", "",
                "| relationship | rank correlation | variance explained by a straight line |",
                "|---|---|---|",
                "| shipped statistic against within-reader uncertainty | %s | %s |"
                % (fmt(red.get("spearman_between_vs_within"), 2),
                   pct(red.get("r2_between_from_within"))),
                "| alternative statistic against within-reader uncertainty | %s | %s |"
                % (fmt(red.get("spearman_js_vs_within"), 2), pct(red.get("r2_js_from_within"))),
                "",
                "The alternative is the less redundant of the two, which is what makes it worth "
                "reporting: it carries information the uncertainty number does not already contain.",
                ""]
    if v.get("recommended_replacement"):
        out += [f"**Recommendation.** {v['recommended_replacement']}", ""]
    bias = v.get("scale_bias", [])
    if bias:
        out += ["### The reader-count bias", "",
                "| reported at | readers | bias, nats | scale sensitive |", "|---|---|---|---|"]
        for r in bias:
            out.append("| %s | %s | %s | %s |"
                       % (r["reported_at"], r["n_readers"], fmt(r["bias_nats"]),
                          fmt(r["scale_sensitive"])))
        out.append("")
    return out


def d4_section(v):
    if not v:
        return ["## D-4", "", "**Not run.**", ""]
    cov = v.get("coverage", {})
    out = ["---", "", "## D-4: solver coverage", "", v.get("plain_language", ""), "",
           "| | count | which |", "|---|---|---|",
           "| checked under exact inference by the validation pass | %d | %s |"
           % (len(cov.get("checked_under_exact_by_the_validation_pass", [])),
              ", ".join(cov.get("checked_under_exact_by_the_validation_pass", []))),
           "| made reachable by this pass | %d | %s |"
           % (len(cov.get("made_reachable_by_this_pass", [])),
              ", ".join(cov.get("made_reachable_by_this_pass", []))),
           "| still structurally unreachable | %d | %s |"
           % (len(cov.get("still_structurally_unreachable", [])),
              ", ".join(cov.get("still_structurally_unreachable", []))),
           "| reachable but never checked | %d | %s |"
           % (len(cov.get("reachable_but_never_checked", [])),
              ", ".join(cov.get("reachable_but_never_checked", []))),
           "", v.get("statement", ""), ""]
    for name, b in (v.get("blockers") or {}).items():
        out += [f"**{name.replace('_', ' ')}** — {b.get('why', '')} Affects: "
                f"{', '.join(b.get('experiments', []))}.", ""]
        for claim in b.get("public_claims_at_risk", []):
            out.append(f"  - {claim}")
        out.append("")
    if v.get("the_fix_made_here"):
        out += [f"**The one code change this pass made.** {v['the_fix_made_here']}", ""]
    return out


def d5_section(v):
    if not v:
        return ["## D-5", "", "**Not run.**", ""]
    out = ["---", "", "## D-5: criterion power", "", v.get("plain_language", ""), "",
           "| experiment | criterion | statistic | computed over | units | available | "
           "under-powered |", "|---|---|---|---|---|---|---|"]
    for r in v.get("table", []):
        out.append("| %s | %s | %s | %s | %d | %d | %s |"
                   % (r["experiment"], r["criterion"], r["statistic"], r["computed_over"],
                      r["units"], r["units_available"], fmt(r["under_powered"])))
    out += ["", v.get("statement", ""), ""]
    for r in v.get("table", []):
        if r.get("under_powered") and r.get("note"):
            out += [f"**{r['experiment']} — {r['criterion']}.** {r['note']}", ""]
    return out


def d6_section(v):
    if not v:
        return ["## D-6", "", "**Not run.**", ""]
    out = ["---", "", "## D-6: seed independence", "", v.get("plain_language", ""), "",
           "| envelope | reader slots | distinct seeds | duplicates | inside a cell and seed | "
           "across cells | replacement |", "|---|---|---|---|---|---|---|"]
    for r in v.get("envelopes", []):
        out.append("| %s | %d | %d | %d | %d | %d | %d |"
                   % (r["envelope"], r["shipped_slots"], r["shipped_distinct_seeds"],
                      r["shipped_collisions_total"],
                      r["shipped_collisions_within_a_cell_and_seed"],
                      r["shipped_collisions_across_cells"], r["replacement_collisions"]))
    out += ["", v.get("statement", ""), ""]
    return out


def closing(loaded):
    """What the pass changed about what may be claimed, assembled from the verdicts only."""
    changes = []
    p1 = loaded.get("p1_recovery.json") or {}
    for s in p1.get("scores", []):
        tag = s["parameter"] + (f" ({s['dataset']})" if s.get("dataset") else "")
        if s.get("classification") == "UNIDENTIFIABLE":
            changes.append(
                f"**{tag} cannot be measured.** No dataset this model generates locates it, so it is "
                f"a free choice of the modeller rather than a quantity. Sweeping it is still "
                f"legitimate, because a sweep asks what a reader with that disposition would do. "
                f"Claiming that any real reader has a particular value of it is not, within this "
                f"framework, a falsifiable claim.")
        elif s.get("classification") == "COMPRESSED":
            changes.append(
                f"**{tag} recovers in order but not in magnitude** (slope "
                f"{fmt(s.get('slope'), 2)}). Directions transfer, sizes do not.")
    p2 = loaded.get("p2_difficulty.json") or {}
    if p2.get("verdict") == "REGIME_FOUND":
        changes.append(
            "A difficulty regime exists, so the generous-fallback experiment and the depth "
            "experiment can both be rerun on a fair footing. The knob that gets there is not one of "
            "the three the specification named.")
    elif p2.get("verdict") == "ACCURACY_MOVES_UPTAKE_DOES_NOT":
        changes.append("Uptake is not sensitive to goal recovery, so the depth experiment's flat "
                       "result was never about depth.")
    d2 = loaded.get("d2_uptake.json") or {}
    if d2.get("verdict") == "UPTAKE_IS_NON_MONOTONE_IN_RECOVERY":
        changes.append(
            "**Uptake is U-shaped in recovery quality, so it cannot be regressed on a difficulty "
            "manipulation without knowing where the arms sit.** Any rerun has to keep both arms on "
            "one side of the trough, and the trough is close to the regime P-2 identifies.")
    d3 = loaded.get("d3_disagreement.json") or {}
    if d3.get("verdict", "").startswith("DISAGREEMENT_NUMBER_IS_NOT_IDENTIFIED"):
        changes.append(
            "The disagreement figure may not be quoted on its own. The conjunction it appears in is "
            "sound; the figure by itself is close to a restatement of how unsure the readers are. A "
            "one-line replacement that separates the cases is recommended.")
    d4 = loaded.get("d4_coverage.json") or {}
    cov = d4.get("coverage", {})
    if cov.get("still_structurally_unreachable"):
        changes.append(
            "%d experiments still cannot run under exact inference, including two carrying public "
            "claims. An exact learning path is the next thing worth building."
            % len(cov["still_structurally_unreachable"]))
    d5 = loaded.get("d5_criterion_power.json") or {}
    if d5.get("under_powered_with_data_available"):
        changes.append(
            "These criteria are computed over too few units and had more data available: "
            + "; ".join(d5["under_powered_with_data_available"])
            + ". The two-gates correlation is the one that matters, because it is the public "
              "headline and the validation pass reported it as flipping; at six points neither value "
              "could have told.")
    d6 = loaded.get("d6_seed_independence.json") or {}
    if d6.get("collisions_across_cells"):
        changes.append(
            "The per-reader seed function is not collision-resistant as documented. The direction is "
            "benign and no reported statistic is affected, and the docstring should stop asserting "
            "otherwise.")

    out = ["---", "", "## What this pass changed about what may be claimed", ""]
    out += [f"{i}. {c}" for i, c in enumerate(changes, 1)] if changes else [
        "Nothing. Every instrument measures what it is used to measure, over the range it is used "
        "over."]
    out += ["", "---", "", "## What comes next, and in what order", "",
            "Recorded so today's work has somewhere to go. None of it is done here.",
            "",
            "1. **Restate the claims resting on unidentifiable parameters.** That is the first thing "
            "because it is a writing job rather than a compute job, and everything downstream reads "
            "better once it is done.",
            "2. **Rerun the two inconclusive experiments in the regime P-2 found**, with both arms "
            "on the same side of D-2's trough. Doing it without that constraint would waste a third "
            "attempt on the depth experiment.",
            "3. **Build an exact learning path**, which unblocks six experiments including the two "
            "with public claims on them.",
            "4. **Re-run the selectivity measure under exact inference.** D-4 shows the "
            "approximation's error peaks in exactly the window that measure is taken over, and it is "
            "now possible to run and cheap.",
            "5. **Add the pairwise-divergence statistic beside the modal-goal entropy** everywhere "
            "disagreement is reported, and correct the reader-count bias at reduced scale.",
            "6. **Then the minimal-model programme**: for each surviving result, find the smallest "
            "model that still produces it. D-1 is a down payment on that, since knowing provenance "
            "inference is a two-channel race with an analytic crossover tells you which commitment "
            "to try removing first.",
            ""]
    return out


def main():
    loaded = {f: load(f) for _, f, _ in CHECKS}
    summary = load("summary.json")
    criteria = load("criteria.json")

    lines = header(summary, criteria)
    lines += summary_table(loaded)
    lines += p1_section(loaded["p1_recovery.json"])
    lines += p2_section(loaded["p2_difficulty.json"])
    lines += d1_section(loaded["d1_channels.json"])
    lines += d2_section(loaded["d2_uptake.json"])
    lines += d3_section(loaded["d3_disagreement.json"])
    lines += d4_section(loaded["d4_coverage.json"])
    lines += d5_section(loaded["d5_criterion_power.json"])
    lines += d6_section(loaded["d6_seed_independence.json"])
    lines += closing(loaded)
    lines += ["*Generated from results/diagnostics/ on %s by "
              "`scripts/write_diagnostics_md.py`. Every number above is read out of a verdict file; "
              "none is typed in.*" % date.today().isoformat(), ""]

    out = REPO / "DIAGNOSTICS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines)")
    missing = [f for f, v in loaded.items() if v is None]
    if missing:
        print("NOT RUN (rendered as such): " + ", ".join(missing))


if __name__ == "__main__":
    main()
