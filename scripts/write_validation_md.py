#!/usr/bin/env python
"""Write VALIDATION.md from the verdict files the validation pass produced.

    python scripts/write_validation_md.py

-----------------------------------------------------------------------------------------
WHY THIS IS A GENERATOR AND NOT A DOCUMENT SOMEBODY WRITES.

The validation spec's own constraint, in as many words: "`VALIDATION.md` is written from the verdict
files after the runs, never from the expectations in this document." A hand-written summary of a
validation pass is a summary of what the person writing it remembers or hopes the pass said, and the
whole point of the pass is that this project has been bitten seven times by exactly that gap between
what an instrument was asked and what it answered.

So every number, every verdict string and every statement in `VALIDATION.md` is read out of
`results/validation/*.json`. The prose that surrounds them is written here, once, and it is written
to be true whichever way each check came out — which is why the branch wording lives in the check
modules rather than here. If a check produced an unwelcome answer, this file renders the unwelcome
answer, because it has nothing else to render.

A missing verdict file produces a NOT RUN row rather than a gap. A blank in a validation table reads
as "fine" to every reader who does not check.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VDIR = REPO / "results" / "validation"

CHECKS = [
    ("V-1", "v1_solver.json", "Does the inference approximation distort the headline results?"),
    ("V-2", "v2_nulls.json", "Would a model of this shape produce these results anyway?"),
    ("V-3", "v3_robustness.json", "Are the headline results knife-edge or robust?"),
    ("V-4", "v4_construction.json", "What is forced by construction?"),
    ("V-5", "v5_superseded_criteria.json",
     "Would any verdict change under the criterion as originally written?"),
    ("V-6", "v6_consistency.json", "Do the versions agree where they measure the same thing?"),
    ("V-7", "v7_seed_and_scale.json",
     "Do the headlines survive a different seed block and twice the scale?"),
    ("V-8", "v8_reimplementation.json",
     "Does the strongest result survive being rebuilt from its own description?"),
    ("V-9", "v9_out_of_sample_prediction.json", "Can this project be wrong about something in "
                                                "advance?"),
]


def load(name: str) -> dict | None:
    p = VDIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def pretty(verdict: str) -> str:
    return verdict.replace("_", " ").lower() if verdict else "not run"


def fmt(x, places=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)):
        return str(x)
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:
        return "not determined"
    if f == 0:
        return "0"
    if abs(f) >= 1000 or abs(f) < 1e-3:
        return f"{f:.2e}"
    return f"{f:.{places}f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------- #
# Sections.
# --------------------------------------------------------------------------- #
def header(summary: dict | None, criteria: dict | None) -> list[str]:
    scale = (criteria or {}).get("scale", {})
    lines = [
        "# What was checked, and what the checks came back with",
        "",
        "This page is generated from the verdict files in "
        "[results/validation/](results/validation/). Nothing in it is written by hand except the "
        "explanations, and none of the explanations can change a number. Regenerate it with "
        "`python scripts/write_validation_md.py`.",
        "",
        "---",
        "",
        "## Read this part first",
        "",
        "**This is exploratory modelling, and all of it is confirmatory by construction.** Every "
        "prediction in this repository came from one prior theory. The simulations write that "
        "theory down as working code and check whether its parts fit together, which means "
        "agreement between the model and the theory is the *expected* outcome rather than evidence "
        "for it.",
        "",
        "That is a legitimate way to work and it has one specific failure. **A model can reproduce "
        "its own assumptions and be indistinguishable, from outside, from a model that discovered "
        "something.** Telling those two apart is the entire job of the checks below.",
        "",
        "The risk is not hypothetical here. Seven times in this project's history an instrument was "
        "quietly answering a different question than the one being asked, and every time the wrong "
        "answer looked completely reasonable. A stale threshold file overrode a requested sample "
        "size. A statistic was computed at the wrong timestep. A lucky seed produced a confirmation "
        "that vanished across four draws. An inference shortcut was confidently blind to the "
        "quantity being measured. Six of those were caught by checks written for other reasons. "
        "**This pass makes the checking systematic instead of lucky.**",
        "",
        "A check that fails is not deleted. It is reported with its failure attached, in the same "
        "place as the claim.",
        "",
    ]
    if scale:
        lines += [
            f"**Scale.** The checks ran at {scale.get('n_observers')} simulated readers and "
            f"{scale.get('n_seeds')} random seeds per cell, with "
            f"{scale.get('random_model_draws')} random-model draws. That is reduced from the "
            f"headline experiments, deliberately and as the specification permits: the question is "
            f"whether a conclusion *survives* a change of solver or of parameter, which needs "
            f"enough readers to resolve the effect rather than the precision the headline number "
            f"was originally quoted at. Every result below carries the scale it ran at.",
            "",
        ]
    if criteria:
        lines += [
            f"**The criteria were fixed before the checks ran** and hash-locked: "
            f"`{criteria.get('content_hash', '')[:16]}`, in "
            f"[results/validation/criteria.json](results/validation/criteria.json). Editing that "
            f"file after the fact makes the whole pass refuse to run.",
            "",
        ]
    if summary and summary.get("quick"):
        lines += ["> **These outputs were produced at development scale and are not reportable.** "
                  "Re-run `python run_validation.py` without `--quick`.", ""]
    return lines


def summary_table(loaded: dict) -> list[str]:
    lines = [
        "---", "",
        "## The nine checks at a glance",
        "",
        "| check | what it asks | how it came back |",
        "|---|---|---|",
    ]
    for code, fname, question in CHECKS:
        v = loaded.get(fname)
        verdict = pretty(v.get("verdict")) if v else "**not run**"
        lines.append(f"| **{code}** | {question} | {verdict} |")
    lines.append("")
    return lines


def v1_section(v: dict | None) -> list[str]:
    if not v:
        return ["## V-1: the solver check", "", "**Not run.**", ""]
    out = ["---", "", "## V-1: does the fast approximation distort the results?", "",
           v.get("plain_language", ""), ""]
    out += [
        "Every experiment before version 5 used pymdp's variational solver, which keeps the "
        "reader's beliefs about different unknowns *separately* and updates each one using an "
        "average over the others. That shortcut is known to have been badly wrong once: version 5 "
        "found it returning the shallow answer for every artifact, confidently, while exact "
        "arithmetic on the same observations recovered depth correctly. Version 5 worked around "
        "that one case. It did not establish that the earlier results were safe.",
        "",
        "So the shortcut was removed rather than worked around. "
        "[`ghostscale/exact.py`](ghostscale/exact.py) carries the reader's belief over every "
        "combination of unknowns at once and updates it by Bayes' rule with no independence "
        "assumption anywhere. The five headline experiments then re-ran **through their own "
        "unmodified code**, twice, with one setting flipped. Anything that moved was moved by the "
        "factorisation and by nothing else.",
        "",
    ]
    sanity = v.get("solver_sanity_check", {})
    if sanity:
        out += [
            f"Before any comparison: on a construction where the shortcut is provably exact, the "
            f"two agents agree to {fmt(sanity.get('max_goal_discrepancy'))}. A disagreement there "
            f"would have meant the new code was wrong rather than the old code, and everything "
            f"below would have been meaningless.",
            "",
        ]
    out += ["| result | quantity | approximate | exact | committed full-scale | survives |",
            "|---|---|---|---|---|---|"]
    for row in v.get("table", []):
        out.append(
            f"| {row['target'].upper()} | {row['quantity'].replace('_', ' ')} | "
            f"{fmt(row['approx_reduced'])} | {fmt(row['exact_reduced'])} | "
            f"{fmt(row.get('committed_full_scale'))} | {fmt(row['agrees'])} |")
    out += ["", f"**{v.get('statement', '')}**", ""]
    moved = [f"{t}" for t, d in (v.get("targets") or {}).items()
             if not d.get("outcome_survives")]
    if moved:
        out += [f"The overall outcome string changed under exact inference for: "
                f"{', '.join(x.upper() for x in moved)}. Those are the rows to read closely.", ""]
    return out


def v2_section(v: dict | None) -> list[str]:
    if not v:
        return ["## V-2", "", "**Not run.**", ""]
    a = v.get("scrambled_provenance", {})
    b = v.get("random_parameter_false_positive_rate", {})
    out = ["---", "",
           "## V-2: would a model of this shape produce these results anyway?", "",
           "### Part one: break the thing the result is supposed to be about", ""]
    out += [
        "The label effect is supposed to be about what a provenance tier tells you about content. "
        "If that is true, then destroying the connection between provenance and content should "
        "remove the effect, and *reversing* it should leave the effect the same size pointing the "
        "other way. Both were run, because the first on its own cannot distinguish \"provenance is "
        "doing the work\" from \"the model is delicate and any change breaks it\".",
        "",
        "| condition | how many times more doubt an honest label leaves |",
        "|---|---|",
        f"| intact | {fmt(a.get('effect_intact'), 1)}x |",
        f"| connection destroyed (every tier equally transparent) | "
        f"{fmt(a.get('effect_flattened'), 2)}x |",
        f"| connection reversed | {fmt(a.get('effect_permuted'), 3)}x |",
        "",
        f"**{a.get('statement', '')}**",
        "",
    ]
    dev = a.get("deviation")
    if dev:
        out += [
            "> **A criterion was restated during this check, and it is logged here rather than "
            f"quietly applied.** {dev.get('what_changed')}. Why: {dev.get('why')} The original "
            f"clause is retained, still computed, and {dev.get('original_clause_outcome', '')}.",
            "",
        ]
    out += ["### Part two: the false-positive rate of the apparatus", "",
            b.get("plain_language", ""), ""]
    arms = b.get("arms") or {}
    if arms:
        out += ["| how the settings were drawn | confident under a false label | disagreement near "
                "its ceiling | clears the whole bar |", "|---|---|---|---|"]
        for name, arm in arms.items():
            cr = arm.get("clause_rates", {})
            out.append(
                f"| {name.replace('_', ' ')} | {_pct(cr.get('confident_under_false_label'))} | "
                f"{_pct(cr.get('disagreement_near_ceiling'))} | "
                f"{_pct(arm.get('false_positive_rate'))} |")
        out.append("")
    out += [f"**{b.get('statement', '')}**", ""]
    if b.get("confident_invention_is_architectural"):
        out += [
            "This is the most consequential thing in the pass and it deserves saying plainly. "
            "**The confident half of the headline is architectural.** A reader built to this shape, "
            "with its settings thrown away and replaced at random, still becomes certain about "
            "machine-made work when a label tells it a person was involved. What a random reader "
            "does *not* produce is the disagreement, the part where no two readers land on the same "
            "answer, and that is what makes the certainty *invention* rather than shared "
            "error. The theory is entitled to the second half. It is not entitled to the first.",
            "",
        ]
    return out


def v3_section(v: dict | None) -> list[str]:
    if not v:
        return ["## V-3", "", "**Not run.**", ""]
    out = ["---", "", "## V-3: knife-edge, or robust?", "", v.get("plain_language", ""), "",
           f"*{v.get('framing', '')}*", ""]
    for name, h in (v.get("headlines") or {}).items():
        out += [f"### {h.get('plain_language', name)}", "",
                f"- holds in **{h.get('holds')} of {h.get('cells_run')}** swept cells "
                f"({h.get('weakens')} weaken, {h.get('flips')} reverse, "
                f"{h.get('unreachable')} could not be built)",
                f"- reported as tuned: **{fmt(h.get('reported_as_tuned'))}**"]
        if h.get("flipped_by"):
            out.append(f"- lost when these change: {', '.join(sorted(set(h['flipped_by'])))}")
        if h.get("peak_locations_observed") is not None:
            out.append(f"- peak locations seen across the sweep: "
                       f"{', '.join(fmt(x, 2) for x in h['peak_locations_observed'])}")
        if not h.get("boolean_gate_applied", True):
            out.append(f"- **caveat:** {h.get('boolean_gate_note')}")
        out += ["", h.get("scope_sentence", ""), ""]
    ns = v.get("not_swept") or {}
    if ns:
        out += ["### What was not swept, said rather than left blank", ""]
        for k, why in ns.items():
            out.append(f"- **{k.replace('_', ' ')}:** {why}")
        out.append("")
    return out


def v4_section(v: dict | None) -> list[str]:
    if not v:
        return ["## V-4", "", "**Not run.**", ""]
    out = ["---", "", "## V-4: what is forced by construction?", "",
           v.get("plain_language", ""), "",
           "| claim | said to depend on | alteration | what happened |", "|---|---|---|---|"]
    for r in v.get("alterations", []):
        out.append(f"| {r['claim']} | {r['downstream_of']} | {r['alteration']} | "
                   f"**{pretty(r['outcome'])}** |")
    out += ["", f"**{v.get('statement', '')}**", ""]
    for r in v.get("alterations", []):
        if r.get("outcome") == "alteration_unreachable":
            out += [f"> *{r['downstream_of']}* could not be altered: the model asserts it at "
                    f"construction and refused. That makes it a definition rather than a setting, "
                    f"and the claim above it is downstream of a definition, which is a weaker "
                    f"thing than a claim downstream of a measurement, and is stated as such.", ""]
    aud = v.get("architectural_audit", {})
    part = aud.get("disjoint_feature_partition", {})
    if part:
        out += ["### The feature partition, audited on its own", "",
                f"**What it is.** {part.get('what_it_is')}", "",
                f"**Chosen on theoretical grounds:** {fmt(part.get('chosen_on_theoretical_grounds'))}. "
                f"{part.get('why_it_was_adopted')}", "",
                "**Everything downstream of it:**", ""]
        for c in part.get("claims_downstream_of_it", []):
            out.append(f"- *{c['claim']}* ({c['experiment']}): {c['why']}")
        out += ["", f"**What a different partitioning would do.** "
                f"{part.get('what_a_different_partitioning_would_do')}", ""]
    eff = aud.get("rebuilt_effort_parameter", {})
    if eff:
        out += ["### The rebuilt effort parameter, audited on its own", "",
                f"**The position taken:** {eff.get('position')}.", "",
                f"**The narrower question that can still be asked:** {eff.get('narrower_question')}",
                ""]
        if eff.get("separation_at_pinned_effort") is not None:
            out += [f"Measured separation with the effort axis pinned: "
                    f"**{fmt(eff.get('separation_at_pinned_effort'))}**.", ""]
        if eff.get("n21_verdict_under_exact_inference"):
            out += [f"The depth-versus-effort null returns "
                    f"`{eff['n21_verdict_under_exact_inference']}` under exact inference.", ""]
        out += [eff.get("reading", ""), ""]
    return out


def v5_section(v: dict | None) -> list[str]:
    if not v:
        return ["## V-5", "", "**Not run.**", ""]
    out = ["---", "", "## V-5: every measurement rule that was changed after the fact", "",
           v.get("plain_language", ""), "",
           "| deviation | what changed | kind | recomputed here | would a verdict change |",
           "|---|---|---|---|---|"]
    for r in v.get("table", []):
        out.append(f"| {r['deviation']} | {r['what_changed']} | {r['kind']} | "
                   f"{fmt(r.get('computable'))} | "
                   f"{fmt(r.get('verdict_would_change')) if r.get('computable') else '—'} |")
    out += ["", f"**{v.get('statement', '')}**", ""]
    for r in v.get("table", []):
        if r.get("computable") and r.get("reading"):
            out += [f"**{r['deviation']}.** {r['reading']}"]
            if r.get("recomputation_note"):
                out.append(f" *{r['recomputation_note']}*")
            out.append("")
        elif r.get("why_not_recomputed"):
            out += [f"**{r['deviation']}.** Not a criterion recomputation. "
                    f"{r['why_not_recomputed']}", ""]
    return out


def v6_section(v: dict | None) -> list[str]:
    if not v:
        return ["## V-6", "", "**Not run.**", ""]
    out = ["---", "", "## V-6: do the versions agree with each other?", "",
           v.get("plain_language", ""), "",
           f"Boundary reductions are the checks that each version can become the previous one "
           f"when its new machinery is switched off. They "
           f"**{'all hold' if v.get('boundary_regressions_pass') else 'DO NOT all hold'}**.",
           ""]
    pairs = v.get("shared_quantities") or []
    if pairs:
        out += ["| quantity measured twice | in | values | agree |", "|---|---|---|---|"]
        for p in pairs:
            out.append(f"| {p['quantity']} | {', '.join(p['measured_in'])} | "
                       f"{' vs '.join(fmt(x) for x in p['values'])} | {fmt(p['agree'])} |")
        out.append("")
        for p in pairs:
            out += [f"*{p['quantity']}:* {p.get('note', '')}", ""]
    out += [f"**{v.get('statement', '')}**", ""]
    return out


def v7_section(v: dict | None) -> list[str]:
    if not v:
        return ["## V-7", "", "**Not run.**", ""]
    out = ["---", "", "## V-7: a different set of random seeds, and twice the readers", "",
           v.get("plain_language", ""), "",
           "| result | condition | reference | re-measured | moved by | conclusion holds |",
           "|---|---|---|---|---|---|"]
    for r in v.get("table", []):
        out.append(f"| {r['headline'].replace('_', ' ')} | {r['arm'].replace('_', ' ')} | "
                   f"{fmt(r['reference_effect'])} | {fmt(r['arm_effect'])} | "
                   f"{_pct(r.get('relative_drift'))} | {fmt(r['verdict_holds'])} |")
    out += ["", f"**{v.get('statement', '')}**", ""]
    return out


def v8_section(v: dict | None) -> list[str]:
    if not v:
        return ["## V-8", "", "**Not run.**", ""]
    out = ["---", "", "## V-8: rebuilt from scratch, from the description alone", "",
           v.get("plain_language", ""), "", f"*{v.get('independence', '')}*", ""]
    orig, re_ = v.get("original", {}), v.get("reimplementation", {})
    out += ["| | original | independent rebuild |", "|---|---|---|",
            f"| how much further the reader moves under a false label | "
            f"{fmt(orig.get('uptake_multiple'), 1)}x | {fmt(re_.get('uptake_multiple'), 1)}x |",
            f"| uptake tracks the reader's own depth estimate | "
            f"{fmt(orig.get('update_tracks_mu_rho'), 2)} | "
            f"{fmt(re_.get('update_tracks_believed_depth_rho'), 2)} |",
            "",
            f"**{v.get('statement', '')}**", ""]
    if re_.get("limitation_reproduced"):
        out += [f"> {re_['limitation_reproduced']}", ""]
    return out


def v9_section(v: dict | None) -> list[str]:
    if not v:
        return ["## V-9", "", "**Not run.**", ""]
    p = v.get("prediction", {})
    out = ["---", "", "## V-9: one prediction, written down before the experiment exists", "",
           v.get("plain_language", ""), "",
           f"**{p.get('title')}**", "", f"*{p.get('status')}*", "",
           f"**Why this one.** {p.get('why_this_one')}", "",
           f"**The setup.** {p.get('the_setup')}", "", "**The prediction.**", ""]
    for k, val in (p.get("the_prediction") or {}).items():
        out.append(f"- **{k.replace('_', ' ')}:** {val}")
    out += ["", "**Named ways it can fail.** All four written before anything was built.", ""]
    for k, val in (p.get("named_failure_branches") or {}).items():
        out.append(f"- **{k}:** {val}")
    out += ["", f"**What is and is not committed.** {p.get('what_is_committed_and_what_is_not')}",
            "", f"**How it gets scored.** {p.get('how_this_gets_scored')}", "",
            f"Locked at `{(v.get('content_hash') or '')[:16]}`; the lock is "
            f"{'intact' if v.get('lock_intact') else '**BROKEN**'}.", ""]
    return out


def _pct(x):
    if x is None:
        return "—"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:
        return "not determined"
    return f"{100 * f:.1f}%"


def closing(loaded: dict) -> list[str]:
    """What the pass changed about what may be claimed. Assembled from the verdicts only."""
    changes = []
    v1 = loaded.get("v1_solver.json") or {}
    if v1.get("verdict") == "PEAK_MOVED_RE_ANCHOR_EVERYTHING":
        changes.append("Every claim anchored to the location of the readability-axis peak is "
                       "re-anchored, and the prediction card written for a human study moves with "
                       "it.")
    elif v1.get("verdict") == "A_VERDICT_FLIPPED_UNDER_EXACT_INFERENCE":
        changes.append("At least one verdict is a property of the inference shortcut rather than "
                       "of the model. Those claims carry the failure in the same cell as the "
                       "claim.")
    elif v1.get("verdict") == "VERDICTS_SURVIVE_MAGNITUDES_MOVE":
        changes.append("The conclusions survive exact inference; some of the specific numbers do "
                       "not, and are quoted as solver-dependent.")
    b = (loaded.get("v2_nulls.json") or {}).get(
        "random_parameter_false_positive_rate", {})
    if b.get("confident_invention_is_architectural"):
        changes.append("Confident belief under a false provenance label is reported as a property "
                       "of this architecture, because a randomly parameterised model of the same "
                       "shape produces it too. What the theory keeps is the disagreement.")
    if b.get("headline_false_positive_rate") is not None:
        changes.append(f"The apparatus's own false-positive rate is "
                       f"{_pct(b['headline_false_positive_rate'])}, and borderline findings are "
                       f"read against that rather than against zero.")
    v3 = loaded.get("v3_robustness.json") or {}
    for name, h in (v3.get("headlines") or {}).items():
        if h.get("reported_as_tuned"):
            changes.append(f"\"{h.get('plain_language', name)}\" is reported as tuned: it depends "
                           f"on where the settings were left.")
    v5 = loaded.get("v5_superseded_criteria.json") or {}
    if v5.get("verdicts_that_would_change"):
        changes.append(f"These deviations do decide a verdict and carry the original outcome "
                       f"beside the claim: {', '.join(v5['verdicts_that_would_change'])}.")
    v7 = loaded.get("v7_seed_and_scale.json") or {}
    if v7.get("under_powered"):
        changes.append(f"Under-powered at this scale and quoted as approximate: "
                       f"{', '.join(v7['under_powered'])}.")
    if v7.get("verdicts_that_flipped"):
        changes.append(f"Does not survive a change of seed block or a doubling of scale: "
                       f"{', '.join(v7['verdicts_that_flipped'])}.")
    v8 = loaded.get("v8_reimplementation.json") or {}
    if v8.get("verdict") == "MECHANISM_REPLICATES_MAGNITUDE_DOES_NOT":
        changes.append("The size of the label effect may not be quoted as though it transfers "
                       "outside this model. The direction may.")
    elif v8.get("verdict") == "DOES_NOT_REPLICATE":
        changes.append("The strongest result is out of the public-facing material until the "
                       "failure to replicate is understood.")

    out = ["---", "", "## What this pass changed about what may be claimed", ""]
    if changes:
        out += [f"{i}. {c}" for i, c in enumerate(changes, 1)]
    else:
        out += ["Nothing. Every headline survived every check at the scale run, which is the "
                "outcome that requires the most scepticism from a reader and the least revision "
                "from the author."]
    out += ["", "---", "",
            "## What this pass does not do", "",
            "- It does not make the work confirmatory-free. Every prediction still came from one "
            "prior theory, and no amount of internal checking changes that. The checks bound how "
            "much of the agreement is the theory and how much is the apparatus; they do not "
            "convert one into the other.",
            "- It does not test anything against people. There is no human data anywhere in this "
            "repository and nothing here is evidence about what real readers do.",
            "- It leaves the withheld experiment withheld, its failing test in the suite, and the "
            "open residual open.",
            "- The out-of-sample prediction in V-9 is written and not yet built. Until it is, the "
            "project has no forward test, and that is the largest single thing it owes.",
            ""]
    return out


def main():
    loaded = {fname: load(fname) for _, fname, _ in CHECKS}
    summary = load("summary.json")
    criteria = load("criteria.json")

    lines = header(summary, criteria)
    lines += summary_table(loaded)
    lines += v1_section(loaded["v1_solver.json"])
    lines += v2_section(loaded["v2_nulls.json"])
    lines += v3_section(loaded["v3_robustness.json"])
    lines += v4_section(loaded["v4_construction.json"])
    lines += v5_section(loaded["v5_superseded_criteria.json"])
    lines += v6_section(loaded["v6_consistency.json"])
    lines += v7_section(loaded["v7_seed_and_scale.json"])
    lines += v8_section(loaded["v8_reimplementation.json"])
    lines += v9_section(loaded["v9_out_of_sample_prediction.json"])
    lines += closing(loaded)
    lines += [f"*Generated from results/validation/ on {date.today().isoformat()} by "
              f"`scripts/write_validation_md.py`. Every number above is read out of a verdict "
              f"file; none is typed in.*", ""]

    out = REPO / "VALIDATION.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines)")
    missing = [f for f, v in loaded.items() if v is None]
    if missing:
        print("NOT RUN (rendered as such): " + ", ".join(missing))


if __name__ == "__main__":
    main()
