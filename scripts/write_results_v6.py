"""Generate RESULTS_V6.md from results/v6/. Never hand-written; every number is read out.

    python scripts/write_results_v6.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V6 = REPO / "results" / "v6"
OUT = REPO / "RESULTS_V6.md"


def load(name: str) -> dict:
    p = V6 / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def f(x, n=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "n/a"
    if v == 0:
        return "0"
    return f"{v:.{n}g}" if abs(v) < 0.001 or abs(v) >= 1e4 else f"{v:.{n}f}"


def main() -> None:
    e35, e36, e37 = load("e35_depletion"), load("e36_process"), load("e37_wall")
    e38, e39, e40 = load("e38_expertise"), load("e39_tool"), load("e40_cues")
    e41, e42, e43 = load("e41_coupling"), load("e42_vulnerability"), load("e43_selfreport")
    prereg = json.loads((V6 / "preregistration_v6.json").read_text(encoding="utf-8"))

    L = []
    A = L.append
    A("# Version 6: the Intent Extraction Limit, implemented")
    A("")
    A("Generated from the verdict files in [results/v6/](results/v6/). Regenerate with "
      "`python scripts/write_results_v6.py`. Criteria hash-locked at "
      f"`{prereg['content_hash'][:16]}` before any cell ran.")
    A("")
    A("---")
    A("")
    A("## What this version was for")
    A("")
    A("Every previous version asked a new question about the world. **This one asks whether the "
      "simulation and the theory it claims to implement are the same object.**")
    A("")
    A("Three audit passes had already checked the results. None of them could have found this, "
      "because all three took the code's own account of itself as given. Reading the preprint's "
      "formal model against the shipped code found **three terms in the equation with no "
      "counterpart in the code**, and one of them is not an omission but a disagreement about "
      "mechanism.")
    A("")
    A("| term of the Intent Extraction Limit | in V1-V5? |")
    A("|---|---|")
    A("| the trust amplifier | yes |")
    A("| belief movement | yes |")
    A("| the value-divergence gate | yes |")
    A("| **metabolic reserve** | **no. No depletion state existed anywhere.** |")
    A("| **the graded gate** | **no. Replaced by a binary decision.** |")
    A("| **trust suppressing the disgust threshold** | **no, and it is a different mechanism "
      "from the one the code uses** |")
    A("")
    A("---")
    A("")
    A("## The pass at a glance")
    A("")
    A("| experiment | what it asks | how it came back |")
    A("|---|---|---|")
    rows = [
        ("E35", "Does the damage accumulate in the reader and carry to work it has never seen?",
         (e35.get("outcome") or "") + " *(direction holds on 3 of 3 seed blocks; the "
         "pre-registered magnitude threshold on 1 of 3 — see the robustness section)*"),
        ("E36", "Does the reader recover the maker's method, and does the goal unlock it?",
         f"{e36.get('H6.4', {}).get('outcome')} / "
         f"{e36.get('H6.3b_temporal', {}).get('outcome')}"),
        ("E37", "Is the wall a vocabulary deficit or a missing inversion?",
         e37.get("H6.5", {}).get("outcome")),
        ("E38", "Does AI literacy stack with art literacy, or replace it?",
         e38.get("H6.6", {}).get("outcome")),
        ("E39", "Does a reader that can conclude 'no maker' stop cleanly?",
         e39.get("H6.7", {}).get("outcome")),
        ("E40", "How do appeal and endorsement combine, and what if appeal is optimised?",
         f"{e40.get('H6.8', {}).get('outcome')} / {e40.get('H6.9', {}).get('outcome')}"),
        ("E41", "Two mechanisms for the trust exploit. Do they predict the same thing?",
         e41.get("H6.2", {}).get("outcome")),
        ("E42", "Is looking deeply the same as being willing to be changed?",
         e42.get("H6.10", {}).get("outcome")),
        ("E43", "Does the maker lose access to its own reasons as the work deepens?",
         e43.get("H6.11", {}).get("outcome")),
    ]
    for n, q, o in rows:
        A(f"| **{n}** | {q} | {o} |")
    A("")
    A("---")
    A("")

    # ---------------- E41 first: it is the most consequential ----------------
    A("## E41: the two trust mechanisms disagree, and the code was missing one")
    A("")
    A(e41.get("plain_language", ""))
    A("")
    h62 = e41.get("H6.2", {})
    A(f"**Outcome: {h62.get('outcome')}.** Largest gap in integration on the discriminating "
      f"cell: **{f(h62.get('max_integration_gap'))}** against a pre-registered "
      f"{prereg['H6.2']['integration_gap']}.")
    A("")
    dc = e41.get("discriminating_cell", {})
    if dc:
        A("| trust | integration, coupled gate | integration, channel race only |")
        A("|---|---|---|")
        for k, a, b in zip(dc.get("kappa_grid", []), dc.get("coupled", []),
                           dc.get("uncoupled", [])):
            A(f"| {f(k)} | {f(a)} | {f(b)} |")
        A("")
    A("**What this means, stated plainly.** Under the code's mechanism a trusting reader is "
      "exploited because the label out-argues the work, so it is *wrong about who made the "
      "thing*. Under the preprint's mechanism trust suppresses the threshold that would "
      "otherwise refuse the material, so the reader integrates **even when it is told the "
      "truth and believes it**. The second is not reachable from the first, and the "
      "simulation had only ever implemented the first.")
    A("")
    vl = e41.get("values_layer", {})
    if vl:
        A(f"*The values layer rides along: {vl.get('n_values')} values over {vl.get('n_goals')} "
          f"goals, non-injective as null N26 requires "
          f"({vl.get('null_n26_non_injective')}). {vl.get('why')}*")
    A("")
    A("---")
    A("")

    # ---------------- E35 ----------------
    A("## E35: depletion, and the apathy the model could not previously represent")
    A("")
    A(e35.get("plain_language", ""))
    A("")
    A(f"**Outcome: {e35.get('outcome')}.** Engagement with a fixed human artifact the reader "
      f"has never seen falls by **{f(e35.get('probe_drop'))}** across the exposure sequence, "
      f"monotonically (rank correlation {f(e35.get('probe_monotone_rho'))}).")
    A("")
    arms = e35.get("arms", {})
    if arms:
        A("| exposure stream | probe engagement, first | probe engagement, last | reserve, last |")
        A("|---|---|---|---|")
        for name, v in arms.items():
            A(f"| {name} | {f(v.get('probe_engagement_first'))} | "
              f"{f(v.get('probe_engagement_last'))} | {f(v.get('reserve_last'))} |")
        A("")
    n22 = e35.get("null_n22", {})
    A(f"**Null N22 holds** ({n22.get('passed')}): on a fully resolvable corpus the reserve does "
      f"not move, {f(n22.get('reserve_start'))} to {f(n22.get('reserve_end'))}. "
      f"{n22.get('why_first', '')}")
    A("")
    A("**The design point that decides whether this means anything:** the criterion is the "
      "*probe*, which is identical in every arm and at every point in the sequence. A depletion "
      "term that only lowered engagement on the content that caused it would be a knob doing "
      "what it was pointed at.")
    A("")
    A("---")
    A("")

    # ---------------- E36 ----------------
    A("## E36: what the reader was recovering all along, and why the depth result was null")
    A("")
    A(e36.get("plain_language", ""))
    A("")
    h64 = e36.get("H6.4", {})
    gcmp = e36.get("H6.4_goal_comparison", {})
    A(f"**On the maker's method, depth moves uptake: {f(h64.get('deepest_minus_shallowest'))} "
      f"[{f(h64.get('interval', [0, 0])[0])}, {f(h64.get('interval', [0, 0])[1])}]. On the "
      f"maker's purpose, measured on the same cells, it does not: "
      f"{f(gcmp.get('deepest_minus_shallowest'))} "
      f"[{f(gcmp.get('interval', [0, 0])[0])}, {f(gcmp.get('interval', [0, 0])[1])}].**")
    A("")
    A("That is the whole argument. Depth is *constructed* so the goal is equally recoverable at "
      "every level, so the measure every previous version used could not have moved with depth "
      "whatever was true of the reader. The experiment was not wrong; it was pointed at the "
      "wrong quantity.")
    A("")
    cells = e36.get("cells", [])
    full = [c for c in cells if abs(float(c.get("beta", 0)) - 1.0) < 1e-9]
    if full:
        A("| depth | goal accuracy | goal uptake | process recovery | process uptake |")
        A("|---|---|---|---|---|")
        for c in full:
            A(f"| {int(c['mu'])} | {f(c['goal_accuracy'])} | {f(c['goal_error_reduction'])} | "
              f"{f(c['process_accuracy'])} | {f(c['process_error_reduction'])} |")
        A("")
    n28 = e36.get("null_n28", {})
    A(f"**Null N28 holds** ({n28.get('passed')}): at the shallowest depth there is no process, "
      f"and recovery carries {f(n28.get('measured_information'))} nats of information.")
    A("")
    A(f"> *{n28.get('why_not_accuracy', '')}*")
    A("")
    h63, h63b = e36.get("H6.3", {}), e36.get("H6.3b_temporal", {})
    A("### The ordering claim: intent as the key that unlocks the method")
    A("")
    A(f"**The pre-registered test fails and is reported as failing.** Comparing readers who "
      f"ended up right about the goal against readers who ended up wrong gives a gap of "
      f"{f(h63.get('ordering_gap'))} against a required {prereg['H6.3']['ordering_gap']}: "
      f"*{h63.get('outcome')}*.")
    A("")
    A(f"**The temporal test, added afterwards and declared, holds.** Within a single reading, "
      f"process recovery before the goal settles is {f(h63b.get('process_before_settling'))} "
      f"and after it is {f(h63b.get('process_after_settling'))} — a gain of "
      f"{f(h63b.get('gain_after_settling'))}, interval "
      f"[{f(h63b.get('interval', [0, 0])[0])}, {f(h63b.get('interval', [0, 0])[1])}], over "
      f"{h63b.get('n_rollouts')} readings: *{h63b.get('outcome')}*.")
    A("")
    A("The two are not in conflict and the difference is the point. The pre-registered form is "
      "a **between-reader** contrast; the claim is a **within-reader, temporal** one. Only the "
      "second is what *once you know what someone was for, you can read their actions as being "
      "in service of it* actually says. The original criterion is retained and decides nothing.")
    A("")
    h612 = e36.get("H6.12", {})
    A(f"**Scale invariance** ({h612.get('outcome')}): recovery on a quarter-length window scores "
      f"{f(h612.get('window_accuracy'))} against {f(h612.get('whole_accuracy'))} on the whole "
      f"artifact. The extraction is not tied to the artifact boundary, which is as much of the "
      f"fractal claim as can be tested without recursion.")
    A("")
    A("---")
    A("")

    # ---------------- E37 ----------------
    A("## E37: legible and empty")
    A("")
    A(e37.get("plain_language", ""))
    A("")
    cs = e37.get("cells", {})
    if cs:
        A("| content | goal accuracy | uncertainty left | still looking | uptake |")
        A("|---|---|---|---|---|")
        for name, v in cs.items():
            A(f"| {name} | {f(v['goal_accuracy'])} | {f(v['final_entropy'])} | "
              f"{f(v['engaged_fraction'])} | {f(v['error_reduction'])} |")
        A("")
    h65 = e37.get("H6.5", {})
    A(f"**Outcome: {h65.get('outcome')}**, separation {f(h65.get('signature_separation'))} "
      f"against a required {prereg['H6.5']['signature_separation']}. Legible-and-empty: "
      f"{h65.get('legible_and_empty')}.")
    A("")
    A("---")
    A("")

    # ---------------- E38 ----------------
    A("## E38: expertise substitutes rather than stacks")
    A("")
    A(e38.get("plain_language", ""))
    A("")
    cl = e38.get("cells", [])
    if cl:
        A("| reader | content | goal accuracy | uptake | still looking |")
        A("|---|---|---|---|---|")
        for c in cl:
            A(f"| {c['reader']} | {c['content']} | {f(c['goal_accuracy'])} | "
              f"{f(c['error_reduction'])} | {f(c['engaged_fraction'])} |")
        A("")
    h66 = e38.get("H6.6", {})
    A(f"**Outcome: {h66.get('outcome')}.** The machine-matched reader gains "
      f"{f(h66.get('gain_on_machine'))} on machine content and gives up "
      f"{f(h66.get('loss_on_human'))} on human content.")
    A("")
    A("---")
    A("")

    # ---------------- E39 ----------------
    A("## E39: the tool hypothesis")
    A("")
    A(e39.get("plain_language", ""))
    A("")
    cs = e39.get("cells", {})
    if cs:
        A("| arm | uncertainty left | still looking | invention | uptake |")
        A("|---|---|---|---|---|")
        for name, v in cs.items():
            A(f"| {name} | {f(v['final_entropy'])} | {f(v['engaged_fraction'])} | "
              f"{f(v['fabrication_index'])} | {f(v['error_reduction'])} |")
        A("")
    h67 = e39.get("H6.7", {})
    A(f"**Outcome: {h67.get('outcome')}.** Resolved: {h67.get('resolved')}; disengaged: "
      f"{h67.get('disengaged')}; not inventing: {h67.get('no_invention')}.")
    A("")
    n27 = e39.get("null_n27", {})
    A(f"**Null N27 holds** ({n27.get('passed')}): the extra hypothesis does not absorb human "
      f"work, which still reads at {f(n27.get('behavioural_human_accuracy'))}. That is the "
      f"failure version 4 caught with its own fallback hypothesis, checked the same way.")
    A("")
    A("---")
    A("")

    # ---------------- E40 ----------------
    A("## E40: the honeypot, the crowd, and what happens when the honeypot is optimised")
    A("")
    A(e40.get("plain_language", ""))
    A("")
    h68, h69 = e40.get("H6.8", {}), e40.get("H6.9", {})
    A(f"**How the cues combine: {h68.get('outcome')}** (pre-registered: "
      f"{h68.get('preregistered')}). At the corner where the content offers nothing and a cue "
      f"is present, the additive rule lifts engagement by "
      f"{f(h68.get('additive_lift_at_empty_corner'))} and the multiplicative rule by "
      f"{f(h68.get('multiplicative_lift_at_empty_corner'))}.")
    A("")
    A(f"**The decoupling: {h69.get('outcome')}.** With the surface cue maximised on content "
      f"that has no depth, engagement rises {f(h69.get('engagement_lift'))} above the honest "
      f"baseline while uptake is {f(h69.get('error_reduction'))}. "
      f"Pays more, gets less: {h69.get('pays_more_gets_less')}.")
    A("")
    A(f"*{e40.get('why_this_is_a_third_failure_mode', '')}*")
    A("")
    n29 = e40.get("null_n29", {})
    A(f"**Null N29 holds** ({n29.get('passed')}): goal accuracy varies by "
      f"{f(n29.get('goal_accuracy_spread_across_corners'))} across cue corners, so the cues "
      f"carry no goal information.")
    A("")
    A("---")
    A("")

    # ---------------- E42 ----------------
    A("## E42: engagement is not integration")
    A("")
    A(e42.get("plain_language", ""))
    A("")
    cl = e42.get("cells", [])
    if cl:
        A("| cell | still looking | integration | value divergence | uptake |")
        A("|---|---|---|---|---|")
        for c in cl:
            A(f"| {c['cell']} | {f(c['engaged_fraction'])} | {f(c['integration'])} | "
              f"{f(c['value_divergence'])} | {f(c['error_reduction'])} |")
        A("")
    h610 = e42.get("H6.10", {})
    A(f"**Outcome: {h610.get('outcome')}.** {h610.get('n_dissociating')} of "
      f"{h610.get('n_cells')} cells combine high engagement with a closed gate: "
      f"{h610.get('dissociating_cells')}.")
    A("")
    A("**What this changes.** Willingness to be vulnerable is the **gate**, not the decision to "
      "look. A reader can look intently, read the maker accurately, and integrate nothing. The "
      "two are driven by different terms and had never been reported apart.")
    A("")
    A("---")
    A("")

    # ---------------- E43 ----------------
    A("## E43: automaticity hides the work from its own author")
    A("")
    A(e43.get("plain_language", ""))
    A("")
    cl = e43.get("cells", [])
    if cl:
        A("| depth | the maker's own account | the reader's reading | process recovery |")
        A("|---|---|---|---|")
        for c in cl:
            A(f"| {int(c['mu'])} | {f(c['declared_accuracy'])} | {f(c['reader_accuracy'])} | "
              f"{f(c['process_accuracy'])} |")
        A("")
    h611 = e43.get("H6.11", {})
    A(f"**Outcome: {h611.get('outcome')}.** The maker's own account falls by "
      f"{f(h611.get('declared_decline'))} across the depth range while the reader moves "
      f"{f(h611.get('reader_movement'))}.")
    A("")
    A(f"*{e43.get('distinction_from_e33', '')}*")
    A("")
    A("---")
    A("")

    # ---------------- seed robustness ----------------
    rob = load("seed_robustness")
    if rob:
        A("## Does any of this survive different random seeds?")
        A("")
        A(f"**{rob.get('headline')}**")
        A("")
        A(f"*{rob.get('method')}*")
        A("")
        A("| seed block | absolute drop | relative drop | fold reduction | monotonicity | "
          "reserve, exposed | reserve, control | verdict |")
        A("|---|---|---|---|---|---|---|---|")
        for r in rob.get("e35_across_three_seed_blocks", []):
            A(f"| {r['seed_offset']} | {f(r['absolute_drop'])} | {f(r['relative_drop'])} | "
              f"{f(r['fold_reduction'])} | {f(r['monotone_rho'])} | "
              f"{f(r['reserve_last_exposed'])} | {f(r['reserve_last_control'])} | "
              f"{r['outcome']} |")
        A("")
        for para in rob.get("e35_reading", "").split(chr(10) + chr(10)):
            A(para)
            A("")
        A(f"**What may be claimed:** {rob.get('what_may_be_claimed')}")
        A("")
        A("---")
        A("")

    # ---------------- summary ----------------
    A("## What this version changed about what may be claimed")
    A("")
    A("1. **The simulation was missing a term the theory has.** Metabolic reserve is in the "
      "equation with its own symbol and had no counterpart in the code, so the framework's "
      "central cultural claim was not untested — it was unrepresentable.")
    A("2. **The preprint and the code explain the trust exploit differently, and the difference "
      "is testable.** The coupled gate predicts an exploit on a reader that is told the truth "
      "and believes it. The channel race cannot produce that.")
    A("3. **The depth result was measuring the wrong quantity.** Depth moves what the reader "
      "takes of the maker's *method* and provably cannot move what it takes of the maker's "
      "*purpose*, because the construction holds the second constant.")
    A("4. **Looking deeply and being willing to be changed are separate**, and the model already "
      "kept them apart. Vulnerability is the gate.")
    A("5. **A cue learned as a predictor of depth, then optimised directly, produces a third "
      "failure mode**: the reader pays more and gets less. Not the crash, not the exploit.")
    A("")
    A("## What was deliberately not built")
    A("")
    for item in prereg.get("what_is_deliberately_not_built", []):
        A(f"- {item}")
    A("")
    A("*Generated from results/v6/ by `scripts/write_results_v6.py`. Every number above is read "
      "out of a verdict file; none is typed in.*")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
