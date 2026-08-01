"""Generate RESULTS_V7.md from results/v7/. Never hand-written."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V7 = REPO / "results" / "v7"
OUT = REPO / "RESULTS_V7.md"


def load(n):
    p = V7 / f"{n}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def f(x, n=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "n/a"
    return f"{v:.{n}g}" if (v and (abs(v) < 0.001 or abs(v) >= 1e4)) else f"{v:.{n}f}"


def main():
    cl, e45 = load("closures"), load("e45_tom_efficiency")
    e46, e47 = load("e46_gate_leak"), load("e47_coverage")
    pre = json.loads((V7 / "preregistration_v7.json").read_text(encoding="utf-8"))
    L, A = [], None
    L = []
    A = L.append

    A("# Version 7: closing what was held back, and going back at the withdrawn claim")
    A("")
    A("Generated from [results/v7/](results/v7/). Regenerate with "
      "`python scripts/write_results_v7.py`. Criteria hash-locked at "
      f"`{pre['content_hash'][:16]}` before any cell ran.")
    A("")
    A("---")
    A("")
    A("## At a glance")
    A("")
    A("| item | question | outcome |")
    A("|---|---|---|")
    for k in ("C-1", "C-2", "C-3", "C-4"):
        d = cl.get(k, {})
        A(f"| **{k}** ({d.get('experiment')}) | {d.get('question','')} | {d.get('outcome')} |")
    A(f"| **E45** | What does modelling a maker actually buy? | "
      f"{e45.get('H7.1',{}).get('outcome')} / {e45.get('H7.2',{}).get('outcome')} |")
    A(f"| **E46** | Can you reject something and be unchanged? | {e46.get('outcome')} |")
    A(f"| **E47** | Does the coverage figure survive the mechanism question? | "
      f"{e47.get('outcome')} |")
    A("")
    A("---")
    A("")

    # ---- E45 ----
    A("## E45: what modelling a maker actually buys")
    A("")
    A(e45.get("plain_language", ""))
    A("")
    h1, h2 = e45.get("H7.1", {}), e45.get("H7.2", {})
    A(f"**Evidence needed to reach {f(pre['H7.1']['competence'],2)} accuracy:**")
    A("")
    A("| reader | examples needed |")
    A("|---|---|")
    A(f"| simulates a maker | **{h1.get('evidence_the_simulator_needs')}** |")
    A(f"| counts co-occurrences | **{h1.get('evidence_the_counter_needs')}** |")
    A("")
    A(f"**On an intention it has never seen:** the simulator scores "
      f"**{f(h2.get('simulator_on_an_unseen_intent'))}**; the counter's best across every "
      f"training size is {f(h2.get('counter_on_an_unseen_intent_best'))}, against chance of "
      f"{f(h2.get('chance'),2)}. {h2.get('note','')}")
    A("")
    A(f"> *{e45.get('relationship_to_e21','')}*")
    A("")
    A("---")
    A("")

    # ---- E46 ----
    A("## E46: the gate cannot fully close")
    A("")
    A(e46.get("plain_language", ""))
    A("")
    A(f"*{e46.get('why_this_is_a_correction_not_an_addition','')}*")
    A("")
    A("| leak | drift after repeated rejection | accumulates? |")
    A("|---|---|---|")
    for k, v in (e46.get("drift_by_leak") or {}).items():
        A(f"| {f(k,2)} | {f(v.get('final_drift'),4)} | rank correlation "
          f"{f(v.get('monotone_rho'))} |")
    A("")
    eng = e46.get("engagement", {})
    A(f"**The reader who studies it carefully drifts {f(eng.get('ratio'),2)}× more than the one "
      f"who skims.** {eng.get('reading','')}")
    A("")
    inv = e46.get("what_a_reader_absorbs_of_its_own_invention", {})
    if inv:
        A(f"**And a reader absorbs its own invention.** On content with no recoverable intent at "
          f"all, drift is {f(inv.get('drift_on_content_with_no_recoverable_intent'),4)}. "
          f"{inv.get('reading','')}")
        A("")
    glr = e46.get("the_graded_gate_already_leaked", {})
    if glr:
        A(f"> *{glr.get('note','')}*")
        A("")
    A("---")
    A("")

    # ---- closures ----
    A("## The four closures")
    A("")
    for k in ("C-1", "C-2", "C-3", "C-4"):
        d = cl.get(k, {})
        A(f"### {k} — {d.get('experiment')}")
        A("")
        A(f"**{d.get('question','')}**")
        A("")
        if k == "C-1":
            m, g = d.get("scored_on_the_method", {}), d.get("scored_on_the_purpose", {})
            h = d.get("history", {})
            A(f"Scored on the **method**: {f(m.get('rho'))}, interval "
              f"[{f(m.get('interval',[0,0])[0])}, {f(m.get('interval',[0,0])[1])}], against a bar "
              f"of {f(m.get('bar'),2)}. Scored on the **purpose**, which the construction holds "
              f"constant: {f(g.get('rho'))}.")
            A("")
            A(f"History: {f(h.get('approximate'))} under the approximate solver, "
              f"{f(h.get('exact'))} under exact arithmetic, {f(h.get('retrofit'))} in the "
              f"retrofit reconstruction. **This is E31's own design.**")
        elif k == "C-2":
            r, m = d.get("on_recovered_depth", {}), d.get("on_the_method", {})
            A(f"The pre-registered contrast gives {f(r.get('dominance_ratio'),2)} against a bar of "
              f"{f(r.get('bar'),2)} and **fails**. On what actually transfers, depth dominates "
              f"effort by {f(m.get('dominance_ratio'),1)}×.")
            A("")
            A("So the reader's *estimate* of depth is contaminated by effort, and what it *takes "
              "away* is not. The original criterion decides and is reported as failing.")
        elif k == "C-3":
            A(f"Peak at {f(d.get('peak_omega'),2)}. The crash signature fires at: "
              f"{d.get('omegas_where_the_crash_signature_fires') or '**nowhere**'}.")
            A("")
            A(f"**{d.get('consequence','')}**")
        elif k == "C-4":
            A(f"Relative drop {f(d.get('relative_drop'))}, monotone at "
              f"{f(d.get('monotone_rho'))}, over {d.get('n_encounters')} encounters. The "
              f"pre-registered absolute clause reads {f(d.get('absolute_drop'))}.")
            A("")
            A(f"> *{d.get('pre_registered_clause',{}).get('why_it_is_retained_and_not_used','')}*")
        A("")
    A("---")
    A("")

    # ---- E47 ----
    A("## E47: does the policy number survive?")
    A("")
    A(e47.get("plain_language", ""))
    A("")
    A(f"**{e47.get('outcome')}.** The threshold sits at "
      f"{f(e47.get('threshold_under_the_channel_race'),2)} under the code's mechanism and "
      f"{f(e47.get('threshold_under_the_coupled_gate'),2)} under the paper's.")
    A("")
    A(f"*{e47.get('this_is_conditional_and_is_reported_as_such','')}*")
    A("")
    A(f"*{e47.get('relationship_to_e16','')}*")
    A("")
    A("---")
    A("")
    A("## What was deliberately not built")
    A("")
    for item in pre.get("not_built", []):
        A(f"- {item}")
    A("")
    A("*Generated from results/v7/ by `scripts/write_results_v7.py`. Every number above is read "
      "out of a verdict file; none is typed in.*")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
