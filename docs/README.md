# The paper trail

## If you have five minutes

1. **[../FINDINGS.md](../FINDINGS.md)** — every question the project asked and where its answer
   stands *today*. One page, current state, no archaeology.
2. **[../EVIDENCE.md](../EVIDENCE.md)** — what the world has published, next to what this
   predicted. Disagreements italicised.
3. **[HISTORY.md](HISTORY.md)** — how the record got that way. Six versions and four audit passes
   in one narrative, so you do not have to read six write-ups.

Everything below is the underlying material. You only need it if you want to check a specific
number or argue with a specific decision.

---

## The five kinds of document, and why they are kept apart

- **[specs/](specs/)** say what was going to be built and what it was going to predict. Written
  before the code, **never edited afterwards**.
- **[writeups/](writeups/)** say what actually happened, including every place the answer disagreed
  with the spec. They describe the record **as it stood when each version shipped** and are not
  edited afterwards either. Read them as history.
- **The four pass documents** at the top level are **generated from verdict files**, never
  hand-written, so an explanation cannot quietly change a number.
- **[decisions/](decisions/)** record design choices signed off before a build, with the evidence.
- **[EXPERIMENTS.md](EXPERIMENTS.md)** is the original plain-language table. **Superseded by
  [../FINDINGS.md](../FINDINGS.md)**, which carries the current verdicts; kept because it is what
  the public README's table was derived from.

Where a spec is now known to be wrong, the correction lives in the write-up that found it, never in
the spec. A spec quietly amended after its own experiment has run is no longer a record of what was
predicted, and this project's claim to being checkable rests on those two staying separable.

---

## The generated pass documents

Read these in order if you want to know how much of the work survived being checked.

| file | the question it asks | how it came back |
|---|---|---|
| [../VALIDATION.md](../VALIDATION.md) | Can the recorded answers be trusted? | Five of nine checks came back against the work |
| [../DIAGNOSTICS.md](../DIAGNOSTICS.md) | Can the instruments answer at all? | Four limits that bound how strongly any number can be stated |
| [../REPAIR.md](../REPAIR.md) | What can be fixed, and what does fixing change? | The uptake measure gained a sign, and the headline reversed with it |
| [../RESULTS_V6.md](../RESULTS_V6.md) | Is the code the same object as the theory? | Three terms of the equation had no counterpart in the code |
| [../RESULTS_V7.md](../RESULTS_V7.md) | What does the withdrawn claim's machinery actually buy? | 128× less evidence, and reading an intent nobody has shown you |
| [../RESULTS_V8.md](../RESULTS_V8.md) | How much of any of this is the theory? | Two of three headlines reproduce in every random model of the same shape |

**Read [../DIAGNOSTICS.md](../DIAGNOSTICS.md) before quoting any specific number**, because that is
where the limits on the numbers live.

---

## The build specs

| file | what it is |
|---|---|
| [specs/SPEC_V1.md](specs/SPEC_V1.md) | The original build. Hypotheses H1–H6, nulls N1–N7, the parameter table, the invariant tests. |
| [specs/SPEC_V2.md](specs/SPEC_V2.md) | The learner, reader heterogeneity, biased machine content. Nulls N8–N12. |
| [specs/PLAN_V2.md](specs/PLAN_V2.md) | The longer working plan behind version 2, including reasoning that did not survive into the spec. |
| [specs/SPEC_V3.md](specs/SPEC_V3.md) | Written to repair the generational experiment under a diagnosis that version 3's own gate then refuted. Nulls N13–N15. |
| [specs/SPEC_V4.md](specs/SPEC_V4.md) | **The reframe.** Changes what machine-made content *is*, from goal-empty to goal-foreign, and tests whether the earlier results survive. Nulls N16–N20. |
| [specs/SPEC_V4_5.md](specs/SPEC_V4_5.md) | The three-gate reader, and the metabolic question promoted to a headline. |
| [specs/SPEC_V5.md](specs/SPEC_V5.md) | Depth replaces effort; provenance becomes evidence rather than a parallel channel. Null N21. |
| [specs/SPEC_V6.md](specs/SPEC_V6.md) | **Not a new question about the world.** Asks whether the simulation and the theory are the same object. Six extensions, two author corrections, nulls N22–N30. |
| [specs/SPEC_V8.md](specs/SPEC_V8.md) | Version 8. The reader gets a hierarchy of its own, a cost for being changed, and a memory that fades. Plus the severity check, a maker that can lie, and the readymade. Nulls N35–N40. |
| [specs/PLAN_V8.md](specs/PLAN_V8.md) | **A planning document, not a spec.** What a sweep of the code and the theory turned up, what should be built next, and the decisions that are the author's. Nothing in it is locked. |
| [specs/SPEC_V7.md](specs/SPEC_V7.md) | Version 7. Closes the four results version 6 would not draw, and attacks E21 on the axis it was never tested on. Nulls N31–N34. |
| [specs/SPEC_VALIDATION.md](specs/SPEC_VALIDATION.md) | The validation pass. Nothing in it asks a new question about the world. |
| [specs/SPEC_DIAGNOSTICS.md](specs/SPEC_DIAGNOSTICS.md) | The diagnostics pass on the instruments, ahead of any repair. |
| [specs/SPEC_REPAIR.md](specs/SPEC_REPAIR.md) | The repair pass. Every change either makes something measurable that was not, or removes something. |
| [specs/SPEC_PUBLIC_ASSETS.md](specs/SPEC_PUBLIC_ASSETS.md) | The public-facing material: the README rewrite and the distribution slides. |

## The write-ups

| file | version | the short version |
|---|---|---|
| [writeups/RESULTS_V1.md](writeups/RESULTS_V1.md) | 1 | The crash, the trust exploit, the labelling trade-off. Seven deviations. |
| [writeups/RESULTS_V2.md](writeups/RESULTS_V2.md) | 2 | Heterogeneity moves from prior to likelihood. Biased content accumulates. |
| [writeups/RESULTS_V3.md](writeups/RESULTS_V3.md) | 3 | The finite-sample diagnosis refuted by its own gate. Estimator bug found and fixed. |
| [writeups/RESULTS_V4.md](writeups/RESULTS_V4.md) | 4 | Goal-empty became goal-foreign, and the metabolic prediction inverted. |
| [writeups/RESULTS_V4_5.md](writeups/RESULTS_V4_5.md) | 4.5 | Four unwelcome results, including the counting classifier that withdrew a claim. |
| [writeups/RESULTS_V5.md](writeups/RESULTS_V5.md) | 5 | Depth replaces effort — and the shortcut caught misreading depth entirely. |
| [../RESULTS_V6.md](../RESULTS_V6.md) | 6 | Generated, not hand-written. The equation had terms the code did not. |
| [../RESULTS_V7.md](../RESULTS_V7.md) | 7 | Generated. Four closures, and what imagining a maker buys. |
| [../RESULTS_V8.md](../RESULTS_V8.md) | 8 | Generated. The reader gets a mind, and the severity check lands. |

## The decisions

| file | what it covers |
|---|---|
| [decisions/DECISIONS_V2.md](decisions/DECISIONS_V2.md) | Version 2's five signed-off decisions, with the measurements behind them. |
| [decisions/DECISIONS_V3.md](decisions/DECISIONS_V3.md) | Version 3's decisions, and an incident worth recording: a smoke run that overwrote committed output, and the two hardening changes that followed. |

---

## Reading order, if you are starting cold and want the whole thing

1. [../FINDINGS.md](../FINDINGS.md) — the current state.
2. [HISTORY.md](HISTORY.md) — how it got there.
3. [../DIAGNOSTICS.md](../DIAGNOSTICS.md) — the limits on the numbers.
4. [../README.md](../README.md) — the public framing, the scope section, and how to run it.
5. A write-up, then its spec, if you want one number's whole provenance.
