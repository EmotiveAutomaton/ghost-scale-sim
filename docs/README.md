# The map

Everything in `docs/` and what it is for. If you only want one page, it is
[FINDINGS.md](../FINDINGS.md); if you want it in pictures, [WALKTHROUGH.md](../WALKTHROUGH.md); if
you want the narrative, [HISTORY.md](HISTORY.md).

Everything below is the underlying material. You need it if you want to check a specific number or
argue with a specific decision.

---

## The two kinds of document, and why they are kept apart

**Versions ask new questions about the world.** Each has a `SPEC.md` written *before* its code and a
`RESULTS.md` written *after* its results, and **neither is edited afterwards.** That is the property
that makes the record checkable rather than merely tidy: where a number was superseded, the old
document still says the old thing and the newer one says what replaced it.

**Audit passes ask whether the existing answers can be trusted.** None asks a new question about the
world. Their `RESULTS.md` files are *generated from verdict files*, never hand-written, so an
explanation cannot quietly drift away from the number it explains.

Version 6 is the only hybrid: it audits the *code against the published theory* rather than the
results against themselves, and then builds what was missing.

---

## The versions

Numbered in run order, named for what each was actually about.

| | name | the question it asked | what came back |
|---|---|---|---|
| **1** | [The Mechanism](versions/v01-the-mechanism/) | can this be written down as a model at all? | three results, two of which still stand |
| **2** | [The Learner](versions/v02-the-learner/) | what if the reader has to *acquire* what machine content looks like? | heterogeneity belongs in the likelihood, not the prior |
| **3** | [The Refuted Repair](versions/v03-the-refuted-repair/) | is the generational leak just sampling noise? | **its own gate said no**, and the experiment stayed withheld |
| **4** | [Foreign Intent](versions/v04-foreign-intent/) | is machine content goal-*empty* or goal-*foreign*? | foreign — **and a headline inverted** |
| **4.5** | [Three Gates](versions/v04.5-three-gates/) | does confident invention require modelling a mind? | **no**, and the claim was withdrawn |
| **5** | [Depth Over Effort](versions/v05-depth-over-effort/) | is it how hard they tried, or how much practice is compressed in? | depth — and the fast solver was caught misreading it |
| **6** | [Code Against Equation](versions/v06-code-against-equation/) | is the simulation the same object as the published theory? | **three terms had no counterpart in the code** |
| **7** | [The Closures](versions/v07-the-closures/) | what does modelling a maker actually buy? | 4 examples against 512, and reading an unseen intent |
| **8** | [The Severity Pass](versions/v08-the-severity-pass/) | how often does a *randomly built* model of this shape do it too? | **two of three headlines: every time** |
| **9** | [Minimal Models](versions/v09-minimal-models/) | which structural commitment is each finding made of? | **every one dies without the maker-model** |
| **10** | [Reader As Defence](versions/v10-reader-as-defence/) | can reading intent defend a learner from content written to be absorbed? | yes — and surface filtering does nothing at all |

Each directory holds `SPEC.md` and `RESULTS.md`. Versions 2, 3 and 8 also carry a `PLAN.md` or a
`DECISIONS.md` — working documents kept because they contain reasoning that did *not* survive into
the spec, which is often worth more than the reasoning that did.

**After version 10 the remaining questions need human subjects or real models, and this apparatus is
neither.**

---

## The audit passes

| | pass | the question it asks | how it came back |
|---|---|---|---|
| **A1** | [Validation](audits/a1-validation/) | can the recorded answers be trusted? | **five of nine checks came back against the work** |
| **A2** | [Diagnostics](audits/a2-diagnostics/) | can the instruments answer at all? | four limits that bound how strongly any number can be stated |
| **A3** | [Repair](audits/a3-repair/) | what can be fixed, and what does fixing change? | the uptake measure gained a sign, and the headline reversed with it |

**Read [the diagnostics pass](audits/a2-diagnostics/RESULTS.md) before quoting any specific number.**
That is where the limits on the numbers live.

---

## Everything else

| file | what it is |
|---|---|
| [HISTORY.md](HISTORY.md) | ten versions and three audit passes as one narrative, so nobody has to read eleven write-ups to learn why a number is what it is |
| [EXPERIMENTS.md](EXPERIMENTS.md) | the original plain-language experiment table. Superseded by [FINDINGS.md](../FINDINGS.md), kept as a record of how the questions were first posed |
| [theory/](theory/) | the preprint this implements, plus the crosswalk from its vocabulary to the code's |
| [assets/SPEC.md](assets/SPEC.md) | the spec for the public-facing material: the README rewrite and the distribution slides |

---

## Conventions that hold everywhere

- **A spec is written before its code and never edited afterwards.** Deviations are logged in the
  matching `RESULTS.md`, with the original criterion retained and still computed.
- **Acceptance criteria are executable and content-hash locked** before any run, so the written
  criterion and the applied criterion are the same object and cannot drift apart.
- **Every headline effect has a null that must come out null.** There are fifty-one, N1 through N51.
  Three of them fail, and they are reported as failing rather than dropped.
- **Generated documents say so at the top.** If a file was produced from verdict files, editing it
  by hand is a bug.
