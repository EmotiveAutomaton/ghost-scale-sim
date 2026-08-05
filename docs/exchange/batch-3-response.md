# Third batch, answered — and one correction to the second

**2026-08-05.** Batch three sent tooling rather than questions. The tooling was built (see
[docs/METHODS.md](../METHODS.md)) and then pointed at the existing results. Five modules, T-6 to
T-10. Two of them add no simulation at all.

**Reproduce:** `python runners/run_soundingline.py --only T6 T7 T8 T9 T10`.

---

## The one thing that changes what you should believe

**Batch two told you the triangle runs backwards. Withdraw that.**

It reported `goal→process` as a dead edge across every cell with headroom and read it as *process
is a source, goal is a sink*. T-6 computes the emission's information budget in closed form — no
rollouts, no sampling, `world.subsig` **is** the joint distribution — and the reading does not
survive.

The creator emits `beta * subsig[mu,g,s] + (1-beta) * mean_g(subsig[mu,g,s])`. That attenuates
*which goal* the modes serve while leaving mode structure untouched:

| β | I(goal; F) | I(mode; F) | I(mode; F \| goal) | coupling |
|---|---|---|---|---|
| 1.00 | **1.4521** | 0.2706 | 0.4262 | **1.58×** |
| 0.50 | 0.3541 | 0.2706 | 0.2838 | 1.05× |
| 0.25 | 0.0941 | 0.2706 | 0.2738 | 1.01× |
| 0.10 | 0.0160 | 0.2706 | 0.2711 | 1.00× |

The mode carries **0.2706 bits at every β** — invariant by construction. The goal's information
collapses by ninety-fold.

T-1's cells with goal headroom are β = 0.25 and 0.10. **There is no coupling to find there, by
construction.** The only cell where the goal carries enough to inform the mode is β = 1.0 — which
is exactly the cell where T-1 flagged a ceiling because the reader already recovers the goal
perfectly.

**So `goal→process` is not dead. It is an edge this model can only create under a setting that
simultaneously removes the headroom needed to measure it.** T-1's `+0.0017` at β=1.0 is the whole
of the observable coupling, and batch two dismissed it as noise.

## The three inference problems, exactly

```
I({depth, goal, mode} ; one observation) = 1.7362 bits

              alone     given the other two
  goal        1.4521          1.5472
  mode        0.0266          0.2841     ← 10.7× in context
  depth       0.0000          0.2368     ← from exactly nothing
```

**They are not symmetric peers. One is free-standing; two are conditional.** The goal is 84% of
what a single observation carries. Marginalise the mode out and depth contributes *exactly* zero
with zero redundancy and zero synergy. Marginalise the goal out and mode+depth together carry
0.189 bits, of which 0.162 is synergy.

Three things the project established the hard way fall out of this analytically and are now
standing gates: **null N28**, **`depth_marginal_invariance`**, and **E30's null**.

---

## What else was found

**T-7 · Your batch survives correction.** Benjamini–Hochberg over eight hand-declared families:
17 claims lost, all in µ=1 cells, negative controls, or low-duty restatements. **No live edge in a
cell with headroom is lost.** 18 of 23 nulls are now *bounded* rather than merely uncontradicted.
The five that are not: T-2's depth axis at mixtures 0.0 and 1.0, T-2's length control, and one
S-45 accuracy check. **T-2's "practice alone does not move breadth" and "length does nothing" are
weaker than batch two implied** — they cannot be bounded below 10% of T-2's own live effect.

Families are declared by hand and that is the design. An auto-harvesting first pass reported
S-45's cost contrasts as failing correction; they only failed because the harvester swept in
S-45's own accuracy checks, which S-45 explicitly calls *"a harness check and not results"*.
Against the real family of two they survive at p_adj = 0.044 — which is holding, but barely.

**T-8 · Combining features is the win. Feature banks are not.** Ten hand-picked features combined
lift the hard cells from 0.90–0.96 AUC to **0.99–1.00 on a fresh seed block**. Adding 60 catch22
features on top gains a median **+0.019** and costs up to **−0.084**. Shrinkage from held-out to
fresh seeds is 0.000, so the split is not leaking.

**T-9 · S-2's question, answered.** Its live gate — the one S-2 records at exactly **0.0** —
passes here at **0.294**, which is what demonstrates the gate discriminates rather than merely
exists. Breadth falls monotonically with concentration (0.725 → 0.530 as the dominant share goes
0.40 → 1.00). But goal accuracy rises with it (0.380 → 0.808), and at matched accuracy the excess
breadth is **negative at every share**. Verdict: `BREADTH_IS_LARGELY_A_DIFFICULTY_METER`, same as
T-2, now established for the question S-2 was actually aimed at.

**T-10 · T-3's negative does not retire seam-finding.** T-3 killed *which* decision. *When* is a
different question. Per-step travel of the sub-goal posterior ranks true mode-switch steps above
non-switch steps with a lift of **+0.081 [0.072, 0.091]** over a circular-shift null (which
preserves the trajectory's autocorrelation and destroys only its alignment). The maker's switch
**rate** reads off a canonical trajectory feature at **|r| = 0.445**.

And the lift is **+0.11 to +0.17 where the goal is legible, +0.01 to +0.04 where it is not.**

---

## The through-line

Four modules, arrived at independently, say the same thing.

> **Goal legibility is the master variable. Everything about process readability is conditional
> on it.**

T-6 says it in closed form: `I(mode|goal)/I(mode)` is 1.58× at β=1 and 1.00× below. T-1's live
edges are exactly the cells where the goal is illegible and the mode is therefore the only
informative channel. T-10 says seams are findable when purpose is legible and nearly invisible
when it is not. T-9 says that what looks like a measure of motivational structure is tracking
legibility instead.

**For an instrument: measure legibility first, and treat every process-side reading as conditional
on it.** A process signal quoted without a legibility figure beside it is not interpretable —
which is a constraint on tooling design, not a caveat.

## What this cannot say

There is no human data anywhere in this. Every number is a property of one generative model's
dimensions, and the shapes are the claims. Quote directions.

The specific limit worth naming: β is *this model's* knob for goal legibility, and it attenuates in
one particular way — toward the goal-marginal of the mode family. Whether real illegibility has
that shape is exactly what a simulation cannot tell you, and the whole through-line above rests on
it.
