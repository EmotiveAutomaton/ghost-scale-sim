# Results from the Ghost Scale Simulation

**Run 2026-08-03, answering `FOR_GHOST_SCALE_SIM.md`.** Five modules, all six questions — S-4 and
S-5 merged, for a reason given below that is itself part of the answer.

Everything lives in `ghost-scale-sim/ghostscale/validation/soundingline/`, writes only to
`results/validation/soundingline/`, and touches nothing versioned. Re-run with
`python runners/run_soundingline.py`.

**None of this says anything about real artifacts.** It is mechanism only, in an environment with
ground truth, and it should not be quoted as evidence about text.

---

## S-1 · The unlock ratio is broken here, and it fails N28 ★

**Run this one's consequences before Gate 3 is interpreted.**

### It fails N28, badly

At `mu = 1` the construction *guarantees* there is no process: every execution mode emits the goal
signature exactly, so the sub-goal posterior never leaves its prior. `process_error_reduction` is
built to read 0 there and does (−0.022, inside its 0.05 tolerance).

A count-style ratio does not.

| statistic | mean at mu = 1 | 95% interval | verdict |
|---|---|---|---|
| **concentration ratio** (threshold-free) | **17.65** | [4.54, 37.43] | **FAILS N28** |
| resolved-count ratio @ 0.90 | 0.11 | [0.00, 0.33] | FAILS N28 |
| resolved-count ratio @ 0.95 | 0.75 | [0.24, 1.32] | passes, on n = 12 of 467 |

**A statistic that never consults the truth reports a 17× "unlock" in a world built so that there
is provably nothing to unlock.** That is the answer to your question and it is not a close call.

### It does not track the quantity it stands in for

Correlation with `process_error_reduction`, same rollouts:

- threshold-free concentration ratio: **r = 0.086** across rollouts
- resolved-count ratio @ 0.95: r = 0.148 across rollouts, r = 0.083 across 9 cells

**Effectively uncorrelated.** Your primary does not inherit E36's support. It has to earn its own.

### It is usually undefined

| threshold | undefined (0/0 or x/0) |
|---|---|
| 0.75 | **467 of 467** |
| 0.90 | 455 of 467 |
| 0.95 | 378 of 467 |

At any threshold strict enough to mean *a decision was recovered*, the denominator is zero. Your
1.111-vs-0.917 control result is consistent with this: those are ratios computed where the
denominator is small and unstable, not measurements of a difference.

### Why — and this is the useful part

**This reader's belief about the maker's execution mode is diffuse the entire way through.** Median
sub-goal posterior entropy sits at **96.5% of maximum**; it never once drops below 75% of maximum in
288 steps. Process recovery here is a *small log-probability edge over a broad posterior*, never a
confident identification.

If Sounding Line's extractor emits discrete "recovered decisions", it is thresholding something
that in this model never crosses a threshold. **The quantity is real and the counting is what breaks
it.** A graded measure against a baseline — which is what `process_error_reduction` is — survives
where a count does not.

**What would have falsified the worry:** a count ratio whose interval covers 1.0 at mu = 1 and which
correlates with `process_error_reduction`. Neither holds.

---

## S-2 · Flattened intent does show up as posterior concentration, and the construct is clean

At matched decision density — same positions, same depth, same number of execution-mode decisions,
only the concentration of the goal mixture differs (one terminal value at 0.70 against four at 0.25):

| | flattened | layered | difference |
|---|---|---|---|
| **purpose_breadth** | **0.450** | **0.558** | **−0.108, interval excludes zero** |
| goal accuracy *(must not fall)* | 0.858 | 0.792 | +0.067, n.s. |
| process recovery *(must not fall)* | −0.012 | −0.007 | −0.005, n.s. |

**Breadth separates them, and neither trap fired.** Accuracy did not drop — it went slightly *up*,
which is what the curator predicted: a flattened maker is easier to read, not harder. Process
recovery held, so "flattened" has not quietly become "shallower". **The construct is not confounded
with depth**, which is the thing no real corpus could have shown you.

*One design note:* this had to be read at CURATOR tier with three glances. On fully transparent work
with a long look the reader scores 1.00 and its posterior entropy collapses to ~1e-10 in **both**
arms — pinned at the floor, and the test measures nothing. If `purpose_breadth` is computed on easy
material in Sounding Line, expect the same floor effect.

---

## S-3 · The leak is readable, and the shield does give the concealer away

**This one does not use the V5 world** — it is a minimal purpose-built two-channel emitter, because
the shared model has one emission channel and adding a second is new generative machinery in a
repository that is otherwise closed. Read it as *what a two-channel emitter does*.

**Can the state be recovered from the leak alone?** Yes: **0.899** against a chance of 0.250. The
layer is readable in principle, so it is not a dead end.

**Does divergence identify concealment?** Yes at every amplification, interval excluding zero
throughout.

**And the curator's counter-intuitive claim holds:**

| amplification of the shield | detector balanced accuracy |
|---|---|
| 1× | 0.845 |
| 2× | 0.843 |
| 4× | 0.897 |
| **8×** | **0.963** |

**Performing louder to cover up makes concealment more detectable, not less.** The shield is a
readout of the thing it opposes, and amplifying it sharpens the readout. *The shield matches the
leak* survives contact with a mechanism.

---

## S-4 and S-5 · Merged, because they are one question — and the answer is cost

**Why merged.** Both ask about the order evidence is consumed in. This reader runs exact inference
over the full joint, and Bayesian updating on the same evidence set is order-independent, so
**neither reordering can change the final posterior.** Your S-5 note says this explicitly; S-4's
does not, and it is the same argument.

**Confirmed empirically, which is the harness check:** anomaly-first minus forward accuracy is
**exactly 0.000** (interval [0.000, 0.000]) over 540 paired rollouts. Reverse differs by 0.002, n.s.
The reorder is a reorder.

**So the answerable half is cost, and there is a real effect:**

| arm | steps to settle | settled fraction | goal accuracy |
|---|---|---|---|
| forward | 9.06 | 0.781 | 0.752 |
| **anomaly-first** | **8.63** | 0.811 | 0.752 |
| **reverse (method-first)** | **8.49** | 0.828 | 0.754 |

| contrast | difference | interval | |
|---|---|---|---|
| reverse − forward | **−0.56 steps** | [−1.104, −0.004] | **excludes zero** |
| anomaly-first − forward | **−0.43 steps** | [−0.807, −0.050] | **excludes zero** |

**Both reach the same answer sooner.** Under a reader that pays per look and can disengage, cheaper
*is* better — so the anomaly entry point is a real efficiency, and it is **not** a better inference.

That is a sharper and more defensible claim than the one Sounding Line is currently making, and it
is the one the evidence supports. It also means **S-4 does not refute the purpose-first loop**: the
loop is correct about where it ends up, and merely not optimal about how it gets there.

*Caveat: about half a step out of nine, so this is a ~5% saving, not a redesign.*

---

## S-6 · All three predictions hold, and the third is the one worth having

**Also a purpose-built construction, not the V5 world.** Both streams cost exactly the same; the
only difference between them is that content decisions can be cached.

| creator | surface slope | content slope | |
|---|---|---|---|
| **practised** | **−0.0370** declines | −0.0057 | surface decays **6.5× faster** |
| novice | −0.0281 declines | −0.0299 declines | **both** decay |
| **synthetic** | **−0.0000 flat** | −0.0000 | **surface does not move** |

- **S-1 there:** surface decays faster than content for a practised creator — content slope minus
  surface slope is **+0.0314**, interval [+0.0311, +0.0317]. ✅
- **S-2 there:** a creator with no budget shows a **flat** surface. ✅
- A novice shows decay in **both** streams. ✅

**The machine signature is not thin depth. It is a surface that does not move.** That is a positive
signature rather than an absence, which is worth more than anything an absence can give you —
every AI detector already looks for absences.

**The honest caveat, and it is a real one:** this construction *builds in* the asymmetry it then
measures. Content is cached and surface is not, by fiat. It is a consistency check on the theory's
arithmetic, not evidence that practice works this way in people. What it genuinely earns is the
**direction of the third prediction**, which does not follow trivially — a budgetless creator comes
out *flat* rather than merely *high*, and flat is a thing you can test for.

---

## What to do with this

**Before Gate 3 is interpreted:** S-1 is not a caution, it is a finding. A statistic that reports
17× unlock where nothing can be unlocked, is uncorrelated with the graded measure it stands in for,
and is undefined in 81–100% of cases, cannot carry a primary. The fix suggested by the mechanism is
to score a **graded log-probability against a baseline** rather than to count recovered decisions.

**Cheap wins:** S-3 and S-6 both came back clean and both sharpen claims you already hold. S-2 says
`purpose_breadth` works but warns you about a floor effect on easy material.

**Scope reduction:** S-4 is answered and does not need building. The purpose-first loop is right
about the destination.

---

*Six questions, five modules, one merge. Two of the six — S-1's N28 check and S-4 — are arguably
Ghost Scale questions rather than Sounding Line ones, and if either changes a claim over there it
will be promoted into that project's walkthrough rather than left here.*
