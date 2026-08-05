# The methodology layer

**What `ghostscale/methods/` is for, why each piece exists, and what it costs to run.**

Every module in here was added because something specific went wrong. None of it computes a
finding; all of it computes a reason to believe or disbelieve one. If a piece of this stops
earning its place, delete it — scaffolding that survives on ceremony is worse than no scaffolding,
because it makes a project look checked.

---

## The rule that makes it safe to add all this

> **Nothing in `ghostscale/methods/` may be required to reproduce a published number.**

`ghostscale/v1` through `v10` are closed: pre-registered, run, reported, left alone. What makes
them reproducible is that `pip install -e .` resolves to the same small set of packages in five
years as it does today. So every third-party dependency here is an **optional extra**, and every
module that leans on one degrades to a recorded skip rather than an exception:

```bash
pip install -e .                 # reproduces every published result. Unchanged.
pip install -e ".[methods]"      # + PID, sensitivity analysis, catch22, FDR, equivalence
pip install -e ".[explore]"      # + tsfresh, arviz, dabest
pip install -e ".[dev]"          # + hypothesis, for the metamorphic tests
```

A skipped gate counts as passing and is **reported separately**, so a skip can never be mistaken
for a check that succeeded.

---

## Gates — `methods/gates.py`

Five kinds of standing control, recorded in every verdict JSON under `"gates"`.

| kind | the relation | the real defect it is aimed at |
|---|---|---|
| `placebo` | a manipulation at **zero** strength reproduces the control **exactly** | a side channel drawing from the rollout's RNG, which moved depth recovery 0.163 → 0.282 on nothing; and a `1/3` that is not uniform in floating point, which flipped an argmax on a near-tied posterior |
| `live` | a manipulation at **full** strength **changes** the output | S-2, whose goal mixture was drawn and discarded — feature streams bit-identical with it off |
| `positive` | a task with a **known answer** returns it, through the whole stack | the gap this repository had. N28 says the instrument does not fire on nothing; only a positive control says it fires on something |
| `identity` | a quantity with a provable symmetry or bound satisfies it | nothing that broke — which is why it is worth keeping. T-1's two goal↔depth edges agree to 3e-16 because both reduce to a symmetric conditional mutual information |
| `no_oracle` | a statistic does not move when a label it should not see is permuted | S-3's threshold, fitted on labelled test data; E45's efficiency result, where the reader held the world's own emission map |

### A gate records; a test fails

This is the whole enforcement design and it is deliberate.

- A gate that **raises mid-run** kills a 300-second sweep over a control that may have been
  expected to fail. That trains people to switch gates off.
- A gate that only **prints** gets ignored. That is how S-2 shipped.

So gates write into the verdict and the run continues. `tests/test_gates.py` then walks every
committed verdict and fails the suite on any broken control — putting the hard stop at the moment
a result would become public, which is where one was needed and missing.

### Three categories, and they are not the same thing

```
failed_names         a control that broke. Stops a release.
documented_failures  a gate marked expected_to_fail: a KNOWN defect, recorded so the
                     evidence travels with the result instead of living in a commit message.
unexpected_passes    a documented defect that has started passing. Somebody fixed it and did
                     not update the gate — the one nobody would otherwise notice.
```

Two gates are currently `expected_to_fail`, and both are load-bearing documentation:

- **S-2 `mixture_reaches_the_reader`** — observed **0.0**. Forcing the mixture off changes nothing,
  because `V5Environment.sample_feature` ignores `artifact.goal` once a creator is bound. This is
  why the module is withdrawn, and the number is now attached to the verdict permanently.
- **S-3 `threshold_is_fitted_on_test_labels`** — the detector threshold is the median of the pooled
  *labelled* divergences, re-fitted per cell. T-4 re-scores it frozen: the headline rise falls from
  +0.125 to +0.046.

A module with a documented failure must also carry `withdrawn`, `WITHDRAWN` or `QUALIFIED` at the
top of its verdict, and `test_documented_failures_are_explained` enforces that — a known defect has
to be legible to someone reading the first screen, not only to someone who scrolls to the gates.

---

## Provenance — `methods/provenance.py`

Every verdict now carries which module wrote it, at what content:

```json
"produced_by": {
  "module": "ghostscale/validation/soundingline/t1_triangle.py",
  "sha256": "3f2a…",
  "git_commit": "b4f5612",
  "git_dirty": true
}
```

`git_dirty` is the field that earns its place: a result from a dirty tree cannot be reconstructed
from the commit alone, and knowing that at read time is the difference between "I can check this"
and "I think I remember".

---

## Separation, not `excludes_zero` — `methods/overlap.py`

`excludes_zero` answers *is the interval on one side of zero*, which is a question about
**precision**. At n = 200 paired rollouts almost anything separates from zero.

The motivating case is in this repository. T-1's `goal→process` edge at µ3/β1.0 returns **+0.0017**
with `excludes_zero: true` — and it is 0.3% of the `process→depth` edge on the same scale, it flips
sign one cell over, and the budget-matched version flickers across duty cycles. A reader has to be
told all of that in prose to know the flag means nothing.

The overlap coefficient is the shared area of the effect and null densities, in [0, 1], unitless —
so a result in nats and a result in AUC are directly comparable. Scored against the strongest null
available, and **the null is named in the output** so a strong claim can never be quoted off a weak
one:

```
placebo  → the manipulation at zero strength. Controls the harness.
permuted → random values through a likelihood that still claims fidelity. Controls the content.
swapped  → another artifact's true values. Strongest: everything but correspondence held fixed.
```

**Two calibration mistakes were made building this, and both are worth knowing about.**

First, the overlap was computed on per-unit values. On those, T-1's *strongest* edge
(`process→depth`, +0.5147) scored 0.335 and read as "not separated" — which is not a fact about
the edge, it is a statement that individual artifacts vary by more than the mean effect, which is
true of nearly every result here. The overlap is now taken on the **bootstrap distributions of the
mean**, which is the sanity-checks paper's actual framing and the scale τ = 0.2 was written for.
The per-unit number is still reported as `per_unit_overlap` because it is a real effect-size
measure — just not a verdict.

Second, `separated` was the wrong name. The scrambled nulls do not sit at zero: a channel that lies
confidently *hurts*, so the swap null for `process→goal` sits at −0.41. Separating from it means
*this effect needs the channel to correspond to the truth* — not *this effect is nonzero*. T-1's
`goal→process` edge at **−0.0012** separates cleanly from a swap null at −0.0514 while being a
negative effect. So there are now three fields:

```
separated_from_null       distinguishable from the scrambled channel
effect_exceeds_null       on the helping side of it
supports_a_positive_edge  both, AND above zero — the only one that licenses "this edge is alive"
```

**And a third thing, which is the one that actually mattered.** None of those fields is a
magnitude, and neither is `standardised_effect` in the way you would hope: at β = 1.0 the reader is
saturated, the null's spread collapses, and the +0.0017 edge scores a standardised effect of
**1.15 — larger than the +0.5147 edge's 0.81**. Every confidence-flavoured statistic here agrees
that the tiny edge is real, and they are all correct. Confidence was never the problem.

So `relative_magnitude()` reports each effect as a fraction of the largest one measured **the same
way in the same family**, and that is what finally says it: the +0.0017 edge is 0.3% of the largest
edge into the same vertex. `excludes_zero` was misleading because it answered a confidence question
that nobody was really asking — and the fix was not a better confidence statistic.

---

## Partial information decomposition — `methods/pid.py`

T-1's superadditivity test is a hand-rolled synergy measure. PID is the principled version, and it
runs on `world.subsig` **exactly, with no rollouts** — the emission likelihood is the joint
distribution.

| depth | total | redundant | unique GOAL | unique MODE | synergy |
|---|---|---|---|---|---|
| µ=1 | 1.4521 | 0.0000 | 1.4521 | 0.0000 | −0.0000 |
| µ=2 | 1.8783 | 0.2966 | 1.1555 | **0.0000** | 0.4262 |
| µ=3 | 1.8783 | 0.2706 | 1.1816 | **0.0000** | 0.4262 |

Two things worth having. **Unique mode information is exactly zero at every depth** — everything
the execution mode contributes is redundant with the goal or readable only jointly with it, which
no previous measure here could state. And **µ=1 returns zero mode information and zero synergy**,
recovering null N28 from the likelihood alone; it is wired in as an `identity` gate on T-1 and as a
positive control in `tests/test_metamorphic.py`, alongside XOR (which must read as one bit of pure
synergy).

---

## Sensitivity — `methods/sensitivity.py`

Plate 5 randomises everything the theory specifies and counts survivals: one scalar, and its honest
reading is "83% of randomly parameterised models of this shape do it too". A survival count cannot
say **which** parameters matter, so an architectural finding and a one-setting finding look alike.

Sobol indices split the variance instead. `S1` is what a parameter explains alone; `ST` includes
every interaction it takes part in. **ST near zero is proof a parameter is not carrying the
result.** A large `ST − S1` means it only matters in combination — the same distinction PID draws
between unique and synergistic, reached from the other direction.

Cost: `N*(2D+2)` evaluations. At ~3 ms a rollout and a dozen parameters, `N = 256` is about a
minute. `morris()` is the cheap ordinal screen for cutting a long parameter list first.

---

## Trajectory features — `methods/trajectory.py`

T-5's best detection signal was not a posterior but how far one **travels**: `subgoal_step_movement`
and whether the reader is allowed to disengage. Both are trajectory statistics; every instrument in
this project and in Sounding Line reads an endpoint.

That result came from about a dozen features written down by hand. `catch22` is a canonical set
distilled from ~7000 candidates by removing redundancy, and a per-step posterior entropy series is
exactly its input. This is the cheapest available route to an exploratory measure nobody proposed.

**With the honesty protocol attached.** Extracting 22 features and reporting the best is a
multiple-comparisons problem wearing a lab coat. In a simulator the fix is free and stronger than a
reusable holdout: `confirm_on_fresh_seeds` re-runs the winner on seeds that did not exist when it
was chosen, and requires the sign to hold **and** half the magnitude to survive. T-2's difficulty
control flipped sign between n=40 and n=200; a feature picked out of 22 is at least that fragile.

---

## Multiplicity and equivalence — `methods/inference.py`

**FDR.** Batch two reports several hundred bootstrap intervals and corrects none of them.
Benjamini–Hochberg is right here and Bonferroni is not: BH controls the expected *proportion* of
false claims among those made, which is the correct target when deliberately scouring a space.
`control_fdr` reads the `{difference, interval}` shape the verdicts already use. Its p-values are
inverted from intervals under a normal approximation — adequate for **ranking**, and flagged in the
output as not quotable in their own right.

**Equivalence.** Half of batch two's findings are nulls. "The interval covers zero" is the absence
of a claim; "the effect is bounded below 0.02 with 95% confidence" is one. `equivalence()` demands
a `bound_source` string and records it verbatim, because an equivalence bound pulled out of the air
is worse than no test — it converts an arbitrary choice into an authoritative-looking verdict.
`smallest_effect_of_interest` builds one from a fraction of a live effect measured on the same axis
in the same run, which is the only construction here that imports no outside convention.

---

## Metamorphic tests — `tests/test_metamorphic.py`

Relations between **two** executions, with Hypothesis generating the inputs. Both batch-one defects
were invisible to every existing test and neither was a statistics problem; what finds them is
"change this and the output must change", "permute that and it must not".

Currently asserted: a zero-nat channel is bit-exactly uniform at every cardinality; channel
information is bounded and monotone; `fidelity_for_nats` inverts `channel_nats`; **a placebo
channel changes nothing end-to-end through the full stack**; **a full-fidelity channel changes
something**; overlap is symmetric, bounded and translation-invariant; `auc(a,b) + auc(b,a) == 1`
(the relation that makes `auc_oriented` well defined); the paired bootstrap never moves its point
estimate; PID atoms sum to the joint information; XOR reads as one bit of pure synergy; and N28
holds in the likelihood.

---

## Running it

```bash
make gates      # the gate + metamorphic suite only, ~40 s
make test       # everything
python runners/run_soundingline.py --only T1 T2 T3 T4 T5
```

## If you add a module here

1. Per-rollout data goes to `*_points.csv` (gitignored); the aggregate goes to `*_summary.csv`.
   **The naming is the policy**, and `test_no_oversized_committed_csv` enforces it.
2. Call `methods.provenance.stamp(verdict, __file__, gate_report)` before writing.
3. Give it at least one `live` gate and one `placebo` or `positive` gate. If you cannot think of a
   known answer your module should return, that is worth an hour before writing any more of it.
4. Add it to `GATED` in `tests/test_gates.py`.
