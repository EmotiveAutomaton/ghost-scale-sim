# RESULTS — V5

V5 exists because a careful read-through of V1–V4.5 by the author surfaced four construct errors
and one omission. **None were found by the simulations.** They were found by someone who knows
the theory checking whether the implementation matched it.

**Status: C1 only.** μ is built, its dissociation null N21 is built and passes, and E30 has run.
C2, C3, C4 and E31–E34 are not started and must pre-register separately.

Criteria were pre-registered and hash-locked in `results/v5_preregistration.json` before N21 ran
at scale and before a line of E30 existed, with every outcome branch written in advance.
`ghostscale/prereg_v5.py` holds them as the functions the experiments and the tests both call,
so the written criterion and the applied criterion are the same object.

---

## E30 — μ, estimated model depth, replaces β

### Verdict: DEPTH_MOVES_NOTHING

**Depth changes what a reader recovers and does not change how much it moves them.** Update
magnitude does not scale with depth — Spearman −0.5 against a required 0.99 — while goal
accuracy holds at **1.000 at every depth**. The observer reads a deep artifact and a shallow one
equally well, correctly identifies which is which, and takes on the same amount from both.

That is the pre-registered `DEPTH_MOVES_NOTHING` branch, written before the run in these words:
*"μ is recoverable and inert, which would mean depth is not what gates uptake and C1's central
claim is wrong."* C1's motivating claim is that what gates uptake is the observer's estimate of
how complex a model the creator had. On this model, it does not.

`results/e30_verdict.json`, `results/e30_cell_stats.csv`, `results/e30_points.csv`,
`figures/e30_depth.png`. 2 attention arms × 7 cells × 20 seeds × 200 observers, 4000 observers
per cell.

### Measured — the depth grid, sustained attention

| true μ | recovered μ | goal accuracy | update (pre-reg) | update (structure) | sub-goal entropy |
|---|---|---|---|---|---|
| 1 — execution only | 1.559 | 1.000 | 3.230 | 3.313 | 1.351 |
| 2 — + sub-goal | **2.253** | 1.000 | 3.253 | **3.656** | 1.211 |
| 3 — + goal plan | 2.239 | 1.000 | 3.223 | 3.583 | 1.230 |

### What held, and it is the thing E28 could not do

**Goal accuracy is 1.000 at every depth.** E28's β collapsed to 0.257 at the bottom of its range,
because low-β content carries no goal information — so an observer that failed to read it
demonstrated nothing κ_p could not already explain, and the predicted category existed over only
part of the range. Depth has no such bottom. Shallow content is exactly as legible as deep
content, by construction and now by measurement, so `DEPTH_IS_LEGIBILITY` — the branch where μ
turns out to be doing κ_p's job under a new name — is ruled out on the evidence rather than by
assertion.

**The negative control is exact.** At δ = 0 the execution modes collapse onto `sig[g]`, so a μ = 3
artifact *is* a μ = 1 artifact and there is nothing to find. Recovered μ came back **2.000, standard
deviation 0.000, posterior entropy ln(3) to machine precision** — the observer sits precisely on
its flat prior. Recovery then rises monotonically with mode contrast (Spearman **1.000**):

| δ (how distinct the stages are) | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| recovered μ | 2.000 | 2.000 | 2.065 | 2.131 | 2.239 |

So whatever E30 reports about depth is a reading of the artifact and not an artifact of the
hypothesis space. That axis is the cleanest measurement in V5.

### What did not hold, beyond the headline

**`depth_is_recoverable` is false, and not for the reason the flag's name suggests.** Spearman on
the μ grid is 0.5 against a required 0.99. But the failure is entirely in the top two levels:
μ = 1 is separated decisively from both (1.559 against 2.253 and 2.239), while μ = 2 and μ = 3 are
tied within 0.014 — far inside their own spread of 0.32.

**The reading, and it is a limitation of the construction rather than of the theory.** The third
level's contribution is goal-diagnostic: at μ = 3 the execution modes are permuted per goal, so
the ORDER of modes tells you which goal you are looking at. But the observer already identifies
the goal perfectly at μ = 2 — accuracy is 1.000 everywhere — so the third layer supplies
information this observer does not need. **Depth saturates at two levels here because the third
one's payload is redundant with something already known.** A design where the deepest layer
carried something not otherwise available would test the top of the range; this one does not, and
the μ grid should be read as a two-level contrast that happens to have three labels.

### The secondary update measure, which changes the reason and not the outcome

`psi_analogue` is `KL(goal posterior ‖ goal prior)`, and depth is constructed so the goal is
exactly as recoverable at every depth. **The pre-registered measure therefore has no headroom
here by design** — it cannot vary with depth whatever is true of the reader. Reporting only it
would answer "does depth move the reader" with a quantity incapable of answering.

`psi_structure` is the same object one level up: how far the observer's belief about the
creator's whole intent structure moved, goal and execution mode jointly, with μ marginalised out
so the manipulation is not scored as its own effect.

| | μ = 1 | μ = 2 | μ = 3 | δ = 0 control |
|---|---|---|---|---|
| pre-registered `psi_analogue` | 3.230 | 3.253 | 3.223 | 3.236 |
| secondary `psi_structure` | 3.313 | **3.656** | 3.583 | 3.236 |

Depth moves structural uptake by about 10% and moves goal uptake by nothing. The control is
clean: at δ = 0 the structural measure sits at 3.236, below every full-contrast cell, so it is
not counting the hypothesis space.

**This does not rescue the verdict and is not offered as doing so.** It shows the same
saturation at μ = 2, its Spearman is also 0.5, and part of what it credits is the observer
knowing which mode is active at the END — transient state rather than durable uptake. It is a
second view, not a better one. `DEPTH_MOVES_NOTHING` stands.

### The attention contrast

| arm | recovered μ at μ = 1 / 2 / 3 | engagement | DEEP steps |
|---|---|---|---|
| sustained (primary) | 1.559 / 2.253 / 2.239 | forced | 24.0 |
| free to stop | 1.798 / 2.094 / 2.096 | 0.03 / 0.19 / 0.17 | 10.5 / 12.7 / 12.4 |

A reader free to stop recovers less depth, and the compression is toward the prior rather than
toward a wrong answer. Depth lives in the ORDER of execution modes, so ten steps of a
three-and-a-half-block artifact is about one and a half blocks. Two things worth noting: the free
arm's recovery is *monotone* where the sustained arm's is not, which is noise at this separation
rather than a finding; and engagement is higher on deep content (0.19 and 0.17) than on shallow
(0.03), so **the observer pays more attention to deeper work without taking more from it.** That
is consistent with E29's finding that engagement here is driven by expected information gain.

### E28 and E30, side by side

V5 decision 8: E30 is a **new design, not a re-run**, and E28 is not deleted. Under a hierarchical
creator there is no version of E28's stimulus to re-run, so the two are two constructs measured
on comparable designs.

| | E28 — β, rationality | E30 — μ, model depth |
|---|---|---|
| recoverable from content alone | yes, Spearman 1.000 | partly: shallow vs deep, not within deep |
| goal accuracy across the range | 0.257 → 1.000 | **1.000 everywhere** |
| update magnitude | **falls with β**, 3.218 → 1.104 | flat, 3.230 / 3.253 / 3.223 |
| verdict | separable over part of the range | moves nothing |

**The honest summary of C1 so far: μ is the better-specified construct and the worse-performing
one.** β's update effect was real but confounded — update fell with β partly because legibility
fell with β, and E28 said so. μ removes the confound, and with the confound removed the update
effect disappears. That is what the correction bought: a cleaner measurement of a smaller effect,
and the framework should say so rather than presenting μ as a straight improvement.

---

## N21 — does the observer recover depth, or effort under a new name?

### Verdict: DEPTH_RECOVERED_NOT_EFFORT. Dominance 5.98× against a required 3.0.

**This is the gate on E30 and it is the single most important item in V5.** C1 motivates μ with a
dissociation — fully committed but trivial is high β and low μ; an offhand sketch by a master is
low β and high μ — and assigns it to no experiment. Without a test of it, μ could have been β with
a new label, E30 would have reproduced E28's numbers, and the correction would have been cosmetic
while looking like a success.

`results/n21_verdict.json`, `results/n21_cell_stats.csv`. 4 cells × 20 seeds × 60 observers.

| | recovered μ | goal accuracy |
|---|---|---|
| β = 1.00, μ = 1 — committed, trivial | 1.552 | 1.000 |
| β = 1.00, μ = 3 — committed, deep | **2.344** | 1.000 |
| β = 0.25, μ = 1 — offhand, shallow | 1.636 | 0.744 |
| β = 0.25, μ = 3 — offhand, deep | 1.841 | 0.760 |

β's ability to manufacture depth — its effect on content that has none — is **−0.083**, against a
μ effect of **+0.499**. β does not make the observer see depth that is not there.

### N21 earned its place before E30 ran

Run first, N21 caught two defects that would each have surfaced in E30 as a small, plausible,
uninformative effect — and would have been read as evidence against C1 rather than against the
machinery.

**1. pymdp's mean-field inference could not read depth at all.** With μ in a separate factor from
the sub-goal, the μ posterior is updated from the expected LOG likelihood under the sub-goal
marginal, which by Jensen penalises any hypothesis carrying latent structure — and μ = 1 carries
none, so it wins on content that refutes it. Measured on identical feature sequences:

| true μ | exact filtering | mean-field agent |
|---|---|---|
| 1 | 1.52 | 1.04 |
| 2 | 2.20 | 1.04 |
| 3 | 2.27 | 1.06 |

The depth was in the artifact; the solver could not see it, and was *confident* — posterior
entropy 0.047. Fixed by merging μ into the sub-goal's factor, at no cost: the state space is 384
either way.

**2. V4.5's β destroys depth.** It mixes toward `sig_EXPLORE`, which is mode-independent, so a low
β washes out the hierarchy along with everything else and "offhand but deep" — the whole point of
C1's 2 × 2 — was unrepresentable. V5's β mixes toward the goal-marginal of the mode family at that
depth: the craft survives, what attenuates is what it was in service of. This **generalises**
V4.5's β rather than replacing it, and the reduction is exact — at μ = 1 the target is
`mean_g(sig[g])`, which IS `sig_EXPLORE`, asserted at 0.0.

---

## The construction, and what is asserted about it

`ghostscale/v5_model.py`. **A scalar μ was considered and rejected**, and the reason is C1's whole
content: β is a mixture weight on a flat categorical, and any scalar μ built the same way
collapses onto the same axis. Route B, composite observations over an F^k alphabet, is documented
as a fallback and was not needed — a 24-step rollout costs 166 ms in the V4.5 model and 657 ms
here, so E30 runs in minutes.

**Depth is correlational, not marginal, and this is enforced rather than hoped for.** The
execution modes average exactly to `sig[g]` and the mode chains are doubly stochastic with a
uniform stationary distribution, so a deep artifact and a shallow one have identical
time-averaged feature histograms. A reader who counts features and ignores their order cannot
tell them apart at all. Asserted at construction, measured at 5.6e-17.

Asserted at every build, each with its own failure message:

- μ = 1 reproduces V4's `A[0]` **elementwise** — measured 0.0, tolerance 1e-12
- β at μ = 1 reduces to V4.5's `sig_EXPLORE` — measured 0.0
- execution modes average exactly to `sig[g]`; chains doubly stochastic — both 0.0
- modes pairwise separated above a Jensen-Shannon floor — 0.192 against 0.10
- modes carry < 0.10 mass in the FOREIGN block — 0.0755, which is V4's own `sig_true` baseline
- three to four mode blocks fit a 24-step artifact — 3.50

**Three construction faults were caught by these assertions rather than by E30's residuals**, and
each would have been invisible in the output: a first mode family separated by 0.019 nats where
two of four modes were near-duplicates; a rebuilt family that put mode mass in the FOREIGN block,
which `assert_c1_properties` cannot see because it only inspects `sig_true`; and cyclic mode
permutations that gave one goal the identity, leaving it unable to distinguish μ = 2 from μ = 3.

### Decision 10 was revised by measurement

C2's evidence path was to be a joint prior over (provenance, μ) in one factor. That is
incompatible with the merge that fixes the solver: putting all four quantities in a single factor
needs a cardinality-1 placeholder to keep attention at factor index 2, and pymdp raises on it. So
provenance stays its own factor and the evidence path becomes a μ prior conditioned on the
observed label — unchanged in substance, arguably more inspectable, and it gives up one thing:
**the observer no longer infers provenance and depth jointly, so a reader who doubts the label
does not automatically doubt the depth estimate that label induced.** E31 manipulates this path
and must report the limitation.

---

## Deviations

**V5-1 — N21's second clause was restated after the first version failed.** The clause as written
averaged β's simple effects across both μ levels; scored that way the null failed at 1.67 against
a required 3.0, and the criterion was restated afterwards to score β's effect on the μ = 1 row.
Nothing was pre-registered at the time — `prereg_v5.py` did not exist and was written immediately
afterwards, before N21 ran at scale — but an operationalisation changed after seeing a
measurement is an operationalisation changed after seeing a measurement.

The substance: β's effect is −0.053 at μ = 1 and +0.421 at μ = 3. β does not invent depth; it
limits how much *real* depth is recoverable, because depth is defined relative to a goal's mode
family and the plan cannot be read without partly knowing the goal. The original clause charged
that legitimate limitation as contamination. The restated clause asks what the null is for: can β
make the observer report depth that is not there.

**The threshold value was not changed** — restating which quantity it scores is the deviation;
moving the number as well would have been a second one. The original clause is retained,
computed, and reported as `dominance_ratio_original` = **1.701, which fails**. It decides nothing.

**V5-2 — E30 gained a second update measure, after the run.** `psi_analogue` has no headroom in
this design by construction, since depth is built so the goal is exactly as recoverable at every
depth. `psi_structure` measures the same thing one level up.

**This cannot manufacture the welcome answer and it was checked before being added:** the δ = 0
negative control must show no structural update either, and it does not — 3.236 there against
3.583 at full contrast. Adding the column is inert with respect to everything else: all 15
pre-existing columns came back **bit-identical** across 630 rows on a re-run, and the computation
consumes no random variates. The pre-registered measure still decides the verdict, and the
secondary agrees with it on the outcome while disagreeing on the reason.

---

## What is not done

- **C2, C3, C4 are not built.** E31, E32, E33, E34 are not started. Each needs its own
  pre-registration; `v5_preregistration.json` covers C1 only and says so in its `scope` field.
- **The literature check (V5 §0) has not been run.** It is explicitly not a spec-implementation
  task and remains outstanding.
- **The README update (V5 §4) is not done**, except that `LICENSE` and `CITATION.cff` were
  already present and the figures/CSV gitignore had already been fixed — §4's premise on both is
  stale. The landing page still describes V3 and lists 19 experiments.
- **E8 remains withheld** with its `xfail(strict)` marker. **E27 stays open.** The null suite is
  unchanged and passes.
- **The μ grid should be redesigned before it is used again.** μ = 2 and μ = 3 are not
  distinguishable in this construction, for the reason given above, and a three-level axis whose
  top two levels are tied is a two-level axis with an extra label.
