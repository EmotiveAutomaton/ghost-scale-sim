# V3 results

Written from the CSVs in `results/`, same discipline as V1/V2: non-replications reported rather
than tuned, deviations collected with evidence, and the pre-registered criteria applied as
written even where they returned an answer their own preconditions do not support.

**Headline: the V3 diagnosis is refuted, and so is its first replacement.** The f = 0 leak is
not finite-sample estimation error — it is flat in per-generation sample size across a 100×
range, flat in the number of observers averaged, and at generation 0 it *grows* with sample
size. E12's own falsification clause fired, E8 did not run, and under §6 it is not reported.
A follow-up hypothesis (that the leak was bounded by observer engagement) was then tested and
refuted in turn.

**What replaced them.** The recursion is a **contraction mapping whose fixed point is the flat
distribution**. Generation g+1 is seeded from a posterior-mean estimate, and a posterior mean is
shrunk toward its prior; each generation re-encodes that shrunk estimate as ground truth, so the
shrinkage compounds. Shrinkage is a bias, which is why it survives more data (E12's N sweep),
more observers (E12's M sweep), and full engagement (E14). This is C4 **outcome 2** — a second,
distinct effect the framework does not predict — and it is flagged as an open problem rather
than explained away.

That is not a wasted round. V3 was built to test a hypothesis rather than assume it. Three
hypotheses died on contact with the data, and the survivor accounts for the V2 leak, the V2
freeze, and the sample-size anomaly in one object.

---

## E12 — Leak versus sample size, and the sample-size decision

`results/e12_leak_vs_samplesize.csv`, `results/e12_slopes.csv`,
`results/e12_threshold.json`, `figures/e12_leak_convergence.png`. Wall clock **231.7 min**.

f = 0 with an honest signal — the exact condition that failed N11 in V2 — at `G_max = 8`,
3 replications, 24 points per cell.

### The N sweep: the leak does not shrink with data

Per-generation leak slope, nats/generation:

| artifacts/generation | 100 | 300 | 1000 | 3000 | 10000 |
|---|---|---|---|---|---|
| **without averaging** (V2 seeding) | +0.0037 | +0.0059 | +0.0042 | +0.0048 | +0.0038 |
| *t* | 3.06 | 4.71 | 2.92 | 3.56 | 2.78 |
| **with C1 averaging** | +0.0037 | +0.0048 | +0.0063 | +0.0065 | +0.0047 |
| *t* | 2.29 | 3.30 | 5.07 | 6.40 | 3.98 |

### N13 — the finite-sample diagnosis, REFUTED

Pre-registered (decision D8): regress `log|slope|` on `log N`; gate on the coefficient being
significantly negative; predicted −1 for 1/N finite-sample error.

```
b = -0.0169   se 0.0611   t = -0.28      predicted -1.0
gate  (b significantly < 0):  REFUTED
1/N band [-1.5, -0.5]:        INCONSISTENT
```

Across a **100× range of sample size the leak is statistically indistinguishable from
constant**, and significant at every single cell. V3 §1 C2 states the consequence, and it is
adopted verbatim:

> if the without-averaging slope does not shrink with N — if it is flat or grows — the leak is
> not finite-sample estimation error and the entire V3 diagnosis is wrong. Report this loudly;
> it would mean the loop is lossy for a structural reason C1 does not address, and E8 must not
> be run until that reason is found.

**No sample size passed the pre-registered N11 criterion.** `sample_size_decision.found =
false`, `e8_may_run = false`. E8 did not run.

### It is a bias, not variance — and D8 is the reason we can tell

Generation 0, before any recursion has occurred, with C1 averaging on:

| artifacts | 100 | 300 | 1000 | 3000 | 10000 |
|---|---|---|---|---|---|
| KL(C_recovered ‖ C_true) | 0.0331 | 0.0419 | 0.0625 | 0.0684 | 0.0746 |
| KL(learned CREATOR column ‖ true) | 0.2885 | 0.4283 | 0.4635 | 0.4555 | 0.4754 |

**More data makes the first generation worse.** Variance shrinks with data; a bias does not,
and more data estimates a bias more sharply. The loop is converging — confidently — on a wrong
answer.

This is the finding that justifies decision D8 in the strongest possible terms. The V3 spec
wrote N13 as strict monotonicity in sample size ("the leak slope must be monotonically
non-increasing"). **A monotonicity null would have read this data as "the leak did not go up"
and PASSED it**, clearing E8 to run on a refuted diagnosis. The log-log exponent is what
caught it, because the diagnosis under test was never "the leak shrinks somewhat" but "the
leak shrinks as 1/N, the signature of finite-sample noise". Testing the real prediction
instead of a weaker proxy changed the verdict of the whole programme.

### The M sweep: C1 works — on a channel that is not the payload

| observers averaged | M = 1 | M = 5 | M = 20 |
|---|---|---|---|
| **payload** leak slope | +0.0059 | +0.0063 | +0.0063 |
| **learned-column** drift slope | +0.0653 | +0.0365 | +0.0243 |
| *t* (column) | 7.56 | 4.83 | 3.86 |

Population averaging does what C1 claims — it cuts drift in the learned CREATOR column by 2.7×
from M = 1 to M = 20 — and that has **no effect whatever on the payload**. The two channels are
dissociated.

So C1 is not wrong; it is aimed at a real drift that is not the one E8 measures. This also
retires V2's stated hypothesis for the honest-signal asymmetry (that single-observer seeding of
a fast-sharpening CREATOR column was the mechanism): fixing exactly that leaves the payload
leak untouched.

**Decision D2 partially confirmed, and it matters.** The pre-registration predicted the 1/M
gain would have a floor, because the corpus is drawn once per generation and shared by all M
observers, so averaging cannot cancel common-mode corpus noise. The column channel does show
the predicted sub-1/M decline (2.7× for a 20× increase in M). The payload channel shows no
gain at all, which is stronger than the floor D2 anticipated.

---

## E13 — The freeze and the leak on one axis

`results/e13_freeze_leak_signature.csv`, `results/e13_verdict.json`,
`figures/e13_shared_signature.png`.

### The C4 classification is UNDEFINED, and this is a deviation

The pre-registered rule returned **outcome 1 (shared axis)**. It should not be reported as
such, because the fit its tolerance test is measured against is not a finite-sample curve:

```
recursion fit:  kl = 0.354 * n_eff ^ (+0.080)
freeze (E9 starvation):  0.344 at n_eff 129.3    predicted 0.522    ratio 0.66  -> "outcome 1"
```

The exponent is **positive**. A flat curve at ≈ 0.5 nats admits anything between 0.26 and 1.04
under a factor-of-2 tolerance, so the test cannot distinguish outcome 1 from outcome 2 — it
would return outcome 1 for almost any freeze value. **The classification is reported as
undefined.**

This is a fault in the criterion, recorded as deviation 2 below, and it is the author's to
own: D7 pre-registered a power-law tolerance test without a precondition that the fitted
exponent be negative. E12 had already shown the exponent would not be, which is precisely when
the check should have been added.

The C4 question — one effect or two — is therefore **unresolved by E13 as specified**. What
resolved it instead was E12's N-sweep and the engagement contrast below.

### What E13 did establish: engagement sets the error LEVEL, and sample count does not

(E14 then showed that the level and the per-generation *decay rate* are separate things —
engagement controls the first and not the second. Read this table as being about the level.)

On the identical quantity through the identical code path (`learning.column_kl`):

| condition | DEEP steps (of 6) | KL(learned CREATOR column ‖ true) |
|---|---|---|
| E9 **control** — engagement forced | 6.00 | **0.0744** |
| E9 **starvation** — engagement free | 1.86 | **0.3442** |
| E12 **recursion** — engagement free | ~2.3 | **~0.55** mean |

Engagement moves the learned-column error by **4.6×**. Sample size moves it by nothing, and in
the wrong direction.

V2's freeze signature replicates exactly: starvation's error is flat in contamination
(0.369 / 0.334 / 0.330 at f = 0 / 0.3 / 0.6), confirming V2's finding that starvation damage is
immediate and contamination-independent.

---

## E14 — Engagement: the hypothesis, and its refutation

`results/e14_engagement_floor.csv`, `results/e14_verdict.json`,
`figures/e14_engagement_floor.png`. Wall clock **~19 min**.

Not in the V3 spec; run because §1 C2 makes finding the cause a precondition for anything
further ("E8 must not be run until that reason is found"). Same f = 0 honest recursion at
N = 1000, M = 5, C1 averaging on, `G_max = 6`, one variable changed.

**The hypothesis it tested.** `learn_step` attributes each observation to the *believed*
`(provenance, goal)`. An unresolved goal posterior therefore misattributes its update, smearing
mass across goal columns; the blurred column produces blurrier posteriors, blurring the column
further. On that account the leak would be bounded by engagement rather than by sample size,
and forcing DEEP should close it.

**It does not.**

| arm | DEEP steps | leak slope | *t* | N11 | column KL (mean) |
|---|---|---|---|---|---|
| forced | 6.00 / 6 | **+0.00724** | 2.91 | FAILS | 0.384 |
| free | 2.17 / 6 | **+0.00548** | 2.56 | FAILS | 0.577 |

Forcing engagement does not reduce the leak — the forced arm's slope is nominally *higher*,
though the two overlap within noise. The pre-registered hypothesis is **refuted**, and it was
my hypothesis, offered in the previous round of reporting. It is recorded here rather than
quietly dropped.

### What E14 established instead, and it is the actual result

Engagement controls single-generation *accuracy* with enormous leverage, and has nothing to do
with generational *transmission*:

| generation | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| payload KL, **forced** | **0.0025** | 0.0171 | 0.0251 | 0.0262 | 0.0307 | 0.0448 |
| payload KL, **free** | 0.0625 | 0.0840 | 0.0736 | 0.0816 | 0.0729 | 0.1058 |
| column KL, **forced** | **0.0848** | 0.2142 | 0.3348 | 0.4646 | 0.5578 | 0.6497 |
| column KL, **free** | 0.4635 | 0.5212 | 0.5225 | 0.6106 | 0.6303 | 0.7121 |

At generation 0 forced engagement recovers the payload **25× more accurately** (0.0025 vs
0.0625) and reproduces E9's control column error (0.085 here, 0.074 in E9 — independent
confirmation across two experiments). Then it degrades *faster* than the free arm (column slope
+0.114, t = 9.5, versus +0.047, t = 4.5).

**Starting from a near-perfect reconstruction does not prevent the decay. It only delays it.**
That is what forecloses every "the observer isn't inferring well enough" explanation, mine
included: at generation 0 the forced arm's inference is essentially exact, and the chain still
loses the payload at the same rate.

### The loop is a contraction toward the observer's prior

Entropy of `C_recovered`, as a fraction of the distance from `H(C_true) = 1.2799` to
`H(uniform) = 1.3863`:

| generation | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| **forced** | 0.13 | 0.47 | 0.57 | 0.54 | 0.62 | 0.76 |
| **free** | 0.87 | 0.89 | 0.93 | 0.89 | 0.90 | 0.96 |

The recursion is a **contraction mapping with its fixed point at the flat distribution**. Free
engagement travels 87 % of that distance in a single generation and then saturates — which is
why its slope *looks* smaller: it has almost nowhere left to fall. Forced engagement starts at
13 % and climbs steadily. Both arms are heading to the same place.

The mechanism is re-encoding, not inference. Generation g+1's creators are seeded from a
posterior-mean estimate, and a posterior mean is shrunk toward its prior. Shrinkage is a
*bias*: it does not vanish with sample size (E12's N sweep), does not vanish with averaging
over observers (E12's M sweep, common-mode), and does not vanish with engagement (E14). Each
generation re-encodes the shrunk estimate as ground truth for the next, so the shrinkage
compounds geometrically toward the prior.

This explains every anomaly in one object: constant slope (a fixed per-generation contraction
ratio), flat in N and M, worse at larger N at generation 0 (the biased fixed point is estimated
more sharply), and the E9 freeze (an arm already at the fixed point cannot move, and is flat in
contamination because the fixed point does not depend on f).

**This is C4 outcome 2, reached by a different route than E13.** There is a second, distinct
mathematical effect that the framework does not currently predict, and it is **not** premature
convergence of a Dirichlet estimate. It is *shrinkage-driven contraction under iterated
re-encoding* — the generational analogue of regression to the mean. It is flagged as an **open
problem in the framework**, in those words, as §5 requires.

**One speculation is retired.** An earlier reading — that more data accelerates disengagement —
is wrong. Engagement *rises* with sample size (mean DEEP 1.96 → 2.31 from N = 100 to 10000) and
with generation (1.81 → 2.54). More looking, and still worse recovery.

E14 identifies a mechanism; it does not repair E8, and §6's bar is unchanged.

---

## N11 — not reached

The repaired N11 gate never ran, because E12 did not clear E8 and stage 3 is gated on stage 1.
The xfail marker on the V2 gate **has been removed** as §3 requires, and the gate now applies
the pre-registered conjunctive criterion (D1). It currently skips, by design, because
`results/e8_*.csv` hold a V2-parameter reference run rather than a V3 E8 run.

E8 is therefore **still not reportable**, for a different and better-understood reason than in
V2, and it remains excluded from E11.

---

## V2 reconciliation

**V2's E8 was correctly withheld, and V3 does not overturn that.** V3 changes the *reason*: V2
diagnosed a lossy loop and hypothesised finite-sample compounding through single-observer
seeding. The first half is confirmed and the second is refuted. The loop is lossy, but not for
that reason, and the fix V2 proposed first (population averaging) does not touch the payload.

V2's E8 numbers are exactly reproducible: `restore_v2_e8.py` regenerates that cell under V2's
parameters and seeding rule and returns f = 0 honest slope **+0.0119, t = 3.75** and unsigned
**+0.0003, t = 0.09**, matching RESULTS_V2.md on both arms. The V3 no-averaging path was
additionally diffed against a verbatim reimplementation of V2's `run_generation`/`run_chain`
and agrees **bit-for-bit** (max |ΔKL| = 0.000e+00 over four generations), so E12's
without-averaging arm is V2's behaviour exactly rather than an approximation of it.

V2's E9 freeze replicates, including its contamination-independence.

---

## The contraction, located: early-step misattribution, scaling as 1/inference

A fourth hypothesis was proposed — that observer inexpertise `d_i` is the smoothing source,
with the contraction being E15's competence ceiling compounding across generations. **It was
already refuted by data on disk before it was run.** E14, E12 and E8 all set `d_i = 0.0`, and
at `d = 0` the observer's `sig` is bit-identical to truth (the N8 fast path draws no variate).
Inexpertise was never present in any recursion experiment, and E14's forced arm *is* the
proposed condition — d = 0 with DEEP forced — which contracted anyway (entropy 0.13 → 0.76,
column slope +0.114, t = 9.5).

That fires the proposed falsification: the smoothing is structural. Following it:

**The seeding path is exact.** Feed a column into `SeededCreator` and read the emission back:
KL = −8e−12, max |Δ| = 2.8e−8. Iterate the seeding path alone six times with no observers at
all and it is a perfect fixed point — entropy 0.99073 at every generation. There is no epsilon
or renormalization bug there.

**The learning step contracts, by a fixed ratio.** One generation, true creators, `d = 0`,
forced DEEP, honest signal, f = 0:

| N | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|
| H(learned column) − H(sig) | +0.035 | +0.061 | +0.083 | **+0.087** |
| contraction ratio *r* | 0.968 | 0.944 | 0.924 | **0.920** |

The learned column is systematically flatter than truth, and *r* **asymptotes to ≈ 0.92 rather
than to 1.0** — the signature of a bias, and the direct explanation of E12's "generation 0 gets
worse with more data".

**The cause is WITHIN-ROLLOUT, not within-corpus.** `learn_step` fires at every timestep with
the *running* posterior, and the first DEEP observation is deposited before the goal has been
resolved — the free initial glance is synthetic and carries no goal information. A fixed
fraction of all evidence, roughly `1/infer_steps`, is therefore filed under the wrong goal.
That fraction is per-rollout, so it is independent of corpus size, common-mode across
observers, and only weakly reduced by engagement — exactly the three invariances measured.

Confirmed by the scaling it predicts:

| inference steps per artifact | 2 | 4 | 6 | 12 | 24 |
|---|---|---|---|---|---|
| contraction 1 − *r* | 0.147 | 0.140 | **0.078** | 0.035 | **0.014** |

Four times more inference, 5.7× less loss, flattening below ~4 steps where the posterior
cannot resolve at all. **V3 ran at `infer_steps = 6`**, i.e. at 1 − *r* ≈ 0.08 per generation.

**An error in the first attempt at this confirmation, recorded because it nearly closed the
question wrongly.** The attribution matrix was first measured from `final_goal_posterior` and
showed only 0.13 % leakage, predicting a contraction of 0.996 against a measured 0.920 — which
read as "mechanism insufficient". That measurement was wrong: updates use the *running*
posterior at every step, not the final one, so it understated the leakage by roughly fifty
times. The mechanism was right and the instrument was wrong.

### What this changes

The contraction is **a property of the measurement protocol, not of the framework**: an
estimator that commits counts before its posterior resolves is biased flat, and iterating
estimate → regenerate → estimate compounds that bias geometrically. It is not a claim about
readers, and it should not be reported as one.

It also gives the first concrete route to a reportable E8. At `infer_steps = 24` the
per-generation contraction falls from 0.078 to 0.014; if the chain leak scales with it, the
f = 0 slope would fall from ≈ +0.0055 to ≈ +0.001 — which is the pre-registered N11 ceiling.
**That is a prediction, not a result**: the scaling above is measured on a single generation,
and whether the chain follows requires running it. Recorded here as the next test rather than
as an outcome.

---

## Follow-ups on the V1/V2 standalone results (E15–E17)

Three experiments run after the V3 programme, sharpening results that do not depend on the
recursion and are therefore untouched by the contraction finding above.

### E15 — E10's competence threshold is a KNEE, not a cliff

`results/e15_competence_cliff.csv`, `results/e15_verdict.json`,
`figures/e15_competence_cliff.png`.

Swept d = 0.40–1.00 rather than the 0.30–0.70 first proposed, because E10's successive
accuracy drops are still accelerating at d = 0.9 — the midpoint lies above 0.7, and the old
range stopped before the collapse it was meant to resolve.

| d | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|
| goal accuracy | 0.9993 | 0.9973 | 0.9851 | 0.9186 | 0.6980 | 0.5110 | **0.2390** |

**The transition is strongly non-linear.** A logistic beats a straight line by ΔAIC 59.7
(R² 0.994 vs 0.612). But logistic and hinge are a dead heat (−111.1 vs −111.0), so the data
cannot even distinguish a smooth sigmoid from a hard knee on shape alone.

**It is not a phase transition, and the width arm is what settles it.** A transition is sharp
because its width shrinks as system size grows; at finite size everything is a crossover.
Across a **16× range of evidence per observer** the fitted width does not move:

| corpus size | 75 | 300 | 1200 |
|---|---|---|---|
| logistic width | 0.0489 | 0.0489 | 0.0473 |

Ratio 0.967. The width is **intrinsic to the model, not an artifact of resolution**, so this is
a smooth crossover of fixed width and **"competence cliff" is not earned**. "Competence knee"
is the defensible term. Had only the fine grid been run, the curve's steepness would have
invited the stronger claim and there would have been nothing in the data to check it.

**Two corrections to V2's E10 write-up.** First, V2 read the threshold as d ≈ 0.5; that is
where the curve *departs from flat*, but the midpoint is **d50 = 0.878** and accuracy is still
falling at d = 1.0 (0.239). V2's sweep ended at 0.9 with the transition unfinished. Second,
**the metrics break at different points**: psi and posterior entropy have d50 ≈ 0.70, accuracy
d50 = 0.878. Belief movement degrades well before choice does — argmax survives a lot of
uncertainty — so "when does extraction fail" has no single answer, and which metric is quoted
changes the number by 0.18 in d.

### E16 — how much AI content must be labelled?

`results/e16_label_coverage.csv`, `results/e16_verdict.json`,
`figures/e16_label_coverage.png`.

Coverage swept 0 → 1 under two regimes (label everything at rate p, versus label only
synthetic work at rate p) crossed with two readers (the V1/V2 observer, versus one that knows
the labelling regime). Threshold = coverage at which the learned GHOST column reaches E7's own
competence criterion.

| arm | coverage to competence | GHOST-column error at full coverage |
|---|---|---|
| **ai_only / regime_aware** | **0.31** | 0.031 |
| ai_only / naive | 0.74 | 0.820 |
| symmetric / naive | 0.42 | 0.031 |
| symmetric / regime_aware | 0.42 | 0.031 |

**Labelling ~31% of synthetic content is sufficient — but only for a reader who knows the
regime exists.** A reader who does not needs 0.74, and never acquires a good model of hollow
content at any coverage (0.820 versus 0.031, a factor of 26).

The mechanism is the contrapositive. Under AI-only disclosure an *unlabelled* artifact is
evidence of humanity, and that inference is unavailable to the V1/V2 observer by construction:
`build_A1_observer` assigns `P(UNSIGNED | provenance)` identically across tiers, so absence of
a label carries exactly zero information at any coverage.

**The symmetric arms are the control that licenses the claim.** 0.418 versus 0.419 — knowing
the regime buys nothing when labelling is uniform, which is correct, because absence really is
uninformative there. The effect appears only where there is an asymmetry to exploit, so the
ai_only difference is attributable to the contrapositive inference rather than to the
likelihood change itself.

**Caveat that must travel with the 31%.** The regime-aware reader is handed the *true*
coverage — the most generous assumption available. It is a **lower bound** on what a real
deployment needs, not a target; a reader with a miscalibrated belief can only do worse.

### E17 — confident fabrication is graded by opacity

`results/e17_points.csv`, `results/e17_tier_stats.csv`, `results/e17_verdict.json`,
`figures/e17_tier_dose_response.png`.

E2's headline ran on a binary CREATOR/GHOST contrast. Run across all four tiers, with the
declared signal claiming human provenance in every case:

| tier | α | within (confidence) | between (disagreement) | fabrication gap |
|---|---|---|---|---|
| CREATOR | 1.00 | 0.0000 | 0.0000 | 0.0000 |
| POLISHED | 0.95 | 0.0000 | 0.0000 | 0.0000 |
| CURATOR | 0.60 | 0.0111 | 0.1083 | 0.0971 |
| GHOST | 0.05 | 0.0894 | **1.3794** | **1.2900** |

Disagreement rises monotonically as intent transmission falls while confidence stays high —
the dissociation is graded, not binary, and the four-tier design carries real signal.
Told the truth instead, GHOST's within-observer entropy jumps to 1.2974: the reader correctly
becomes uncertain rather than confidently wrong.

**Four tiers resolve three levels.** POLISHED (α = 0.95) is indistinguishable from CREATOR on
every measure. The dose axis is α — unevenly spaced by construction — so plotting against tier
index would manufacture a straight line out of that spacing. Worth knowing before the
four-tier scale is defended as four rather than as a graded scale with three resolvable points
in this model.

---

## E18 — Fixing the estimator: the contraction goes, a different residual stays

`results/e18_deferred_estimator.csv`, `results/e18_verdict.json`,
`figures/e18_deferred_estimator.png`.

The f = 0 honest chain at **V3's own `infer_steps = 6`**, forced DEEP, C1 averaging on, with
the estimator as the only variable. Raising the inference budget was deliberately *not* used:
it divides the bias rather than removing it, and choosing the budget at which the residue
clears a gate is choosing an operating point to pass one's own test — decision D1's failure in
a new costume. Deferred commitment removes the bias at any budget (flat from four steps on),
so nothing below depends on where the budget was set.

| generation | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| column KL, **online** | 0.0848 | 0.2142 | 0.3348 | 0.4646 | 0.5578 | **0.6497** |
| column KL, **deferred** | 0.0009 | 0.0013 | 0.0017 | 0.0024 | 0.0035 | **0.0038** |
| H(C_recovered), online | 1.2937 | 1.3302 | 1.3408 | 1.3376 | 1.3454 | **1.3608** |
| H(C_recovered), deferred | 1.2737 | 1.2869 | 1.2855 | 1.2711 | 1.2765 | **1.2934** |
| payload KL, online | 0.0025 | 0.0171 | 0.0251 | 0.0262 | 0.0307 | 0.0448 |
| payload KL, deferred | 0.0011 | 0.0019 | 0.0041 | 0.0037 | 0.0062 | 0.0067 |

**The contraction is gone.** The learned-column error at the final generation falls from 0.6497
to 0.0038 — a factor of **171** — and `H(C_recovered)` stops climbing toward uniform: it
oscillates *around* `H(C_true) = 1.2799` (1.2737 / 1.2869 / 1.2855 / 1.2711 / 1.2765 / 1.2934)
instead of travelling 63 % of the way to flat. The contraction mapping reported above was an
estimator artifact, and it is now removed at its source rather than divided.

**N11 still FAILS, and it is not close enough.** Deferred leak slope **+0.00116, t = 2.14**,
against a pre-registered ceiling of 0.001 and |t| < 2. Both conjuncts miss, narrowly. V3 §6
says "no exceptions, no 'close enough'", and a slope 16 % over its ceiling at t = 2.14 is
precisely the case that rule exists for. **E8 remains withheld.**

### Two independent routes hit the same floor

`results_e14_steps24/`. The diagnostic re-run of E14 at `infer_steps = 24` — the tuned route,
run to bound the mechanism and explicitly not to report E8 — lands on the same residual:

| route | intervention | leak slope | *t* | column KL (final) |
|---|---|---|---|---|
| E14 @ 24 steps, forced | keep the online estimator, quadruple inference | **+0.00116** | 2.12 | 0.0278 |
| E18 deferred @ 6 steps | fix the estimator, keep V3's budget | **+0.00116** | 2.14 | 0.0038 |

Two interventions with nothing in common but their target agree to five decimal places, and
the single-generation scaling predicted +0.0010 against +0.00116 measured — a 16 % miss on a
prediction registered before the run. Early-step misattribution is confirmed as *a* mechanism
and quantified; the floor beneath it is confirmed as *not* that mechanism.

**The column channel and the payload channel come apart again.** Deferred commitment leaves a
learned-column error seven times smaller than the 24-step route (0.0038 vs 0.0278), and the
payload leak is nevertheless identical. The residual therefore does not travel through the
learned likelihood at all — it lives in the `C_recovered` → `allocate_creator_goals`
re-encoding path. That is the same dissociation E12's M-sweep found, now localised to one
code path rather than inferred from an invariance.

**The free-engagement arm is the control that makes this readable.** At 24 steps it uses only
2.33 of them and its numbers barely move (leak +0.00630 vs +0.00548 at 6 steps; column KL
0.5666 vs 0.5767). A larger inference budget helps only an observer that spends it — which is
what rules out "more compute per artifact" as a general explanation of either result.

**Both routes also FAIL N11** (0.00116 > 0.001, t ≈ 2.1). Worth stating plainly: the tuned
configuration does not pass either, so there was never a tempting number here to report.

### The residual is a different animal, and that matters

The online arm shows KL growing *and* entropy rising monotonically — the signature of a
contraction toward uniform. The deferred arm shows **KL growing while entropy stays put**,
wandering above and below the true value rather than climbing. Payload KL rises roughly
linearly in generation (0.0011 → 0.0067), which is what diffusion does.

That is a **random walk on the simplex at roughly constant entropy**, not a contraction. The
two were superimposed, and the estimator fix separated them: bias removed, variance remaining.

**Which reopens V3's original hypothesis, and this is stated as a hypothesis given that four
have now died.** A random walk in the re-encoding loop is exactly the finite-sample
compounding V3 §1 C2 proposed, and it should shrink as 1/N and with observer averaging — the
two things E12 measured and found flat. But E12 ran the ONLINE estimator, where a contraction
bias roughly six times larger sat on top of any diffusion and would have masked its scaling
entirely. The V3 diagnosis may have been right about the mechanism that remains, and wrong
only because it was measured through a much larger effect.

**The test that settles it**: re-run E12's N-sweep and M-sweep under the deferred estimator
and read the exponent. If `b` goes negative and lands near −1, the residual is finite-sample,
C1 and larger N are the right tools after all, and E8 becomes reachable on the merits. If `b`
stays flat, a third mechanism is present. Not run here; recorded as the next experiment.

**A caution for whoever runs it.** "Pick N large enough that the slope clears the ceiling" is
the same failure as picking `infer_steps = 24`. What licenses a scale choice is a *measured*
1/N law plus V3's existing rule that a null is evaluated at the scale of the experiment it
gates — not the observation that the number went under the line at some N.

---

## The methodological finding: check the instrument before accepting the story

This is the most transferable thing in the V3 round, and it is recorded as a finding rather
than as housekeeping because it changed the reported conclusion **three separate times**. In
each case a plausible, coherent story was one step from being written down, and what stopped
it was measuring the instrument instead of the phenomenon.

**1. The V2 reproduction that "proved" environmental drift.** Re-running V2's E8 cell returned
slope +0.0067 against the reported +0.0119, twice. The available story was good: different
numpy/BLAS build, different hardware rounding inside pymdp's inference flipping engagement
decisions, chaotic divergence. It was drafted into the write-up in those terms. The actual
cause was `resolve_sample_size` consulting a leftover `e12_threshold.json` regardless of its
`require_e12` argument, silently running at 120 artifacts instead of 300. Deleting one stale
file reproduced V2 **exactly** — +0.0119, t = 3.75, both arms. Had the drift story been
accepted, a real bug would have shipped and V2's evidence would have been recorded as
irreproducible.

**2. The monotonicity null that would have passed a refuted diagnosis.** V3 §3 specified N13
as "the leak slope must be monotonically non-increasing in sample size". The measured leak is
flat in N — which *satisfies* monotonicity. The gate would have gone green and E8 would have
run on a diagnosis the same data refutes. What caught it was testing the exponent the
diagnosis actually predicts (1/N) rather than a weaker proxy that the diagnosis merely
implies. **An instrument that cannot fail is not an instrument.**

**3. The mechanism "refuted" by a mis-measured attribution matrix.** The early-step
misattribution hypothesis was checked by building the attribution matrix from
`final_goal_posterior`. It showed 0.13 % leakage and predicted a contraction of 0.996 against
a measured 0.920 — comfortably "mechanism insufficient", and that is nearly what was written.
But Dirichlet updates use the *running* posterior at every timestep, not the final one, so the
measurement understated leakage roughly **fifty-fold**. The hypothesis was right; the
instrument was wrong. Re-measured against the prediction it actually makes — that contraction
scales as 1/`infer_steps` — it holds cleanly (0.147 / 0.140 / 0.078 / 0.035 / 0.014 at
2 / 4 / 6 / 12 / 24 steps).

**The common shape.** In all three, the measurement apparatus was quietly answering a
different question from the one being asked: a stale file substituting its own parameter, a
null testing a weaker claim than the hypothesis, a statistic computed at the wrong timestep.
None was detectable from the result alone — each produced a plausible number, and two produced
plausible *stories*. What exposed them was asking what the instrument would report if the
hypothesis were false, and checking that it could.

That question is cheap. In this round it cost minutes each time, and it changed the headline,
the gate, and the mechanism respectively. **When a result is surprising, suspect the
instrument before the phenomenon — and when a result is unsurprising, suspect it harder**,
because a confirmation is exactly the case where nobody looks.

---

## Deviations

1. **E14 was added, and is not in the V3 spec.** Justified by §1 C2's requirement that the
   structural reason be *found* before E8 is attempted, and by E13's classification proving
   undefined. Its own stated hypothesis was refuted; the result is reported as a refutation,
   and the finding that survived it (contraction toward the prior) came from the arm contrast
   rather than from the predicted effect. It repairs nothing and E8's bar is unchanged.

2. **E13's pre-registered C4 classifier returned a verdict its preconditions do not support.**
   The factor-of-2 tolerance test around a fitted power law is only discriminating when the
   exponent is negative; at b = +0.080 it is near-vacuous. Outcome 1 was returned and is **not
   reported as the result**. The criterion needed a precondition on the sign of the exponent,
   and D7 did not include one.

3. **V2's E8 CSVs and figure were overwritten during development** by a `--quick` run launched
   without `--out`, and were regenerated (reproducing V2 exactly, above). Two hardening
   changes followed: `--quick` now writes to `results_quick/` and cannot reach `results/` or
   `figures/`, and opting out of the E12 gate now means not consulting it — a stale
   `e12_threshold.json` had been silently overriding an explicit sample size with no error.
   See DECISIONS_V3.md, "An incident worth recording".

4. **E16's primary outcome was changed after the first run, on pre-existing grounds.** The
   module was written with `creator_mi` as primary; E7's own documentation already records
   that the D1-seeded learner starts at ~95% of oracle MI, leaving ~5% of the range available,
   so it cannot resolve a threshold. The primary was moved to `ghost_col_err` on that
   documented ground rather than because of how the numbers came out, and both measures are
   reported in the CSV so the change is auditable.

5. **E17's monotonicity criterion was changed after the first run.** Spearman penalises ties,
   and CREATOR/POLISHED tie at exactly 0.000; a perfectly monotone result scored −0.80. The
   criterion was moved to tie-aware weak monotonicity with a tolerance of 1e-3 nats — set to
   the measurement scale, against a between-observer sd of ~0.016. The violating step was
   −2.3e-6 nats. Flagged because changing a criterion after seeing data is the same fault
   recorded in deviation 2, and it should be visible even when the change is defensible.

6. **The E12 gate blocks stage 2 as well as stages 3–5.** E13 is not gated on E12 in the spec's
   run order, but `run_all_v3.py` exits at the stage-1 gate before reaching it. E13 was run
   separately. The stage ordering should be amended so a failed E12 blocks only E8 and E11.

---

## Wall clock

| stage | experiment | wall clock | notes |
|---|---|---|---|
| 1 | E12 | **231.7 min** | 10000-artifact cell retained; ~69% of the sweep's cost |
| 2 | E13 | ~14 min | run separately; see deviation 4 |
| 3 | N11 re-run | — | not reached (E12 gate) |
| 4 | E8 | — | **did not run**; §6 |
| 5 | E11 re-pool | — | not reached |
| — | E14 | **~19 min** | added; see deviation 1 |
| — | E17 | ~1 min | follow-up on E2 |
| — | E16 | ~13 min | follow-up on E7 |
| — | E15 | ~26 min | follow-up on E10 |

The D3 estimate of ~3.4 h for E12 was accurate (231.7 min = 3.9 h). The V3 spec §4 estimate for
the same experiment was 30–60 min.

Total compute spent: **~4.4 h**, against a programme budgeted at ~1.5 h that would have run
~5–6 h had every stage executed. E8, the most expensive stage, was never reached — the gate
that stopped it did so after 3.9 h rather than after the full run, which is the ordering in §2
doing its job.
