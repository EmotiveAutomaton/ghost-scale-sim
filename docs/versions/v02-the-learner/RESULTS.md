# RESULTS — V2

Written from the CSVs in `results/`, same discipline as V1: non-replications are reported
rather than tuned, and every deviation is collected with the evidence that motivated it.
Design decisions D1–D5 were signed off **before** implementation and are recorded in
`DECISIONS_V2.md`.

---

## Pre-registration compliance — E6b

Required by V2 spec §5. This is the first thing a skeptical reader should be able to check.

**The bound was computed and written to disk before the run, and locked.**
`results/e6b_preregistration.json`, content hash
`93011f34141581b4053a92625c8d98801283572804e8afd31c3fbbe98fe69f47`, written by
`--prereg-only` before any inference. `preregistration.assert_prereg_locked` recomputes the
hash at run time and aborts on a mismatch; `write_preregistration` refuses to overwrite a
differing file. Both mechanisms are tested
(`tests/test_nulls_v2.py::test_N9_prereg_is_locked_against_tampering`).

**Per decision D3a**, the file records the full 200-seed scan — every candidate synthetic
draw with its favoured goal, lean and entropy — plus the stated selection rule: *for each
favoured goal k, the scanned seed whose lean is closest to the scan median (0.9611)*, which
holds bias magnitude roughly constant so that `k` is the only quantity varying across draws.

| draw seed | favours | lean | H(synth) | pre-registered bound at f=0.8 |
|---|---|---|---|---|
| 17 | G0 (C_true 0.40) | 0.952 | 0.226 | 0.501 |
| 187 | G1 (C_true 0.30) | 0.957 | 0.633 | 0.680 |
| 109 | G2 (C_true 0.20) | 0.963 | 0.190 | 0.948 |
| 45 | G3 (C_true 0.10) | 0.959 | 0.203 | 1.436 |

### Compliance table

| item | pre-registered | observed |
|---|---|---|
| naive KL at f=0.8, biased arm (pooled over k) | 0.15 – 0.60 | **0.1238** |
| falsification threshold | < 0.10 ⇒ falsified | **not crossed** |
| observed below its bound, every k | required | **yes, all four** |

**Verdict: NOT falsified, but the point prediction was NOT met.** The observed pooled value
of 0.1238 sits in the gap between the falsification threshold (0.10) and the bottom of the
predicted range (0.15). Downstream stages were therefore run, as the spec's gate directs, but
the prediction itself is recorded as **unconfirmed at the pooled level**.

### What the stratification revealed — and why D3a earned its cost

The pre-registered prediction is met for one draw and missed for the others:

| synth favours | C_true[k] | naive KL at f=0.8 | bound | in predicted range? |
|---|---|---|---|---|
| G0 | 0.40 | 0.0458 | 0.501 | no |
| G1 | 0.30 | 0.1374 | 0.680 | no |
| G2 | 0.20 | 0.1016 | 0.948 | no |
| **G3** | **0.10** | **0.2103** | 1.436 | **yes** |

`corr(KL, bound)` across the four draws at f=0.8 is **0.873**. Corruption scales with the
realized lean, which is the mechanism C2 claims.

**G3 is the goal the committed V1 synth seed happens to favour.** Had E6b been run as
specified — one draw, the default seed — it would have returned 0.21, landed inside the
predicted range, and been reported as a clean confirmation. Pooled honestly across which goal
the synthetic content happens to favour, it is 0.124 and the prediction is missed. The
apparent confirmation was an artifact of which goal the draw pointed at, and the
stratification is the only reason that is visible.

---

## E6b — Corpus corruption under biased synthetic content (V2 C2)

Oracle aggregator, 2000 artifacts, full V1 grid (κ ∈ {0.1,0.5,0.9} × signing ∈ {0,0.5,1.0}),
5 contamination levels × 3 replications × 4 biased draws + symmetric control.
`figures/e6b_corpus_biased.png`. Wall clock **24.7 min** (10 workers).

Naive-aggregator KL(C_recovered ‖ C_true):

| f | biased | symmetric (V1 control) | ratio |
|---|---|---|---|
| 0.0 | 0.0012 | 0.0007 | 1.58 |
| 0.2 | 0.0094 | 0.0070 | 1.35 |
| 0.4 | 0.0343 | 0.0236 | 1.46 |
| 0.6 | 0.0733 | 0.0504 | 1.45 |
| 0.8 | **0.1238** | **0.0826** | 1.50 |

**The V1 post-mortem is partially supported, and weaker than it claimed.** Removing
symmetrization does increase corruption, monotonically and at every contamination level, by a
strikingly stable factor of **~1.5×**. But the post-mortem's framing — that symmetrization
was *what suppressed* the V1 effect — implies something much larger. A 50% increase does not
carry that reading. The bias axis is real; it is not the whole explanation for V1's small E6
effect.

**The symmetric control reproduces V1 (null N9).** 0.0826 here against V1's measured 0.066 at
f=0.8, on a different creator-allocation scheme and a re-implemented aggregator. The new
effect comes from the bias axis and not from the refactor.

**Provenance-weighting remains near-immune, and this is the result that strengthens.**

| f | naive (biased) | provenance-weighted (biased) |
|---|---|---|
| 0.4 | 0.0343 | 0.0023 |
| 0.6 | 0.0733 | 0.0027 |
| 0.8 | 0.1238 | 0.0080 |

A 15× protection factor at f=0.8 under the *harder* biased condition. V1's core E6 finding —
that provenance-aware aggregation protects the alignment payload — survives the change that
was designed to break it.

### A confound, stated plainly

The between-arm comparison is **confounded by entropy**. Symmetrization redistributes mass
across all four goal pairs and necessarily raises entropy: H(synth) is **1.515** in the
symmetric arm against **0.313** in the biased arm. So the 1.5× between-arm ratio cannot be
attributed to lean alone.

The **within-biased-arm k-dependence is not confounded**, and it is where the C2 claim
actually rests: all four draws are matched on lean (0.952–0.963), and the KL ordering follows
the *bound* rather than the entropy — draw G1 has by far the highest entropy (0.633 vs
~0.20) yet a *higher* KL than the lower-entropy G2 draw, which is the opposite of what an
entropy artifact would produce. This is why D3a's stratification is load-bearing rather than
decorative.

### Harm does not track KL (early E11 evidence)

At f=0.8, across the four draws:

| synth favours | KL | regret | argmax preserved |
|---|---|---|---|
| G0 | 0.046 | 1.5 | 1.00 |
| G1 | 0.137 | 10.0 | 0.00 |
| G2 | 0.102 | 15.3 | 0.22 |
| G3 | 0.210 | 23.2 | 0.11 |

KL spans 4.6× while regret spans 15×, and the ordering is not the same (G2 has *lower* KL
than G1 but *higher* regret). See E11.

---

<!-- Sections below are filled from the CSVs as each stage completes. -->

## E10 — The expertise gradient (the RLHF result)

Inexpertise `d` swept 0 → 0.9 on a corpus of **uncontaminated, genuinely intent-dense human
artifacts** — no synthetic content at any level, asserted in the worker (null N12). 300
artifacts × 20 observers × 5 replications per level. `figures/e10_expertise_gradient.png`.
Wall clock **5.1 min**.

| d | psi (intent density) | goal accuracy | posterior entropy | KL(C_rec‖C_true) | regret | argmax kept | sycophancy |
|---|---|---|---|---|---|---|---|
| 0.0 | 3.232 | 1.000 | 0.000 | 0.002 | 1.09 | 1.00 | 0.592 |
| 0.2 | 3.242 | 1.000 | 0.001 | 0.002 | 1.10 | 1.00 | 0.592 |
| 0.4 | 3.214 | 0.999 | 0.007 | 0.002 | 1.09 | 1.00 | 0.592 |
| 0.5 | 3.170 | 0.997 | 0.023 | 0.002 | 1.15 | 1.00 | 0.594 |
| 0.6 | 3.047 | 0.985 | 0.085 | 0.007 | 1.67 | 0.97 | 0.622 |
| 0.7 | 2.803 | 0.931 | 0.190 | 0.029 | 2.72 | 0.88 | 0.601 |
| 0.8 | 2.522 | 0.750 | 0.308 | 0.088 | 6.06 | 0.66 | 0.595 |
| 0.9 | 2.329 | 0.530 | 0.395 | 0.151 | 9.63 | 0.46 | 0.564 |

**CONFIRMED — the central claim.** Extracted intent density falls monotonically with
inexpertise on a corpus whose quality is held *perfectly* constant: slope **−0.968 nats per
unit d, t = −34.2**, Spearman −0.73. Goal-recovery accuracy falls from 1.000 to 0.530 and
downstream regret rises 9× (1.09 → 9.63), with argmax preservation collapsing from 1.00 to
0.46. Not one synthetic artifact was involved. **The extractor's own competence is a hard
ceiling on recoverable intent, independent of data quality**, and the falsification criterion
(a flat gradient) is not remotely approached.

**The gradient has a knee, and that matters.** Nothing happens until d ≈ 0.5 — psi is flat to
three decimals and accuracy is ≥ 0.997 across the entire first half of the range — and then
collapse is rapid. The claim "more expertise helps" is therefore *false over most of the
range* in this model. What the result actually says is that there is a **competence
threshold**, below which extraction degrades catastrophically and above which additional
expertise buys nothing measurable. For the RLHF reframing this is a sharper claim than the
linear one: raters do not need to be experts, they need to be *above threshold*, and the
harm from being below it is severe and non-linear.

**NOT REPLICATED — the sycophancy prediction.** E10 predicted that `C_recovered` from
high-`d` observers would produce measurably *higher* sycophancy downstream. It does not:
slope **−0.0065, t = −0.37**, Spearman +0.002 — flat, and if anything faintly negative. The
prediction fails, and the reason is visible in the same table. Sycophancy in this model rises
with the *flatness* of `C_recovered` (H5's mechanism: a flat prior supplies no counter-
evidence to a misleading user signal). But inexpertise does not flatten `C_recovered` — it
produces a **sharp and wrong** one. KL rises to 0.151 and the argmax flips in over half the
populations, while the recovered distribution stays concentrated. A confidently wrong
preference prior resists a misleading signal just as well as a correct one; it simply resists
it toward the wrong answer.

That is a genuine finding rather than a null: **inexpertise produces confident error, not
uncertainty** — the same signature E2 reported for hallucination, now appearing one level up
in the recovered preference distribution. Harm from inexpertise shows up in regret and argmax
flips, not in sycophancy. Reported as measured.

---

## D2 — prior_strength calibration: **FAILED, fallback applied**

The criterion was written to `results/v2_calibration_criterion.json` **before** the sweep ran,
and the sweep got exactly one pass, as pre-committed.

Oracle reference: G(DEEP) − G(SKIM) on CREATOR content = **+1.754**.

| prior_strength | learner gap | ratio to oracle | (a) gap ≤ 2× oracle | GHOST engagement after convergence | (b) < 1.0 |
|---|---|---|---|---|---|
| 1.0 | 133.17 | 75.9 | ✗ | 0.00 | ✓ |
| 4.0 | 34.62 | 19.7 | ✗ | 0.00 | ✓ |
| 16.0 | 9.98 | 5.7 | ✗ | 2.40 | ✗ |
| 64.0 | 3.82 | 2.2 | ✗ | 2.70 | ✗ |
| 256.0 | 2.28 | 1.3 | ✓ | 2.50 | ✗ |
| 1024.0 | 1.89 | 1.1 | ✓ | 2.30 | ✗ |

**No value satisfies both criteria, and the two are structurally anti-correlated.** A prior
weak enough to keep the parameter info-gain term in scale is weak enough that accumulated
counts quickly dominate it; a prior strong enough to tame the term is strong enough that the
learner never updates away from believing GHOST content is CREATOR-like, so it never
disengages and E9's starvation arm ceases to exist. The failure is clean and it is a property
of pymdp's uncoefficiented parameter info-gain term, not of a badly chosen grid.

**Per the pre-committed one-pass limit, the term was switched off rather than retuned.**
`learning.use_param_info_gain = false` for every learner experiment.

Validated afterwards — explicitly a validation of the fallback, not a second calibration
pass: with the term off the learner's DEEP−SKIM gap is **1.765 against the oracle's 1.754**
(ratio 1.01), so criterion (a) is vacuous, and criterion (b) passes at `prior_strength = 1.0`
(GHOST engagement 0.90 < 1.0).

**What is lost.** C3 wanted the term so "a learner has a reason to engage with GHOST content
that an oracle does not." Under D1's seeding the learner still has that reason — it believes
GHOST is CREATOR-like until it learns otherwise — so the motivation survives; what is lost is
the specifically *epistemic* framing of it. Recorded as a deviation.

## E7 — Can the GHOST column be learned without labels?

Learner observers (D1 seeding, `use_param_info_gain=false` per the D2 fallback), biased
synthetic content, 600 artifacts × 8 observers × 3 replications, crossed with
`signing_rate ∈ {0, 0.5, 1.0}` × κ ∈ {0.1, 0.9} × f ∈ {0.3, 0.6}.
`figures/e7_learn_ghost.png`. Wall clock **3.2 min**.

Final state of the learned model (κ = 0.9):

| signing | f | GHOST-column error (nats) | human-column MI / oracle | time to competence |
|---|---|---|---|---|
| 0.0 | 0.3 | 3.612 | **0.820** | never (600) |
| 0.0 | 0.6 | 2.718 | **0.667** | never (600) |
| 0.5 | 0.3 | 1.465 | 0.906 | 592 |
| 0.5 | 0.6 | 0.513 | 0.883 | 288 |
| 1.0 | 0.3 | **0.036** | 0.935 | **100** (first checkpoint) |
| 1.0 | 0.6 | **0.029** | 0.905 | **100** (first checkpoint) |

**CONFIRMED, and this is the result that restores the strong claim.** Both halves of the
prediction hold, cleanly:

- *Without labels the learner folds synthetic features into its model of human intent.* The
  human columns lose goal-discriminability — MI falls to **0.667 of the oracle's** at f=0.6,
  and the GHOST column is never acquired (error 2.7–3.6 nats, competence never reached within
  600 artifacts).
- *With honest labels it acquires a clean GHOST column quickly and the human columns stay
  sharp.* GHOST-column error drops to **0.029 nats** — a **~100× reduction** — competence is
  reached by the first checkpoint, and human-column MI holds at 0.90–0.94.

V1 showed the provenance signal was metabolically useful to an aggregator that already knew
what goalless output looks like, and V1's own §0 called that a question answered before the
experiment began. E7 removes the answer key and asks whether you can come to know without it.
**You largely cannot.** An unlabelled learner does not merely fail to identify synthetic
content — it absorbs synthetic structure into its model of *human* intent, losing a third of
its goal-discriminability on genuine work. That is an epistemic loss, not a metabolic one,
and it is the strong claim V1 could not support.

**NOT REPLICATED — "degradation is worse at high κ".** κ is essentially inert without a
signal (GHOST-column error 3.68 at κ=0.1 vs 3.61 at κ=0.9 with `signing_rate=0`), exactly as
V1's null N5 requires — an observer cannot be affected by a channel that is silent. With
honest labels κ is strongly *protective* rather than harmful (error 1.67 at κ=0.1 vs 0.036 at
κ=0.9). The prediction appears to have imported E4's trust-exploit intuition, where high κ is
dangerous *because the signal can lie*; E7 uses honest signals, so trusting them is simply
correct. High κ is a liability only when honesty is not guaranteed.

---

## E9 — Poisoning versus starvation

Four arms over 400 artifacts × 8 observers × 3 replications × f ∈ {0, 0.3, 0.6}.
`figures/e9_poison_starve.png`. Wall clock **6.2 min**.

| arm | f | shape (KL from true A) | flatness (entropy) | human-column MI | DEEP per genuine artifact |
|---|---|---|---|---|---|
| control | 0.0 | 0.143 | 1.020 | 1.007 | 6.00 |
| control | 0.6 | 0.165 | 1.023 | 0.989 | 6.00 |
| poisoning only | 0.0 | 0.143 | 1.020 | 1.007 | 6.00 |
| poisoning only | 0.3 | 0.171 | 1.025 | 0.966 | 6.00 |
| poisoning only | 0.6 | **0.355** | 1.003 | 0.847 | 6.00 |
| starvation only | 0.0 | **0.243** | 1.036 | 0.882 | 1.87 |
| starvation only | 0.3 | 0.238 | 1.038 | 0.880 | 1.88 |
| starvation only | 0.6 | 0.238 | 1.017 | 0.934 | 1.79 |
| both | 0.6 | **0.593** | 0.990 | **0.520** | 1.75 |

**The two channels are cleanly separable — but not on the axis the spec predicted.**

*Poisoning* behaves exactly as described: it distorts **shape**, and does so as a function of
contamination (0.143 → 0.355, a 2.5× rise in per-column KL from f=0 to f=0.6), while leaving
flatness untouched (1.020 → 1.003).

*Starvation* does **not flatten**. Its entropy signature is indistinguishable from the
control's (1.036 vs 1.020, and both drift the same way with f). What it does instead is
**freeze the model at its prior**: engagement collapses to 1.87 DEEP steps of a possible 6,
so genuine content stops producing updates, and the learned model simply stays where it
started — shape error 0.243 and human-column MI 0.88, *flat in contamination* (0.243 / 0.238 /
0.238 across f = 0 / 0.3 / 0.6).

That flatness-in-f is the real starvation signature and it is arguably more diagnostic than
the predicted one: **starvation's damage is immediate and contamination-independent.** It is
already fully present at f = 0, because disengagement is driven by the *metabolic* trade-off
rather than by how much synthetic content is in the corpus.

**Why the predicted signature did not appear, and it traces to D1.** "Starvation flattens"
presumes an uninformative prior that stays uninformative for want of evidence. Under D1's
seeding the prior is *informative* (the shared goal→feature family), so disuse cannot flatten
it — it can only prevent it from being corrected. Disuse ossifies rather than blurs. The
diagnostic still separates the channels; the discriminating axis is
**shape-versus-contamination** (poisoning scales with f, starvation does not) rather than
shape-versus-flatness.

**NOT REPLICATED — "starvation is the larger effect at realistic contamination levels".**
The two cross over. At f = 0 starvation dominates decisively (0.243 vs 0.143). At f = 0.6
poisoning dominates (0.355 vs 0.238). The crossover sits near f ≈ 0.35. The predicted
reasoning — that disengagement is fast and total while fabrication needs a dishonest signal
present and trusted — is correct about *speed* but not about *ceiling*: starvation saturates
immediately and then stops getting worse, whereas poisoning keeps accumulating with
contamination. The two together are worse than either (0.593, with human-column MI halved to
0.520), and super-additive: 0.593 exceeds the sum of the individual increments over control.

## E8 — Recursive degradation: **NOT REPORTABLE (N11 failed)**

`G_max = 4`, f ∈ {0, 0.3, 0.6}, signal ∈ {absent, honest}, learner observers, 300 artifacts ×
5 observers × 3 replications. The experiment ran to completion in **14.3 min** and its CSVs
are in `results/e8_*.csv`. **Its results are not reported as findings, because N11 failed.**

V2 spec §3 states the condition and the consequence:

> **N11 — Zero-contamination recursion.** With `f = 0`, E8 must show **no** degradation across
> generations. *This is the most important new null.* If generational decay appears without any
> contamination, the recursion loop itself is lossy and every E8 result is an artifact of the
> implementation rather than a finding.

Per-generation KL slopes at **f = 0**:

| signal | slope (nats/generation) | t | significant |
|---|---|---|---|
| absent | +0.0003 | 0.09 | no |
| **honest** | **+0.0119** | **3.75** | **YES** |

**The loop leaks, specifically in the honest-signal arm.** With zero synthetic content and a
perfectly honest provenance signal, the payload still degrades from KL 0.042 to 0.075 across
four generations. There is nothing in the world for it to degrade *from*: every artifact is
CREATOR-provenance, produced by a real creator policy, and correctly labelled.

**The likely mechanism, stated as a hypothesis and not as a result.** An honest signal
concentrates the observer's provenance posterior on CREATOR, so Dirichlet updates land almost
entirely on the CREATOR column and it sharpens fast around a *finite-sample* estimate. That
estimate then seeds the next generation's creators, whose output is re-estimated, and the
error compounds as a random walk. Without a signal, updates smear across provenance columns
and the CREATOR column stays nearer its (correct) prior — which is why the unsigned arm looks
clean. If that is right, the honest signal is not causing damage; it is *removing the
regularisation that the unsigned arm accidentally enjoys*. Confirming that requires work
E8 has not done, and no claim is made here.

**What this costs.** E8's headline predictions — monotone degradation at f > 0, attenuated by
an honest signal, superlinear in generation — cannot be evaluated. The observed f > 0 numbers
are not obviously null (f=0.3, signal absent: KL slope +0.0137, t=2.27; regret slope +0.534,
t=3.28), and they run in the predicted direction. **They are not reported as support for the
hypothesis**, because a lossy loop produces drift in every arm and there is no way, from this
run, to separate real recursive degradation from the leak. The superlinearity prediction is
likewise untestable here: the quadratic terms are small and negative (−0.010 to +0.107)
across conditions, which would ordinarily argue *against* superlinearity, but that reading is
not available while the null is failing.

**What must happen before E8 can be reported.** The loop needs a fixed point at f = 0 under an
honest signal. The most promising directions, in order: (1) seed generation g+1 from the
*population-averaged* learned column rather than one observer's, which averages out the
per-observer estimation error before it can compound; (2) increase artifacts per generation
until the per-generation estimate is tight enough that the random walk is negligible, and
demonstrate the slope shrinking with sample size (which would confirm the finite-sample
diagnosis directly); (3) regularise the seeded creator back toward the prior by a fixed
amount. Each is a design change with its own justification burden, and none should be applied
without re-running N11 as the acceptance test.

The failing null is kept **visible in the test suite** as a strict xfail
(`tests/test_nulls_v2.py::test_N11_gate_on_full_scale_e8_output`) rather than deleted or
weakened, so that a future fix forces the marker off.

### A methodological finding worth keeping

The **first** version of N11 re-simulated a short chain at reduced scale and **passed** — in
both signal conditions, even after being strengthened to `G_max=4`. The full-scale run
failed. The compounding estimation error simply is not visible at 120 artifacts and 3
observers.

**A null checked at a smaller scale than the experiment it gates is not a gate.** The N11 test
was therefore rewritten to evaluate `results/e8_trends.csv` — the actual output of the actual
run — and the reduced-scale re-simulation was demoted to an explicitly-labelled smoke check.
Any other null in this suite that is only ever exercised at `--quick` scale carries the same
risk.

---

## E11 — Regret versus KL

Cross-cutting analysis over **E6b and E10** (n = 2350 recovered preference distributions).
`figures/e11_regret_vs_kl.png`. Wall clock **< 0.1 min**.

**E8 is excluded from the pool.** Spec §2 lists it as an input, but N11 failed, so pooling
E8's numbers would launder implementation artifacts into a cross-cutting result.

| quantity | value |
|---|---|
| Pearson(KL, regret) | **0.813** |
| Spearman(KL, regret) | 0.375 |
| R² — KL alone | 0.661 |
| R² — argmax preservation alone | 0.678 |
| R² — both | 0.818 |
| mean regret, argmax preserved | **1.56** |
| mean regret, argmax flipped | **13.81** |

**NOT CONFIRMED as stated.** The prediction had two parts and both fail:

- *"The relationship is weak."* It is not. Pearson correlation between KL and regret is
  **0.813** — KL alone explains 66% of the variance in harm.
- *"The argmax-flip boundary explains far more variance than KL magnitude."* It explains
  **slightly** more (0.678 vs 0.661) — a 2.6% relative margin, well inside the 10% threshold
  pre-set for "materially more". Reporting that split as a win would be reading noise.

**What is true, and it is the useful half of the prediction.** The argmax flip is a large,
clean *discontinuity*: mean regret is **1.56** when the argmax is preserved and **13.81** when
it is not — an **8.8× jump**. And the two predictors are strongly complementary: together they
explain 0.818, materially more than either alone, which means each carries signal the other
misses.

So the honest conclusion is narrower than "KL is the wrong harm metric": **KL is informative
about harm but not sufficient, and it systematically misses the argmax-flip cliff.** The
practical recommendation still stands and is unchanged — report regret alongside KL, and never
report KL alone — but the justification is complementarity rather than KL being uninformative.
E6b's own f=0.8 rows show the failure mode concretely: the G2 draw has *lower* KL than the G1
draw (0.102 vs 0.137) and *higher* regret (15.3 vs 10.0).

---

## Null conditions

| null | condition | required outcome | result |
|---|---|---|---|
| N1–N7 | V1 suite, unchanged | as V1 | **pass** (17 tests) |
| **N8** | `d_i = 0` for all observers | V2 reproduces V1 | **pass** — exact. `build_observer_model` short-circuits to the world model, the `sig_i` draw consumes no variate, and the D-prior stream is bit-identical at d ∈ {0, 0.3, 0.9} |
| **N9** | `goal_symmetric: true` | E6b reproduces V1's E6 | **pass** — 0.0826 at f=0.8 against V1's 0.066, and below its bound. Pre-registration tamper-locking also tested |
| **N10** | `lr_pA = 0` | learner ≡ fixed-`A` observer | **pass** — *exactly*: `A[0]` bit-identical after 20 updates, identical actions and posteriors to a V1 observer, and matches the true `A[0]` to 1.7e-16 |
| **N11** | `f = 0` recursion | **no** degradation | **FAIL** — significant at f=0 with an honest signal (t=3.75). E8 not reported. See above |
| **N12** | E10 corpus | no GHOST artifacts | **pass** — asserted in the worker and verified in the CSV |
| invariant | every `pA` update leaves `A` column-stochastic | — | **pass** — asserted unconditionally inside `learning.learn_step`, exercised across all four tiers |
| D1 evidence | uniform prior is unidentifiable | — | **pass** — kept as a live test so the deviation is revisited if it ever stops being true |

Suite: **38 passed, 1 xfail** (the N11 gate, deliberately visible).

---

## Deviations from spec

Each was decided **before** implementation except where noted, and each is recorded in
`DECISIONS_V2.md` with its evidence.

1. **D1 — the Learner's prior is uninformative over PROVENANCE, not over everything.**
   C3's literal reading is not implementable. Measured: with DEEP forced so disengagement is
   impossible, a uniform-`pA[0]` learner holds `MI(features;goal)` at exactly 0.0000 nats
   after 400 artifacts and all four learned goal columns stay bit-identical. It is an
   *identifiability* deadlock, not a disengagement one, so no forced-engagement warmup
   rescues it. Framed in the README as a theoretical commitment — observers share a
   likelihood family because they share a body plan — and it scopes E7's claim to "can you
   learn which sources are hollow", which is what §0's third diagnosis actually asks.

2. **D2 — `use_param_info_gain` switched OFF; calibration failed on its one permitted pass.**
   Criteria (a) and (b) are structurally anti-correlated; no swept `prior_strength` satisfies
   both. Full sweep table in the D2 section above. The one-pass limit was pre-committed and
   honoured — no retuning, no grid widening.

3. **D3a — the biased arm sweeps four synth draws stratified by favoured goal**, rather than
   the single default draw. This was not cosmetic: the default draw favours G3, the rarest
   goal, and would have produced an apparent confirmation (0.21, inside the predicted range)
   that the pooled result (0.124) does not support.

4. **D3b — a `1e-3` support floor on `noise_free_synth`.** The raw Dirichlet(0.03) draw has
   hard zeros; a zero mass makes those features impossible under GHOST, so observing one
   would be a *proof* of non-GHOST provenance — a perfect provenance channel bypassing the
   Ghost Scale, which would have biased *against* the C2 effect. It also makes E9's
   per-column KL infinite. A no-op for V1's symmetrized draw (min mass 0.007), so N8/N9 are
   unaffected.

5. **D4 — regret and argmax computed in closed form; sycophancy uses a real pymdp agent.**
   For a bandit-shaped task a utility-only agent's policy posterior *is* `softmax(γ·C)` — the
   identity `creators.HumanCreator` already relies on — so the closed form is a computation,
   not a shortcut. Sycophancy needs genuine inference from an unreliable signal and gets a
   real agent.

6. **D5 — generation-g+1 creators keep the `HumanCreator` POMDP**, with
   `C = log(A_learned[:, CREATOR, goal, DEEP])`. Preserves §4.2's load-bearing property that
   human artifacts come from a reward-optimising policy. Verified as an exact fixed point
   given a perfect model (`test_N11_seeded_creator_is_lossless_given_a_perfect_model`).

7. **Creator goals are allocated by largest remainder, not sampled i.i.d.** *(discovered
   during implementation, not pre-decided.)* V1's E6 sampled creator goals i.i.d. and took
   `C_true` to be the realized empirical distribution; that vector can contain a **zero** (35%
   of the time at 10 creators, ~0.5% at 50), and KL against zero support diverges. E6b first
   reported KL ≈ 5.1 nats in *both* arms — above its own pre-registered bound, which the
   prereg file names as the signature of an accumulation bug. Largest-remainder allocation
   gives full support by construction and makes the realized `C_true` equal the population
   distribution the bound is defined against. This is a fix to an estimator instability
   inherited from V1, not a change of subject.

8. **E7's time-to-competence is defined on the GHOST column, not human-column MI.** Under D1
   seeding the learner starts at ~95% of oracle MI, so an MI-based clock reads zero in every
   condition and would measure the seeding rather than the learning.

9. **`infer_policies` is skipped on forced steps when EFE is not being recorded.** Its output
   is discarded there, so this is exact rather than an approximation — verified by
   bit-identical E6 CSVs — and it cut the E6 run from 24:20 to 3:52 (**6.3×**). Recorded
   because it changes the cost model in §4, not the results.

10. **Spec §4's cost lever 1 (corpus caching) was measured and NOT implemented.** In this
    implementation an `Artifact` is metadata only — features are sampled during the rollout —
    so `draw_corpus(200)` costs **0.9 ms** against **5091 ms** of rollouts for the same 200
    artifacts, a ratio of 5544×. Caching it would have delivered nothing. The lever that
    actually mattered is deviation 9.

11. **E11 excludes E8** because N11 failed. Spec §2 lists E8 as an input; pooling artifacts
    into a cross-cutting analysis would launder them.

12. **E6b retains the full V1 κ × signing grid** rather than invoking cost lever 2, since
    deviation 9 bought the headroom to honour "identical to V1 E6 except `goal_symmetric`"
    literally.

---

## V1 reconciliation

**What V2 overturns:** nothing in V1's reported results. No V1 finding is contradicted.

**What V2 refines:**

- *The V1 post-mortem's diagnosis of E6 is only partially supported.* The post-mortem held
  that symmetrizing the synthetic distribution turned contamination into noise and was
  therefore *what suppressed* the V1 effect. Removing symmetrization does increase corruption
  monotonically at every contamination level — but by a stable **~1.5×**, not by the order of
  magnitude that framing implies (0.124 vs 0.083 at f=0.8). The bias axis is real and it is
  not the whole story. E6b's pre-registered point prediction was **not met** at the pooled
  level, though the falsification threshold was not crossed either.
- *Corruption scales with which goal the synthetic content happens to favour* — `corr(KL,
  bound) = 0.873` across four lean-matched draws, a 4.6× spread from the commonest to the
  rarest goal. This is new, and it means single-draw results in this family are not
  trustworthy.

**What stands, and is strengthened:**

- *Provenance-aware aggregation protects the alignment payload.* Under the harder biased
  condition the provenance-weighted aggregator stays at KL 0.008 against the naive
  aggregator's 0.124 at f=0.8 — a **15× protection factor**. This survived the change designed
  to break it.
- *V1's E6 finding — that the provenance signal is metabolically rather than epistemically
  valuable **for an oracle aggregator** — is not overturned by V2. It is scoped.* Those are
  the words §5 asks for, and E7 is what scopes it: remove the answer key and the signal
  becomes **epistemically** load-bearing. An unlabelled learner never acquires a clean GHOST
  column (error 2.7–3.6 nats, competence never reached) and folds synthetic structure into its
  model of human intent, losing a third of its goal-discriminability on genuine work
  (MI ratio 0.667). With honest labels the column is acquired almost immediately (0.029 nats)
  and the human columns stay sharp (0.94). The signal's value is metabolic *only* to someone
  who already knows what goalless output looks like.
- *V1's E2 hallucination signature reappears one level up.* E10 finds that inexpertise
  produces **confident error, not uncertainty**: `C_recovered` from high-`d` observers is
  sharp and wrong (KL 0.151, argmax flipped in over half the populations) rather than flat —
  which is why the predicted sycophancy rise did not materialise. That is the same
  low-within-entropy / high-between-error dissociation V1 reported for goal inference, now
  visible in the recovered preference distribution.

**The strongest new standalone claim** is E10: on a corpus whose quality is held perfectly
constant, with zero synthetic content, extracted intent density falls with the reader's
inexpertise (t = −34.2) and downstream regret rises 9×. The extractor's competence is a hard
ceiling on recoverable intent. With the caveat that the gradient has a **knee** near d ≈ 0.5
rather than being linear — which sharpens the RLHF reading into a threshold claim rather than
a "more expertise is always better" one.

---

## Compute

Required by spec §4 so the next iteration can be planned. Machine: 12 logical cores,
10 worker processes.

| stage | experiment | wall clock | notes |
|---|---|---|---|
| 1 | E6b (+ pre-registration) | **24.7 min** | full V1 κ×signing grid × 4 draws + control; 2000 artifacts |
| 2 | E10 | **5.1 min** | 10 `d` levels × 20 observers × 5 reps |
| 3 | D2 calibration | **0.2 min** | one pass, failed, fallback applied |
| 4 | E7 | **3.2 min** | |
| 5 | E9 | **6.2 min** | |
| 6 | E8 | **14.3 min** | ran to completion; results not reportable (N11) |
| 7 | E11 | **< 0.1 min** | post-processing |
| — | test suite | ~9 min | 38 passed, 1 xfail |
| | **total** | **~54 min** | |

Against spec §4's ~16 h estimate. Two reasons for the gap: deviation 9's exact 6.3× speedup,
and the fact that the spec's own cost model (lever 1, corpus caching) mis-identified the
bottleneck — the cost is entirely in policy and state inference, not artifact generation.

**Implication for the next iteration:** compute is not the binding constraint it was assumed
to be. There is ample headroom to run E8 at the much larger per-generation sample sizes that
the N11 diagnosis calls for, and to raise E7/E9 observer counts. The binding constraints in
this round were design questions (D1, D2, C5) and one failed null, not wall clock.
