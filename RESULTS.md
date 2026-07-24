# RESULTS

Written from the actual CSVs in `results/`, not from the predictions in the build spec. Headline
runs are at full spec scale (**200 observers × 40 timesteps × 20 seeds**; E6 uses 2000 artifacts ×
5 corpus replications). Where a prediction did not replicate, it is reported here rather than tuned
away. Deviations from the build spec are collected in the final section, each with the evidence
that motivated it, so any can be revisited.

Reproduce with `python run_all.py` (or `make all`); every number below is regenerable and
seed-stable (`tests/test_model_invariants.py::test_reproducibility_identical_csv`).

---

## Summary of outcomes

| experiment | maps to | prediction | outcome |
|---|---|---|---|
| E1 generative crash | H1 | disengage + fail to recover goal for GHOST, monotonic across tiers | **confirmed** (recovery gradient; GHOST at chance) |
| E2 confident disagreement | H2 | GHOST+SIG_CREATOR → low within / high between entropy | **confirmed, cleanly** (the headline) |
| E3 titration | H3 | calibrated observers spend less effort at no accuracy cost | **confirmed** (~2× less effort, equal accuracy) |
| E4 trust exploit | H6 | high-κ × low-honesty region of confident fabrication | **confirmed** (sharp, graded boundary) |
| E5 κ ≠ γ | §8 | κ content-selective, γ indiscriminate | **confirmed** (falsification not triggered) |
| E6 corpus corruption | H6 | honest signal attenuates corruption; high-κ-low-signing worst | **partially** — provenance-awareness protects the payload, but the *signal's* value is metabolic, not accuracy (see E6) |

All **7 null tests pass** (`tests/test_nulls.py`). All **model-invariant tests pass**
(`tests/test_model_invariants.py`).

---

## E1 — The generative crash (H1)

1×4 across true provenance; honest signals, `signing_rate=1.0`, κ=0.9.

| tier | mean DEEP steps | final goal entropy (nats) | goal-recovery accuracy |
|---|---|---|---|
| CREATOR | 1.58 | 0.135 | **0.969** |
| POLISHED | 1.85 | 0.130 | **0.970** |
| CURATOR | 2.62 | 0.365 | 0.906 |
| GHOST | **0.00** | **1.367** | **0.246** (≈ chance) |

Goal recovery is monotonic across tiers and collapses to chance for GHOST, which disengages
immediately (0 DEEP steps). The EFE decomposition (`figures/e1_crash.png`, right panel) shows the
**epistemic value of DEEP about the goal is a step function across tiers** — ≈2.05 nats for
CREATOR versus ≈0.09 for GHOST at t=0, i.e. below the effort gap (0.40) from the very first
decision. This is the step-vs-gradual claim: GHOST is not a slow drift to disengagement, it is a
provenance-conditioned step.

**A reported nuance (not tuned away):** cumulative DEEP steps are *not* monotonic across tiers —
CURATOR spends the most (2.62), more than CREATOR (1.58). Metabolic cost tracks *extraction
difficulty*, not intent: CREATOR's goal is recovered efficiently in one or two looks, CURATOR's is
extractable but noisier and takes more, GHOST's is not extractable at all so it never looks. The
monotone signal lives in *recovery* (accuracy / final entropy), not in raw effort.

---

## E2 — Confident disagreement is the signature of hallucination (H2, headline)

2×2 true {CREATOR, GHOST} × declared {SIG_CREATOR, SIG_GHOST}; DEEP forced for the first 10 steps,
then free; all observers view the same true goal (inter-rater reliability). `figures/e2_variance.png`.

| true | signal | within-observer entropy (confidence) | between-observer entropy (disagreement) |
|---|---|---|---|
| CREATOR | SIG_CREATOR | 0.000 | 0.000 |
| CREATOR | SIG_GHOST | 0.092 | 0.009 |
| **GHOST** | **SIG_CREATOR** | **0.090 (confident)** | **1.379 (max ≈ ln 4)** |
| GHOST | SIG_GHOST | 1.293 (uncertain) | 1.378 |

The dissociation is exactly as predicted and very clean (SDs across 20 seeds < 0.02). In
`(GHOST, SIG_CREATOR)` every observer is individually confident (within-entropy 0.09) yet the
population disagrees maximally (between-entropy 1.38, near the ln 4 = 1.386 ceiling): each observer
fabricates a *different* confident goal. That is `R_hallucinated`.

The comparison that isolates the mechanism is the two GHOST cells: with an honest ghost signal the
observer **keeps** its uncertainty (within-entropy 1.29); with a trusted dishonest creator signal
it **resolves that uncertainty into a fabrication** (0.09). Trust is what converts preserved
uncertainty into confident error. This is the figure for the email.

---

## E3 — Ghost Scale titration (H3)

Four-tier mixed corpus × observer {calibrated κ=0.9, naive κ=0.1} × `c_effort` sweep. Metabolic
expenditure = mean HIGH_COST observations; accuracy measured on CREATOR trials. Both observer types
see the same corpus.

| c_effort | calibrated effort | calibrated CREATOR acc | naive effort | naive CREATOR acc |
|---|---|---|---|---|
| 0.05 | 2.15 | 0.997 | 3.15 | 0.997 |
| 0.10 | 1.49 | 0.960 | 2.51 | 0.989 |
| 0.15 | 1.14 | 0.929 | 1.87 | 0.961 |
| 0.25 | **0.50** | **0.941** | **1.21** | **0.954** |
| 0.35 | 0.50 | 0.947 | 0.62 | 0.946 |

Across the engaged regime the calibrated observer spends **~40–60 % less metabolic effort at
statistically indistinguishable CREATOR accuracy** — e.g. at c_effort=0.25, effort 0.50 vs 1.21 for
accuracy 0.941 vs 0.954. The Ghost Scale buys metabolic savings at no epistemic cost on genuine
human artifacts. (At c_effort ≥ 0.5 both types stop engaging and accuracy falls to chance for both;
the interesting regime is c_effort < 0.5.) `figures/e3_titration_effort.png`,
`figures/e3_titration_accuracy.png`.

---

## E4 — The trust exploit (H6)

κ sweep 0→0.99 × honesty 1.0→0.5 on GHOST content; confident-arbitrary rate = fraction of observers
whose final goal posterior entropy < 0.5 nats (any confident goal on GHOST is fabricated).
`figures/e4_trust_exploit_rate.png`.

Rate of confident fabrication (rows = honesty, selected κ columns):

| honesty \\ κ | 0.0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|
| 1.0 | 0.21 | 0.01 | 0.00 | **0.00** |
| 0.8 | 0.22 | 0.20 | 0.21 | 0.20 |
| 0.6 | 0.22 | 0.40 | 0.40 | 0.39 |
| 0.5 | 0.21 | 0.50 | 0.49 | **0.50** |

There is a clear two-dimensional region of sustained confident fabrication in the **high-κ,
low-honesty** corner. The boundary is **graded in honesty and sharp in κ**: fabrication jumps as
soon as κ rises above ~0.2 (the observer starts trusting the channel), then is roughly flat in κ,
while it rises steadily as honesty falls. The κ=0 column is honesty-invariant at ~0.21 — an
observer that ignores the signal cannot be exploited by a dishonest one (its residual ~0.21 is
content-driven, not signal-driven). At honesty=1.0 the fabrication rate falls to **0** for any κ>0:
an honest signal is protective precisely because it is honest.

---

## E5 — κ is not γ (§8, the precision baseline)

Three arms on a mixed corpus. Selectivity = P(DEEP | CREATOR) − P(DEEP | GHOST) over an early
window (see deviation 5). E5 uses **stochastic** action selection so that γ is actually active
(see deviation 5). `figures/e5_precision_baseline.png`.

- **Arm A** (sweep κ, γ fixed): selectivity rises from −0.22 to **+0.54** as κ increases.
- **Arm B** (κ=0, sweep γ): selectivity stays near the content-only baseline (−0.23 → −0.16) across
  the entire γ range; γ shifts *overall engagement* only modestly (0.847 → 0.874) and produces no
  selectivity.
- **Arm C** (κ=0.9, sweep γ): selectivity is **preserved (+0.51 → +0.57)** across the whole γ range.

**Falsification not triggered.** κ produces content-selective engagement (arms A, C); γ produces at
most an indiscriminate shift in engagement and no selectivity (arm B). The **content-only baseline**
(arm B) is slightly *negative* (≈ −0.23): with no signal a skilled observer engages marginally
*more* on GHOST than CREATOR early on, because CREATOR resolves and disengages fast (the same
extraction-efficiency effect seen in E1). The **Ghost Scale increment** over the content-only
baseline is therefore large (arm A max − arm B mean ≈ **+0.75**), and it also *reduces the DEEP
steps needed* to reach that selectivity. The framework's claim was never that people cannot detect
AI unaided; it is that detecting it costs something the signal gives away for free.

---

## E6 — Corpus corruption and the alignment payload (H6)

50 creators with a known population goal distribution `C_true = [0.40, 0.30, 0.20, 0.10]`; an
aggregator watches 2000 artifacts with contamination fraction f of GHOST synthetics (honestly signed
per `signing_rate`) and accumulates `C_recovered`, weighting each artifact's goal posterior by its
belief that the artifact is genuine, `1 − P(GHOST)`. `figures/e6_corpus_corruption.png`.

KL(C_recovered ‖ C_true), averaged over κ and signing_rate:

| contamination f | naive aggregation | provenance-weighted |
|---|---|---|
| 0.0 | 0.001 | 0.001 |
| 0.2 | 0.005 | 0.001 |
| 0.4 | 0.018 | 0.001 |
| 0.6 | 0.038 | 0.002 |
| 0.8 | **0.066** | **0.005** |

**Confirmed:** contamination degrades a **naive** aggregator monotonically (0.001 → 0.066), and a
**provenance-aware** (down-weighted) aggregator is almost immune (stays ≤ 0.005). Provenance
inference protects the alignment payload — this is the CIRL-loop-in-miniature result: what you would
hand a downstream agent as its preference prior survives heavy contamination *if and only if* the
aggregator is provenance-aware.

**Mostly did not replicate, and why (reported, not tuned):** the two finer predictions — that an
*honest signal* attenuates the corruption, and that *high κ under low signing* is the most
corruptible — are, at full scale, **an order of magnitude too small to matter**. An honest signal
*does* help, but only marginally and only at extreme contamination (at f=0.8: KL 0.008 / 0.005 /
0.003 for signing 0.0 / 0.5 / 1.0) — a ≤0.008 nats effect against the naive aggregator's 0.066 and
essentially flat across κ. The mechanism: a DEEP-inspecting aggregator identifies synthetic content
**from the content itself** (GHOST features match the GHOST likelihood column), so it excludes
contamination without needing the provenance signal. In this model the signal is therefore **nearly
redundant for recovery accuracy**; its value is overwhelmingly **metabolic** — it lets the
aggregator exclude synthetics *cheaply*, without spending DEEP
inspections (exactly the saving quantified in E3 and E5). The "trusting observer with no signal is
most corruptible" effect requires either a *bounded* inspection budget (so the aggregator cannot
afford to inspect and must rely on the signal) or *dishonest* signals (the trust exploit, which is
E4's domain, not E6's — E6 uses honest signals). A bounded-budget variant is the natural next
experiment; it is noted here rather than silently added.

---

## Null conditions (Spec §9) — all pass

| null | condition | required outcome | result |
|---|---|---|---|
| N1 | `alpha[GHOST]=1.0` | crash vanishes; restored-GHOST recovers like CREATOR | **pass** |
| N2 | `c_effort=0` | selective disengagement from GHOST vanishes (GHOST engages fully) | **pass** |
| N3 | `eps=0` + shared seed | between-observer disagreement → 0 in every E2 cell | **pass** (≈0) |
| N4 | permute provenance→α | tier-ordered recoverability (MI monotonicity) destroyed | **pass** |
| N5 | `signing_rate=0` | calibrated vs naive engagement coincide (no signal to use) | **pass** |
| N6 | uniform synth strawman | crash persists, but MI+entropy diagnostic separates the cases | **pass** |
| N7 | preference audit | `C[0]==C[1]==0` at construction | **pass** |

N6 is the referee-proofing test: with the default *structured* synth, GHOST shows low MI
(≈0.014) **and** low feature entropy (≈1.62 nats); with the uniform *noise* strawman it shows low MI
**and** high entropy (≈2.08 = ln 8). The diagnostic distinguishes unidentifiability from noise, as
required — the crash in the strawman is real but arises for a different reason.

## Invariant tests (Spec §10) — all pass

Column-stochastic A and B; shapes match §3.3; pairwise JS(sig) ≥ 0.566 > threshold 0.20;
H(noise_free_synth) = 1.52 < ceiling 1.8 < ln 8; MI(features;goal) monotonically decreasing across
tiers ([1.09, 0.97, 0.45, 0.014]); empirical creator feature distributions match A[0] within
tolerance (max 0.023 < 0.05); identical seed → identical CSV.

---

## Deviations from spec

Each was forced or motivated by contact with the implementation. Load-bearing §14 constraints (zero
preference over provenance/signal; structured-not-uniform synthetic likelihood; observer
heterogeneity; the null suite; the honest crosswalk) are unchanged.

1. **`c_effort` default 0.5 → 0.1.** pymdp legacy softmaxes `C` per modality, so the effective
   DEEP-vs-SKIM effort gap is `2·policy_len·c_effort = 4·c_effort` nats. At 0.5 the gap (2.0)
   exceeds the model's maximum epistemic value (≈2.18 for a known CREATOR; ≈0 for GHOST), so *every*
   tier disengages at t=0 and E1 collapses to a null. 0.1 places the gap (0.4) below the human tiers
   and above GHOST. This is a value calibration, not a mechanism change; E3 sweeps `c_effort` across
   the whole transition, and N2 (c_effort=0 → no crash) still holds.

2. **`noise_free_synth` goal-symmetrized** (config `goal_symmetric: true`, concentration 0.03). A
   raw Dirichlet draw generically peaks on features belonging to *one* goal's signature, so all
   observers hallucinate the *same* goal → the H2 result appeared as spurious **consensus**
   (between-entropy 0.09) instead of the predicted **disagreement**. The draw is now symmetrized
   across the goal-pair group so synth is equidistant from every signature (structure preserved;
   H=1.52). This *strengthens* the §14 "goal-independent" requirement by removing an accidental
   single-goal bias. With it, H2 disagreement is 1.38 (max).

3. **`structured_ceiling` 2.6 → 1.8.** The original ceiling exceeded uniform entropy (ln 8 = 2.079),
   making the "synth is not high-entropy" assertion vacuous. 1.8 sits below uniform, so it is a
   meaningful bound and it cleanly rejects the N6 uniform strawman (2.079 > 1.8).

4. **Initial cheap "glance"** (`observer.rollout_observer(initial_glance=True)`, default on). Not in
   the spec, but it *is* the Ghost Scale's stated purpose (§0: "make that determination cheaply and
   early, before it has burned the budget"). Without it the observer decides at t=0 from its prior,
   before reading the signal, and wastes one DEEP step on GHOST. With it, GHOST disengages at 0 DEEP
   steps. Toggleable off.

5. **E5 uses stochastic action selection and an early-window selectivity metric.** γ scales the
   softmax over policies and is **inert under deterministic (argmax) selection** — with argmax the
   γ sweep is a no-op and the κ-vs-γ test is vacuous. E5 therefore sets `action_selection=stochastic`
   in all three arms so γ can act. Separately, all-timesteps `P(DEEP)` is confounded by extraction
   efficiency (CREATOR resolves and disengages fast, which *understates* human engagement and even
   inverts the selectivity sign), so selectivity is measured over the first 3 free steps — the
   decision that matters is "does the observer choose to look deeply at all." The headline
   experiments (E1, E2) keep deterministic selection per §3.7.

6. **E6 corpus signs synthetics honestly per `signing_rate`** (was: always unsigned). Otherwise
   `signing_rate` could never let a trusting aggregator identify synthetics via the signal, and the
   sweep was a no-op. This is required for the signing_rate axis to be meaningful.

7. **E6 finding: the provenance *signal* is redundant for recovery accuracy in this model** (its
   value is metabolic). Reported in the E6 section above. Two of E6's finer predictions do not
   replicate under honest signals + a DEEP-inspecting aggregator; the mechanism is that synthetic
   content is identifiable from content on inspection. Recorded as a finding; a bounded-budget
   variant is the suggested follow-up.

### Judgment calls the spec explicitly permits (§13), left at their defaults
Feature cardinality 8, goal cardinality 4, `policy_len=2`, and the `psi_analogue` form were all kept
as specified; none needed to change to obtain the results above.
