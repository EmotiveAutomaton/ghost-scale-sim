# Ghost Scale Simulation — V3 Specification

**Author of spec:** Abraham Haskins, PhD
**Target:** an autonomous coding agent, extending the existing `ghost-scale-sim` repository
**Depends on:** V1 spec, V2 spec, `RESULTS_V2.md`, `DECISIONS_V2.md`, and the existing codebase.
This is a **delta focused on one broken experiment (E8) and one unexplained effect (the E9
freeze)**. Everything in V1 §14 and V2 §6 (load-bearing constraints) still holds unchanged.

---

## 0. Scope

V2 produced a working model and several confirmed results. One experiment could not be
reported and one effect was left unexplained. V3 addresses exactly those two things, and the
working hypothesis is that **they are the same thing**.

1. **E8 (recursive generational degradation) failed its acceptance null (N11).** With zero
   contamination and an honest provenance signal, the recovered payload still degraded across
   generations (slope +0.0119 nats/gen, t = 3.75). A lossy loop produces drift in every arm,
   so real recursive degradation cannot be separated from the leak, and E8's findings were
   correctly withheld.

2. **E9 produced an unpredicted "freeze."** Under disuse (starvation arm), the learned model
   did not blur toward flatness as predicted; it **ossified at its finite-sample estimate**
   (shape error frozen at ~0.24, contamination-independent across f = 0 / 0.3 / 0.6).

**The V3 hypothesis, stated up front so the whole build is organized around testing it:**
both are symptoms of a single mathematical effect — *premature convergence of a Dirichlet
estimate that stops receiving corrective evidence*. In E8 the honest-signal arm concentrates
updates on the CREATOR column, which sharpens fast around a finite-sample estimate that then
seeds the next generation and compounds as a random walk (the West 1992 / Al Labadi–Zarepour
Brownian-bridge regime for re-estimated Dirichlet parameters). In E9 the same premature
convergence locks the decoder in place once genuine content stops producing updates. If this
is right, the fix for E8 and the explanation for E9 are one result. V3 is built to confirm or
refute that in a single pass.

**Author's theoretical commitment, which the build must respect (D4).** Real human decoders
do **not** freeze — the world supplies too much corrective signal for a decoder to ossify. The
primary real-world degradation channel is **value divergence**: as an observer's model of what
others want drifts, and because decoding skill is entangled with that value model, decoding
ability drifts *with it, through the same mechanism* — encoder divergence is a **secondary
consequence** of value divergence, not an independent frozen state. It follows that a
*correctly specified* model should not exhibit a standalone freeze; the E9 freeze is therefore
predicted to be an artifact of the same finite-sample lossiness that breaks E8. V3 tests this
prediction rather than assuming it. Confirming it **validates the theory by earning the
conclusion**, instead of asserting it.

---

## 1. The three changes

### C1 — Lossless baton-passing via population averaging (V2 decision-carryforward D1: A + B)

The V2 leak entered because generation `g+1` creators were seeded from **one observer's**
finite-sample learned column. Per-observer estimation error then compounded generation over
generation as a random walk.

**Change.** Seed generation `g+1` creators from the **population-averaged** learned column, not
any single observer's:

```
A_seed[:, CREATOR, goal, DEEP] = mean_i( A_learned_i[:, CREATOR, goal, DEEP] )
```

Averaging across the generation's observers cancels the zero-mean per-observer estimation
error before it can compound. This is the standard variance-reduction move and it is the
primary fix.

**Preserve the D5 property.** Creators remain real `HumanCreator` POMDP agents, with
`C = log(A_seed[:, CREATOR, goal, DEEP])`. The averaging changes *what* column seeds the
creator, not *that a reward-optimizing policy produces the artifact*. V2's §4.2 load-bearing
property (human artifacts come from a policy, never sampled from a distribution) is unchanged.
Option B from V2 D5 (marginalizing over `C_recovered`) remains explicitly forbidden.

### C2 — Sample-size sweep proving the leak is finite-sample (V2 D1: B; V2 D2: B)

Averaging alone is a patch. To show it is the *correct* patch, the finite-sample diagnosis
must be demonstrated directly: **the zero-contamination leak must shrink toward zero as
per-generation sample size grows.**

Add a standalone experiment, **E12**, run *before* the real recursive experiment:

- Sweep `artifacts_per_generation ∈ {100, 300, 1000, 3000, 10000}`.
- At each sample size, run the recursion at **f = 0 with an honest signal** (the exact
  condition that failed N11 in V2).
- Measure the per-generation KL slope of the recovered payload.
- Run this both **with** C1 averaging and **without** it (single-observer seeding, the V2
  behavior), so the plot shows two curves.

**Predicted:** without averaging, the slope decreases roughly as `1/N` (the random-walk step
variance shrinks with sample size) but is nonzero at every finite N. With averaging over `M`
observers, the slope is an additional factor of `~1/M` smaller at every N and reaches
statistical insignificance at a modest N. The crossing of the "insignificant" threshold
defines the **minimum viable per-generation sample size** for E8.

**Falsification of the diagnosis:** if the without-averaging slope does **not** shrink with N
— if it is flat or grows — the leak is not finite-sample estimation error and the entire V3
diagnosis is wrong. Report this loudly if it occurs; it would mean the loop is lossy for a
structural reason C1 does not address, and E8 must not be run until that reason is found.

`E12` writes `results/e12_leak_vs_samplesize.csv` and
`figures/e12_leak_convergence.png`, and its output **sets the sample size** used by the
re-run N11 and E8. Wire this dependency explicitly; do not hardcode E8's sample size.

### C3 — Two-channel degradation tracking (D4)

V2's E8 tracked only value drift (`KL(C_recovered ‖ C_true)`). V3 tracks **both** channels the
author's theory predicts, and their relationship:

- **Value divergence (primary):** `KL(C_recovered ‖ C_true)` per generation. Predicted to be
  the dominant effect.
- **Encoder divergence (secondary):** the observer's intent-*decoding* ability per generation,
  measured as `MI(features; goal)` on a held-out set of **known-genuine, uncontaminated**
  probe artifacts — i.e. can this generation's observer still read intent from clean human
  work? Predicted to degrade *as a consequence of* value divergence, through the shared model,
  and therefore to **lag and track** the value-divergence channel rather than lead it.

**The specific relationship to test.** The author's claim is that encoder divergence is not
independent but is downstream of value divergence through a shared representation. Operationalize
as: **per generation, encoder divergence should be predictable from value divergence** (regress
the decoding-ability drop on the payload KL across generations and conditions; a strong
generation-wise relationship supports the shared-mechanism claim; independence between the two
channels would refute it). Report the regression, not just the two time series.

**No freeze channel is instrumented as a *primary* effect**, per D4 — the author's position is
that a correctly specified model will not freeze. But see C4: the freeze is not ignored, it is
made a diagnostic target.

### C4 — Instrument the freeze as a diagnostic, not a phenomenon

The E9 freeze is treated as a **symptom to be traced to its cause**, and the specific cause
under test is that it is the *same* finite-sample premature-convergence effect that C2
diagnoses in E8.

Add a diagnostic, **E13**, that connects the two directly:

- Re-run the E9 starvation arm, but track, per observation, the **effective sample count** the
  learned column has accumulated (the total Dirichlet concentration mass minus prior) and the
  **rate of change** of the learned column.
- Simultaneously track the same two quantities in E12's f = 0 honest-signal recursion.
- **The test:** the freeze in E9 and the leak in E8 should share a signature — the learned
  column's rate of change collapses once effective sample count crosses a threshold, and the
  *residual* estimation error at that threshold should match between the two experiments (up to
  the known scaling with sample size). If the E9 freeze error and the E8 per-generation step
  error, corrected for sample size, fall on the same curve, they are the same effect.

**Three outcomes, all reportable:**

1. **Same effect (predicted).** The freeze and the leak fall on one finite-sample curve. C1
   averaging fixes E8, and the E9 freeze is confirmed as an artifact of premature convergence
   — which is exactly what the author's theory predicts a correctly specified model's freeze
   *would* turn out to be. The theory is validated by the diagnosis rather than by assertion.
2. **Different effects.** The E9 freeze does not track the E8 leak's sample-size scaling. Then
   there is a second, distinct mathematical effect the framework does not currently predict, and
   it is flagged as an open problem rather than explained away. This is the author's own stated
   possibility ("some other unknown mathematical effect we need to understand") and it must not
   be suppressed if the data show it.
3. **Freeze vanishes under correct specification.** If, once C1 and adequate sample size are in
   place, the starvation arm no longer freezes at all, that is the cleanest possible
   confirmation of D4: the freeze was never real, only under-sampled. Report the disappearance.

`E13` writes `results/e13_freeze_leak_signature.csv` and
`figures/e13_shared_signature.png`.

---

## 2. Experiments

E1–E7, E9, E10, E11 are unchanged from V2. E8 is re-run under the fix. E12 and E13 are new and
run **first**, because E8 is gated on their output.

### Run order (E8 is gated, do not reorder)

| stage | experiment | purpose | gates |
|---|---|---|---|
| 1 | **E12** | leak-vs-sample-size; sets E8's sample size | must show `1/N` shrink or V3 diagnosis is refuted |
| 2 | **E13** | freeze/leak shared-signature diagnostic | classifies the E9 freeze (outcomes 1/2/3) |
| 3 | **N11 re-run** | acceptance gate for the loop, at E12-determined scale | must pass at full E8 scale |
| 4 | **E8** | the recursive result, finally reportable | only if N11 passes |
| 5 | **E11 re-pool** | fold E8 back into the harm analysis | only if E8 is reportable |

### E8 — Recursive degradation (re-run under C1)

Design as V2 §E8 (`f ∈ {0, 0.3, 0.6}`, signal ∈ {absent, honest}, learner observers) with
three changes:

- **C1 population-averaged seeding.**
- **Per-generation sample size** set by E12's convergence threshold (expected substantially
  larger than V2's 300; compute is cheap, see §4).
- **`G_max` raised to 8** (V2 used 4). Six-plus generations give the compounding room to show
  its shape; four could not establish a trend. Rationale: the author's prediction is
  *superlinear* degradation, which needs enough generations to distinguish a curve from a line.
- **Two-channel tracking (C3):** value divergence and encoder divergence, plus the
  generation-wise regression between them.

**Predicted, and now testable because the loop is clean:**

- At **f = 0**, both channels flat across all 8 generations (this is the re-run N11, and it
  must hold at full scale, not merely at reduced scale — the V2 methodological lesson).
- At **f > 0**, value divergence rises monotonically, attenuated by an honest signal,
  **superlinear** in generation (generation 6's damage exceeds 6× generation 1's, because a
  degraded reader produces degraded artifacts that further degrade the next reader).
- Encoder divergence **tracks value divergence** across generations (the C3 regression), rather
  than leading it or moving independently.

### E12, E13 — as specified in §1 (C2, C4).

---

## 3. Nulls and invariants — additions

Existing N1–N12 must still pass unchanged, **except N11, which V3 repairs.**

- **N11 (repaired).** Re-defined to run at the **E12-determined full E8 sample size** and to
  require the f = 0 honest-signal slope to be statistically insignificant *at that scale*. The
  V2 methodological finding is now a spec rule: **a null must be evaluated at the scale of the
  experiment it gates.** The reduced-scale re-simulation remains only as an explicitly labeled
  smoke check. The xfail marker on the V2 N11 gate must be **removed by this fix** — if it is
  still xfailing, E8 stays unreported.

- **N13 — Convergence monotonicity (new).** In E12, the without-averaging leak slope must be
  monotonically non-increasing in sample size across the swept range. *Guards the finite-sample
  diagnosis itself:* if the leak does not shrink with data, C1 is not the right fix and the
  whole V3 story is wrong. A failure here blocks E8.

- **N14 — Averaging null (new).** With a single observer per generation (`M = 1`), C1 averaging
  must reduce **exactly** to the V2 single-observer seeding, reproducing the V2 leak. *Proves
  the fix is the averaging over `M > 1` and not an incidental change to the seeding code.*

- **N15 — Probe-set purity (new).** E8/E13's encoder-divergence probe set must contain **zero**
  contaminated or synthetic artifacts, asserted in the worker and verified in the CSV.
  Otherwise encoder divergence would be measuring the probe corpus, not the observer.

**Invariant:** every population-averaged seed column remains a valid categorical distribution
(non-negative, sums to 1) after averaging and renormalization. Assert it.

---

## 4. Compute budget

V2's full suite ran in ~54 minutes against a 16-hour estimate; the bottleneck was policy/state
inference, not artifact generation, and V2 deviation 9's exact 6.3× speedup (skipping discarded
`infer_policies` on forced steps) carries forward. **Compute is not the binding constraint**,
which is what makes V3 affordable — E12's largest cells (10000 artifacts/generation) and E8 at
`G_max = 8` with large populations were out of reach under the V2 cost assumptions and are now
routine.

**Expected costs (order-of-magnitude, for planning):**

| experiment | rough cost | note |
|---|---|---|
| E12 | ~30–60 min | dominated by the 3000 and 10000 cells × with/without averaging |
| E13 | ~10 min | re-runs E9 starvation + instruments E12's f=0 arm |
| N11 re-run | folded into E8's f=0 arm | |
| E8 (G_max=8, large N) | ~30–45 min | larger than V2's 14 min but well within budget |
| E11 re-pool | < 1 min | post-processing |

**Cost levers if needed, in order:** (1) E12's 10000-artifact cell can be dropped if the `1/N`
trend is already unambiguous by 3000; (2) reduce replications on E12's exploratory cells while
keeping full replications on the threshold-setting cell; (3) the `--quick` scale must remain
functional for every new experiment. Do **not** reduce `G_max` below 6 — the superlinearity
claim needs the generations.

Report wall-clock per experiment in `RESULTS_V3.md`.

---

## 5. Reporting

`RESULTS_V3.md`, same discipline as V1/V2: written from the CSVs, non-replications reported
rather than tuned, deviations collected with evidence.

Required sections:

- **E12 convergence result and the sample-size decision.** State the leak slope at each sample
  size, with and without averaging, and the threshold chosen for E8. If the leak does not shrink
  with N, say so and stop — E8 is not run.
- **E13 freeze/leak classification.** State explicitly which of the three C4 outcomes obtained.
  If the freeze and the leak are the same effect, say so and note that this validates the D4
  prediction by diagnosis. If they are different, flag the second effect as an open problem in
  the framework, in those words.
- **N11 repair confirmation.** The f = 0 honest-signal slope at full E8 scale, its significance,
  and confirmation that the xfail marker is removed. This is the first thing a skeptical reader
  checks.
- **E8 findings, if reportable.** The two-channel result: value divergence per generation across
  conditions, encoder divergence per generation, and the C3 regression relating them. State
  whether superlinearity holds. If N11 still fails, E8 remains unreported and the reason is
  stated.
- **V2 reconciliation.** What V3 changes about the V2 write-up: specifically, that V2's E8 was
  correctly withheld and V3 either reports it (loop fixed) or explains why it still cannot be
  (loop still lossy for a newly identified reason).

---

## 6. What may not change

Everything in V1 §14 and V2 §6, plus:

- **C1 preserves the policy-produced-artifact property.** Creators stay real `HumanCreator`
  agents. V2 D5 Option B remains forbidden.
- **E8 is gated on E12 and the repaired N11.** It is not run, and its results are not reported,
  until the f = 0 honest-signal slope is insignificant at full E8 scale. No exceptions, no
  "close enough."
- **The freeze is diagnosed, not suppressed and not assumed-artifact.** E13 must be capable of
  returning outcome 2 (a genuinely distinct effect) and reporting it as an open problem. The
  author's theoretical prediction is that it is an artifact; the build must not encode that
  prediction as a foregone conclusion.
- **Nulls are evaluated at the scale of the experiment they gate.** N11's repair is exactly
  this rule; it applies to any null in the suite.
- **The encoder-divergence probe set is uncontaminated (N15).**
