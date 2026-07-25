# V3 design decisions — signed off before implementation

Recorded 2026-07-24, before any V3 experiment was run. Each decision names the option chosen,
the reason, and what it obliges the implementation to do — same discipline as `DECISIONS_V2.md`.

These arose from a viability review of `GHOST_SCALE_SIM_V3_SPEC_1.md` against the existing
codebase. Four of them (D1, D2, D5, D7) change *what gets built*; the rest change *how*. The
acceptance criteria they fix are machine-readable in `ghostscale/prereg_v3.py` and are written
and content-hashed to `results/v3_preregistration.json` before any experiment runs.

---

## D1 — Repaired N11: **conjunctive criterion, replications fixed in advance**

N11 passes only if **both** `|t| < 2` **and** `|slope| < 0.001` nats/generation, with
`n_replications` fixed across every E12 cell.

**Why.** V2's `|t| < 2` alone is power-dependent in a direction that works against the
experiment. Raising per-generation sample size `N` shrinks the leak; raising replications
shrinks the standard error. So "insignificant at the scale of the experiment" is not monotone
in `N`, and the verdict can be moved by the replication count alone. V3 §1 C2 then defines the
minimum viable sample size *as* the crossing of that criterion — so the criterion would be
selecting its own operating point.

`0.001` is ~12× below V2's measured leak (+0.0119) and ~40× below the f=0 baseline KL it sits
on (0.042).

**Obligation.** Written to the pre-registration before E12 runs; `prereg_v3.n11_verdict` is
the single implementation, called by E12, E8 and the N11 test, so the written criterion and the
applied criterion cannot drift apart.

## D2 — Compute split: **raise N, keep M = 5, and measure the averaging floor**

E8 runs at the E12-determined `N` with `n_observers = 5`. E12 adds an `M ∈ {1, 5, 20}` sweep at
fixed `N` to measure what averaging actually buys.

**Why, and this is a correction to the spec's stated mechanism.** V3 §1 C1 predicts averaging
makes the slope "an additional factor of ~1/M smaller at every N". That holds for the
observer-side term and fails for the rest: `run_generation` draws the corpus **once per
generation** and all `M` observers read that same corpus — only feature emissions are redrawn
per observer. The corpus's own goal-composition and signal noise is therefore common-mode
across the population and averaging cannot cancel it. **The 1/M gain has a floor that only
larger N can lower.**

This does not weaken C1 — the V2 leak was localised to the single-observer seeding path, which
is exactly what C1 fixes — but it does mean C2's sample-size sweep is load-bearing rather than
confirmatory decoration, and that the M sweep should be *measured* rather than assumed.

**Rejected:** giving each observer an independent corpus draw. It would make the 1/M claim
literally true, but it changes what a "generation" *is* and costs M× per generation; that is a
semantic change to the model deserving its own justification, not a fix smuggled in under C1.

## D3 — Compute budget: **the §4 estimates are optimistic; the run is overnight**

Measured on this machine: 41 artifact-rollouts/s/core solo, **331 rollouts/s aggregate** at 18
workers (24 CPUs). BLAS thread pinning (`OMP_NUM_THREADS=1`) changed nothing, so the ~45%
parallel efficiency is not oversubscription.

| | spec §4 estimate | at measured throughput |
|---|---|---|
| E12 (g_max 8, M 5, 3 reps, both arms, N-sweep + M-sweep) | 30–60 min | **~3.4 h** |
| E8 (N ≈ 3000, g_max 8) | 30–45 min | **~1.8 h** |

§4's "compute is not the binding constraint" survives — this is an overnight run, not a week —
but the plan is not built on the 30–60 min figure. The 10,000-artifact cell alone is ~69% of
E12's N-sweep, and remains the first cost lever exactly as §4 orders it.

## D4 — E12 runs at `g_max = 8`, matching E8

The spec never states E12's generation count, but E12 measures a per-generation slope and E8 is
judged by one. A threshold derived at a different `g_max` than the experiment it gates does not
transfer. Costs 2× versus `g_max = 4`; accepted.

## D5 — Encoder divergence is a **decoding** measure, not a corpus statistic

Measured as `MI(observer's modal inferred goal ; true goal)` over a fixed, uncontaminated probe
set, with mean posterior mass on the true goal reported alongside.

**Why.** V3 §1 C3 words it as "`MI(features; goal)` on a held-out set of known-genuine probe
artifacts". Read literally that measures **the corpus**: it is a property of how those
artifacts were generated and returns the same number for a perfect decoder and a destroyed
one — the exact failure N15 exists to prevent, reached from the other direction.

**Obligations.** (a) Probes are drawn from the **generation-0** creator bank and held fixed for
the whole chain, for the same reason `c_true` is fixed — a probe set that drifts alongside the
observer makes a flat result ambiguous between a stable decoder and two cancelling drifts.
(b) Every probe set is decoded **twice**, engagement-free and DEEP-forced, so model drift and
disengagement are separable; C3's claim is about the model, and E9/E13 own engagement.
(c) Probe decoding never learns, and draws from an independent RNG stream so that enabling the
second channel cannot shift any other draw.

## D6 — The C3 relationship is tested by **partial correlation**, not a pooled regression

Reported: partial correlation of encoder divergence with value divergence controlling for
generation; within-generation correlation across conditions and replications; lag-1
cross-correlation in both directions. The pooled regression is reported and explicitly labelled
**not evidence**.

**Why.** Both channels trend with generation by construction, so a pooled regression across
generations comes out significant *even if the channels are independent*. It would confirm the
shared-mechanism claim automatically and could never refute it. A test that cannot fail is not
evidence, and §5's "non-replications reported rather than tuned" applies to the analysis as
much as to the code. The lag statistic also operationalises "lags and tracks rather than
leads", which the spec asserts but never defines.

## D7 — E13: **one shared quantity, and outcome 2 expected before the run**

Both halves of E13 report the identical quantity through the identical code path — mean over
goals of `KL(learned CREATOR/DEEP column ‖ true column)` — against effective Dirichlet sample
count. The classification tolerance is pre-registered.

**Why the framing is softened.** §0 hypothesises the freeze and the leak are *the same effect*.
V2's own numbers argue they are opposite ends of one axis:

- **E9 starvation** — engagement collapses to 1.87 of 6 DEEP steps, so genuine content stops
  producing updates and the column never departs from its informative D1-seeded prior.
  → **low** effective sample count; the error is **prior-anchoring**.
- **E8 honest arm** — the signal concentrates updates on the CREATOR column, which sharpens
  fast around a finite-sample estimate that then compounds.
  → **high** effective sample count; the error is **sampling noise**.

So §0's claim is restated as **"a shared finite-sample axis, on which they may sit at opposite
ends"**, and **C4 outcome 2 (two distinct effects) is recorded as the expected result before
the run**.

**This is not a failure of the redo.** It is the redo establishing that the framework has *two*
finite-sample failure modes where it assumed one — which §6 requires E13 to be capable of
returning, and §5 requires to be reported as an open problem in those words rather than
explained away.

## D8 — N13 tests the **1/N prediction**, not monotonicity

`log|slope| = a + b·log N` over the without-averaging arm. Gate: `b` significantly negative.
Reported: whether `b` falls in `[-1.5, -0.5]`, against a predicted `-1`.

**Why, and the reason matters more than the fix.** The spec states N13 as strict monotonicity
in sample size. Five noisy slope estimates break that by luck, and N13 blocks E8 — so as
written it fails for statistical rather than substantive reasons. But the correction is not
merely "relax the null to avoid false failures":

> Monotonicity says *"the leak didn't go up."* The regression coefficient says *"the leak
> shrinks as 1/N, which is the specific signature of finite-sample noise."* The second is the
> actual diagnosis.

A leak shrinking as `1/√N` or `1/log N` passes monotonicity while arguing *against* the
finite-sample story; the exponent catches that and the monotonicity test cannot. So N13 now
**confirms or refutes** the V3 diagnosis rather than guarding it. The band is reported, not
gated — a `b` of −0.5 clears the gate while arguing the error is not the 1/N kind, and that
tension is a finding to state, not to suppress.

**Obligation.** `tests/test_nulls_v3.py` asserts the criterion can fail (flat and growing
leaks refute) and that it distinguishes `1/N` from `1/√N`.

---

## Smaller resolutions, recorded for completeness

- **N14 restated.** As written it is near-vacuous: V2 already seeded from one observer, so
  `M = 1` averaging is trivially identical. Implemented as (a) a bit-exactness test on the
  seeding function — with an explicit single-observer pass-through, since renormalising an
  already-normalised column perturbs the last bits and can flip an `rng.choice` — plus (b) a
  guard that averaging is *not* inert at `M > 1`, without which the fix could pass N14 by
  doing nothing.
- **Run-order contradiction resolved.** §2 lists the N11 re-run as stage 3 and E8 as stage 4,
  while §4 says N11 is "folded into E8's f=0 arm"; both cannot be literal. `run_all_v3.py`
  runs E8's f=0 arm **alone at full scale** as stage 3 and gates on it, then runs the full grid
  as stage 4 (the f=0 cells are seeded identically and reproduce the gated numbers exactly).
  Nothing is evaluated at reduced scale.
- **Staleness guard.** The V2 gate read `results/e8_trends.csv` and would have judged a
  leftover file. Stage 3 deletes E8's outputs before regenerating them.
- **E8's sample size is a refusal, not a default.** `resolve_sample_size` raises if
  `results/e12_threshold.json` is absent or does not clear E8. A silent fallback to the config
  placeholder is precisely how a hardcoded sample size would come back (§1 C2).

## An incident worth recording

During development a `--quick` E8 run was launched without `--out` and **overwrote V2's
`results/e8_{raw,summary,trends}.csv` and `figures/e8_recursive.png`**. `restore_v2_e8.py`
regenerates that cell under V2's parameters and seeding rule, and it **reproduces the reported
V2 run exactly**: f=0 honest slope **+0.0119**, **t = 3.75**, matching RESULTS_V2.md to the
reported precision. The V2 numbers are intact and the V3 premise stands unaltered.

Two things came out of the incident that are worth more than the incident itself.

**1. A silent bug that would have shipped.** The first two restore attempts returned +0.0067
(t = 1.31) rather than +0.0119, which looked like environmental drift and was very nearly
written up as such. It was not. `resolve_sample_size` consulted `results/e12_threshold.json`
whenever the file existed, *regardless of `require_e12`* — so a leftover threshold from an
earlier `--quick` E12 (120 artifacts) silently overrode the caller's 300. No error, no warning,
a plausible-looking wrong number. Two consequences, both now fixed:

- Opting out of the E12 gate now means opting out of consulting it at all.
- **`--quick` no longer writes into `results/` or `figures/`.** `run_all_v3.py --quick` writes
  to `results_quick/`, and `ensure_dirs` redirects figures alongside any explicit `--out`. A
  smoke run must not be able to overwrite a reported run's evidence, and it must not be able
  to *gate* one either — a quick threshold sitting in `results/` would have set the real E8's
  sample size.

The general lesson, and it is the same one V2 recorded about nulls: **a smoke-scale artifact
that lands in the same path as a reportable one is not a harmless leftover — it is an input.**

**2. The no-averaging arm is verified as a control.** The V3 no-averaging path was diffed
against a verbatim reimplementation of V2's `run_generation`/`run_chain` on this machine and
agrees **bit-for-bit** (max |ΔKL| = 0.000e+00 over four generations). E12's without-averaging
arm is therefore V2's behaviour exactly, not a differently-broken variant — which is what makes
the two-curve comparison in E12 mean what §1 C2 says it means. This is a stronger check than
N14's unit test, which exercises only the seeding function.

Getting there required one deliberate change: `average_seed_column` passes a single column
through **untouched** rather than renormalising it. `SeededCreator` already clips and
renormalises, and normalising an already-normalised column perturbs it by ~5e-17 — enough to
flip an `rng.choice` and send the chain down a different realisation. N14 asks for *exact*
reduction to the V2 seeding, so the M=1 path is a literal pass-through.
