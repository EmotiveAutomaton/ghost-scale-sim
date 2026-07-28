# Ghost Scale Simulation — V2 Specification

**Author of spec:** Abraham Haskins, PhD
**Target:** an autonomous coding agent, extending the existing `ghost-scale-sim` repository
**Depends on:** V1 spec, `RESULTS.md`, and the existing codebase. This document is a **delta**,
not a replacement. Everything in V1 §14 (load-bearing constraints) still holds.

---

## 0. Why V2 exists

V1 confirmed five of six predictions. E6 (corpus corruption) mostly did not replicate, and the
post-mortem identified two coupled specification errors rather than a theoretical failure:

**Error 1 — heterogeneity in the wrong place.** V1 gave observers heterogeneous *priors* (`D`) and
a single shared *likelihood* (`sig[g]`). The theory says the opposite: observers share a *family*
of likelihoods because they share a body plan, but each individual's mapping from artifact to
inferred goal is built from their own sensory history. You partly find yourself in the artist you
are reading. V1 could not represent that, so when the synthetic distribution leaned toward one
goal, the shared likelihood dominated the small prior perturbations and every observer hallucinated
the *same* goal. The rescue was to symmetrize the synthetic distribution (V1 deviation 2).

**Error 2 — symmetrization made contamination into noise.** Synthetic content equidistant from
every goal signature averages out across a large corpus and contributes almost nothing net. Real
generative output is a regression to the statistical mean: a systematic pull with a direction,
which accumulates linearly in sample size rather than cancelling. The choice that rescued E2 is the
same choice that made E6 toothless.

**Error 3 — the aggregator was given the answer key.** V1's aggregator arrived holding a correct
`A[0]` column for GHOST content. It already knew what goalless output looks like. Asking whether a
provenance label added anything was therefore a question that had been answered before the
experiment began. The actual alignment problem is that **you do not have that column** and must
learn it from a corpus that is already contaminated and unlabeled.

V2 fixes all three, adds recursion, and adds the expertise axis.

---

## 1. The five changes

### C1 — Observer-specific likelihoods (expertise)

Each observer `i` gets its own `sig_i[g]`, drawn as a perturbation of a shared latent signature:

```
sig_i[g] = normalize( (1 - d_i) * sig_true[g] + d_i * rng_i.dirichlet(ones(F)) )
```

`d_i ∈ [0, 1]` is the observer's **inexpertise**. `d_i = 0` is a perfect reader of this domain;
`d_i → 1` is a novice whose intent-reading template is essentially arbitrary. Expertise is
`1 - d_i`.

This is the single most important change in V2 and it is load-bearing for three separate results
(E2 heterogeneity, E10 expertise gradient, E8 recursive decay). `D` heterogeneity (V1 §3.6) stays,
but it is no longer the primary source of between-observer variance.

**Invariant:** with `d_i = 0` for all observers, V2 must reproduce V1's E2 numbers within
tolerance. Assert this in `tests/test_v1_regression.py`. If it does not, the refactor broke
something.

### C2 — Biased synthetic content as the default for corpus experiments

`goal_symmetric` becomes an experiment-level axis rather than a global default.

- `goal_symmetric: true` — V1 behavior. Synthetic distribution equidistant from every signature.
  **Retained as a named control condition.** E2 keeps it.
- `goal_symmetric: false` — the synthetic distribution is drawn once from a low-concentration
  Dirichlet and lands wherever it lands, generically leaning toward some goal `k`. Record `k` and
  the lean magnitude in every CSV.

Report both arms everywhere the axis is relevant. The finding to state is the axis itself:
**symmetric synthetic content is noise and averages out; biased synthetic content is an attractor
and accumulates.** Neither condition is "correct"; they are claims about different generator
architectures, and distinguishing them empirically becomes a prediction the framework makes.

### C3 — Learned likelihoods (`pA` / Dirichlet)

The observer no longer arrives knowing what GHOST content looks like. pymdp legacy supports this
directly:

```python
pA = utils.dirichlet_like(A, scale=prior_strength)
agent = Agent(A=A, pA=pA, B=B, C=C, D=D, control_fac_idx=[2],
              policy_len=2, lr_pA=lr, use_param_info_gain=True, ...)
qs = agent.infer_states(obs)
agent.update_A(obs)          # Dirichlet concentration update
```

Two agent classes, both required:

- **Oracle observer** — V1 behavior, correct `A[0]` supplied. Retained as the ceiling condition.
- **Learner observer** — starts with a weak, uninformative `pA[0]` over the provenance dimension
  and must accumulate concentration from experience.

`use_param_info_gain=True` gives the learner an epistemic drive to look at things it does not yet
have a model of, which matters: a learner has a reason to engage with GHOST content that an oracle
does not.

**This change is what converts E6 from "does the label help someone who already knows?" into "can
you learn to tell without labels?"** It is the core of V2.

### C4 — Recursive generations

A generation loop. Generation `g`:

1. Observers learn `A` from a corpus with contamination fraction `f`.
2. A subset of those observers become **creators** for generation `g+1`. Their creator `C` is
   seeded from their own `C_recovered`, and their production likelihood is seeded from their
   learned `A` (this is the H4 motor-model claim: what you absorbed shapes what you make).
3. Generation `g+1`'s corpus is those artifacts, plus fresh synthetic contamination at fraction
   `f`.

Bounded to `G_max = 4` by default. **Compute is the binding constraint** (see §4); the goal is to
detect a significant per-generation trend, not to run to equilibrium.

Track per generation: mean expertise `1 - d̄`, `MI(features; goal)` on genuine artifacts,
`KL(C_recovered ‖ C_true)`, mean DEEP engagement on genuine artifacts, and behavioral regret (§C5).

### C5 — Harm as behavioral regret

Add `metrics.behavioral_regret`. Instantiate a downstream agent with `C = C_recovered`, run it in a
task environment where `C_true` defines the optimal policy, and measure:

- **cumulative regret** against a `C_true` agent over a fixed horizon
- **argmax preservation** — binary, does `argmax(C_recovered) == argmax(C_true)`
- **a sycophancy analogue** (H5): in a forced choice between an outcome the *simulated user*
  signals a preference for and the outcome `C_true` actually rewards, how often does the
  `C_recovered` agent follow the signal against the reward?

Report regret alongside KL everywhere. **A KL of 1.0 that preserves the argmax may be harmless; a
KL of 0.05 that flips it is catastrophic.** Distributional distance is a statistics metric; regret
is an alignment metric. The headline harm claim uses regret.

---

## 2. Experiments

Existing E1–E5 are unchanged except for the C1 refactor and the V1 regression test. New and
rewritten below.

### E6b — Corpus corruption, biased synthetic content — **RUN THIS FIRST, ALONE**

Identical to V1 E6 except `goal_symmetric: false` and the oracle aggregator retained. Nothing else
changes. This is the cheap decisive test of whether the V1 post-mortem was correct.

**Mandatory pre-registration.** Before running, compute and write to
`results/e6b_preregistration.json`:

```
bound(f, k) = KL( (1-f)*C_true + f*onehot(k)  ||  C_true )
```

for the `k` the synthetic draw actually favors, at every `f` in the sweep. With
`C_true = [0.40, 0.30, 0.20, 0.10]` and `f = 0.8`, this bound is **0.50 nats if k=0** and **1.44
nats if k=3**. The observed value will fall below the bound, since the aggregator weights soft
posteriors rather than deltas.

**Predicted:** naive-aggregator KL at f=0.8 between **0.15 and 0.6 nats** (against V1's 0.066).
Provenance-weighted stays near flat.

**Falsification:** if naive KL at f=0.8 stays **below 0.10**, the noise-cancellation diagnosis is
wrong, symmetrization was not what suppressed the V1 effect, and every downstream V2 experiment
needs rethinking before it is run. Report this outcome loudly if it occurs.

### E7 — Can the GHOST column be learned without labels?

Learner observers, biased synthetic content, contamination `f`, crossed with signal availability
`signing_rate ∈ {0.0, 0.5, 1.0}` and κ.

**Measure:** convergence of the learned `A[0][:, GHOST, :, DEEP]` toward the true synthetic
distribution over exposure; `MI(features; goal)` of the learned human columns; time-to-competence.

**Predicted:** without labels the learner folds synthetic features into its model of human intent —
the human columns acquire synthetic mass and lose goal-discriminability — and the degradation is
worse at high κ. With honest labels the learner acquires a clean GHOST column quickly and the human
columns stay sharp.

**This is the experiment that restores the strong claim.** V1 showed the signal is metabolically
useful to someone who already knows. E7 asks whether you can come to know without it. State the
result either way.

### E8 — Recursive degradation across generations

`G_max = 4`, contamination `f ∈ {0.0, 0.3, 0.6}`, signal `∈ {absent, honest}`. Learner observers.

**Measure:** the C4 panel per generation, plus behavioral regret.

**Predicted:** monotone degradation with generation at `f > 0`, attenuated by an honest signal, and
**superlinear** — generation 3's damage exceeds three times generation 1's, because a degraded
reader produces degraded artifacts which further degrade the next reader.

**Report the trend and its significance, not an equilibrium.** With `G_max = 4` the honest claim is
"a significant per-generation effect in the predicted direction," and that is sufficient. Do not
extrapolate the curve.

### E9 — Poisoning versus starvation

Two corruption channels, run independently and together:

- **Poisoning only** — hallucinated goal posteriors (high κ, dishonest signals) are written into
  `pA` updates. Engagement policy frozen so disuse cannot occur.
- **Starvation only** — honest signals, so no fabrication, but disengagement means no `pA` updates
  on genuine content. `pA` learning enabled, engagement free.
- **Both** — the default condition.

**Measure:** shape of the learned `A` (KL from the true `A` per column) versus *flatness* of the
learned `A` (entropy per column). **Poisoning distorts shape. Starvation flattens.** These are
distinguishable signatures and the diagnostic must separate them.

**Predicted:** starvation is the larger effect at realistic contamination levels, because
disengagement is fast and total while fabrication requires a dishonest signal to be present and
trusted.

### E10 — The expertise gradient (the RLHF result)

Sweep observer inexpertise `d_i` from 0 to 0.9 on a corpus of **uncontaminated, genuinely
intent-dense human artifacts**. No synthetic content at all.

**Measure:** extracted intent density (`psi_analogue`, and `KL(Q(goal|τ) ‖ P0)`), goal-recovery
accuracy, and behavioral regret of a downstream agent trained on the resulting `C_recovered`.

**Predicted:** recovered intent density falls monotonically with inexpertise **even though corpus
quality is held perfectly constant**, and the `C_recovered` from high-`d` observers produces
measurably higher sycophancy in the downstream agent.

**Why this matters more than the rest.** It says the extractor's own competence is a hard ceiling
on recoverable intent, independent of data quality. If true, no amount of better data or more
raters repairs RLHF, because the instrument is the limit rather than the sample. It also closes the
self-sealing loop: corrupted `A` lowers extractable intent from genuine work, which lowers the
reward for engaging, which drives disuse, which further corrupts `A`.

Run E10 even if compute forces cuts elsewhere. It is cheap (no contamination, no recursion) and it
is the strongest standalone claim in V2.

### E11 — Regret versus KL

Not a separate simulation. A cross-cutting analysis over E6b, E7, E8 outputs: scatter
`KL(C_recovered ‖ C_true)` against behavioral regret, colored by argmax preservation.

**Predicted:** the relationship is weak and the argmax-flip boundary explains far more variance
than KL magnitude. If so, report that KL is the wrong harm metric and say so plainly.

---

## 3. Nulls and invariants — additions

Existing N1–N7 must still pass unchanged.

- **N8 — Expertise null.** With `d_i = 0` for all observers, V2 E2 reproduces V1 E2 within
  tolerance. *Guards the refactor.*
- **N9 — Symmetric control.** With `goal_symmetric: true`, E6b reproduces V1's E6 numbers.
  *Proves the new effect comes from the bias axis and nothing else.*
- **N10 — Learning null.** With `lr_pA = 0`, the learner observer must behave identically to a
  fixed-`A` observer with the same starting `A`. *Proves learning is doing the work in E7.*
- **N11 — Zero-contamination recursion.** With `f = 0`, E8 must show **no** degradation across
  generations. *This is the most important new null.* If generational decay appears without any
  contamination, the recursion loop itself is lossy and every E8 result is an artifact of the
  implementation rather than a finding.
- **N12 — Clean-corpus expertise.** E10's gradient must persist with zero synthetic content
  present. Already the design, but assert it: no GHOST artifacts in the E10 corpus.

**Invariant:** every `pA` update leaves `A` column-stochastic. Assert after each update.

---

## 4. Compute budget — a hard constraint

V1's E6 took roughly two hours on the target machine. V2 is substantially larger, so cost control
is part of the specification rather than an afterthought.

**Required ordering.** Each stage is conditional on the previous one.

| stage | experiments | rough cost | gate |
|---|---|---|---|
| 1 | E6b alone | ~2h | if falsified (KL < 0.10), stop and report |
| 2 | E10, N8, N9, N10 | ~2h | cheap, no recursion; E10 is a headline |
| 3 | E7 | ~3h | |
| 4 | E9, N11 | ~3h | |
| 5 | E8 (recursive) | ~6h+ | run last, overnight |
| 6 | E11 analysis | minutes | pure post-processing |

**Cost reduction levers, in order of preference:**

1. Cache creator corpora to disk and reuse across conditions. Artifact generation is deterministic
   given a seed and is currently regenerated per condition. This is likely the single largest win.
2. Reduce observer count where between-observer variance is *not* the measured quantity. E2 needs
   200. E6b, E7, E9 do not — 50 is sufficient for a population mean. Only E2 and E8's variance
   panel require the full 200.
3. Reduce seeds from 20 to 8 for exploratory arms; keep 20 for headline runs only.
4. Vectorize the observer loop across the seed axis before reaching for more workers.
5. `--quick` scale must remain functional for every new experiment.

**Do not** reduce `G_max` below 3; two generations cannot establish a trend.

Report wall-clock per experiment in `RESULTS_V2.md` so the next iteration can be planned.

---

## 5. Reporting

`RESULTS_V2.md`, same discipline as V1: written from the CSVs, non-replications reported rather
than tuned, deviations collected with the evidence that motivated them.

Two required additions:

- A **pre-registration compliance section** for E6b: the bound computed before the run, the value
  observed, and whether the falsification threshold was crossed. This is what keeps the V2
  redesign a model rather than a rationalization, and it is the first thing a skeptical reader
  should be able to check.
- A **V1 reconciliation section** stating plainly which V1 conclusions V2 overturns, which it
  refines, and which stand. V1's E6 finding — that the provenance signal is metabolically rather
  than epistemically valuable *for an oracle aggregator* — is not overturned by V2. It is scoped.
  Say so in those words.

## 6. What may not change

Everything in V1 §14, plus:

- The pre-registered bound for E6b, computed and written to disk **before** the run.
- N11. Zero-contamination recursion must show zero degradation.
- The symmetric arm retained as a named control wherever the bias axis is used.
- Honest reporting of E7 and E10 in whichever direction they come out. E10 in particular is a
  strong claim with a clean falsification: if recovered intent density is flat in expertise on a
  fixed clean corpus, the expertise-gating hypothesis is wrong and the RLHF reframing does not
  follow.
