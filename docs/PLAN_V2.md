# Ghost Scale Simulation — V2 Implementation Plan

**Status:** proposed, awaiting sign-off on §1 (four decisions) before code is written.
**Scope:** implements `GHOST_SCALE_SIM_V2_SPEC.md` against the existing `ghost-scale-sim`
repository. V1 §14 load-bearing constraints are treated as inviolable throughout.

---

## 0. What I did before planning

Rather than plan from the spec text alone, I ran three validation spikes against the
installed `inferactively-pymdp==1.0.2` legacy API and the existing model. Four of the
results change the plan materially. They are in §1. The rest of this document assumes
those decisions.

Spikes run (scratch only, nothing written to the repo):
1. **Learner-agent mechanics** — does `pA` / `update_A` / `use_param_info_gain` actually
   work for this model shape, and does learning survive the per-artifact `agent.reset()`?
2. **Cold-start engagement** — does a learner with an uninformative `pA[0]` have any
   reason to choose DEEP?
3. **Biased synth characterisation** — what does `goal_symmetric: false` actually produce
   at the current config, and what is the resulting E6b pre-registered bound?

---

## 1. Four findings that change the spec, and the decisions they force

### F1 — Cold-start deadlock in C3. **This is the one that would have silently killed E7/E8/E9.**

Spec C3 says the learner "starts with a weak, uninformative `pA[0]`". Taken literally
(uniform over features in every column), the learner never engages:

```
FULLY uninformative pA[0], observer facing CREATOR content, c_effort=0.1:
  use_param_info_gain=True    G(DEEP)=+7.1599   G(SKIM)=+7.5599   gap=-0.4000  -> SKIM
  use_param_info_gain=False   G(DEEP)=-6.8402   G(SKIM)=-6.4402   gap=-0.4000  -> SKIM
```

The gap is **exactly** `−0.4` (= the effort cost, `4·c_effort`) in both cases. Reason: with
a uniform `A[0]`, DEEP and SKIM predict identical observations, so the state info gain *and*
the parameter info gain are identical across the two policies and cancel exactly. The
epistemic drive C3 relies on does not discriminate DEEP from SKIM at cold start. The learner
SKIMs forever, accumulates counts only on the SKIM columns, and never learns a DEEP column
at all. E7, E8 and E9 would all return null results that look like findings.

**The fix is in the spec's own words.** C3 says "uninformative `pA[0]` *over the provenance
dimension*". Read that literally instead: the learner *does* know the shared goal→feature
family (it has a body plan, per C1's own argument), and what it does *not* know is how
provenance modulates it — i.e. it does not know `α`. So seed:

```
pA[0][:, p, g, DEEP] = strength · sig_i[g]     for EVERY provenance p
pA[0][:, p, g, SKIM] = strength · synth
```

Every tier starts out looking CREATOR-like. This is exactly right theoretically — the naive
reader assumes everything they read was meant — and it makes E7's question the sharp one:
*can you learn that some tiers are hollow, from content alone?* Measured:

```
Provenance-uninformative pA[0]:
  use_param_info_gain=True    gap=+133.84  -> DEEP
  use_param_info_gain=False   gap=  +1.78  -> DEEP
  (oracle, V1 behaviour, for reference:  gap=+1.03  -> DEEP)
```

**Decision required (D1):** adopt provenance-uninformative seeding as the Learner
definition, documented as a deviation in `RESULTS_V2.md`. My recommendation: yes. The
literal reading is not implementable.

### F2 — The parameter info gain term is wildly out of scale and pymdp gives no weight on it.

See the numbers above: `+133.84` against the oracle's `+1.03`. `calc_pA_info_gain` is added
to the EFE raw, and pymdp exposes no coefficient for it. Left uncalibrated, the learner
engages DEEP on *everything* including GHOST, forever — which destroys E9's starvation
channel (disengagement can never occur) and makes E1/E3's metabolic result inapplicable to
learners. The only lever is `prior_strength`, and it is monotone and usable:

| prior_strength | G(DEEP) − G(SKIM) |
|---|---|
| 0.5 | +265.9 |
| 1.0 | +133.8 |
| 4.0 | +34.8 |
| 16.0 | +10.0 |
| 64.0 | +3.8 |
| *(oracle)* | *+1.03* |

**Decision required (D2):** add a `prior_strength` calibration step as a gated task before
E7 (W6 below), with the criterion **pre-committed before it is run**:

> choose the smallest `prior_strength` such that (a) the learner's DEEP−SKIM gap on CREATOR
> content is within 2× the oracle's, and (b) a learner whose `pA[0]` GHOST column has
> converged disengages from GHOST content (`cum_deep` < 1 over 20 free steps), i.e. the
> starvation channel is alive.

Written to `results/v2_calibration.json` with the sweep that produced it, and reported as a
deviation. Without this the E9 "starvation" arm cannot exist and E8's engagement panel is
meaningless. If no `prior_strength` satisfies both, that is itself reportable and E9 gets
run with `use_param_info_gain=False` as a documented fallback.

### F3 — The default biased draw lands on the single most flattering goal.

`goal_symmetric: false` at the current config (`concentration=0.03`, `seed=20240719`) gives:

```
raw synth = [0.028 0.045 0.190 0.000 0.000 0.064 0.000 0.674]   H=0.997 nats
goal-pair mass = [0.073, 0.190, 0.064, 0.674]  ->  favoured k=3
pre-registered bound(f=0.8, k=3) = 1.436 nats
```

Two problems. First, `k=3` is the **rarest** goal in `C_true = [0.40, 0.30, 0.20, 0.10]`, so
the single default draw yields the **maximum** bound of the four possible (0.50 / 0.68 /
0.95 / **1.44**). A skeptical reader — the reader §5 explicitly writes for — will say the
synth was drawn to point at the rarest goal. Second, the draw contains **hard zeros**
(features 3, 4, 6): those features become literally impossible under GHOST, so observing one
is a proof of non-GHOST provenance. That is an unintended information channel and it will
also produce infinities in the per-column KL diagnostics E9 needs.

**Decision required (D3):**
- (a) Make the biased arm a **sweep over `synth_draw_seed`** (8 draws), not one draw. Record
  `k` and lean magnitude per draw; pre-register the bound *per draw*; report naive KL as a
  function of the realized lean. Cost is small (E6b is the cheap stage) and it converts
  E6b's headline from "one lucky draw" into "KL scales with lean, across draws".
- (b) Add a `synth_floor` (default `1e-3`, renormalized) so the synth has full support.
  Required for E9's per-column KL regardless.

I recommend both. (b) is close to mandatory.

### F4 — Mechanics that DO work (no decision needed, but they constrain the design)

Verified against pymdp 1.0.2 legacy:

- `Agent.reset()` executes `self.A = utils.norm_dist_obj_arr(self.pA)` when `pA` is set. So
  **learning survives the per-artifact `reset()`** already in `rollout_observer` — confirmed,
  `A[0]` identical across a reset. Good: no rollout-loop surgery needed for accumulation.
- **Corollary and trap:** that same line means `A[1]` (the κ-precision signal likelihood)
  and `A[2]` (deterministic effort) get *overwritten from `pA`* on every reset. If `pA[1]`
  and `pA[2]` are left unset the constructor rejects it; if set naively, κ is destroyed.
  Fix: set `pA[m] = FIX_SCALE · A[m]` for `m ∈ {1,2}` with `FIX_SCALE = 1e6`. Verified drift
  after 12 learning steps: **exactly 0.0** for both, and it drives their parameter info-gain
  contribution to ~0 so it does not pollute the EFE.
- `modalities_to_learn=[0]` correctly confines learning to `A[0]`.
- `A[0]` stays column-stochastic after every `update_A` (the §3 invariant) — asserted over 12
  steps, holds. Cheap to assert in production.
- `spm_wnorm` adds `EPS` before dividing and the result is masked by `(pA > 0)`, so exact
  zeros in `pA[2]` produce **no NaN**. EFE finite throughout. Confirmed.
- **N10 is exactly satisfiable:** `lr_pA=0.0` leaves `A[0]` bit-identical to its start
  (max drift `0.0`) and identical to the oracle `A[0]` to `1.7e-16`. N10 can be an exact
  test, not a tolerance test.
- The spec's own pre-registration arithmetic checks out: bound(f=0.8) = 0.5007 / 0.6804 /
  0.9480 / 1.4357 nats for k = 0/1/2/3. The spec's "0.50 if k=0, 1.44 if k=3" is correct.

---

## 2. Architecture

The central structural change is C1, and it breaks an assumption the current code is built
on: **`GenerativeModel` currently serves double duty** as the observer's model *and* as the
world's ground truth (`Environment` and `creators` both read `gm.sig`, `gm.noise_free_synth`,
`gm.alpha`). Once each observer has its own `sig_i`, those must separate.

```
                        V1                                  V2
  build_shared_model -> GenerativeModel  ->  build_world_model    -> WorldModel   (sig_true, synth, alpha)
       |                    |                                          |  used by Environment + creators
       +-> A for ALL        +-> Environment                            |
           observers        +-> creators                 build_observer_model(world, d_i, rng)
                                                                   -> ObserverModel (sig_i, A_i)
                                                                          |  one per observer
```

`WorldModel` keeps the exact V1 semantics and is what `Environment`/`creators` consume, so
the world is unchanged by the refactor — which is what makes N8 checkable.
`ObserverModel` is per-observer and is what `make_agent` consumes.

**Backward compatibility:** `build_shared_model` is retained as a thin shim returning a
`WorldModel` with `d=0` observer semantics, so E1/E3/E4/E5 need no edits beyond the
observer-construction call site.

### File map

| file | action | contents |
|---|---|---|
| `ghostscale/generative_model.py` | **modify** | split `WorldModel`/`ObserverModel`; `build_observer_signature` (C1); `goal_symmetric:false` + `synth_floor` + lean diagnostics; `make_learner_agent` |
| `ghostscale/learning.py` | **new** | `build_pA`, `learn_step` (with the column-stochastic assert), learned-`A` diagnostics: per-column KL from true `A` (**shape**), per-column entropy (**flatness**), GHOST-column convergence, `MI(features;goal)` on learned human columns |
| `ghostscale/regret.py` | **new** | C5: downstream task env, `cumulative_regret`, `argmax_preserved`, `sycophancy_rate` |
| `ghostscale/generations.py` | **new** | C4 recursion driver; per-generation panel |
| `ghostscale/corpus_cache.py` | **new** | cost lever #1: deterministic corpus cache keyed by (seed, f, signing, synth_draw, n) |
| `ghostscale/creators.py` | **modify** | `SeededCreator` — `C` from `C_recovered`, emission from learned `A` (H4 motor model) |
| `ghostscale/environment.py` | **modify** | corpus draw over an arbitrary creator population; generation-g corpora |
| `ghostscale/observer.py` | **modify** | per-observer `ObserverModel`; optional `learn=True` in the rollout loop |
| `ghostscale/metrics.py` | **modify** | `column_kl` / `column_entropy` / `time_to_competence` helpers |
| `config/default.yaml` | **modify** | new `expertise`, `learning`, `regret`, `generations` blocks + `experiments.e6b/e7/e8/e9/e10`; `quick` entries for all |
| `ghostscale/experiments/e6b_corpus_biased.py` | **new** | stage 1 |
| `ghostscale/experiments/e10_expertise.py` | **new** | stage 2 |
| `ghostscale/experiments/e7_learn_ghost.py` | **new** | stage 3 |
| `ghostscale/experiments/e9_poison_vs_starve.py` | **new** | stage 4 |
| `ghostscale/experiments/e8_recursive.py` | **new** | stage 5 |
| `ghostscale/experiments/e11_regret_vs_kl.py` | **new** | stage 6, pure post-processing |
| `tests/test_v1_regression.py` | **new** | N8 |
| `tests/test_nulls_v2.py` | **new** | N9, N10, N11, N12 + the `pA`-invariant |
| `run_all_v2.py` | **new** | staged, gated runner per spec §4 |
| `RESULTS_V2.md` | **new** | written from CSVs at the end |

---

## 3. Workstreams

Ordered by dependency. W1–W3 are prerequisites for everything; W4 gates the whole project.

### W1 — Model refactor for C1 (observer-specific likelihoods)

```python
def build_observer_signature(sig_true, d_i, rng):
    if d_i == 0.0:
        return sig_true.copy()          # exact, no RNG consumed  <-- see hazard below
    out = np.empty_like(sig_true)
    for g in range(sig_true.shape[0]):
        out[g] = normalize((1 - d_i) * sig_true[g] + d_i * rng.dirichlet(np.ones(F)))
    return out
```

**Hazard (N8-critical):** the `sig_i` perturbation must be drawn from a **dedicated RNG
stream**, not the one that draws `D`. If it shares a stream, then at `d=0` the draws consume
different numbers of variates than V1 did, every `D` shifts, and N8 fails for a reason that
has nothing to do with the refactor. Implement as
`rng_sig = default_rng(observer_seed(...) ^ SIG_SALT)` and leave the `D` stream untouched.
Additionally short-circuit at `d_i == 0` so no variate is consumed at all. **N8 is the
acceptance test for W1 and must be exact-to-tolerance against the committed V1 CSVs.**

Also in W1: `build_noise_free_synth` gains `goal_symmetric: false`, `synth_floor`, and
returns `(synth, k_favoured, lean_magnitude)`; `check_signature_invariants` must still pass
in both arms (`H(synth)=0.997 < 1.8` in the biased arm — verified, it passes).

**Acceptance:** N8 passes; full existing test suite passes unchanged; E2 rerun at `d=0`
reproduces `RESULTS.md` E2 within tolerance.

### W2 — Learner observer (C3)

`make_learner_agent(observer_model, D, cfg, prior_strength, lr_pA, use_param_info_gain)`,
building `pA` per F1/F4 (provenance-uninformative seeding; `FIX_SCALE` pinning of
modalities 1–2; `modalities_to_learn=[0]`). Oracle path unchanged.

`ghostscale/learning.py` owns the diagnostics, and the E9 **shape vs flatness** split is the
design constraint on it: per-column `KL(A_learned[:,p,g,att] ‖ A_true[:,p,g,att])` measures
*shape distortion* (poisoning); per-column `H(A_learned[:,p,g,att])` measures *flatness*
(starvation). Both are recorded for every column in every learning experiment so E9's
diagnostic is a report rather than a new measurement.

**Invariant (spec §3):** `assert_column_stochastic(agent.A)` after **every** `update_A`.
Verified cheap and true; make it unconditional, not debug-gated.

**Acceptance:** N10 exact (`lr_pA=0` ⇒ bit-identical to fixed-`A`); the column-stochastic
invariant asserted in every learning path.

### W3 — Corpus cache (spec §4 cost lever 1)

Artifact generation is deterministic given a seed and is currently regenerated per
condition. Cache `list[Artifact]` to `results/_cache/corpus_<hash>.npz`, keyed by
`(base_seed, seed_rep, n_artifacts, contamination, signing_rate, honesty, synth_draw_seed,
goal_symmetric, creator_goal_vector_hash)`. Invalidate on key change; `--no-cache` flag.
Add a test that a cached and an uncached run produce identical CSVs — the cache must be a
pure speedup and never a source of drift.

Deliberately built **before** the experiments so every one of them gets the win.

### W4 — E6b + pre-registration (spec stage 1). **This gates everything downstream.**

The good news: E6b is mostly config and reporting. The oracle aggregator, the corpus draw,
and the accumulation loop already exist in `e6_corpus_corruption.py` and are correct for
this purpose. Work is: the biased-synth axis (W1), the `synth_draw_seed` sweep (D3), the
pre-registration writer, and the report.

**Order is mandatory and enforced in code:** `e6b_preregistration.json` is written, and its
hash recorded, **before** a single rollout runs. The runner refuses to proceed if the file
already exists with a different content hash — that is what makes the pre-registration
mean something.

Gate: if naive KL at `f=0.8` stays **below 0.10**, **stop**. Do not start W5+. Report the
falsification loudly, per spec §2. I will bring that result back rather than proceeding.

### W5 — E10 + behavioural regret (spec stage 2)

E10 is cheap, has no contamination and no recursion, and per the spec is the strongest
standalone claim — but it depends on C5, which **does not exist in any form** and is the
largest piece of genuinely new design in V2. Building it here (rather than in E8, where it
is also needed) front-loads the risk onto the cheap experiment.

Proposed C5 design, for review:

- **Task env.** A downstream agent chooses among `num_goals` actions over horizon `H`.
  Action `g` yields stochastic reward with mean `C_true[g]`. This makes `C_true` the optimal
  policy by construction, as C5 requires.
- **Cumulative regret.** `Σ_t (max_g C_true[g] − C_true[a_t])` for the `C_recovered` agent,
  against the `C_true` agent over the same horizon and seed. Reported normalized by `H`.
- **Argmax preservation.** `argmax(C_recovered) == argmax(C_true)`, binary. Reported
  alongside every KL in every V2 CSV — this is the pairing E11 exists to test.
- **Sycophancy analogue (H5).** A forced binary choice between outcome `j` that a simulated
  user *signals* a preference for (an observation the agent's model links to `j`) and
  outcome `argmax(C_true)` that is actually rewarded, with `j ≠ argmax(C_true)`. Metric:
  P(agent follows the signal against the reward). A flatter / mis-shaped `C_recovered`
  supplies less counter-evidence to the signal, so sycophancy should rise with `d`.

**Open question (D4) — is a pymdp agent required for the downstream agent, or is a softmax
policy over `C_recovered` acceptable?** A pymdp `Agent` is more faithful to the rest of the
build and to the CIRL framing; a softmax bandit policy is ~50× cheaper and makes regret an
analytic near-closed-form. Given E8 needs regret at every generation × condition, cost
matters. **My recommendation: pymdp agent, since C5 says "instantiate a downstream agent"
and this is the headline harm metric — but cache the per-`C` regret, since regret depends
only on `C_recovered` and the seed, so identical `C` vectors need computing once.** That
recovers most of the cost difference. Flagging it rather than deciding unilaterally.

E10 itself: sweep `d ∈ {0, 0.1, ..., 0.9}` on a clean corpus. **N12 asserted in code**: the
E10 corpus is constructed with an assert that no artifact has `provenance == GHOST` and
`contamination == 0.0`.

### W6 — `prior_strength` calibration (D2)

Small gated task between stage 2 and stage 3. Pre-committed criterion (F2), sweep written
to `results/v2_calibration.json`, result reported as a deviation. Blocks E7/E8/E9.

### W7 — E7 (stage 3)

Learner × `signing_rate ∈ {0, 0.5, 1}` × κ, biased synth, contamination `f`. Measures:
GHOST-column convergence toward the true synth over exposure; `MI(features;goal)` of the
learned human columns; time-to-competence (first exposure at which `MI` crosses a
pre-registered fraction of the oracle's). Observers reduced to 50 per cost lever 2.

### W8 — E9 + N11 (stage 4)

Three arms, mapping cleanly onto existing switches:
- **Poisoning only:** high κ, dishonest signals, `force_deep_k = T` (engagement frozen, so
  disuse cannot occur), `lr_pA > 0`.
- **Starvation only:** honest signals, engagement free, `lr_pA > 0`.
- **Both:** default.

Report per-column KL (shape) against per-column entropy (flatness), which W2 already
records. N11 runs here as a zero-contamination recursion check before the expensive E8.

**N11 is the most important null and the most likely to fail for implementation reasons.**
At `f = 0` the generation loop must be *exactly* lossless. The realistic failure modes are:
resampling a creator population per generation (sampling noise reads as decay), finite
corpus size, `argmax`-collapsing `C_recovered` into the next generation's `C`, and RNG reuse
across generations. Design the loop so `f=0` is provably a fixed point — reuse the creator
goal vector across generations, seed each generation from an independent stream, and pass
`C_recovered` forward as a full distribution, never an argmax. **If N11 fails, E8 does not
run and I report the loop as lossy.**

### W9 — E8 (stage 5) and E11 (stage 6)

E8: `G_max = 4`, `f ∈ {0, 0.3, 0.6}`, signal ∈ {absent, honest}, learner observers, C4 panel
+ regret per generation. Reported as a **per-generation trend and its significance** (a
mixed-effects or per-seed OLS slope with CI), explicitly **not** an equilibrium and with no
extrapolation, per spec §2. The superlinearity prediction is tested as a quadratic term, not
asserted.

E11 is pure post-processing over the E6b/E7/E8 CSVs: scatter KL against regret, colour by
argmax preservation, and report the variance explained by the argmax-flip boundary versus by
KL magnitude. Requires only that every upstream CSV carries `kl`, `regret`, and
`argmax_preserved` — which is why W5 lands before all of them.

---

## 4. Compute plan

Spec §4's staging is adopted verbatim, with the gates enforced by `run_all_v2.py` rather
than by discipline. Cost levers, in the spec's own priority order:

1. **Corpus cache (W3)** — built first, benefits every stage.
2. **50 observers** for E6b/E7/E9; 200 retained only for E2 and E8's variance panel.
3. **8 seeds** for exploratory arms; 20 for headline runs (E6b's headline cell, E10).
4. **Vectorize the observer loop over the seed axis** before adding workers — noted as the
   next lever if stages 3–5 overrun; not built pre-emptively.
5. `--quick` kept functional for every new experiment (a `quick:` block entry per experiment
   is part of each workstream's definition of done, not an afterthought).

`G_max` stays at 4 and will not be reduced below 3. Wall-clock is recorded per experiment
and reported in `RESULTS_V2.md` §compute.

**Honest estimate:** the spec's ~16h of compute is plausible *if* the cache lands the win it
should. The schedule risk is not compute, it is W5 (regret, new design) and W6
(calibration) — both of which can consume real time before any headline number exists.
That is why W5 is attached to the cheap stage-2 experiment rather than to E8.

---

## 5. Reporting

`RESULTS_V2.md`, V1 discipline, written from the CSVs. Two required sections per spec §5:

- **Pre-registration compliance (E6b):** the bound per synth draw computed *before* the run,
  the observed value, and whether the falsification threshold was crossed. First thing a
  skeptical reader checks, so it goes near the top.
- **V1 reconciliation:** which V1 conclusions V2 overturns, refines, and leaves standing.
  Including, in these words, that V1's E6 finding — the provenance signal is metabolically
  rather than epistemically valuable *for an oracle aggregator* — is **not overturned by V2,
  it is scoped**.

Deviations collected with the evidence that motivated them. On current count the deviations
list already opens with three: D1 (provenance-uninformative seeding), D2 (`prior_strength`
calibration), D3 (synth draw sweep + support floor).

---

## 6. Decisions I need from you

Six questions. D1 and D3b are near-forced by measurement; D2, D3a, D4 and D5 are real
trade-offs where your call changes the science, not just the code.

---

### D1 — How is the Learner observer's `pA[0]` initialised?

**Why this is being asked.** C3 says the learner "starts with a weak, uninformative `pA[0]`
over the provenance dimension". That sentence has two readings and they behave completely
differently.

**Option A — literal/uniform.** `pA[0]` uniform over features in every column. The learner
knows nothing about how any content maps to any goal.

**Option B — provenance-uninformative.** `pA[0][:, p, g, DEEP] = strength · sig_i[g]` for
*every* provenance `p`. The learner knows the shared goal→feature family but not `α` — it
does not know which tiers carry intent, and assumes all of them do.

**Measured implications.** Option A fails twice over. First it deadlocks engagement: DEEP and
SKIM score identically under a uniform `A[0]`, so the gap is exactly `−4·c_effort` and the
learner skims forever. Second — and this is the one that closes the question — **forcing DEEP
does not rescue it.** With disengagement made impossible by construction, over 400 artifacts:

```
Option A: MI(features;goal) = -0.0000 nats at n = 100, 200, 300, 400
          max spread between the four learned goal columns = 0.0000
Option B: MI = 1.034 / 1.017 / 1.024 / 1.025      (ceiling, true A[0]: 1.089)
```

Learning `A[0]` attributes each observed feature to the *believed* `(provenance, goal)`. Under
a uniform `A[0]` the goal posterior never leaves the prior, so every observation deposits ~¼
of a count into all four goal columns. They converge to the same marginal and stay bit-
identical forever. It is an **identifiability** deadlock, not merely a disengagement one, so
the obvious workaround — a forced-DEEP warmup — does not exist. Option B reaches 94% of the
oracle ceiling.

**What each choice costs you.** Option B concedes that the learner arrives knowing the
goal→feature family. That is a real concession and it must be stated plainly in
`RESULTS_V2.md`, because it narrows E7's claim from "can you learn intent-reading from
scratch?" to "can you learn *which sources are hollow* from content alone?". The narrower
claim is still the one §0 Error 3 actually asks for — you do not have the `A[0]` GHOST column
and must learn it from an unlabelled contaminated corpus — and it is consistent with C1's own
argument that observers share a likelihood *family* because they share a body plan.

**Recommendation: B.** A is not implementable; the measurement above is what I would cite in
the deviations section.

---

### D2 — How is `prior_strength` chosen, and does `use_param_info_gain` stay on?

**Why this is being asked.** `calc_pA_info_gain` is added to the EFE raw and pymdp exposes no
coefficient on it. At the natural prior strength it is ~130× the scale of everything else:

| prior_strength | G(DEEP) − G(SKIM) | | prior_strength | G(DEEP) − G(SKIM) |
|---|---|---|---|---|
| 0.5 | +265.9 | | 16.0 | +10.0 |
| 1.0 | +133.8 | | 64.0 | +3.8 |
| 4.0 | +34.8 | | *oracle* | *+1.03* |

Uncalibrated, the learner engages DEEP on everything forever. That deletes E9's **starvation**
arm (disengagement can never occur, so there is nothing to measure), and makes E8's engagement
panel and E1/E3's metabolic results inapplicable to learners.

**Option A — calibrate.** A gated sweep before E7, criterion committed in writing first:
smallest `prior_strength` such that (a) the learner's DEEP−SKIM gap on CREATOR content is
within 2× the oracle's, and (b) a learner with a converged GHOST column disengages from GHOST
(`cum_deep < 1` over 20 free steps).

**Option B — fix by fiat.** Pick a value from the table, no calibration run, no criterion.

**Option C — `use_param_info_gain=False`.** Drop the term entirely.

**Implications, and the coupling you should know about.** `prior_strength` does **two jobs at
once**: it sets the EFE scale of the parameter info-gain term *and* it is the inverse learning
rate — a strong prior is a slow learner. So calibrating it for engagement also silently sets
how fast E7's convergence curve rises, which is E7's headline measurement. If that bothers
you, the clean fix is to decouple them: calibrate `prior_strength` for the EFE scale and use
`lr_pA` to set learning speed independently. I would do this, and it costs nothing.

Option C is cheaper than it looks: C3 wants param info gain so "a learner has a reason to
engage with GHOST content that an oracle does not" — but under D1-B the learner *already* has
that reason, because it believes GHOST is CREATOR-like until it learns otherwise. The
epistemic drive may be redundant given B. Turning it off makes the learner's engagement
directly comparable to the oracle's, which is arguably better for E9. Against that: it drops
a mechanism the spec names explicitly.

**Recommendation: A, with `lr_pA` decoupled**, and C held as a documented fallback if no
`prior_strength` satisfies both criteria. Option B I would avoid — an uncalibrated magic
number in the load-bearing position is exactly the kind of thing V1's post-mortem was about.

---

### D3a — Does the biased-synth arm use one draw or several?

**Why this is being asked.** At the current config (`concentration=0.03`, `seed=20240719`),
`goal_symmetric: false` produces:

```
raw synth = [0.028 0.045 0.190 0.000 0.000 0.064 0.000 0.674]
goal-pair mass = [0.073, 0.190, 0.064, 0.674]  ->  favoured goal k=3
pre-registered bound(f=0.8) = 1.436 nats
```

`k=3` is the **rarest** goal in `C_true = [0.40, 0.30, 0.20, 0.10]`, so this single draw
yields the largest of the four possible bounds (0.50 / 0.68 / 0.95 / **1.44**). Nothing was
rigged — it is the committed V1 seed — but a reader cannot distinguish that from rigging.

**Option A — one draw** (spec literal). Cheapest. E6b's headline rests on a draw that happens
to point at the rarest goal, and the honest caveat has to be written into the results.

**Option B — 8 random draws.** `k` varies across draws; report naive KL as a function of
realized lean. Converts the headline from "one draw gave 0.3 nats" into "KL scales with lean,
across draws" — a strictly stronger claim, and the one E6b's falsification criterion is
really about. Costs ~8× the stage-1 grid, partly recoverable by cutting `n_replications`
from 5 to 2 (E6b already aggregates 2000 artifacts, so replication buys little).

**Option C — 4 draws stratified by `k`.** Scan seeds, take one draw favouring each goal.
Cheapest route to the lean-scaling claim and it guarantees the full range of `k` is covered
rather than hoping 8 random draws span it. The selection is deliberate and must be disclosed —
but it is a *balanced* selection, which is easier to defend than a single arbitrary one.

**Implications.** This is the experiment §5 says a skeptical reader checks first. A is
defensible only if the caveat is prominent. B and C both make the result robust; C is cheaper
and cleaner, B is less open to "you chose the seeds".

**Recommendation: C**, with the seed-scan procedure and its output written into the
pre-registration file *before* the run, so the stratification is auditable.

---

### D3b — Does the synthetic distribution get a support floor?

**Why this is being asked.** The biased draw contains **hard zeros** at features 3, 4 and 6.
V1 never hit this because symmetrization produced `[0.243 0.007 ...]` — no zeros. The biased
arm does.

**Implications of leaving zeros in.** Two problems, and the first cuts against your own
hypothesis. (1) A zero means those features are *impossible* under GHOST, so observing one is
a **proof** of non-GHOST provenance — an unintended perfect provenance channel that bypasses
the Ghost Scale entirely. It would let the oracle aggregator exclude synthetics for free and
*suppress* the E6b effect. (2) Per-column KL against a zero-support reference is infinite,
which breaks E9's shape-vs-flatness diagnostic outright.

**Options.** Floor at `1e-3` (my default), or `1e-2`, or none. The floor is renormalized and
must keep `H(synth)` below `structured_ceiling = 1.8`; at `1e-3` the current draw sits at
`H = 0.997`, comfortably clear.

**Recommendation: floor at `1e-3`.** I would treat this as near-mandatory rather than a
preference — the zeros are a modelling artifact of `Dirichlet(0.03)`, not a claim about
generative output.

---

### D4 — What is the downstream agent in the C5 regret metric?

**Why this is being asked.** C5 says "instantiate a downstream agent with `C = C_recovered`".
Regret is the headline harm metric and is reported alongside KL in every V2 CSV, including at
every generation × condition of E8 — so its cost multiplies.

**Correction to my earlier recommendation.** I previously argued a pymdp `Agent` is more
faithful. On inspection that is largely false: for a bandit-shaped task, a utility-only pymdp
agent's policy posterior **is** `softmax(γ · C)`. Your own `creators.py` already relies on
exactly this identity — `HumanCreator` sets `C = log(sig[g])`, `gamma=1`, and reads the
emission distribution straight off the policy posterior, documented as "the policy posterior
equals sig[g]". So for the regret and argmax-preservation components the two options are
mathematically the same object, and the pymdp version is just a slower way to compute it.

Where they genuinely differ is the **sycophancy analogue**: that involves an observation (the
simulated user's signal) that has to be *inferred from*, which is a real POMDP and not a
bandit.

**Option A — pymdp throughout.** Faithful-looking, ~50× more expensive, no fidelity gain on
two of the three metrics.
**Option B — softmax throughout.** Cheapest; loses the inference structure the sycophancy
metric actually needs.
**Option C — split by metric.** Closed-form softmax for cumulative regret and argmax
preservation (provably identical to A for these); a real pymdp agent for the sycophancy
analogue, where inference is the point.

**Recommendation: C**, and I would state the softmax/pymdp equivalence explicitly in
`RESULTS_V2.md` so the cheap path is visibly a computation, not a shortcut.

---

### D5 — In E8, what exactly does a creator inherit from its learned `A`?

**Why this is being asked.** C4 step 2 says the next generation's creators have "their
production likelihood seeded from their learned `A`". Several readings are available and they
determine whether E8 shows decay at all — and whether **N11 can pass**.

**Option A — `emission = A_learned[:, CREATOR, g, DEEP]`.** The creator produces features from
its own learned model of what a CREATOR pursuing goal `g` looks like. Clean, and **N11-safe**:
at `f = 0` a perfectly-learned `A` returns `sig_true` exactly, so the generation loop is a
provable fixed point.

**Option B — marginalise the learned `A` over `C_recovered`.** The creator's output reflects
its whole recovered preference distribution rather than one goal. Blurs goal-specific content
even at `f = 0`, so **N11 would likely fail by construction** — generational decay would appear
with zero contamination, which spec §3 says invalidates every E8 result.

**Option C — keep the `HumanCreator` POMDP, set `C = log(A_learned[:, CREATOR, g, DEEP])`.**
Preserves the "a real policy produced this" property that §4.2 calls the entire theoretical
content of the model. Slightly more expensive; reduces to A when `γ=1`, by the same identity
as D4.

**Implications.** A and C are equivalent in output and both make N11 achievable; C keeps the
architectural commitment that human artifacts are produced by an optimising policy rather than
sampled from a distribution, which is a claim V1 deliberately made and defended. B is the
reading that quietly guarantees the result — I flag it because it is the most natural reading
of the sentence as written, and it is the one that would make E8 an artifact.

**Recommendation: C** (equivalently A, plus the policy machinery), and N11 built as the
acceptance test for whichever is chosen.

---

### Summary

| # | question | recommendation | how forced |
|---|---|---|---|
| D1 | Learner `pA[0]` seeding | **B — provenance-uninformative** | near-forced by measurement |
| D2 | `prior_strength` calibration + param info gain | **A, with `lr_pA` decoupled** | real trade-off |
| D3a | one synth draw or several | **C — 4 draws stratified by `k`** | real trade-off (cost vs credibility) |
| D3b | synth support floor | **`1e-3`** | near-forced |
| D4 | downstream regret agent | **C — split by metric** | real trade-off, my earlier advice corrected |
| D5 | E8 creator inheritance | **C (≡ A) — never B** | real trade-off, N11 is the test |

---

## 7. Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| N8 fails from RNG-stream drift, not from a real refactor bug | **high** | high — burns time chasing a phantom | dedicated `sig` RNG stream + short-circuit at `d=0` (W1); this is the first thing built |
| N11 fails — the recursion loop is lossy at `f=0` | **medium** | very high — invalidates all of E8 | design the `f=0` fixed point explicitly (W8); run N11 in stage 4, *before* the 6h E8 |
| No `prior_strength` satisfies both W6 criteria | medium | medium | documented fallback to `use_param_info_gain=False`, reported as a deviation |
| E6b falsified (naive KL < 0.10 at f=0.8) | low–medium | **project-defining** | this is the design working as intended: stop at stage 1, report loudly, bring it back before rethinking downstream |
| C5 regret design is contested after implementation | medium | medium | design reviewed at D4 *before* code; regret is additive to KL, never a replacement, so a revision does not invalidate collected CSVs |
| Learner engages DEEP on GHOST forever, killing the starvation channel | medium | high | exactly what W6's criterion (b) tests for; caught before E9 |
| Stage 5 (E8, 6h+) overruns | medium | low | run last and overnight, as the spec already orders; cost lever 4 (seed-axis vectorization) held in reserve |

---

## 8. Sequencing summary

```
W1 refactor (C1) ──► N8 gate ──► W2 learner ──► W3 cache
                                     │
                                     ▼
                       W4  E6b + pre-registration   ◄── STAGE 1 GATE (falsification)
                                     │
                                     ▼
                       W5  E10 + C5 regret          ◄── stage 2 (headline, cheap)
                                     │
                                     ▼
                       W6  prior_strength calibration ◄── gate for all learner experiments
                                     │
                                     ▼
                       W7  E7 ──► W8 E9 + N11 ──► N11 GATE ──► W9 E8 ──► E11 ──► RESULTS_V2.md
```

Work happens on a `v2` branch off the current `ghost-scale-sim` HEAD, with V1's committed
CSVs preserved as the N8 reference. I will stop and report at the two hard gates (stage 1
falsification, N11) rather than proceeding past them.
