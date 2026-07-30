# V2 design decisions — signed off before implementation

Recorded 2026-07-24, before any V2 code was written. Each decision names the option chosen,
the reason, and what it obliges the implementation to do. Evidence for D1/D2/D3 is the three
validation spikes described in `GHOST_SCALE_SIM_V2_PLAN.md` §1.

---

## D1 — Learner `pA[0]` initialisation: **Option B (provenance-uninformative)**

`pA[0][:, p, g, DEEP] = strength · sig_i[g]` for every provenance `p`; `SKIM` seeded with the
synthetic distribution. The learner knows the shared goal→feature family; it does not know
`α`, i.e. it does not know which tiers carry intent, and initially assumes all of them do.

**Why not the literal uniform reading.** Measured: with DEEP forced (disengagement made
impossible by construction) a uniform-`pA[0]` learner shows `MI(features;goal) = -0.0000`
nats after 400 artifacts and all four learned goal columns are bit-identical. Learning `A[0]`
attributes each observed feature to the *believed* `(provenance, goal)`; under a uniform
`A[0]` the goal posterior never leaves the prior, so every observation deposits ~¼ count into
all four goal columns and they converge to a common marginal. It is an identifiability
deadlock, not only a disengagement one, so a forced-DEEP warmup does not rescue it.

**Obligation.** This is a **theoretical commitment, not a shortcut**, and must be named as
such in the README and in `RESULTS_V2.md`: *observers share a likelihood family because they
share a body plan; what they do not share, and must learn, is which sources carry intent.*
This is the same commitment C1 makes when it derives `sig_i` as a perturbation of a shared
latent signature. E7's claim is correspondingly scoped: not "can you learn intent-reading
from scratch" but "can you learn which sources are hollow, from content alone, without
labels".

## D2 — `prior_strength`: **calibrate, with `lr_pA` decoupled — and a one-pass limit**

Calibrate `prior_strength` for the EFE scale of the parameter info-gain term; set learning
speed independently via `lr_pA` so that one knob does not silently do two jobs.

**The criterion is written to `results/v2_calibration_criterion.json` BEFORE the sweep runs.**

**Hard limit (caveat as given):** if calibration does not succeed in **one pass**, do not
iterate. Switch to `use_param_info_gain=False` for all learner experiments and record it as a
deviation. Under D1-B the learner already has a reason to engage GHOST content — it believes
GHOST is CREATOR-like until it learns otherwise — so the parameter info-gain term is not
load-bearing for that motivation.

## D3a — Biased synth arm: **Option C (draws stratified by favoured goal `k`)**

Four draws, one favouring each goal, so `k` spans `{0,1,2,3}` and the reported effect is
KL-as-a-function-of-lean rather than a single draw's value.

**Obligation (as directed):** the **full seed scan** — every seed examined, its favoured `k`,
its lean magnitude, and the selection rule — is written into the pre-registration file
**before the run**, not just the four chosen seeds. The stratification must be auditable end
to end.

## D3b — Synthetic support floor: **1e-3**

Applied to `noise_free_synth` in both arms, then renormalized. A no-op for V1's symmetrized
draw (min mass 0.007), so N8/N9 are unaffected.

**Two reasons.** (1) The biased draw contains hard zeros at features 3, 4, 6; a zero makes
those features impossible under GHOST, so observing one is a *proof* of non-GHOST provenance —
an unintended perfect provenance channel that bypasses the Ghost Scale and would suppress the
E6b effect. (2) Per-column KL against zero support is infinite, which breaks E9's
shape-vs-flatness diagnostic.

## D4 — Downstream regret agent: **split by metric**

- `cumulative_regret`, `argmax_preserved` — closed-form softmax policy over `C_recovered`.
- `sycophancy_rate` — a real pymdp `Agent`, because it must *infer from* a user signal.

**Obligation.** State the equivalence explicitly in `RESULTS_V2.md`: for a bandit-shaped
task a utility-only pymdp agent's policy posterior **is** `softmax(γ·C)`. The repo already
relies on this identity — `creators.HumanCreator` sets `C = log(sig[g])`, `γ=1`, and reads
the emission distribution straight off the policy posterior. The closed form is a
computation, not a shortcut.

## D5 — E8 creator inheritance: **Option C — creators stay real agents; never Option B**

Generation `g+1` creators keep the `HumanCreator` POMDP machinery, with
`C = log(A_learned[:, CREATOR, goal, DEEP])`. Preserves §4.2's load-bearing property that
human artifacts are produced by a reward-optimising *policy*, not sampled from a
distribution. Reduces to the direct-emission reading at `γ=1`, by the same identity as D4.

**Explicitly rejected:** marginalising the learned `A` over `C_recovered` (Option B). It
blurs goal-specific content even at `f = 0`, which would make N11 fail by construction and
turn every E8 result into an implementation artifact.

**Obligation.** N11 (zero-contamination recursion shows zero degradation) is the acceptance
test for the generation loop and gates E8.
