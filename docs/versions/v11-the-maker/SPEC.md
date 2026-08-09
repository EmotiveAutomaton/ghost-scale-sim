# V11 — The Maker. Specification

**Written before the code, 2026-08-08. Not edited after.** Results go to `RESULTS.md` beside this
file when the runs complete; deviations are logged there with originals retained.

## Why this version exists

Three sources converged on one missing object. T-6 returned VOID on the values vertex — *"the
model could not represent a quantity that is only defined across artifacts."* T-1's committed
verdict carries a SUBSTITUTION field: the triangle it measured was goal–process–**depth**, and
*"the values vertex has to be BUILT before T-1 can be asked as posed."* And batch four from the
sibling project (S-12 through S-15) requests exactly that build, with the fidelity list naming a
values factor that is not a coarsening of the goal, a drive-availability mask, and a stage
structure. Every version to ten varied the reader; the maker stayed a per-artifact goal draw.
**V11 gives the world a persistent maker** and runs the first three of the experiments that
unblocks.

**Scope of this build.** The world (§1), three modules — **S-15** (convergence), **S-14**
(aperture), **S-12** (three-locus smear, standalone) — and four repairs (§5). Queued and NOT in
this build, listed so their absence is a decision rather than an accident: T-11 (off-ceiling
restatement of T-1), T-12 (the missing supply arms, G47/G56), T-13 (the habit-residue estimator
duel and the domain-change separator — the habit channel and domains are **not** built here),
T-14 (bard/concealer), T-15 (flattened intent), AL-6 (anti-capture toy), SV-T (severity mini-pass
over the T/S layer, which the new modules are owed as much as the old ones).

**Method, not mechanism, throughout.** Values are built into this world, so every recovery result
is about *identifiability, sample complexity, and estimator choice given existence* — never
evidence that human values are like this. Each module's verdict carries a
`what_must_hold_in_the_real_environment` field, per batch four's standing demand. The one partial
exception is S-15's construction contrast, whose two arms make predictions that land on
measurements the sibling has already taken on real text.

## §1. The world — `ghostscale/v11/maker.py`

Base objects are V1's: the four goal signatures over eight features (`build_goal_signatures`),
the frozen synthetic distribution, the published tier transmission α (CREATOR 1.0, POLISHED 0.95,
CURATOR 0.6, GHOST 0.05). Labels are honest everywhere in V11 — provenance deception is not this
version's axis.

**A maker is a value profile `w` over the four goal channels, fixed for life**, drawn from a
named finite family. The finite family is deliberate and load-bearing: the theory's answer to the
impossibility results is that human empathy works inside a bounded, shared hypothesis space
(*"convergent midbrains"*), and the family IS that bound, made literal. Removing it is an
experimental arm, not a fix.

    THE PROFILE FAMILY (K = 6)
      uniform      [.25 .25 .25 .25]
      peaked-k     0.70 on goal k, 0.10 elsewhere        (four of these, k = 0..3)
      bimodal      [.40 .40 .10 .10]

**Two constructions of what a profile does, built as two emitters, because the theory holds both
and they cannot both be right** (triple inference §3 vs G54):

- **Construction A — amplification.** Per artifact one goal is drawn, `g ~ Cat(w)` — attention
  amplifying one component for the occasion — and the artifact emits V1's tier mixture
  `α·sig[g] + (1−α)·synth` for 24 steps. The profile touches the surface **only through which
  goals get drawn**, so it is visible only across artifacts.
- **Construction B — conjunctive satisfaction.** Every emission partially satisfies every
  unmasked channel at once: the emission distribution is the weighted geometric mean
  `poe(w) ∝ exp(Σ_g w[g]·log sig[g])`, mixed through the tier channel the same way. The profile
  is in every marginal, so it is visible within one artifact.

**The reader** is the exact-inference reader over the same generative objects, forced deep — the
closed-form Bayes the static factors admit, which V-1 established as the reference solver. Its
per-artifact goal likelihood is `Π_t (α·sig_r[g] + (1−α)·synth)[o_t]`, with `sig_r` the reader's
own signature (expert `d = 0` reads `sig_true`; inexpert readers use `build_observer_signature`
at `d > 0`). Across artifacts it carries an explicit posterior over the profile family:
`P(w | a₁..aₙ) ∝ P(w)·Π_a Σ_g w[g]·L_a(g)` under A, and the direct emission likelihood under B.
This omits the attention policy and the metabolic budget deliberately: for an identifiability
question the forced-deep exact reader is the **ceiling**, so every error floor reported here is a
lower bound on what a costlier reader would leave. That framing is part of the claim, not a
caveat discovered later.

**The drive mask (S-14).** A masked channel is absent, not merely down-weighted, and the
distinction is enforced by the theory's own mechanism: **instruction amplifies multiplicatively**.
An artifact commissioned toward goal k is produced under
`w' = normalize(w ⊙ exp(A_amp·e_k))` — attention can only amplify a component that exists, so
`w[k] = 0` makes the amplification a no-op and the maker routes around the absence, while a
present-but-unused trace (`w[k] = ε`) is amplified to dominance. Commissioned emissions carry a
**compliance channel**: `λ·sig[k] + (1−λ)·poe(w')` — both maker types deliver the commissioned
surface; they differ only in how the rest is pursued. Constants, fixed here before any run:
`ε = 0.02`, `A_amp = 4.0`, `λ = 0.5`.

Seeds derive from `zlib.crc32` of named strings (house rule); the V11 offset is 1,100,000.

## §2. S-15 — does recovery error shrink with more artifacts, and what is left at the end?

*Batch four's S-15, the project's disagreement with the impossibility literature made runnable.
The theorems say a reward is not uniquely identifiable and the ambiguity survives infinite data.
The claim here is about a convergence rate and a residual, which the theorems constrain and
nobody has measured.*

**Design.** 60 makers (10 per profile), 50 artifacts each, construction A unless stated.
Recovery error is read at n ∈ {1, 2, 3, 5, 8, 12, 20, 30, 50} artifacts (prefixes of one stream,
so the curve is paired). Two readouts: **accuracy** (argmax profile equals truth) and **L1**
(‖posterior-mean profile − w‖₁, defined in every arm including those where the truth is outside
the reader's family).

    ARMS
      bounded_clean       CREATOR tier, expert reader, the 6-profile family.  The theory's case
      bounded_noisy       CURATOR tier (α=0.6), inexpert reader d=0.25.       The corpus-pricing case
      unbounded_family    as bounded_clean, but the reader's family is 64 random Dirichlet(1)
                          profiles, truth not included.                       Convergent midbrains removed
      wrong_expertise     as bounded_clean, reader at d=0.5.                  Known transition model removed
      construction_B      B-world, CREATOR tier, expert reader, 6 profiles.   The discriminating arm
      shuffled            bounded_clean with artifacts permuted across makers. The null

**Pre-registered criteria** (executable in `prereg_v11.py`, hash-locked before any run):

- **C1, convergence.** In `bounded_clean`: accuracy at n=1 ≤ 0.60, accuracy at n=50 ≥ 0.95, and
  Spearman ρ(accuracy, n) ≥ 0.90 over the grid. *If accuracy is flat in n, the theorems bite at
  practical scale and the project must say so.*
- **C2, the price of the assumptions.** L1 at n=50: `bounded_clean` beats `unbounded_family` and
  beats `wrong_expertise`, each by ≥ 0.05. *The gap is the measured value of the bounded family
  and of expertise — the two assumptions the theory shares with the proofs' requirements.*
- **C3, construction discrimination.** Accuracy at n=1: `construction_B` − `bounded_clean`
  ≥ 0.25. *Under B one artifact carries the profile; under A it cannot. The sibling's real-text
  record (every single-artifact values attempt failed; every within-maker multi-work design
  worked; no acceleration at stacked motivations) sides with A, and this arm turns that record
  into a discriminating measurement.*
- **C4, the corpus price.** Report n* = the smallest n with accuracy ≥ 0.90 in `bounded_noisy`.
  Reported, not thresholded: it is the power analysis for the sibling's follower-corpus design.

**Gates.** Positive: `bounded_clean` accuracy at n=50 within 0.05 of 1.0. No-oracle:
`shuffled` accuracy at n=50 within 0.12 of chance (1/6). Positive: a 12-maker all-uniform world
reads uniform at n=50 (accuracy ≥ 0.95 → within 0.05 of 1.0). Live: the C3 gap ≥ 0.25 (the
construction manipulation must reach the measurement).

**What must hold in the real environment:** a maker's profile is stable on the timescale of the
works sampled, and the reader's channel family overlaps the maker's (the convergent-midbrains
premise itself — which is exactly what C2's arms price).

## §3. S-14 — the motivational aperture: is an absent drive recoverable?

*Batch four's S-14. The claim: you can only route attention onto drives you possess, so the drive
a maker lacks constrains what they produce — the absent drive is as informative as the present
one, and it is the only proposal in the program that treats an absence as a measurable.*

**Design.** Per target channel k: makers of three kinds — **masked** (`w[k] = 0`, rest uniform),
**unused** (`w[k] = ε`, rest uniform), **present** (peaked 0.70 on k). Two evidence regimes:
**spontaneous** (standing emissions, `poe(w)`, no commission) and **commissioned** (instructed
toward k, compliance channel as §1). 20 makers per kind per k, 12 artifacts each, 24 steps.
The reader discriminates masked from unused per maker by exact likelihood over the two-hypothesis
family, at n = 1..12 artifacts.

**Pre-registered criteria:**

- **C5, the aperture is revealed by commission.** Commissioned discrimination accuracy
  (masked vs unused, n=12) ≥ 0.85, AND commissioned − spontaneous accuracy ≥ 0.20. *The absence
  is near-invisible in spontaneous work and visible under instruction, because instruction
  amplifies what exists and cannot amplify what does not.*
- **C6, the discriminator is the how-channel.** With λ = 1 (pure compliance, no how-channel),
  commissioned discrimination collapses to ≤ 0.60. *If it does not collapse, the reader is
  reading something other than the pursuit, and the mechanism claim fails.*

**Gates.** Positive: present-vs-masked commissioned discrimination within 0.05 of 1.0 (the
easy known answer must be easy). Placebo: the λ=1 arm's discrimination minus 0.5, tolerance 0.10
(C6 as a gate). Live: commissioned-minus-spontaneous gap ≥ 0.20 (the commission manipulation
reaches the measurement).

**What must hold in the real environment:** commissioned work with a known brief exists and can
be identified; instruction acts multiplicatively on standing drives (the theory's own §3
mechanism); and the compliance/pursuit split is at least partly separable in real artifacts.

## §4. S-12 — does a three-locus structure with a noisy middle read as a single mid peak?

*Batch four's S-12, standalone as it advises — an abstract emitter, no contact with the maker
world.* The published depth-profile instrument averages across units at each position; real units
put their loci at different positions (the sibling's own L14: peak at layer 2 of 29 in one
family, 47 of 49 in another). The question is whether that instrument smears three loci into the
field's consensus mid peak, and whether the residual statistic G22 proposes separates what the
profile cannot.

**Design.** Units are synthetic depth-profiles over 30 positions, 200 units per world, 40
repetitions per world. **Three-locus world:** early bump (position ~U(4,12), width 2, amp 1.0),
middle bump (~U(10,20), width 4, amp 2.0, extra position-noise σ 0.8 — high activity, low
coherence), late bump (~U(18,26), width 2, amp 1.0); **early and late share a per-unit gain**
`g_u ~ LogNormal(0, 0.4)` (the curator's early–late ratio structure); the middle amplitude
`m_u ~ LogNormal(0, 0.8)` is independent; base noise σ 0.25. **One-locus world:** a single
mid bump (~U(12,18), width 5, amp matched to the three-locus mean profile's peak), per-unit gain
`m_u`, matched base noise.

    INSTRUMENT 1  (the field's)   mean profile across units; count modes above 10% prominence
    INSTRUMENT 2  (G22's)         fit each unit a single Gaussian, take residuals; the statistic
                                  is corr across units of (early-third residual mean,
                                  late-third residual mean)

**Pre-registered criteria:**

- **C7, the smear.** The three-locus world's mean profile reads **unimodal** in ≥ 80% of
  repetitions. *Then the field's mid-layer consensus is consistent with a three-locus truth.*
- **C8, the separation.** Instrument 2 separates the two worlds at AUC ≥ 0.80 across
  repetitions. *Then the residual carries what the profile cannot, and the sibling has an
  instrument the field does not.*

**Gates.** Identity: a well-separated arm (position spreads halved, noise halved, loci at fixed
5/15/25) reads **trimodal** in ≥ 80% of repetitions — the instrument sees loci that do not
overlap, so the smear is about overlap, not blindness (observed = trimodal fraction, expected
1.0, tolerance 0.2). Placebo: instrument 2 between two independent batches of the one-locus
world sits at AUC 0.5 ± 0.15.

**Severity rider, complying with the miniature rule this version introduces (§5.4):** 20 random
redraws of the generative constants (bump widths, amplitudes, gains, noises within ×/÷2) report
the fraction reproducing C7 and C8, in the verdict.

**What must hold in the real environment:** per-unit locus positions genuinely vary (L14 says
they do), and the early–late shared-gain structure is the right parametric form for "ratio
variance relationships between early and late" — which is the part a simulation cannot certify.

## §5. Repairs shipped with this version

1. **`build_subgoal_chains_v5b`.** The V5 chain builder assigns goals 0 and 3 the same cyclic
   order (`step = (g % (n_sub−1)) + 1`), contradicting its own docstring — logged as deviation
   **V5-5** in the V5 results document. The v5b builder derives successor maps from derangements
   disjoint from the emission derangements, asserts pairwise-distinct chains, and is used by new
   work only. Closed versions are untouched; a regression test documents the old collision and
   the new builder's properties.
2. **T-10's positive gate.** The committed gate compares a number to itself and cannot fail. It
   is replaced by a live gate — the easiest cells (dwell 4.0, β 1.0) must show travel lift ≥ 0.02
   over the circular null — and T-10 is re-run at its standard scale.
3. **S-1 and S-6 gain gates, and their exemption ends.** S-1: identity (process gain at μ=1 is
   zero by construction, |mean| ≤ 0.02) and live (process gain at μ=3, β=1.0 exceeds 0.02 —
   the harness must reproduce the phenomenon whose statistic it audits). S-6: placebo (the
   synthetic creator's surface trajectory is flat to exact tolerance, which its infinite budget
   guarantees) and live (the practised creator's surface slope is a real decay, |slope| ≥ 0.005).
   Both re-run; `tests/test_gates.py`'s GATED set extends to them and to the three new modules.
4. **The miniature rule**, added to `CLAUDE.md`: a purpose-built miniature ships its own
   random-draw severity check or its verdict carries *"miniature — architecture untested."*
   S-12 complies by construction (§4); S-14 and S-15 run on the shared V11 world and carry the
   marker until SV-T covers them.

## §6. What this version does not claim

The habit residue, domains, and the residue-estimator duel (T-13) are **not built** — the
"values live in the residue of expertise" account is untouched by V11 and remains open. The
off-ceiling restatement of T-1 (T-11) and the missing supply arms (T-12) run on this world in a
later pass. Nothing here reads a label dishonestly, so V11 says nothing new about the trust
exploit. And no result in this version is evidence that human values are profiles over four
channels; the claims are about what a bounded-family reader can and cannot recover from a maker
that has one.
