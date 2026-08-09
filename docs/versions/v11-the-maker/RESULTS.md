# V11 — The Maker. Results

**Written after the runs, 2026-08-08.** Spec beside this file, written first; criteria
hash-locked in `ghostscale/prereg_v11.py` before any run; verdicts under
`results/validation/soundingline/` (`s12_three_locus.json`, `s14_aperture.json`,
`s15_convergence.json`) and the lock under `results/v11/`. Full test suite green over the
committed tree.

## The one-paragraph version

The world gained a persistent maker — a value profile over the goal channels, fixed for life,
drawn from a named six-member family — and the first three experiments it unblocks all returned
answers. **Recovery of the profile converges with more artifacts and its residual is small under
the theory's assumptions (S-15)**; removing the bounded family costs 0.24 L1, which is the first
measured price of the convergent-midbrains assumption; removing the reader's expertise cost
nothing at this observation length, and that half of the criterion **fails as pre-registered**.
**An absent drive is recoverable — under commission, and only through how the goal is pursued
(S-14)**, which gives made-under-duress its first working mechanism. And **the field's
depth-profile instrument smears a three-locus structure into one mid peak in every run, while
the residual instrument the theory proposed separates the worlds at AUC 0.87 (S-12)**.

## S-15 — the convergence curve

| criterion | locked bar | outcome |
|---|---|---|
| C1 convergence | acc(1) ≤ 0.60, acc(50) ≥ 0.95, Spearman ≥ 0.90 | **PASS** — 0.533 → 0.983, Spearman 1.00 |
| C2 assumption price | bounded beats unbounded AND wrong-expertise by ≥ 0.05 L1 | **FAIL** — unbounded margin **0.240**; wrong-expertise margin **0.0003** |
| C3 construction gap | B minus A at n = 1 ≥ 0.25 | **PASS** — 0.967 − 0.533 = **0.433** |
| C4 corpus price | report n\* at 90%, CURATOR tier, inexpert reader | **n\* = 20** artifacts per maker |

**C2 is reported as failing and the original decides.** The bounded-family half behaved exactly
as predicted: with 64 random profiles replacing the six-member family, the L1 asymptote rises
0.009 → 0.249. The expertise half did not: a reader with signatures corrupted halfway to random
(d = 0.5) recovers profiles as well as the expert (0.0096 vs 0.0093). Diagnosis, recorded rather
than acted on: construction A routes all profile evidence through goal identification, and at 24
observations per artifact goal identification saturates through a half-corrupted template — the
same goal-side ceiling T-1's flagship cell sits at. **The expertise assumption's price is
invisible at this observation length, not shown to be zero**; the shorter-artifact restatement
belongs with T-11's off-ceiling work and is queued. Until it runs, C2 may be quoted only as
*half-established: the family is priced, expertise is not.*

**The construction discrimination is the sharpest result in the version.** Under conjunctive
satisfaction (G54's account — every emission partially satisfies all channels) one artifact
identifies the profile at 0.97. Under amplification (§3's account — one goal drawn per occasion
from the standing weighting) one artifact reaches 0.53 and fifty are needed. The sibling's
real-text record — single-artifact values attempts all failed, within-maker multi-work designs
worked, no stacking acceleration — is the amplification signature. A simulation cannot say which
account is true of people; it can say the record already on file is what one of them predicts
and the other forbids.

Gates: all four green, including the shuffled-maker null landing at exactly chance (0.167
against 1/6) and the all-uniform placebo world reading uniform.

## S-14 — the aperture

| criterion | locked bar | outcome |
|---|---|---|
| C5 aperture | commissioned ≥ 0.85 AND commissioned − spontaneous ≥ 0.20 | **PASS** — 1.00 commissioned, 0.61 spontaneous, gap 0.39 |
| C6 how-channel | λ = 1 arm collapses to ≤ 0.60 | **PASS** — exactly 0.50 |

A hard-zero channel against an ε-trace of the same channel: nearly invisible in spontaneous
standing work, perfectly separable under commission toward the missing channel — because
instruction amplifies multiplicatively and a zero cannot be amplified, so the masked maker
routes the pursuit through substitute drives and the routing is what the reader reads. Strip the
pursuit channel (pure compliance) and discrimination collapses to exact chance: the discriminator
is *how the goal is pursued* and nothing else. Gates green, including present-vs-masked at 1.00
(the easy known answer is easy).

## S-12 — the smear, and the residual instrument

| criterion | locked bar | outcome |
|---|---|---|
| C7 smear | three-locus mean profile unimodal in ≥ 80% of reps | **PASS** — 100% |
| C8 separation | residual statistic separates the worlds at AUC ≥ 0.80 | **PASS** — 0.87 |

The position-averaging instrument every published depth profile uses cannot see three loci whose
positions vary by unit; the residual instrument the theory folder proposed (fit each unit its own
single peak, correlate early-third and late-third residuals across units) can. **The severity
rider bounds both:** the smear reproduces in 100% of twenty random re-parameterisations —
architectural, which for C7 is the point (any variable-locus world smears, so the field's
mid-peak consensus is uninformative against a three-locus truth) — while the separation
reproduces in only **25%**, so the residual instrument works where the shared early–late gain is
strong relative to noise and is not a free detector. Its real-model deployment should be
calibrated against that.

## Deviations

**D-V11-1.** S-12's identity arm (well-separated loci must read trimodal) failed its own gate at
0.0 on first construction: the arm kept the default middle bump (amplitude 2.0, width 4), whose
shoulder outgrows the side peaks, so even fixed loci read unimodal. The arm was rebuilt with
matched narrow bumps and passes at 1.00. The locked criteria C7/C8 were not touched, and the
first construction's failure is retained in the verdict (`identity_arm_note`) because it is
independent evidence for the smear mechanism: amplitude alone can produce it.

No other criterion, threshold, or arm changed after any result was seen. C2 fails and stands.

## Repairs shipped with this version (SPEC §5)

- **V5-5 logged and pinned.** Goals 0 and 3 share an identical execution chain in the closed V5
  builder (a pigeonhole fact of cyclic steps); deviation logged in the V5 results, a one-line
  caveat added to the README limits, `build_subgoal_chains_v5b` added for new work (derangement
  successors disjoint from the emission derangements, pairwise distinctness asserted), and
  `tests/test_v5b_chains.py` pins the original collision so it cannot be silently "fixed" in the
  closed record.
- **T-10's self-comparing positive gate replaced** with a planted-seam liveness check; re-run,
  passes at 0.141 against a 0.02 bar, and the re-run reproduced the committed numbers to the
  digit.
- **S-1 and S-6's gate exemption ended.** S-1 now gates its harness reproducing E36's phenomenon
  (μ=1 gain zero at 0.017 within 0.02; μ=3 gain alive at 0.054) — the package's
  reproduce-before-you-audit rule, formalised. S-6 gates its synthetic creator's exactly-flat
  surface (the placebo its construction guarantees) and its budget actually reaching the emitter.
  Both re-runs reproduce the committed verdicts.
- **The miniature rule** added to CLAUDE.md: purpose-built miniatures carry their own random-draw
  severity check or the *architecture untested* marker. S-12 is the template; S-14 and S-15 carry
  the marker pending SV-T.

## What this version leaves open, in order

1. **The expertise price at honest difficulty** — C2's failed half, jointly with T-11's
   off-ceiling restatement of the triangle. One harness, two debts.
2. **T-12, the missing supply arms** (values — G47, mechanics — G56), now askable for the first
   time.
3. **T-13, the residue duel and domains** — the habit channel is still unbuilt, and the
   values-in-the-residue account remains untouched by V11.
4. T-14 (bard/concealer), T-15 (flattened intent), AL-6, and SV-T over the T/S layer including
   the two new marker-carrying modules.
