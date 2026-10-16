# V23 implementation review — October 14, 2026

**Disposition: changes needed before scientific admission.** The scientific
questions and broad sequence are implementable. The proposed confirmation score
does not control false discoveries for the stated raw log-loss gain. The required
ToMpathy schema is also not implemented in the inspected checkout. This document
proposes corrections; it does not change the original specification or accept
new scientific results.

## 1. Correct the confirmation statistic

The proposed operation replaces a world gain below -0.1 by -0.1. This raises its
mean and can produce apparent evidence under a true null. An exact counterexample
uses margin 0.02, gain -1 with probability 0.1 and gain 2/15 with probability 0.9.
The raw mean is exactly 0.02. Flooring makes it 0.11. The proposed half-maximum
bet is 25/6, so the expected one-world factor is **11/8 = 1.375**, already greater
than one. Independent repetition compounds the error. An observed discovery
minimum does not bound future gains.

The e-LOND formula itself is correct: at nomination j use
`alpha_j = 0.10 * (1 / (j * (j + 1))) * (D_previous + 1)` and reject when the
valid e-value is at least `1 / alpha_j`. Its guarantee requires valid input
e-values; it cannot repair upward-biased inputs. See Xu and Ramdas,
[Online multiple testing with e-values](https://proceedings.mlr.press/v238/xu24a.html),
definition 1 and theorem 1.

**Proposed replacement.** Retain the original, uncapped paired gain and margin.
For each frozen comparison, certify an actual support lower bound L from the
generator and readers before reserved-world exposure. Finite supports and
strictly positive predictive probabilities can supply such a bound; an empirical
minimum cannot. Preserve genuine infinite losses; a comparison without the
required finite bound is confirmation-unavailable, not a failed scientific
prediction. A violation of a certified support bound is an instrument failure.
Use nonnegative predictable bets no
larger than `1 / (2 * (m - L))` where L < m, and document the sampling assumptions
under which each null conditional mean is at most m. If no valid bound and
sampling argument are available, leave confirmation unavailable; do not claim a
bounded-score test confirms the uncapped quantity. A very loose bound may have
low power, which must remain visible.

The world sampler must be part of that proof. Requiring 16 distinct laws by
sampling without replacement does not automatically meet an independent-world
betting argument. Either derive the appropriate without-replacement method or
use independent complete, balanced law blocks as the betting units and cost them
in pilots. Do not silently reinterpret 128 worlds as 128 independent blocks.
The nomination order is frozen from discovery evidence before any confirmation
outcomes; D counts earlier completed confirmations only. Confirmation revisits
the nominated primary reader pair. The newly refitted direct rival is a separate,
predeclared comparison, not a replacement hypothesis after nomination.

## 2. Pin the actual canonical ledger

At inspection, ToMpathy HEAD was `a4dae2479b5f87c03000d5a4c62690eaad0c5d70` on
`codex/0.2-development`; `engine/tompathy_engine/ledger/schema.py` was absent.
Its October 14 engine specification describes future implementation. This is a
dependency, not an import path that Ghost can use today.

The two prose tables also disagree. The ToMpathy specification includes Revision,
Evidence and Provenance; `source_dataset`, content references, correction revision
links, context session links and revision-scoped stated goals are absent or named
differently in Ghost's abbreviated table. Ghost must implement adapters against
the actual canonical schema, not fork a second schema from its summary table.

**Proposed resolution.** Keep ownership in ToMpathy. Once its schema is available,
record its commit, schema version and byte hash; document the mapping of every
family, using all required canonical types and provenance. Freeze a read-only
dependency snapshot for the campaign. Keep truth sidecars inaccessible to readers,
and explicitly whitelist prediction-time fields: an outcome field or future
correction can leak the target even when a random canary is absent. Independent
canaries, identifier permutation, time masking and native/ledger prediction
agreement are complementary checks. JSONL transport must have an agreed envelope;
a header is not an undeclared schema row. Real CoAuthor smoke execution remains
Sounding Line's responsibility; Ghost retains its receipt, not private human text.

## 3. Separate controls from hypotheses

Some proposed known answers are plausible empirical predictions rather than
identities. They must not be used to disqualify contrary valid results.

- **N1:** a population with one true type can still benefit from pooling while
  individual histories are short. Use a matched-information identity as the
  no-pooling-gain instrument control; retain the one-type setting as an experiment.
  Treat convergence within 0.02 at 512 records as a prediction, not guaranteed
  instrument correctness. Define the drift target (current versus historical
  type) and condition alias claims on the full observation law and prior.
- **N3:** finite-temperature softmax corrections can decrease utility; they do
  not imply a hard half-space constraint on preference. Keep the half-space
  reader as a deliberately misspecified competitor and use the exact noisy
  correction likelihood as the reference. Add the stated all-local-fix negative
  control explicitly; local-fix share 1 is absent from the proposed main grid.
- **N4:** recipe error below 0.05 at 512 records is an empirical prediction.
  A finite enumerated or supplied-truth fixture can validate reconstruction
  without declaring this performance target an instrument gate.
- **N6:** zero stakes changes payoff, not necessarily warmth/competence forecast
  information. A payoff null must test payoff. Validate the forecasting null by
  making the diagnostic observations conditionally identical.
- **R2:** the claimed conditional accuracy identity requires proof for the full
  sequential policy, including later contradictory evidence and release. Until
  proved, retain it as a falsifiable prediction rather than a known-answer gate.

These changes preserve the original predicted outcomes on the family reports.
They do not remove losing conditions or weaken a scientific criterion to pass.

## 4. Freeze the adaptive design completely

The seven-batch budget and the stated sequence 16, 48, 144 worlds leave later
rungs unspecified. A proposed bounded schedule is **16, 16, 48, 144, 144, 144,
144 worlds per active variant**, halving the eligible survivors with upward
rounding from batch 3. Predeclared late-effect variants remain exempt through
batch 4; their additional cost must be included. Freeze eligibility, ties,
protected variants and complete rung rules before calibration, not after outcomes.

Calibrate the full policy, including selection, parked variants, slow starters,
and dependence caused by shared worlds or training. Use distinct tuning and
validation seeds, state the number of simulations and report binomial uncertainty
for both target error rates. Freeze the permitted prior-scale grid in advance.
Failure to meet targets produces an unresolved calibration, not an unreported
search for another threshold. The posterior probability for the selected
highest-mean variant is not the probability that any variant exceeds the margin.

Sounding Line's companion currently uses a different nomination rule (probability
of positive gain above 0.975) and a different calibration alternative. A Ghost
calibration does not validate that rule automatically. Evaluate the separately
specified rule or identify the bridge as uncalibrated; do not silently harmonize
the two projects. Grid quadrature needs convergence checks; a finite numerical
grid is an approximation, even for an analytically specified model.

## 5. Bound the complete program before admission

The core variant counts sum to **1,210**, including optional N4b but excluding
N8. N8 adds 24, for **1,234**, before twin grids and extra controls. The individual
budget caps sum to 130 CPU hours; the 90-hour program ceiling binds. The original
priority order determines deferrals, with all setup, failure, model fit, replay,
service and verification costs charged and 18 hours protected. Pilots must measure
the largest latent supports and history cells; V22's case rate alone is not a
reliable forecast for these worlds.

N1's precision-extension loop requires a finite cap and a recorded budget stop.
Twins require bounded grids and authenticated summary sources before claims about
realistic shapes. Full chess trees at branching five and depth ten can exceed
twelve million nodes per tree; exact enumeration must be piloted before admission.
Only derived CoAuthor summaries may cross into this public simulator repository.
Every optional or budget-deferred cell remains visible on the coverage board.

## Launch order and owner decision

The [work order](WORK_ORDER.json) retains the specified sequence. No worker may
start merely because V21 is idle: require its published final report, hash-bound
closure and native absence of scientific workers. Day 0 is the actual report
publication time, not an estimated finish time. A delayed external dependency
does not reset the seven-day limit without a separately recorded amendment.

**Decision requested:** adopt the corrections above, preserving the original
scientific questions and uncapped primary gains, then build and admit V23 only
after V21 closure, the canonical schema pin and the corrected validity gates.
Until resolved, this is a prepared implementation order, not an executable queue.
