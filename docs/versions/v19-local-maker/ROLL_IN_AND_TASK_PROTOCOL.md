# Roll-in training and objective-dependent artifact assessment

These successors implement the two previously prepared alternatives. They use the
existing single-worker queue, no tiny-model setting and no reserved test or
confirmation lineage. Their source, inputs, schedules and caps freeze before new
outcomes. Earlier results and source snapshots stay unchanged.

## B2: one fixed roll-in pass

B1's independent-source free-running forecasts have approximately 0.024 nats more
loss than the exact-previous-bank diagnostic. This can reflect input distribution,
initialization and teacher privilege; it does not by itself establish increasing
drift with length. B2 tests whether a single training pass on self-generated states
reduces that gap. No longer horizon is silently added.

Reuse all B1 streams, endpoint teachers, fixed input transformations, training
draws, feature seeds and initial mean bank. Fit the original ridge head once on
new-source prefixes with the exact preceding bank. Roll that head through the
training streams from the common mean bank, skipping copied sources. Record the
bank immediately before every observation. Fit one new head to those inputs and
the identical target distributions at identical new-source prefixes. Separately
refit the exact-teacher head as a two-pass computation control. No mixing, iteration,
checkpoint selection or development-data selection. Three fits per paired draw
and feature seed are charged: shared initial fit, roll-in fit and control refit.

Each head has 129 coefficients for each of 32 output categories. The fixed
128-coordinate tanh input map, ridge penalty 0.01, unpenalized intercept and fixed
probability repair are unchanged. Save all heads, training roll-in inputs,
free-running and teacher-forced forecasts and invalid raw probability flags.
The first head must reproduce B1 on retained inputs. A duplicated teacher refit
must be identical; a failure is an instrument failure, not a scientific difference.

Evaluate all heads on paired independent/copy streams at 8 and 32 observations.
Retain B1's exact-teacher diagnostic definition: at a duplicate it passes through
the exact preceding bank. Such positions are oracle identity checks, not learned
update scores. Report independent-source gaps separately. Same target labels do
not mean equal input privilege. The rolled-in head changes the training-input
distribution and adds a fit; comparisons include the equal-pass exact-teacher
control and explicit sunk preparation costs. No joint historical-process claim.

Population: B1's 32 training and eight development lineages, two draws, two fixed
feature seeds, eight streams per lineage and condition. Paired lineage uncertainty
conditions on those fits. The seven-CPU-hour B2 cap includes checks and both full
replays. A later 128-observation or reset comparison requires a new frozen stream
manifest; it is not part of this packet.

## E2: task assessment and hidden dependency

Reuse the eight admitted local-world trajectory blocks with complete probability
weights; do not sample or change governing/local goals. For every governing-purpose
and maker-skill pair, marginalize the two belief-error and two routine settings
equally. Functional success requires claim-on and evidence matching that claim;
presentation success requires display-on. Report both, joint success and their
disagreement mass for every stratum. These are declared evaluative functions.

Compare intact final artifact versus a projection hiding the evidence unit, crossed
with context absent/present. Context supplies the initial artifact and requested
purpose, not the actual adopted purpose. The context's initial evidence value
remains visible when present; degradation hides only the final evidence unit.
Exact conditional inference and the uniform compatible-trajectory baseline receive
the identical visible projection. Neither receives the true skill or purpose.
The exact reference additionally knows the supplied generative distribution; the
uniform template baseline has the same legal candidate family, different weights.

Compute governing-purpose logarithmic loss, squared probability error, true-purpose
candidate coverage and ambiguity abstention (more than one purpose in support).
All are expectation values over the complete trajectory distribution. Preserve
every visible-group denominator and posterior in evaluator evidence. Reader export
contains only deduplicated visible packets, content hashes, passage anchors and
declared domain semantics, without lineage/maker identifiers or posterior truth.

Known-answer controls include a no-information purpose fixture, an identifiable
purpose fixture, projection coarsening consistency, full mass, and the fact that
hiding final evidence cannot lower exact expected logarithmic loss within either
context condition. This latter relation is population-level; it need not hold for
each individual maker-purpose stratum. The four-CPU-hour E2 cap includes checks
and both full replays. It is an exact method/mechanism diagnostic, not a learned
reader production-skill or practice-benefit test. A reversal supports the prepared
new-dependency discriminator only within these stipulated objectives.
