# Source mixture: frozen implementation choices before native outcomes

This implements the prepared source-uncertainty protocol using every available
retained posterior, all eight laws, both posterior source conditions and both
observation draws. True retained source identities select the reports in both
conditions; this is evaluator information, not restored provenance supplied to a
reader. Repeated copies of one source count once. Every original source time,
context and endpoint remains bound to the saved stream.

For each source, the new report copies its endpoint with probability 0, 0.25,
0.5, 0.75 or 1, otherwise drawing an independent endpoint from the same past
maker state. The always-independent and always-copy rivals receive the same
posterior and supplied law. The report distribution is the declared mixture.
All eight endpoints are enumerated, including impossible outcomes and rival
support failures. No model is fit and no data or seed is added.

The complete likelihood factors determine all updates. At an endpoint different
from the copied one, every possible correct update equals the independent
update, while the copy rival has zero support. At the copied endpoint, the correct
posterior is a convex combination of the old posterior and the independent
update. The copied fraction is alpha divided by the mixture report probability.
This identity permits exact reuse across all five settings without sampling an
endpoint or dropping a forecast. A sparse schedule-increment operator evaluates
every remaining-time/four-context/eight-endpoint forecast difference between the
independent and copy posteriors; scalar controls expand all updates directly.

Raw factors retain source identities, independent and correct report probabilities,
support masks, mixture coefficients, future-group total variation, maximum future
coordinate difference and mean future total variation. Full posterior weights,
laws and schedules remain frozen inputs: together they reconstruct every report
posterior and forecast. Future errors are measured here, unlike the preceding
single-report sufficiency bound. Float summation uses the existing 1e-12 tolerance.

For each rival, report support-failure probability separately from error mass
on supported reports. Conditional supported error requires division by supported
mass; impossible rival cases are never renormalized away into an unconditional
finite score. Average sources within each posterior first, then retain every
128-row draw stratum and pair both draws within each law. Repeated source queries
are not independent laws. No practical-success threshold is added after outcomes.

The prospective inclusive cap is 3,600 CPU seconds including tests, failures,
original execution, both complete replays, independent review and final regroup.
Admission requires a complete structural synthetic benchmark before native outcome
consumption. The 48 unavailable checkpoint strata remain unavailable. No protected
lineage or shared tiny setting is consumed. Two-report dependence and bounded
provenance intervals remain independently prepared alternatives.
