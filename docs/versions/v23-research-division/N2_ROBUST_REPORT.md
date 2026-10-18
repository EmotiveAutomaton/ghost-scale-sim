# D10: robust timing-weight reanalysis

Does timing still help after generic tempering and robust pooling? In D10, 43 of 120 nonstructural settings (35.8%) retain observed gains above 0.02 nats over a validation-tuned tempered reader. The variance-floored pooled gain is 0.00203 nats. All 253,952 retained cases, 248 intervals and full replay verify. This is a descriptive constructed-method result; N2's frozen stop remains unchanged.

Generic tempering absorbs part of the timing reader's apparent benefit. It does
not remove the heterogeneous pattern: observed per-setting gains range from
-3.45213 to 12.44301 nats. These extremes are descriptive, selected after
discovery, and cannot support a universal timing benefit.

The table compares original naive and validation-tempered baselines against
the same retained timing-weighted reader. “Above margin” counts equal-weight
32-world point estimates exceeding .02 nats; the pooled mean uses inverse
variance with the across-setting median as a floor.

| Baseline | Above margin, of 120 | Variance-floored pooled gain (nats) | Largest normalized setting weight |
|---|---:|---:|---:|
| Original naive | 50 | 0.00350773 | 1.4771% |
| Generic tempered | 43 | 0.00203093 | 1.4865% |

The earlier N2 report's 48/120 count uses inverse-variance combination of the
two burn-ins. D10 instead gives all 32 complete worlds equal weight before
the across-setting robust pool, yielding 50/120 on the original comparison.
This declared descriptive aggregation change does not revise the frozen card.
Sixty of 120 variances are floored in each comparison; their medians are
9.81804e-6 and 5.26860e-6. No tiny-error cell can dominate the weighted summary.

Each setting chooses one fixed exponent from 1/.75/.5 using only its 33 or 34
labels in the retained pre-discovery validation bank. Among 120 nonstructural
settings, 77 choose 1, three choose .75 and 40 choose .5. This small validation
allocation and restricted exponent grid limit the comparator; it is not an
optimized arbitrary calibration family. Neither tuning nor variance flooring
uses new discovery observations, and no scores are clipped or floored.

Complete original and fresh-process replay agree. The producer reconstructs
original naive/timing probabilities, stable losses and published world gains
before scoring the new comparator. A separate reviewer independently rebuilds
all case losses, identities, coverage, 248 whole-world intervals, variance-floor
weights, pooled means and the 43-setting count from retained rows. All four
structural controls remain visible and excluded from the 120-setting count.
The count is not a formal partial-conjunction test or multiplicity-corrected
claim. Constructed-method evidence does not establish human behavior.

[Protocol](N2_ROBUST_PROTOCOL.md), [complete summary](../../../results/v23/n2-robust-review/SUMMARY.json),
[all-setting curve](../../../results/v23/n2-robust-review/SPECIFICATION_CURVE.csv),
[independent review](../../../results/v23/n2-robust-review/REVIEW.json),
[provenance](../../../results/v23/n2-robust-review/PROVENANCE.json).
