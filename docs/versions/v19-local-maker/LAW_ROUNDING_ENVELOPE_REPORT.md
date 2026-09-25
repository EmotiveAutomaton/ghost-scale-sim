# Supplied-law rounding envelopes — 25 September 2026

We tested how much probability error comes from storing a supplied law at lower precision. Across eight retained laws, half-precision storage uses 1,024 instead of 4,096 bytes and bounds every one-step probability error by 0.00012148, with no positive support lost. Independent scalar reconstruction, all 192 strata and both complete replays verify this constructed-method ruler. It does not establish accumulated inference accuracy, learned access, historical process correspondence or human intent.

The comparison reuses all eight development laws, sixteen maker states, four
query contexts and eight endpoints. Three storage precisions cross direct casts
and one float64 row normalization. Every normalized nonnegative mixture of maker
states lies within the retained coordinate extrema; convexity bounds its sum of
squared forecast errors by the largest squared law-row error. This is a finite
algebraic bound over all state mixtures, without a new population inference.

The table gives storage for one full law, the largest absolute probability error
over all retained laws/states/contexts/endpoints, and the largest squared-error
bound. The final column averages each law/context's squared bound equally over
eight laws and four contexts. Direct forecasts may have nonunit mass; squared
forecast error and expected one-hot loss were checked separately.

| Stored precision and reconstruction | Law bytes | Largest probability error | Largest squared bound | Mean squared bound |
| --- | ---: | ---: | ---: | ---: |
| Float64, direct | 4,096 | 0 | 0 | 0 |
| Float64, normalized once | 4,096 | 5.55112e-17 | 1.27111e-32 | 7.36743e-33 |
| Float32, direct | 2,048 | 1.46717e-8 | 4.68135e-16 | 3.00716e-16 |
| Float32, normalized once | 2,048 | 1.40056e-8 | 3.56018e-16 | 2.27200e-16 |
| Float16, direct | 1,024 | 0.00012147624 | 3.04410e-8 | 1.90367e-8 |
| Float16, normalized once | 1,024 | 0.00012147624 | 3.04410e-8 | 1.42230e-8 |

No cast removes positive support in these laws. Half-precision direct row-mass
drift reaches 0.0002746582; normalization reduces that maximum to 1.11023e-16.
Normalization lowers the mean squared bound but leaves its overall maximum
unchanged. No practical equivalence threshold was specified for this ruler.
The float64 normalized difference records existing floating row-sum roundoff,
not mathematical ambiguity. Empty positive rows would fail without a floor.

**Validation.** The 23 pre-admission controls include exact/deterministic and rare
endpoints, permutation and corrupted-support checks, and deliberately undersized
bounds. The three executions match every scientific output byte. The independent
audit uses scalar binary casts, scalar coordinate extrema and all attaining
states, exact support masks, and 65-digit explicit expected one-hot losses.
Its largest loss discrepancy is 2.23e-20, below its stated 5e-19 absolute tolerance.
Scalar squared sums use relative tolerance 2e-15 and absolute tolerance 1e-45;
support and coordinate-extremum masks compare exactly. Row normalization is
also reconstructed in the original float64 convention; scalar row sums provide
a separate rounding check. All 192 original law/precision/mode/context rows and
24 equally weighted law cells reproduce. The 377 vertex, rational two-state and
uniform-mixture checks per variant accompany the convex bound.

These checks do not show that an extremal mixture is reachable in an admitted
history, quantify realized posterior error, or verify repeated likelihood
updates. Eight reused laws do not establish a universal numerical guarantee.
No reader receives new evidence; laws, states, raw arrays and attaining masks
remain scientific/evaluator material. No new fit, observation or protected
lineage was used. **Warrant:** exact finite supplied-law ruler with recorded
floating tolerances. **Pursuit:** continue the independent memory-allocation
comparison and prepare the relative-likelihood accumulation diagnostic.

[Frozen protocol](LAW_ROUNDING_ENVELOPE_PROTOCOL.md).
[Complete numerical review](../../../results/v19/G19-B-law-rounding-envelope-1/NUMERICAL_REVIEW.json).
[Scientific export](../../../results/v19/G19-B-law-rounding-envelope-1/EXPORT_MANIFEST.json).
[Allocation contract](SOURCE_MASS_BYTE_IMPLEMENTATION.md).
[Likelihood alternative](LAW_LIKELIHOOD_ENVELOPE_PROTOCOL.md).

[Independent robust-source alternative](SOURCE_PRIOR_ROBUST_MASS_PROTOCOL.md) is prepared before allocation outcomes are consumed; it optimizes worst retained mass only within a frozen finite candidate library.
