# Accumulated likelihood-rounding bounds — 25 September 2026

We tested how rounding a supplied observation law can accumulate across repeated observations. Across eight retained laws, half-precision storage admits a worst-case posterior total-variation bound of at most 0.028781 after 128 observations; single precision bounds it by 0.000003798. Independent scalar reconstruction, all 72 law/precision/length strata and both complete replays verify this constructed-method ruler. These are conservative bounds under the same prior and transition law, not attained inference errors; learned access, historical process correspondence and human intent remain unestablished.

Total variation is half the sum of absolute changes in normalized state
probabilities, ranging from zero to one. The result bounds that distance between
inferences using original and rounded observation laws, provided both use the
same prior and transition law. It does not measure an observed posterior error.

All eight retained development laws have sixteen maker states, four query
contexts and eight endpoints. Original and cast rows are normalized once in
double precision, with both normalization changes retained. For each supported
context/endpoint, the instrument bounds the range of log likelihood ratios over
states. The largest range times history length bounds the trajectory log-ratio
range; the posterior bound is the hyperbolic tangent of one quarter of that range.
The extremal two-point derivation and explicit likelihood-product controls are
in the [frozen implementation contract](LAW_LIKELIHOOD_IMPLEMENTATION.md).

The table reports conservative total-variation bounds, not measured errors.
Rows identify storage precision and observation count. Bytes count the shared
law alone, excluding posterior state and schedules. Mean, minimum and maximum
give equal weight to the eight laws; they are descriptive, not confidence limits.

| Law storage | Observations | Bytes | Mean bound | Smallest bound | Largest bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single precision | 8 | 2048 | 0.0000002120 | 0.0000001907 | 0.0000002374 |
| Single precision | 32 | 2048 | 0.0000008480 | 0.0000007629 | 0.0000009495 |
| Single precision | 128 | 2048 | 0.0000033919 | 0.0000030517 | 0.0000037980 |
| Half precision | 8 | 1024 | 0.0016679269 | 0.0015748667 | 0.0017993035 |
| Half precision | 32 | 1024 | 0.0066716140 | 0.0062993886 | 0.0071970974 |
| Half precision | 128 | 1024 | 0.0266804786 | 0.0251925559 | 0.0287809359 |

Double precision uses 4,096 bytes and returns zero relative to its identically
normalized reference. Half precision uses 1,024 bytes; single uses 2,048 bytes.
No positive support is lost in these eight laws. Support-loss fixtures produce
an explicitly unbounded log ratio and the trivial total-variation bound of one;
unreachable observations remain excluded rather than being assigned a floor.

The bound permits the most fragile context/endpoint at every step and permits
statewise extrema that may not be jointly reachable. It remains a valid
conservative ruler without an attainment claim. Summing context-specific ranges
along a fixed query schedule is the prepared successor; reversing equal counts
must produce the same bound and cannot establish a transition-order effect.

**Validation.** All 943 producing source files, source archive, environment,
plan and original input hashes verify, as do the accepted parent law bindings.
Original, adjacent replay and extracted-source portable replay have identical
output bytes. Independent standard-library binary casts, scalar normalization,
logs, extrema and an exponential form of the bound reconstruct all 24 raw
array sets, all 72 original law/precision/length rows and nine equal-law cells.
Finite arithmetic comparisons use absolute and relative tolerance 2e-13;
original and cast bytes and support masks compare exactly. This is numerical
verification, not an outward-rounded interval-arithmetic proof. All 34 isolated
review/producer controls pass, including deliberate corruptions of ratios,
extrema, support and bounds. No new reader input, fit, observation or protected
lineage is consumed. Evaluator laws are not available as artifact-only evidence.

**Warrant:** verified finite supplied-law error bound. **Pursuit:** test known
query composition and finite storage allocation; primary learning conclusions
remain unchanged. Nothing here establishes historical process correspondence.

V19 remains active with 92 independently accepted scientific batches. The accumulated likelihood bound is verified. Query-schedule likelihood bounds passed 20 isolated controls and complete-support timing; their original was observed running through the existing serial queue, and original plus both complete replays now have execution receipts. Their independent numerical review belongs to completion events. Robust source-prior retention and context-specific precision allocation remain two prepared independent alternatives. Protected lineages, the 16-hour reserve and immutable reporting times are unchanged.

[Scientific record](../../../results/v19/G19-B-law-likelihood-envelope-1/README.md),
[schedule protocol](SCHEDULE_LIKELIHOOD_ENVELOPE_PROTOCOL.md),
[robust-source alternative](SOURCE_PRIOR_ROBUST_MASS_PROTOCOL.md),
[precision-allocation alternative](CONTEXT_PRECISION_ALLOCATION_PROTOCOL.md).
