# Fixed query-schedule likelihood bounds — 25 September 2026

We tested whether known query composition tightens the error bound from rounding a supplied observation law. Across eight retained laws, balanced queries lower the largest half-precision posterior total-variation bound after 128 observations from 0.028781 to 0.027783. Independent scalar reconstruction, all 144 law/precision/order/length strata and both complete replays verify this constructed-method ruler. Reversing the same context counts leaves the bound unchanged. These are conservative bounds under the same prior and transition law, not attained inference errors; learned access, historical process correspondence and human intent remain unestablished.

Total variation is half the sum of absolute differences in normalized state
probabilities, from zero to one. Here it bounds the posterior difference caused
by replacing a supplied observation law with its stored, once-normalized
counterpart. Both inferences must share the prior and transition law.

The prior universal bound permits the largest context/endpoint log-ratio range
at every observation. The new bound sums each context's largest supported
endpoint range along two frozen balanced schedules. Their counts match, so
reversal must leave the result unchanged. This is a statement about the bound,
not about transition-order effects or attained posterior differences.

The table compares equal-law mean bounds and the largest schedule bound across
the eight laws. Rows identify storage precision and observation count. Bounds
are conservative total-variation limits, not measured errors or confidence
intervals. Both schedule orders yield the same displayed numbers.

| Law storage | Observations | Mean universal bound | Mean schedule bound | Largest schedule bound |
| --- | ---: | ---: | ---: | ---: |
| Single precision | 8 | 0.0000002120 | 0.0000002031 | 0.0000002203 |
| Single precision | 32 | 0.0000008480 | 0.0000008124 | 0.0000008811 |
| Single precision | 128 | 0.0000033919 | 0.0000032494 | 0.0000035246 |
| Half precision | 8 | 0.0016679269 | 0.0015974157 | 0.0017368715 |
| Half precision | 32 | 0.0066716140 | 0.0063895808 | 0.0069473810 |
| Half precision | 128 | 0.0266804786 | 0.0255530698 | 0.0277828196 |

Half precision stores the law in 1,024 bytes, single in 2,048, and double in
4,096. State and schedules are additional costs. Double precision has zero
relative error against its identically normalized reference. No positive
support is lost in the actual eight laws. Support-loss fixtures retain an
infinite log range and vacuous bound one. A rounded observation impossible
under every state has an undefined posterior, not a meaningful rounded answer.

**Validation.** All 949 source pins, the archive, plan, environment and original
law inputs verify. Original, adjacent and extracted-source portable replay
outputs agree byte for byte. Independent standard-library binary casts, scalar
normalization/logarithms and an exponential bound reconstruct all 24 complete
raw array sets, all 144 original strata and 18 equal-law cells. The checker
retains all statewise ratios, support masks, context maxima and attaining
endpoint sets, integer schedules and universal-versus-schedule differences.
All 30 isolated producer/reviewer/runtime/dispatch controls pass. Floating
comparisons use absolute and relative tolerance 2e-13; this is not an
outward-rounded interval proof or a claim that extrema are jointly reachable.

**Warrant:** verified finite supplied-law bound. **Pursuit:** optimize precision
within a finite byte budget and test robustness to unknown query composition.
No fitted reader, new observation or protected lineage is consumed. Evaluator
laws remain scientific inputs; there are no new artifact-only reader inputs.
A1 was independently regrouped without rerunning its science, and both tiny
training settings remain consumed by Ghost with matching sibling ownership.

V19 remains active with 93 independently accepted scientific batches. Query-schedule bounds are verified. Context-specific precision allocation passed 31 isolated controls and complete-support timing; the existing serial queue dispatched its frozen original and both full replays, which now have execution receipts. Its numerical acceptance remains pending completion-event review. Robust source-prior retention and precision allocation under uncertain query composition remain two prepared independent alternatives. Protected lineages, the 16-hour reserve and immutable reporting times are unchanged.

[Scientific record](../../../results/v19/G19-B-schedule-likelihood-envelope-1/README.md),
[frozen schedule protocol](SCHEDULE_LIKELIHOOD_ENVELOPE_PROTOCOL.md),
[allocation implementation](CONTEXT_PRECISION_IMPLEMENTATION.md),
[robust source alternative](SOURCE_PRIOR_ROBUST_MASS_PROTOCOL.md),
[uncertain query alternative](ROBUST_CONTEXT_PRECISION_PROTOCOL.md).
