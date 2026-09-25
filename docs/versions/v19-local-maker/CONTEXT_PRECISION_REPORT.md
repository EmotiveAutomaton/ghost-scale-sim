# Context-specific precision allocation — 25 September 2026

We tested whether assigning storage precision by query context tightens a supplied-law error bound. At 1,536 bytes, the largest posterior total-variation bound after 128 balanced observations falls from 0.027783 for feasible uniform precision to 0.013637. All 648 assignments, 72 strata and both complete replays verify this constructed-method result. Seven computed optimizer ties change under scalar summation, without a material bound change. These are conservative bounds, not attained inference errors; learned access, historical process correspondence and human intent remain unestablished.

Total variation is half the sum of absolute differences in normalized state
probabilities, from zero to one. The bound compares inference under the original
and stored observation law, with identical priors and transitions. It is not a
measured posterior error. Every context can use half, single or double precision;
the complete library contains 81 assignments per law. All eight retained laws,
three byte budgets and balanced lengths 8, 32 and 128 are included. Posterior
state and schedules cost additional storage.

The table reports equal-law mean and largest selected bounds, the largest bound
for the best feasible uniform precision, and how many of the eight laws improve.
Rows identify law-storage budget in bytes and observation count. Uniform storage
uses 1,024 bytes at the two lower budgets, so it leaves some permitted bytes unused;
it uses 2,048 bytes at the highest budget. This is an equal-budget comparison,
not an equal-used-byte comparison. All laws choose uniform single precision at
the highest budget and tie that baseline.

| Byte budget | Observations | Mean selected bound | Largest selected bound | Largest uniform bound | Laws improved |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,280 | 8 | 0.0011804852 | 0.0012946237 | 0.0017368715 | 8/8 |
| 1,280 | 32 | 0.0047219075 | 0.0051784512 | 0.0069473810 | 8/8 |
| 1,280 | 128 | 0.0188855058 | 0.0207110280 | 0.0277828196 | 8/8 |
| 1,536 | 8 | 0.0007635542 | 0.0008523754 | 0.0017368715 | 8/8 |
| 1,536 | 32 | 0.0030542077 | 0.0034094890 | 0.0069473810 | 8/8 |
| 1,536 | 128 | 0.0122162525 | 0.0136371635 | 0.0277828196 | 8/8 |
| 2,048 | 8 | 0.0000002031 | 0.0000002203 | 0.0000002203 | 0/8 |
| 2,048 | 32 | 0.0000008124 | 0.0000008811 | 0.0000008811 | 0/8 |
| 2,048 | 128 | 0.0000032494 | 0.0000035246 | 0.0000035246 | 0/8 |

**Validation.** All 956 frozen source pins, archive, environment, plan and input
bindings verify. Original, adjacent replay and extracted-source portable replay
agree byte for byte. Independent binary casts, scalar normalization, support and
likelihood-range reconstruction verify all 24 law array sets. Independent exhaustive
enumeration verifies all eight allocation sets, 648 assignments, byte costs,
dominated alternatives, recorded minimizing ties, uniform baselines, 72 original
rows and nine equal-law cells. All 28 isolated producer/reviewer/runtime controls
pass. All assignments remain retained, including dominated choices.

The first independent check failed because it required discrete ties to survive
a different summation order. This failed review is preserved and charged.
Seven of eight laws at 1,280 bytes have a recorded two-way tie that scalar
normalization splits. Context ranges differ by at most 4.45e-16 in those cases;
the largest accumulated log-range regret for a recorded tied choice at 128
observations is 1.42e-14. The final review separately verifies the original exact
computed-float tie rule using independently checked ranges, and records the
scalar tie changes. No producer code, score, raw result or frozen source changed.
This does not certify exact real-number optimizer identities or outward-rounded
bounds. Numeric comparisons use the inherited 2e-13 tolerance.

**Warrant:** finite supplied-law storage decision, with numeric ties qualified.
**Pursuit:** determine whether knowing query composition is necessary for the gain.
No new fitted reader, observation or protected lineage was consumed. Evaluator
laws remain scientific inputs; there are no new artifact-only reader inputs.

V19 remains active with 94 independently accepted scientific batches. The interim checkpoint is reconciled with 92 accepted batches at its cutoff and two afterward. Context precision allocation is verified with scalar tie sensitivity retained. Robust precision under uncertain query composition passed 35 isolated controls and complete-support timing; the existing serial queue dispatched its frozen original and both full replays. Their execution receipts are complete; numerical acceptance belongs to the next completion review. Robust source-prior retention and minimax regret allocation remain two prepared alternatives. Protected lineages, the 16-hour reserve and final time are unchanged.

[Scientific record](../../../results/v19/G19-B-context-precision-allocation-1/README.md),
[frozen implementation](CONTEXT_PRECISION_IMPLEMENTATION.md),
[uncertain-query alternative](ROBUST_CONTEXT_PRECISION_PROTOCOL.md),
[source-prior alternative](SOURCE_PRIOR_ROBUST_MASS_PROTOCOL.md).
