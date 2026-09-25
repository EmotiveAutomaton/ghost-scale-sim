# Shared endpoint-law precision: engineering disposition

We tested whether shared-law precision and highest-probability source retention could fit their complete V19 comparisons. Both implementations pass controls, but inclusive timing projects 4,094.10 and 6,866.40 CPU seconds against their separate 3,600-second caps. Neither scientific comparison was admitted. These are scoped engineering blockers, not scientific nulls or evidence that the week is exhausted.

This branch asks whether lowering the precision of the shared future-response
law damages forecasts while report-state precision stays exact. All saved
posteriors, sources, five copying probabilities and future coordinates remain in
scope. The exact report law is unchanged. Float64, float32 and float16 storage
cross direct casts and one float64 row normalization. Explicit expected one-hot
loss change is evaluated even for unnormalized direct forecasts; squared error,
normalization drift, law underflow and future support loss remain separate.
An empty positive law row fails without a floor. A zero future prediction has a
defined squared loss and is not mislabeled an unsupported report.

The initial implementation passed 61 isolated controls. One storage repair keeps
hashes of exact group/joint arrays reconstructible from the complete immutable
posterior/law/source inputs, rather than serializing those duplicate arrays in
every result. This version passed 62 controls. The next identified cost was
repeated propagation at each copying probability. An exact mixture factorization
propagates the prior and eight independent-report states once, then evaluates
every original probability/report/future coordinate. Integer support propagation
preserves structural zero cells. That version also passed 62 controls. A separate
structural inspection found only seven repeated future columns in the long
schedules; no temporal roster reduction was implemented.

The three source-bound timing estimates were
4,550.31,
4,164.41 and
4,065.59 CPU seconds
before each benchmark's own charge and later work. The final inclusive snapshot
above includes those subsequent metered controls and benchmarks. Each estimate
uses all seven available structures and both source densities, three 32-row
synthetic batches per shape, regular/random/single-context schedules, complete
raw and summary serialization, a 50% margin, 120 seconds per execution for
inputs/setup, three executions, independent reconstruction and original-row
regroup. No native outcomes were used to select a repair. The final packaging
check stores the three law dtypes in a single archive covered by the existing
completion manifest; 62 controls pass after this binding fix. That packaging-only
change has no new timing claim and cannot make the failed admission pass.

All source versions, logs and timings are retained in the engineering export.
No scientific original, adjacent replay or portable replay was dispatched.
Independent scalar tests cover every fixture report and future coordinate,
proper loss, sparse/deterministic/constant/rare endpoints, permutations,
normalization, state reconstruction, corruption and complete native fixtures.
They establish fixture arithmetic, not native performance, learned access,
historical process correspondence or human intention.

[Frozen question](SHARED_LAW_PRECISION_PROTOCOL.md).
[Engineering evidence](../../../results/v19/G19-continuation-admission-20260925/BLOCKER.json).
