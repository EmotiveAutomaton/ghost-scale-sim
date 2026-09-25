# Precision regret — 25 September 2026

We tested whether minimizing regret relative to query-informed precision choices improves on minimizing the worst absolute bound, and found no improvement in these eight laws. All 648 assignments, 72 strata and both complete replays verify. Seven regret-optimum tie sets change under scalar arithmetic, with objective differences below 5e-14. This is a constructed-method result about supplied-law error bounds; attained inference accuracy, learned access, historical process correspondence and human intent remain unestablished.

Regret here is excess accumulated log-likelihood-ratio range relative to the
best feasible precision assignment chosen with query composition known. It is
neither posterior loss nor measured prediction error. The fixed uncertainty set
contains every convex mixture of four permutations of counts (13,1,1,1), scaled
to 16, 64 and 128 observations. All 81 assignments of half, single and double
precision to four contexts are retained for each law. The maximum regret over
the convex set occurs at a vertex because it is a maximum of linear contrasts.

The table reports maxima across eight laws, equally weighted in the accompanying
mean summaries. Rows identify byte budget and observation count. Columns compare
minimax regret, the regret of absolute-bound-optimal choices, and the best feasible
uniform precision; the last column is the greatest added posterior total-variation
bound from a regret-optimal choice. Total variation is half the summed absolute
probability difference, from zero to one. All ties are retained. Posterior state
and schedules are outside these law-byte budgets.

| Law bytes | Observations | Minimax regret | Absolute-choice regret | Uniform regret | Added absolute bound |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,280 | 16 | 0.0107946094 | 0.0107946094 | 0.0116941602 | 0.0 |
| 1,280 | 64 | 0.0431784376 | 0.0431784376 | 0.0467766407 | 0.0 |
| 1,280 | 128 | 0.0863568751 | 0.0863568751 | 0.0935532814 | 0.0 |
| 1,536 | 16 | 0.0101935741 | 0.0101935741 | 0.0125937110 | 0.0 |
| 1,536 | 64 | 0.0407742966 | 0.0407742966 | 0.0503748438 | 0.0 |
| 1,536 | 128 | 0.0815485931 | 0.0815485931 | 0.1007496877 | 0.0 |
| 2,048 | 16 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0 |
| 2,048 | 64 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0 |
| 2,048 | 128 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0 |

No retained law improves over either the absolute-bound or balanced-query optimum,
and regret selection adds no absolute-bound cost. Zero regret at 2,048 bytes means
the uniform single-precision assignment is vertex-optimal in this finite library;
it does not mean zero probability error. The synthetic counterexample proves that
absolute minimax and minimax regret can disagree outside these retained laws.

All 971 source pins, source archive, environment, plan, eight law inputs and
original/adjacent/portable output identities verify. Independent scalar casts,
normalization, support and log ratios reconstruct 24 law arrays. Independent
enumeration reconstructs eight allocation arrays, every byte cost, every vertex
regret at three lengths, all regret/absolute/balanced/vertex optima and ties,
uniform baselines, 72 original rows and nine equal-law cells. All 33 isolated
review controls pass, including corrupted arrays and undefined regret handling.
Infinity minus infinity is undefined and cannot win; actual retained regrets
are finite. No producer score or frozen source changed.

Scalar normalization changes seven regret-optimum tie sets in addition to seven
absolute, seven balanced and fourteen vertex-specific sets. Maximum regret-choice
objective difference at 128 observations is 4.98e-14. Exact computed-float tie
identities verify separately from arithmetic sensitivity. No exact-real optimizer
or outward-rounded certification is claimed. No new fit, observation or protected
lineage was used. Scientific evaluator laws remain separate from reader inputs;
this batch supplies no new reader input.

**Warrant:** descriptive equality of two finite supplied-law criteria; miniature —
architecture untested. **Pursuit:** uncertain source-prior retention is independent;
interval sensitivity can distinguish unstable identities from unstable bounds.
These controls do not test learned law acquisition, actual filtering errors,
historical execution-chain recovery or generalization beyond the declared laws.

V19 remains active with 96 independently accepted scientific batches. Precision regret is independently verified. Robust source-prior retention passed 41 isolated controls and complete-support timing; the existing serial queue dispatched its frozen original and both full replays. Their execution receipts are complete; independent numerical acceptance belongs to completion-event review. Precision-choice stability and source-prior regret remain two prepared alternatives. The immutable interim population remains 92; four later acceptances are dated separately. Protected lineages, the 16-hour reserve and final time are unchanged.

[Scientific record](../../../results/v19/G19-B-context-precision-regret-1/README.md),
[frozen contract](CONTEXT_PRECISION_REGRET_IMPLEMENTATION.md),
[source-prior alternative](SOURCE_PRIOR_ROBUST_MASS_PROTOCOL.md),
[precision stability](PRECISION_STABILITY_PROTOCOL.md).
