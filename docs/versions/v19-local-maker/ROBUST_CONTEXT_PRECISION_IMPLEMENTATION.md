# Robust context precision: frozen implementation contract

Implements the prepared finite alternative using all 81 context assignments of
half, single and double precision. Eight existing development laws, sixteen
states and eight endpoints per context are retained. Capacity and byte budgets
are unchanged: each context costs 256, 512 or 1,024 bytes; complete law budgets
are 1,280, 1,536 and 2,048 bytes. State and schedules are additional costs.

The query-count uncertainty set is the convex hull of four permutations of
(13, 1, 1, 1), scaled by 1, 4 and 8 for lengths 16, 64 and 128. Accumulated
log-range is linear in the count vector, so its maximum over the hull occurs
at a vertex. The bound is monotone in that range. No arbitrary unseen query
distribution or attained inference-error claim follows from this set.

Select all assignments minimizing the worst vertex log-range within each byte
budget. Also retain every balanced optimum, every separate vertex optimum,
every assignment's vertex regret and every feasible uniform choice. Both best
and worst robust bounds across balanced ties are reported. Ties use exact
computed-float equality; lexical assignment order is half, single, double in
context order. Scalar-reconstruction tie sensitivity must be recorded separately,
as the preceding allocation review demonstrates. No real-number tie certification.

The teacher is the supplied evaluator law. There is no fitted architecture,
training label schedule, new observation, protected lineage or tiny-model fit.
All 24 law variants and eight allocation arrays retain every range, bound,
support mask, choice and byte count. Positive support loss remains infinite
log-range and vacuous bound one; infinity-minus-infinity regret is explicitly
undefined, never zero. Actual-law rows must be finite before JSON serialization.

Controls cover independent scalar exhaustive minimax, all denominator-four
convex vertex mixtures, equal-context/exact-cast identity, a known case where
balanced allocation is worse, matched context/vertex permutations, infeasible
budgets, support loss and undercharged bytes. Full synthetic-support timing,
source freeze, original and two complete replays precede independent numerical
review and all 72 original-row/nine equal-law-cell reconstruction. The separate
inclusive cap remains 1,800 CPU seconds within existing family/global ceilings.

Robust source-prior retention and minimax regret allocation are separate prepared
alternatives. No previous failed comparison's cap is enlarged.
