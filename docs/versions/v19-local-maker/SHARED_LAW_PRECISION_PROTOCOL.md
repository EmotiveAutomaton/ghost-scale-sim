# Precision of a shared endpoint law: prepared alternative

Does quantizing the shared observation law damage forecasts even when the retained
report state is exact? Hold the accepted aggregate group and joint-report masses
and integer copy counts in their original precision. Quantize only the 16 by 4 by
8 law used to answer future queries, to float64, float32 and float16. Compare
direct casts with one row normalization in float64. Preserve zeros; a positive
law row that becomes empty is a failure, with no floor or evaluator repair.

Use all saved posteriors, eight laws, both evidence conditions, two paired draws,
five copying probabilities, all reports and every future coordinate. The exact
report law and Bayes update remain unchanged: this isolates future-response
arithmetic from retained-state error. Record future normalization drift, underflow,
probability error and explicit expected one-hot loss difference. For unnormalized
direct forecasts, do not substitute the normalized squared-distance identity.
Separate actual byte savings for the one shared law from per-history state cost.

Prepared, unimplemented and unadmitted. Require scalar future-law evaluation,
constant/deterministic/rare-endpoint fixtures, endpoint/context permutation,
normalization and support failures, and complete-support synthetic timing before
source and plan freeze. No fit, new observation or protected lineage. The inclusive
3,600 CPU-second cap covers failures, controls, three executions, independent
reconstruction and original-row regroup within the existing ceilings. No learned
law, historical correspondence or human-intent claim follows.
