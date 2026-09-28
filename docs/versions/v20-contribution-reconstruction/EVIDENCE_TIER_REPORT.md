# V20 evidence tiers and learning curves

We tested whether richer records and short histories give the structured reader an advantage. None of the 60 new diagnostic conditions beats both direct rivals and the legal template under the declared comparison rule. With 16 complete past records and 32,768 labels, its capped log loss is 1.06422, versus 0.80049 for direct frequencies and 0.79957 for the legal template; lower is better. All 27,525,120 forecasts verify and four complete reference blocks replay exactly. These are constructed-method results; the context primary remains failed and confirmation remains untouched.

## What was compared

The new packets complete almost all artifact-only, sparse-record and complete-record curves at histories of zero, one, four and sixteen episodes, with 128, 512, 2,048, 8,192 and 32,768 labels. The zero-history complete-record condition at 8,192 labels was already reviewed in the opening wave and is reused. One additional new packet begins the artifact-only 64-episode curve at 128 labels. There are eight training worlds, 32 discovery worlds, seven readers and 2,048 queries per discovery world. Fit seed zero is fixed.

Artifact-only input reveals the endpoint. Sparse records add permitted context and the proposal route. Complete records add selection, revision and operation order, while inspection remains latent in these cards and private aims remain withheld. The full past observations come from the same persistent maker. All readers receive identical supplied-law candidate support; this is an input privilege shared across methods.

Capped log loss is the negative natural logarithm of saved probability on the true joint answer, floored at one trillionth. Lower is better; it is not an uncapped proper score. The comparison rule requires a 0.02 improvement against each direct rival and the legal template, a positive lower whole-world interval and no increase in confident wrong attribution. Applying that rule to these tiers is diagnostic: the registered primary is context only, and none of these results can substitute for it.

## Larger-budget evidence and history curve

Table: rows give the available evidence and number of past episodes at 32,768 training labels. Columns give mean capped joint log loss for the structured account, direct frequency table, joint neural reader, independent-bit neural reader, legal uniform template and supplied-law oracle. All means use the same 32 whole worlds.

| Evidence | Past episodes | Structured | Frequencies | Joint neural | Independent bits | Legal template | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| Artifact only | 0 | 3.97036 | 3.92568 | 3.96570 | 4.00045 | 4.14532 | 3.84758 |
| Artifact only | 1 | 3.81196 | 3.74101 | 3.79614 | 26.52543 | 3.99116 | 3.62277 |
| Artifact only | 4 | 3.61226 | 4.81910 | 3.55295 | 24.71570 | 3.83026 | 3.26405 |
| Artifact only | 16 | 3.74907 | 3.59258 | 3.44400 | 22.94819 | 3.72986 | 2.96788 |
| Sparse record | 0 | 3.25398 | 3.17773 | 3.20981 | 3.23835 | 3.23565 | 3.05794 |
| Sparse record | 1 | 3.05331 | 4.49087 | 2.98791 | 3.01785 | 3.02941 | 2.78890 |
| Sparse record | 4 | 2.75968 | 2.74421 | 2.63492 | 20.80479 | 2.74228 | 2.37875 |
| Sparse record | 16 | 2.70905 | 2.38504 | 2.33030 | 15.67962 | 2.39028 | 2.11470 |
| Complete record | 0 | 1.66279 | 1.62858 | 1.63737 | 1.68925 | 1.64786 | 1.56031 |
| Complete record | 1 | 1.45512 | 2.66162 | 1.38859 | 13.12365 | 1.39966 | 1.26393 |
| Complete record | 4 | 1.21313 | 1.08506 | 1.07399 | 11.89069 | 1.08387 | 0.91450 |
| Complete record | 16 | 1.06422 | 0.80049 | 0.81431 | 7.43435 | 0.79957 | 0.72226 |

Richer observation lowers the larger-budget structured loss at every displayed history, but does not establish superiority over the matched rivals. With 16 past complete records, even the legal template beats the structured account. Artifact-only structured loss improves from zero to four episodes and worsens at sixteen; information available to the oracle and information successfully used by a learned reader remain distinct. The 64-episode artifact condition at 128 labels has structured loss 26.38892, close to the cap, and is a severe decoder failure: 95.00% of true answers fall below the score floor, 90.55% of queries receive a confident wrong answer, and exact joint accuracy is 3.67%. No true answer has stored zero probability. The full five-budget curves and all secondary scores are in the [aggregates](../../../results/v20/learning-curve-wave-1/AGGREGATES.json).

Table: each row compares the structured reader with a rival for 16 complete past records at 32,768 labels. Improvement is rival loss minus structured loss, so positive favors the structured reader. Intervals are descriptive 95% intervals from 4,096 paired whole-world bootstrap resamples. Confident wrong increase is structured minus rival error with at least 90% confidence.

| Rival | Improvement | World interval | Confident wrong increase |
|---|---:|---|---:|
| Direct frequencies | -0.26374 | [-0.29674, -0.23071] | 0.04697 |
| Joint neural | -0.24992 | [-0.27684, -0.22313] | 0.04697 |
| Legal template | -0.26465 | [-0.29768, -0.23161] | 0.04697 |

Whole coefficient worlds are paired across methods and curves; rows and repeated fits are not independent. These intervals condition on the shared training worlds and a single fixed fit seed. They are descriptive within the finite generator, with no confirmatory claim across the many comparisons. [Paired curves](../../../results/v20/learning-curve-wave-1/PAIRED_CURVES.json).

## Why an exact-match table can lose

The frequency reader uses counts only when the entire evidence vector matches a training vector; otherwise it uses its training-frequency prior. We reconstructed every forecast across five label budgets for complete records with one past episode and artifact-only inputs with four past episodes. Unmatched queries reproduce the prior, within floating-point tolerance. Every excess-loss contribution is accounted for by matched queries. This identifies a failure of this exact-match estimator; it does not establish that all nonparametric readers fail.

Table: each row is one of the two audited evidence/history conditions at 32,768 labels. Match coverage is the fraction of test queries with an exact training-vector match. Excess loss is direct-table minus prior capped log loss; positive is worse. The final column gives the descriptive paired whole-world interval.

| Evidence and history | Match coverage | Excess loss | World interval |
|---|---:|---:|---|
| Complete record, 1 past | 0.57870 | 1.26055 | [1.24229, 1.27831] |
| Artifact only, 4 past | 0.95387 | 1.12975 | [1.10624, 1.15342] |

The mechanism is available in the estimator and verified against saved probabilities: a sparse matching count can move probability away from the true answer, while the fallback prior avoids that update. Matching is not randomized, so this decomposition is not a causal comparison between matched and unmatched populations. It does not isolate sampling sparsity, smoothing or differences between training and evaluation worlds as the cause of poor matched forecasts. The complete one-episode, 32,768-label reference block was replayed for this interpretation. No smoothing, scoring, criterion or training setting was changed. [Exact-match audit](../../../results/v20/learning-curve-wave-1/TABLE_MATCH_AUDIT.json).

## Verification and limits

All 60 COMPLETE records, plan/source/environment bindings, fit/decode costs and blind/evaluator roles verify. Independent review reconstructs every saved forecast, all secondary scores and summaries. No saved true answer receives exact zero probability in any of the 420 new method-condition arms; below-floor probabilities are reported separately, so severe capped loss is not mistaken for omitted support. Joint squared probability loss remains proper. [Zero and floor receipt](../../../results/v20/learning-curve-wave-1/ZERO_MASS.json).

Three complete 16-episode, 128-label blocks, one per evidence tier, were selected before reading the new outcomes and replayed from frozen source. The additional complete-record interpretation replay uses 32,768 labels and one past episode. Each matches all 548 scientific files and its summary. Earlier full G1/G4/G5/G6, saved-score repair and long-history replays are reused. There are now 96 numerically verified packets, with no new scientific execution failure; all previous failures and costs remain retained. Twelve frozen confirmation diagnostics remain untouched.

Review and reference replay used 1143.07812 CPU seconds; the retained-data audit and interpretation replay used 139.84375. The initial audit driver failed before analysis because older binding receipts lack a branch field, consuming 0.01562 CPU seconds. Its source and failed attempt are retained; the replacement reads that field from immutable packet plans. Producer code, saved forecasts and criteria are unchanged. Costs and conservative control-plane overhead remain inside the 90-hour ceiling, with 18 hours protected. [Costs](../../../results/v20/learning-curve-wave-1/COSTS.json).

Measured evidence-tier costs revise the remaining useful forecast from 27.15 to 30.20 hours across 246 discovery blocks. No refill is needed at this boundary; 669 designs remain conditional. Combined finite demand is below the 1.25-times remaining wall-window margin, an explicit forecast shortfall rather than a reason to inflate runtime. The unchanged admitted curves will test larger label budgets, longer histories and matched shifts. [Forecast](../../../results/v20/learning-curve-wave-1/FORECAST.json).

Same-host replay does not verify a fresh installation or another platform. The structured reader is a naive-Bayes conditional-evidence account, not an exact learned causal interpreter. The fixed optimizer and one fit seed do not isolate representation from optimization; only the two neural heads are approximately parameter matched. Thirty-two coefficient redraws probe this miniature, not arbitrary architectures. No simulator result establishes human intent.
