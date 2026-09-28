# V20 retained-history learning curves

We tested whether longer retained histories give the structured reader a primary advantage. None of the 18 new conditions passes. At 32,768 labels and 64 past episodes, its capped log loss is 4.18870, versus 2.58658 for direct frequencies and 2.52141 for the legal template. All 8,257,536 new forecasts verify, and a complete 64-episode block replays exactly. These are constructed-method results; confirmation remains untouched.

## Comparison

The 18 conditions cover 1, 4, 16 and 64 retained episodes. The first three complete the one-episode curve; the other histories each use 128, 512, 2,048, 8,192 and 32,768 training labels. Each condition has 32 discovery lineages, 2,048 test cases per lineage and seven matched readers. Training uses eight separate lineages and one fixed fit seed. Whole worlds are the units of uncertainty; the same worlds recur across label and history settings. Rows and repeated fits do not supply additional independent worlds.

Capped log loss is the negative natural logarithm of the probability assigned to the true joint answer, with probability floored at one trillionth; lower is better. It is not an uncapped proper score. The primary requires the structured reader to beat both direct rivals and the legal template by at least 0.02, a positive lower descriptive world interval and no increase in confident wrong attribution. None passes.

Table: each row is a training-label budget and retained past-episode count. Columns are mean capped joint log loss for the structured account, direct frequency table, joint neural reader, independent-bit neural reader, legal uniform template and supplied-law oracle. All primary criteria fail.

| Labels | Past episodes | Structured | Frequencies | Joint neural | Independent bits | Legal template | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 1 | 3.99613 | 5.19785 | 3.45202 | 23.39706 | 3.37592 | 3.13541 |
| 8,192 | 1 | 3.49600 | 5.48345 | 3.38144 | 21.28275 | 3.37592 | 3.13541 |
| 32,768 | 1 | 3.39833 | 4.62937 | 3.36665 | 25.07344 | 3.37592 | 3.13541 |
| 128 | 4 | 16.27918 | 3.28349 | 4.99844 | 25.22045 | 3.08879 | 2.72526 |
| 512 | 4 | 10.39433 | 3.25841 | 3.73556 | 18.38499 | 3.08879 | 2.72526 |
| 2,048 | 4 | 4.76061 | 3.17474 | 3.12561 | 25.24677 | 3.08879 | 2.72526 |
| 8,192 | 4 | 3.36777 | 3.16264 | 3.04421 | 24.14562 | 3.08879 | 2.72526 |
| 32,768 | 4 | 3.10032 | 3.17557 | 3.02992 | 23.30268 | 3.08879 | 2.72526 |
| 128 | 16 | 22.38371 | 2.94362 | 4.27172 | 16.92504 | 2.73679 | 2.46121 |
| 512 | 16 | 17.35851 | 2.89835 | 3.56065 | 18.62352 | 2.73679 | 2.46121 |
| 2,048 | 16 | 8.86518 | 2.82059 | 2.85578 | 16.90582 | 2.73679 | 2.46121 |
| 8,192 | 16 | 4.12843 | 2.80720 | 2.76861 | 17.82280 | 2.73679 | 2.46121 |
| 32,768 | 16 | 3.07255 | 2.80277 | 2.74882 | 19.26735 | 2.73679 | 2.46121 |
| 128 | 64 | 24.29146 | 2.73466 | 3.60669 | 14.17230 | 2.52141 | 2.39208 |
| 512 | 64 | 22.31726 | 2.67954 | 2.97060 | 18.12915 | 2.52141 | 2.39208 |
| 2,048 | 64 | 15.62728 | 2.60480 | 2.67744 | 19.60740 | 2.52141 | 2.39208 |
| 8,192 | 64 | 8.38777 | 2.59252 | 2.61604 | 14.37667 | 2.52141 | 2.39208 |
| 32,768 | 64 | 4.18870 | 2.58658 | 2.59493 | 15.39040 | 2.52141 | 2.39208 |

Table: rows give the largest training budget for each history and each primary rival. Improvement is rival loss minus structured loss; positive favors the structured account. Intervals are descriptive 95% intervals from 4,096 whole-world bootstrap resamples.

| Past episodes | Rival | Improvement | World interval |
|---:|---|---:|---|
| 1 | Direct frequencies | 1.23104 | [1.20471, 1.25752] |
| 1 | Joint neural | -0.03168 | [-0.04105, -0.02218] |
| 1 | Legal template | -0.02241 | [-0.03893, -0.00619] |
| 4 | Direct frequencies | 0.07525 | [0.04386, 0.10729] |
| 4 | Joint neural | -0.07040 | [-0.08804, -0.05207] |
| 4 | Legal template | -0.01152 | [-0.04429, 0.02117] |
| 16 | Direct frequencies | -0.26978 | [-0.31290, -0.22576] |
| 16 | Joint neural | -0.32373 | [-0.35602, -0.29077] |
| 16 | Legal template | -0.33576 | [-0.38079, -0.29170] |
| 64 | Direct frequencies | -1.60212 | [-1.70768, -1.49893] |
| 64 | Joint neural | -1.59377 | [-1.69753, -1.49179] |
| 64 | Legal template | -1.66729 | [-1.77171, -1.56285] |

At every fixed history, the structured reader improves as the training-label budget grows, but remains worse than the legal template even at 32,768 labels. At that budget, structured loss is lowest with sixteen past episodes and rises at sixty-four; the legal template and supplied-law oracle improve with longer histories. Extra history supplies useful information in this world, while this learned decoder fails to use it reliably. This is a distinction between available information and learned readout, not evidence that all structured approaches fail.

The direct frequency table also performs poorly with one episode, including loss 5.48345 at 8,192 labels. The independently trained bit reader has severe failures throughout: capped losses range from 14.17230 to 25.24677. These are numerically verified outcomes, not failed execution. Fixed optimization, sparse counts and representation assumptions remain possible explanations; there is no hyperparameter search or isolated causal diagnosis.

## Probability floor and secondary outcomes

Exact zero truth probability is zero for every reader in all 18 conditions. Small positive mass below the log floor remains distinct from exact-zero candidate omission. At the worst observed condition, 84.98688% of structured truth probabilities and 90.62347% of independent-bit truth probabilities fall below the floor. These maxima come from different conditions and are not independent trials. Joint squared probability loss, accuracy, goal/tool/order/inspection loss and confident wrong attribution are retained for every arm in [aggregates](../../../results/v20/history-wave-1/AGGREGATES.json); all 54 primary paired comparisons and both learning/history contrast sets are retained. No secondary outcome replaces the primary.

## Verification, costs and continuation

All 18 COMPLETE records, plan/source/environment identities, training parameter counts, decoding costs and reader/evaluator separation verify. Every forecast and secondary score reconstructs from saved probabilities, and every summary matches the raw records. The full 64-episode, 128-label block was selected for replay before reading its outcomes. All 548 scientific files and its summary reproduce exactly from frozen source. Previous complete opening-family and saved-score-repair replays remain valid and are reused. No implementation or scientific criterion changed, and no whole test suite was rerun for this checkpoint.

Thirty-one packets are now verified. The original scoring failure and earlier replay-helper failure remain retained with their costs. This review and replay used 329.45312 CPU seconds, charged within the original 90-hour ceiling and 18-hour protected reserve. Full attempt costs are in the [cost receipt](../../../results/v20/history-wave-1/COSTS.json).

Publication triggered automatic Git packing, which used parallel CPU and violated the intended single-thread operating limit. Its last native measurement was 1373.68750 CPU seconds; it exited naturally before a targeted stop could act. Final CPU was unavailable, so accounting charges a conservative 4031.22350-second upper bound, using all 24 logical processors over the entire unobserved tail plus parent overhead. Completed campaign attempts then total 2.61541 charged CPU hours, excluding the resumed worker. The 18-hour reserve remains protected. The event is retained as a resource overrun; scientific outputs are unchanged. Subsequent publication commands disable automatic maintenance without changing shared repository configuration.

Measured full-block costs reduce the remaining admitted forecast from 32.06 to 25.43 hours, including review, across 287 discovery blocks. Twelve frozen diagnostic confirmation blocks remain untouched. No refill is needed now; the next review should recalibrate after the five 256-episode blocks and refill from the finite forest if needed. The 693 conditional designs represent 78.68 forecast hours, not promised execution. The combined forecast is below the requested 1.25-times wall-window backlog margin; this shortfall is recorded without inflating estimates or adding designs.

Same-host replay does not verify a fresh installation or another platform. The 32 coefficient redraws test sensitivity within this miniature, not arbitrary architectures. One fit seed per condition limits optimization inference. Execution, numerical acceptance, failed discovery criteria and untouched confirmation remain separate. The fixed confirmation boundary, science cutoff and final delivery are unchanged. No simulator result establishes human intent.

[Validity](../../../results/v20/history-wave-1/VALIDITY.json), [paired curves](../../../results/v20/history-wave-1/PAIRED_CURVES.json), [forecast](../../../results/v20/history-wave-1/FORECAST.json).
