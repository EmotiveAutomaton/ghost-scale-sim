# V20 longest-history learning curve

We tested whether 256 retained episodes give the structured reader a primary advantage. None of the five training budgets passes. At 32,768 labels, its capped log loss is 9.90534, versus 2.55699 for direct frequencies and 2.49182 for the legal template. At the smallest budget, 36.15% of true answers receive stored zero probability from numerical underflow. All 2,293,760 forecasts verify and the complete reference block replays exactly. These are constructed-method results; confirmation remains untouched.

## Comparison and findings

Each condition retains 256 past episodes at the permitted-context evidence tier, with 128, 512, 2,048, 8,192 or 32,768 training labels. Each uses eight training lineages, 32 discovery lineages, 2,048 test cases per discovery lineage and seven matched readers. Fit seed zero is fixed. The same whole lineages recur across conditions; rows and repeated fits are not independent observations.

Capped log loss is the negative natural logarithm of the saved probability of the true joint answer, floored at one trillionth; lower is better. It is not an uncapped proper score. The primary requires an improvement of at least 0.02 against both direct rivals and the legal template, a positive lower descriptive whole-world interval, and no increase in confident wrong attribution. All five conditions fail.

Table: each row gives a training-label budget. Columns give mean capped joint log loss for the structured account, direct frequency table, joint neural reader, independently trained bit reader, legal uniform template and supplied-law oracle, all with 256 retained episodes.

| Labels | Structured | Frequencies | Joint neural | Independent bits | Legal template | Oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 24.72247 | 2.70495 | 3.44918 | 18.80658 | 2.49182 | 2.38750 |
| 512 | 23.93714 | 2.64969 | 2.89665 | 8.95794 | 2.49182 | 2.38750 |
| 2,048 | 22.48954 | 2.57526 | 2.67251 | 19.89804 | 2.49182 | 2.38750 |
| 8,192 | 14.33522 | 2.56301 | 2.57359 | 21.43161 | 2.49182 | 2.38750 |
| 32,768 | 9.90534 | 2.55699 | 2.57738 | 16.37002 | 2.49182 | 2.38750 |

More training labels reduce structured loss, but its 32,768-label loss rises from 4.18870 with 64 episodes to 9.90534 with 256. The legal template improves from 2.52141 to 2.49182 and the supplied-law oracle also improves. Available information and learned use of that information remain distinct. The direct frequency table has the same mean loss as the training-frequency prior at every new budget. The bit reader also fails severely and nonmonotonically. These outcomes do not isolate optimizer, sample sparsity or representation effects.

Table: each row compares the structured reader with a primary rival at 32,768 labels. Improvement is rival loss minus structured loss; positive favors structured. The interval is a descriptive 95% interval from 4,096 paired whole-lineage bootstrap resamples. Confident wrong increase is the structured minus rival fraction of answers that are wrong with at least 90% confidence.

| Rival | Improvement | World interval | Confident wrong increase |
|---|---:|---|---:|
| Direct frequencies | -7.34835 | [-7.87166, -6.83147] | 0.18622 |
| Joint neural | -7.32796 | [-7.84996, -6.81297] | 0.18622 |
| Legal template | -7.41352 | [-7.93558, -6.89689] | 0.18622 |

## Stored zeros and capped scores

Table: each row is a training-label budget. The columns give the fraction of structured true-answer probabilities exactly zero in storage, and the fraction below the declared log-loss floor (including zero).

| Labels | Exact zero | Below log floor |
|---:|---:|---:|
| 128 | 0.36148071 | 0.88006592 |
| 512 | 0.18827820 | 0.84284973 |
| 2,048 | 0.00148010 | 0.71405029 |
| 8,192 | 0.00000000 | 0.26765442 |
| 32,768 | 0.00000000 | 0.05693054 |

Every exact-zero truth remains in the common legal support. An independent reconstruction of smoothed training counts finds finite log weights and truth-to-modal odds that underflow when exponentiated, for every such row. Thus these stored zeros are a numerical consequence of extreme learned concentration, not logical exclusion of the true candidate. They differ from forced candidate omission in G5. The saved forecasts, cap and criteria are unchanged; no repair or substitute uncapped verdict is introduced. Zero mass is absent for the other six readers in these five conditions. The independent-bit reader still assigns many extremely small positive probabilities.

Joint squared probability loss, accuracy, goal/tool/order/inspection losses, confidence and confident wrong attribution are retained in the [aggregates](../../../results/v20/long-history-wave-1/AGGREGATES.json). Squared probability loss remains proper. No secondary replaces the failed primary. [Zero-mass audit](../../../results/v20/long-history-wave-1/ZERO_MASS_AUDIT.json).

## Verification and continuation

All five COMPLETE records and plan/source/environment bindings verify. Every saved forecast, secondary score and summary reconstructs independently, including fit/decode costs and blind reader versus evaluator roles. The complete 256-episode, 128-label scientific block was selected for replay before its outcomes were read: all 548 scientific files and its summary match from frozen source. Earlier full G1/G4/G5/G6, saved-score-repair and 64-episode replay evidence is reused. There is no new scientific execution failure. Thirty-six packets are verified; original failed attempts remain retained and charged.

The review and complete replay cost 467.21875 CPU seconds; the zero audit cost 7.37500. All overhead remains charged within 90 CPU hours, with 18 protected. [Cost receipt](../../../results/v20/long-history-wave-1/COSTS.json).

Full longest-history costs reduce the pre-refill forecast from 23.38 to 21.06 hours. Twenty-four frozen shift comparisons add 8.05 hours, leaving 29.12 measured forecast hours across 306 discovery blocks. They compare 256-episode histories at two label budgets and 16-episode histories at the larger budget, under changed global execution rules and presentation shifts across four evidence tiers. The former changes both routes and is not a tool-specific skill manipulation. This tests whether the observed history failure transfers across shifts and evidence access; it adds no seeds or settings. [Selection](../../../results/v20/long-history-wave-1/SELECTION.json).

The 669 remaining conditional designs represent 59.62 forecast hours, not promised execution. Combined finite demand remains below the requested 1.25-times remaining wall-window margin; this shortfall is recorded without inflation. Twelve frozen diagnostic confirmation blocks remain untouched and retain priority at the fixed opening. The science cutoff and final delivery remain unchanged. [Forecast](../../../results/v20/long-history-wave-1/FORECAST.json).

Same-host replay does not establish another-platform or fresh-installation reproduction. Thirty-two coefficient redraws test sensitivity within this miniature, not arbitrary architectures. Fixed settings and one fit seed limit optimization conclusions; no simulator result establishes human intent.
