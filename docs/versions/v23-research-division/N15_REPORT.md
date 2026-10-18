# N15: source-specific trust or general vigilance

Can checking behavior distinguish source-specific trust from general vigilance? In N15, failures of both the focal and second source improve true-mechanism log loss by 0.009–0.535 nats across twelve settings; focal-source failures alone gain 0.005–0.293 nats and preserve the declared exact alias between source tracking and global vigilance. Added failures become less informative after 32 records. All 12,288 cases, 108 independently reconstructed intervals, six frozen rivals and full replay verify. This is a constructed-mechanism result, not evidence about human trust or attention.

The table crosses four stipulated checking rules with 0, 8 and 32 public records.
Combined-probe gain is the reduction in true-mechanism log loss after focal-source
and second-source failures. Focal-only gain is the declared secondary ablation.
Parentheses are descriptive 95% intervals from whole reliability-law blocks.
Forecast losses compare the exact reader with the best of six fixed/fresh direct
rivals for the same withheld responses; lower is better, but the mechanism primary
does not establish predictive superiority over every rival.

| Stipulated mechanism | History records | Combined-probe gain (95% interval) | Focal-only gain (95% interval) | Exact forecast loss | Best direct-rival loss |
|---|---:|---:|---:|---:|---:|
| fixed precision | 0 | 0.155433 (0.106213, 0.201518) | 0.111856 (0.075061, 0.147909) | 0.601831 | 0.598115 |
| fixed precision | 8 | 0.067387 (0.038175, 0.094968) | 0.031989 (0.005395, 0.057575) | 0.600189 | 0.603761 |
| fixed precision | 32 | 0.026532 (0.010458, 0.041074) | 0.013953 (0.001529, 0.026049) | 0.585613 | 0.593808 |
| source-specific tracking | 0 | 0.188871 (0.138944, 0.239059) | 0.042997 (0.028581, 0.058212) | 0.607485 | 0.607877 |
| source-specific tracking | 8 | 0.046543 (0.028731, 0.064236) | 0.024863 (0.009140, 0.040455) | 0.639483 | 0.648321 |
| source-specific tracking | 32 | 0.008633 (0.000044, 0.017314) | 0.004985 (-0.001660, 0.012369) | 0.639268 | 0.651531 |
| global vigilance | 0 | 0.139212 (0.114965, 0.165013) | 0.042997 (0.028397, 0.058191) | 0.692841 | 0.691216 |
| global vigilance | 8 | 0.047843 (0.017999, 0.076602) | 0.024863 (0.008784, 0.040112) | 0.655050 | 0.664107 |
| global vigilance | 32 | 0.011778 (0.000770, 0.022818) | 0.004985 (-0.001859, 0.012478) | 0.645010 | 0.656865 |
| stakes only | 0 | 0.535359 (0.465062, 0.599483) | 0.293410 (0.231269, 0.354222) | 0.597370 | 0.592616 |
| stakes only | 8 | 0.174224 (0.132236, 0.211868) | 0.094430 (0.064382, 0.125814) | 0.526924 | 0.574509 |
| stakes only | 32 | 0.016746 (0.008758, 0.026539) | 0.008647 (-0.000526, 0.018543) | 0.488703 | 0.548747 |

The same-source alias is a design constraint, not a missed detection: histories
concern source A, so source-specific tracking and global vigilance have the same
public counts when only A fails. A failure of source B changes the global-vigilance
law but not source-A tracking, which supplies the separating observation. Even the
combined probe adds little after 32 records in some settings because one new failure
is diluted by the retained history; the smallest combined mean is 0.00863 nats.

Controls verify the exact alias, the second-source law change, probability/log-space
agreement, canonical ledger round trips, evaluator-label exclusion and planted
recovery above 0.9968 in every reliability law. The independent extension review
recomputed all 108 primary, secondary and forecast intervals from the 12,288 raw
cases. Every source, fit, complete and replay hash matches, including all six frozen
rivals and their independent validation receipts.

These response laws are stipulated: checking probability is not a fitted human
psychophysical function, public failures are binary, probes reset, reliability is
uniform response noise, and source-specific precision is represented only through
checking. The miniature architecture, mechanism prevalence, real-text observability
and engine-module behavior remain untested. This conditional study is outside
confirmation and does not admit an attention or source-trust module.

[Protocol](N15_PROTOCOL.md), [complete summary](../../../results/v23/n15-decision-review/SUMMARY.json),
[independent review](../../../results/v23/n15-decision-review/REVIEW.json),
[controls](../../../results/v23/n15-decision-review/CONTROLS.json),
[provenance](../../../results/v23/n15-decision-review/PROVENANCE.json).
