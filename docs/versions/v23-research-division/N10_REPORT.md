# N10: residual correction structure in a supplied model

Can unexplained correction directions reveal a missing goal? In N10, adding residual structure improves hidden-goal log loss by 0.00276–0.06192 nats across twelve settings. Hidden-goal detection AUC is 0.643–0.988 where positives exist; soft support never empties. All 12,288 cases and full replay verify. This is a constructed-mechanism result with a supplied hidden-goal model, not general protection against model misspecification.

Detection AUC is the probability that a planted-positive case outranks a matched
negative (ties count half); .5 is chance and 1 is perfect. Cosine compares the
estimated and planted full hidden-goal direction, conditional on a hidden goal;
1 means aligned, 0 orthogonal and -1 opposite. Zero-share settings have no
positive cases: AUC and conditional direction are unavailable, not zero or one.

The table crosses hidden share, catalog correlation and noise (0=low,1=high).
The primary column is residual-over-soft true-mechanism log-loss gain in nats
with a descriptive 95% whole-law interval. Detection and shuffled AUC are means
over sixteen laws, as is the conditional direction cosine.

| Declared settings | Gain (95% interval) | Detection AUC | Shuffled AUC | Hidden-direction cosine |
|---|---|---:|---:|---:|
| {'noise': 0, 'rho': 0.0, 'share': 0.0} | 0.031455 (0.008208, 0.051935) | unavailable | unavailable | unavailable |
| {'noise': 1, 'rho': 0.0, 'share': 0.0} | 0.008878 (0.001463, 0.015940) | unavailable | unavailable | unavailable |
| {'noise': 0, 'rho': 0.5, 'share': 0.0} | 0.037538 (0.022023, 0.052906) | unavailable | unavailable | unavailable |
| {'noise': 1, 'rho': 0.5, 'share': 0.0} | 0.003557 (-0.002110, 0.008147) | unavailable | unavailable | unavailable |
| {'noise': 0, 'rho': 0.0, 'share': 0.1} | 0.022261 (0.007241, 0.037795) | 0.8274 | 0.8053 | 0.7461 |
| {'noise': 1, 'rho': 0.0, 'share': 0.1} | 0.003455 (-0.000937, 0.008098) | 0.6692 | 0.6578 | 0.3574 |
| {'noise': 0, 'rho': 0.5, 'share': 0.1} | 0.020186 (0.008136, 0.031591) | 0.7932 | 0.7805 | 0.8076 |
| {'noise': 1, 'rho': 0.5, 'share': 0.1} | 0.002763 (-0.000167, 0.006092) | 0.6430 | 0.6350 | 0.6276 |
| {'noise': 0, 'rho': 0.0, 'share': 0.3} | 0.049101 (0.032072, 0.068567) | 0.9881 | 0.9757 | 0.9648 |
| {'noise': 1, 'rho': 0.0, 'share': 0.3} | 0.061917 (0.044332, 0.078264) | 0.9133 | 0.8693 | 0.8828 |
| {'noise': 0, 'rho': 0.5, 'share': 0.3} | 0.050989 (0.036156, 0.066156) | 0.9813 | 0.9624 | 0.9798 |
| {'noise': 1, 'rho': 0.5, 'share': 0.3} | 0.052518 (0.039307, 0.066394) | 0.8270 | 0.7628 | 0.8798 |

Soft likelihood support is positive in every case; the strict hard comparator
empties in 97.85–100% of cases across settings. No hard prediction is invented
after that happens. The four zero-share settings can still gain by correctly
reducing the probability of the hidden alternative; no positive-class AUC exists.
All reader forecasts, six fitted rivals and privileged oracle scores are retained.

At correlation .5, detection AUC is .643–.981, rather than a wholesale collapse.
This does not establish the packet's stronger predicted confident attribution
to a catalog goal: catalog-attribution confidence is not the primary measure.
The exact residual reader knows the alternative model family and correlation;
an unmodelled arbitrary direction need not behave this way. The context shuffle
is one fixed cyclic permutation, not a calibrated randomization test. Positive
primary gains do not show superiority to every predictive rival.

Complete source/fit bindings, 12,288 retained cases, full fresh-process replay,
independent proper scores and 204 whole-law intervals verify. A separate rank-sum
calculation matches pairwise AUC in every block and checks conditional direction
denominators. Additional AUC/direction intervals resample whole laws. Every
secondary diagnostic is replay-verified; no claim is made that an independent
reviewer reimplemented the full hidden-vector generative architecture.

Constructed mechanisms, miniature—architecture untested; outside confirmation.
[Protocol](N10_PROTOCOL.md), [complete summary](../../../results/v23/n10-decision-review/SUMMARY.json),
[secondary review](../../../results/v23/n10-decision-review/SECONDARY_REVIEW.json),
[provenance](../../../results/v23/n10-decision-review/PROVENANCE.json).
