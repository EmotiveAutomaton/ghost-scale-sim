# N11: learned habit priors, precision and reset probes

Can reset probes separate learned habit priors from goal precision? In N11, all-probe true-mechanism log-loss gain is 0.270–0.724 nats across twelve settings. Pressure, context-shift and devaluation probes each have positive whole-law lower bounds in every setting. Automatizing effort falls from 0.603 to 0.137 by 32 records, while the three fixed mechanisms remain constant. All 12,288 cases, 168 independently reconstructed intervals and full replay verify. These are constructed mechanisms, not evidence about human automaticity or effort.

The table crosses four stipulated mechanisms with 0, 8 or 32 ordinary-history
records: 0 is goal-dominated, 1 habit-dominated, 2 automatizing and 3
capture-prone. Gain is reduction in true-mechanism log loss after all six reset probes;
the parenthesized values are descriptive 95% whole-law intervals. The next three
columns are single-probe-family mean gains. Effort is the model-defined KL from
the normalized habit prior before response noise, not reported or experienced
effort. It is unavailable—not zero—when there is no history.

| Declared setting | All-probe gain (95% interval) | Pressure | Context shift | Devaluation | First → last effort |
|---|---|---:|---:|---:|---:|
| {'history': 0, 'mechanism': 0} | 0.470225 (0.416392, 0.519702) | 0.082091 | 0.302323 | 0.302119 | unavailable |
| {'history': 8, 'mechanism': 0} | 0.675645 (0.616387, 0.736065) | 0.164298 | 0.369507 | 0.424914 | 0.603 → 0.603 |
| {'history': 32, 'mechanism': 0} | 0.723890 (0.652981, 0.801147) | 0.212241 | 0.371773 | 0.470534 | 0.603 → 0.603 |
| {'history': 0, 'mechanism': 1} | 0.624682 (0.518023, 0.720225) | 0.064081 | 0.510804 | 0.365368 | unavailable |
| {'history': 8, 'mechanism': 1} | 0.592034 (0.513394, 0.673824) | 0.028075 | 0.510197 | 0.270375 | 0.000 → 0.000 |
| {'history': 32, 'mechanism': 1} | 0.499740 (0.381136, 0.614316) | 0.034880 | 0.419160 | 0.231595 | 0.000 → 0.000 |
| {'history': 0, 'mechanism': 2} | 0.470225 (0.416703, 0.518686) | 0.082091 | 0.302323 | 0.302119 | unavailable |
| {'history': 8, 'mechanism': 2} | 0.374099 (0.329049, 0.417012) | 0.022259 | 0.248257 | 0.096756 | 0.603 → 0.211 |
| {'history': 32, 'mechanism': 2} | 0.354097 (0.329052, 0.381491) | 0.031436 | 0.206751 | 0.105165 | 0.603 → 0.137 |
| {'history': 0, 'mechanism': 3} | 0.403168 (0.334946, 0.463965) | 0.074635 | 0.149275 | 0.249160 | unavailable |
| {'history': 8, 'mechanism': 3} | 0.289678 (0.252471, 0.327325) | 0.034144 | 0.145171 | 0.143136 | 0.047 → 0.047 |
| {'history': 32, 'mechanism': 3} | 0.269662 (0.236750, 0.304920) | 0.033650 | 0.138520 | 0.131396 | 0.047 → 0.047 |

Goal-dominated and automatizing mechanisms are exactly aliased at zero history,
including their post-probe posterior for paired cases. At 32 records, history
alone only weakly separates the habit/automatization pair: conditional within
that pair, the true-mechanism posterior share averages 0.552 for habit worlds
and 0.557 for automatizing worlds. After all reset probes those shares rise to
0.821 and 0.844. This measures the
predicted long-history confusion instead of assuming it away.

The falling-effort prediction is selective. Automatizing worlds fall from 0.603
to 0.211 after eight records and to 0.137 after 32. Goal-dominated worlds stay at
0.603, habit-dominated worlds at 0.000336, and capture-prone worlds at 0.0465.
These constants and trajectories arise from the declared E and gamma values;
they do not validate an empirical measure of human effort.

Pressure, context shift and devaluation each add information in this miniature:
their means span 0.022–0.212, 0.139–0.511, and 0.097–0.471
nats respectively, with all twelve lower bounds above zero. The exact reader
knows the four candidate mechanisms and sixteen reliability laws. Only one
mechanism learns its habit prior; ordinary history favors action zero; pressure
changes the stipulated choice law; devaluation is a supplied value change; and
probe resets prevent one probe from training the next. Those design choices,
uniform response noise and the miniature architecture limit generalization.

All seven forecasts per setting (the exact reader plus six fixed/fresh direct
rivals), 12,288 raw cases, source and fit hashes, proper scores, complete replay,
and 168 primary/secondary/forecast intervals independently reconstruct. Positive
mechanism-posterior gain does not establish superiority to every predictive
rival. This conditional study is outside confirmation.

[Protocol](N11_PROTOCOL.md), [complete summary](../../../results/v23/n11-decision-review/SUMMARY.json),
[secondary review](../../../results/v23/n11-decision-review/SECONDARY_REVIEW.json),
[provenance](../../../results/v23/n11-decision-review/PROVENANCE.json).
