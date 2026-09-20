# V18.4 adaptive memory, 96-step streams

Does uncertainty over changing roles remain useful in longer streams? With independent observations in the 96-step comparison, the mixture improves prediction after skill changes: logarithmic loss is 0.67629 versus 0.68863 for static inference, reversing the shorter packet's ordering. After goal changes, its loss is 0.71549 versus 1.62893 for static inference and 0.75456 for fixed fast-role updating; expected artifact matching is 79.56%, 67.11% and 79.48%, respectively. The skill benefit disappears with copied observations. A small stationary cost remains, and the mixture still does not repair an omitted decision rule. These are descriptive constructed-method results; the horizons use separate generated traces, so their difference is not a paired causal estimate of extra observation time.

The second adaptation packet contains 8,640 assigned 96-step streams over the same
nine conditions, three copy spans, 20 coefficient draws and 16 nominal cells.
The source-sharing cell flag remains inactive because explicit copy span replaces
it. Horizon enters the trace seed, and change time scales with horizon; the two
horizon packets are not nested observations of one event. Their within-packet
method contrasts share traces, but the cross-horizon contrast also changes the
generated trace and change schedule. No confidence interval or independent-world
sample-size claim is made from the repeated cells.

The table shows mean logarithmic prediction loss in nats for independent sources;
lower is better. Each row names the planted change, and each column a reader.

| Planted change | Static | Fixed fast-role update | Role/rate mixture |
|---|---:|---:|---:|
| None | 0.65235 | 0.73115 | 0.65631 |
| Goal | 1.62893 | 0.75456 | 0.71549 |
| Skill | 0.68863 | 0.76660 | 0.67629 |
| Decision rule outside the supplied family | 2.01365 | 2.09008 | 2.01981 |

All raw forecasts, native programs and outcomes are retained. Independent
source streams carry the skill benefit: with copy spans three and six the mixture
instead has excess loss 0.00487 and 0.01393 over static inference. Other beneficial
change conditions retain a prediction advantage over static across copy spans;
stationary and omitted-rule streams retain a penalty. Independent
extracted-source verification reconstructed 15,120 means and exactly replayed
eight fixed whole streams. Export reassembly checked every scientific file hash.
This is bounded replay, not independent retraining or acquired craft.

The skill reversal motivates a matched-prefix/change-time diagnostic before
attributing it specifically to longer evidence. The persisting rule failure
keeps uncertainty over explanatory families and paid model revision high on the
backlog. Existing 32-step results remain intact.

[Scientific evidence](../../results/v18/exploratory-loop/U-hazard-horizon96-1/SCIENTIFIC_MANIFEST.json).
