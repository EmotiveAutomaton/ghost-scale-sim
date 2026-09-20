# V18.4 matched-prefix adaptation

Does the skill-update reversal persist when shorter and longer scores use the same trajectory? It does for the later skill switch with independent evidence: mean logarithmic prediction loss, where lower is better, changes from 0.70717 for the role/rate mixture versus 0.70292 for static inference in the first 32 steps to 0.66483 versus 0.68067 over 96 steps. With the earlier switch, the mixture already leads at 32 steps. Copied evidence usually removes the skill benefit, while goal-change benefits and omitted-rule failures remain. Evidence age and switch timing therefore matter within matched trajectories; longer observation alone does not guarantee an adaptive advantage. These are descriptive constructed-method results, miniature — architecture untested.

The packet contains 3,360 assigned 96-step trajectories: twenty fresh coefficient
draws crossed with eight active architecture cells, three source-copy spans and
seven change/time conditions. Stationarity uses one window anchor; goal, skill
and omitted-rule changes occur after 12 or 28 observations. Shared seeds exclude
condition, switch time, copy span and horizon. Each trajectory supplies nested
32/64/96-step scores; readers and prefixes are not independent replications.
The inactive nominal source-sharing bit is omitted. Every other reader, cell,
matching outcome and scoring window remains in the portable raw record.

The table reports differences in mean logarithmic prediction loss in nats.
Negative values favor the role/rate mixture; positive values favor static
inference. Rows cross switch timing with the number of repeated reports per
unique source. Columns score nested prefixes of the same trajectory.

| Skill switch after this many observations | Reports per source | Mixture minus static, first 32 | First 64 | First 96 |
|---|---:|---:|---:|---:|
| 12 | 1 | -0.00369 | -0.00528 | -0.00364 |
| 12 | 3 | +0.01899 | +0.01264 | +0.00950 |
| 12 | 6 | +0.03295 | +0.02000 | +0.01469 |
| 28 | 1 | +0.00426 | -0.01481 | -0.01584 |
| 28 | 3 | +0.02082 | +0.00489 | -0.00056 |
| 28 | 6 | +0.03562 | +0.01791 | +0.01021 |

The later independent-source switch reproduces a within-trajectory sign reversal,
while the earlier switch already favors the mixture at the shortest prefix. With
three reports per source and the later switch, the 96-step advantage is only
0.00055 nats; no significance or practical-superiority conclusion is attached to
it. The other copied-evidence skill comparisons retain a mixture penalty at 96
steps. No extra seeds are used to inflate this descriptive comparison.

Equal post-change windows separate evidence age from the proportion of the prefix
that occurred before a switch. The table gives mean logarithmic prediction loss
with independent sources; rows name the switch time, and paired columns compare
static inference with the mixture in two fixed 16-step windows.

| Skill switch | Static, first 16 after change | Mixture, first 16 | Static, steps 48–63 after change | Mixture, steps 48–63 |
|---|---:|---:|---:|---:|
| After 12 observations | 0.69129 | 0.67417 | 0.63062 | 0.62784 |
| After 28 observations | 0.72556 | 0.68607 | 0.64757 | 0.63036 |

The mixture's early post-skill-change losses are lower under both switch times;
the later switch's first-32-step penalty includes 28 pre-change steps. Thus the
prefix reversal combines changing evidence age with changing pre/post-change
weights. It is not an isolated causal effect of elapsed time. Static inference
also improves after the switch, and the contrasts differ with initial evidence.

At 96 steps, goal-change loss favors the mixture over static inference at both
switch times and every copy span. With independent observations and the later
goal switch, losses are 0.66609 versus 1.39073. Yet 48–63 steps after that change,
static loss is 0.62659 versus mixture 0.63801: a retained cumulative advantage
does not imply permanent local superiority. Stationary and omitted-rule losses
remain worse for the mixture in all copy conditions. At the later independent
rule switch, 96-step loss is 1.99542 versus static 1.98841. The already admitted
family-selection/averaging/expansion experiment addresses that distinct failure.

All recorded instrument gates passed. Independent extracted-source verification
reconstructed 16,464 means and exactly replayed eight fixed whole trajectories;
all 3,360 units had independent scientific score/native-execution checks during
aggregation. Source, plan and raw hashes match; portable reassembly verified every
scientific file. This is bounded replay, not replay of every whole trajectory or
a new confirmatory campaign. Native artifact matching is not learned uptake.
Historical V15 C11/M01 instrument failures remain unchanged.

Science used 584.890625 charged CPU seconds and 603.165643 wall seconds; adjacent
verification added 29.28125 charged CPU seconds and 30.572842 wall seconds.
The combined observed duration was 10.56 minutes. Future tests should separate
pre-change learning from post-change information arrival, or replace the omitted
law family, instead of merely extending already completed horizons.

[Portable scientific evidence](../../results/v18/exploratory-loop/U2-matched-prefix-1/SCIENTIFIC_MANIFEST.json).
