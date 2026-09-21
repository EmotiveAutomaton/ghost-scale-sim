# B2: the single roll-in pass destabilizes prediction

Does training on its own predictions stabilize the updater? One roll-in pass instead increases forecast loss by 5.74016 nats at 32 independent observations. The unchanged teacher-refit control, independent numerical checks and both complete replays verify this failure. This is a constructed-method result, miniature — architecture untested; it does not establish that roll-in training fails generally.

Logarithmic loss is prediction error measured in nats; lower is better. This
eight-development-lineage comparison retains two training draws and two fixed
feature seeds. It changes the training-input distribution once, then tests the
resulting frozen head. Each head has 4,128 coefficients and receives the same
12,288 observable forecast teachers per fit. The initial fit, rolled-input fit
and exact-teacher refit control cost three fits per paired draw/feature seed.
The comparison does not hold total fitting cost or input privilege equal to the
single initial fit; the refit control exposes that distinction.

Rows below separate independent source observations from copied source pairs.
Columns give the evaluated prefix length, loss for the original one-step and
rolled-in heads, and their paired difference with a 95% lineage-resampling
interval. Seeds, draws and streams remain inside lineage rather than becoming
independent sample units. These intervals condition on the retained fits.

| Evidence stream | Observations | One-step loss | Rolled-in loss | Rolled-in minus one-step, 95% interval |
|---|---:|---:|---:|---:|
| Independent episodes | 8 | 1.95684 | 5.30523 | +3.34839 [3.11154, 3.60397] |
| Independent episodes | 32 | 1.91879 | 7.65895 | +5.74016 [5.49795, 6.00383] |
| Adjacent copied pairs | 8 | 1.96863 | 2.01258 | +0.04395 [0.03018, 0.05818] |
| Adjacent copied pairs | 32 | 1.93490 | 7.46025 | +5.52535 [5.30677, 5.75388] |

At 32 independent observations the increase is positive for all four paired
draw/feature-seed combinations, ranging from 4.14582 to 6.94477 nats. Every scored
rolled-in output distribution has invalid raw probabilities in both stream types
at length 32. The declared clipping, floor and normalization repair still produces
valid distributions, but cannot make them accurate. The original one-step arm's
invalid fraction is 1.76% for independent observations at that length.

Teacher-forced evaluation of the rolled-in head is also worse: +0.07805 nats at
eight independent observations and +0.08942 at 32. This diagnostic supplies an
exact preceding bank. At copied scored positions every teacher arm simply passes
through the exact preceding bank, yielding identical scores. Those identities
are controls, not evidence of learned competence. The free/teacher contrast mixes
initialization, oracle privilege and input distribution; it does not isolate drift.

The exact-teacher refit is byte-identical to the initial head, and all initial-head
forecasts reproduce B1. Independent review reconstructs all 48 forecast files,
65,536 training-rollout prefixes and 768 score rows, with maximum score difference
below 2e-15. Both heads satisfy their saved ridge normal equations to residual
below 1.6e-11. Adjacent and independently extracted-source replays match all 90
deterministic files. Component timing is retained separately. The independent
review does not refit heads or regenerate the full endpoint laws; normal-equation
checks and complete source replay have those narrower stated roles.

**Warrant:** exploratory reversal against the proposed stabilization benefit;
the declared procedure fails scientifically while its reproduction checks pass.
**Pursuit:** the [bounded reset successor](BOUNDED_RESET_PROTOCOL.md) tests one
fixed public-history reset and the longer 128-observation horizon, sharing the
original B2 cap. No extra training, interval search or confirmation claim is made.
This result supplies no historical-process correspondence or human inference.

[Scientific evidence and reader-role reference](../../../results/v19/G19-B2-roll-in-1/EXPORT_MANIFEST.json).
