# Purpose-switch filtering: conditional prediction and state-recovery advantage

Can an account of a possible purpose change preserve current-state information? After 128 independent observations with a purpose switch, the coherent mixture lowers fresh-endpoint logarithmic loss by 0.05473 nats versus static filtering and 0.02530 versus a fixed reset. Its 90% credible set contains the current maker state in 96.48% of cases, versus 33.20% for static filtering. Independent reconstruction and both complete replays verify this exploratory constructed-method result. The true law and possible switch time are supplied; historical process correspondence and human intent remain unestablished.

The comparison asks whether a coherent account of one possible purpose change
retains information about the current maker better than a static account or a
fixed history reset. All three methods receive the exact endpoint law. The
mixture receives the possible change time and gives equal prior mass to change
and no change. Actual switch status and maker identity remain hidden. This is a
supplied-law ruler for recursive learning, not a learned reader or a claim about
goals within an individual episode.

All 110,592 original three-operation native paths remain retained, including
local goals selected before actions and complete transient execution state.
Eight admission lineages cross all sixteen makers equally, two fixed observation
draws, lengths 32/128, switch/no-switch and independent/duplicated sources. Purpose
changes between episodes after observation 16/64; skill, belief and routine stay
fixed. The 2,048 streams contain 163,840 observations and 33,792 scored rows.

Endpoint logarithmic loss is expected negative log probability of a fresh endpoint
under the current maker, averaged equally over four contexts; smaller is better.
It predicts neither a future switch nor the historical local-goal path. The table
reports final-checkpoint paired differences: negative favors the first filter.
Each row keeps length, switch and source condition separate. Brackets are 95%
intervals from 10,000 paired coefficient-lineage resamples, seed 190501, with both
observation draws averaged within lineage. These are exploratory conditional
intervals from eight lineages, without simultaneous-comparison correction.

| Length | Purpose change | Sources | First minus second filter | Endpoint loss difference [95% interval], nats |
|---:|---|---|---|---:|
| 32 | No | Independent | Last sixteen minus Static | +0.01318 [+0.00947, +0.01639] |
| 32 | No | Independent | Possible-switch mixture minus Static | +0.00494 [+0.00162, +0.00749] |
| 32 | No | Independent | Possible-switch mixture minus Last sixteen | -0.00824 [-0.01160, -0.00558] |
| 32 | No | Duplicated | Last sixteen minus Static | +0.00923 [+0.00593, +0.01247] |
| 32 | No | Duplicated | Possible-switch mixture minus Static | +0.00660 [+0.00305, +0.00920] |
| 32 | No | Duplicated | Possible-switch mixture minus Last sixteen | -0.00263 [-0.00530, +0.00054] |
| 32 | Yes | Independent | Last sixteen minus Static | -0.03573 [-0.04652, -0.02716] |
| 32 | Yes | Independent | Possible-switch mixture minus Static | -0.04406 [-0.05502, -0.03468] |
| 32 | Yes | Independent | Possible-switch mixture minus Last sixteen | -0.00833 [-0.01095, -0.00601] |
| 32 | Yes | Duplicated | Last sixteen minus Static | -0.01924 [-0.02678, -0.01291] |
| 32 | Yes | Duplicated | Possible-switch mixture minus Static | -0.04154 [-0.05144, -0.03371] |
| 32 | Yes | Duplicated | Possible-switch mixture minus Last sixteen | -0.02230 [-0.02755, -0.01704] |
| 128 | No | Independent | Last sixteen minus Static | +0.03427 [+0.02730, +0.04124] |
| 128 | No | Independent | Possible-switch mixture minus Static | +0.00141 [+0.00052, +0.00250] |
| 128 | No | Independent | Possible-switch mixture minus Last sixteen | -0.03285 [-0.04051, -0.02520] |
| 128 | No | Duplicated | Last sixteen minus Static | +0.02906 [+0.02532, +0.03248] |
| 128 | No | Duplicated | Possible-switch mixture minus Static | +0.00194 [+0.00104, +0.00268] |
| 128 | No | Duplicated | Possible-switch mixture minus Last sixteen | -0.02711 [-0.03055, -0.02307] |
| 128 | Yes | Independent | Last sixteen minus Static | -0.02943 [-0.03975, -0.01931] |
| 128 | Yes | Independent | Possible-switch mixture minus Static | -0.05473 [-0.06565, -0.04485] |
| 128 | Yes | Independent | Possible-switch mixture minus Last sixteen | -0.02530 [-0.02954, -0.02130] |
| 128 | Yes | Duplicated | Last sixteen minus Static | -0.03463 [-0.04349, -0.02521] |
| 128 | Yes | Duplicated | Possible-switch mixture minus Static | -0.05384 [-0.06349, -0.04375] |
| 128 | Yes | Duplicated | Possible-switch mixture minus Last sixteen | -0.01921 [-0.02279, -0.01605] |

At 128 independent observations after a switch, the mixture's loss advantage
over static is 0.05473 nats, interval [0.04485, 0.06565], and over reset is 0.02530,
interval [0.02130, 0.02954]. Both clear the frozen 0.02-nat practical margin.
With 32 independent observations its static advantage also clears the margin,
but the 0.00833-nat advantage over reset is practically small. In duplicated
streams, mixture-versus-reset intervals cross the practical margin at both final
lengths; those cells do not establish a practical winner at that threshold.

Without a purpose change, the mixture adds 0.00494 nats at 32 independent
observations and 0.00141 at 128; both full intervals lie within the practical
margin. The fixed reset instead loses 0.03427 nats at 128 independent observations,
with its full interval above the margin. Retaining an alternative change account
and discarding old observations have different stationary costs in this model.

Current-state recovery is a separate outcome. In the next table state log loss
is negative log mass on the true persistent maker state; true-state mass is the
average probability assigned to that state. Credible-set inclusion is the fraction
of cases whose 90%-posterior set contains truth; size is its mean number of states.
Boundary ties within 1e-14 are included, so nominal 90% is not a fixed size or a
guaranteed coverage rate under each conditional switched population.

| Length | Purpose change | Sources | Filter | State log loss | True-state mass | Credible-set inclusion | Set size |
|---:|---|---|---|---:|---:|---:|---:|
| 32 | No | Independent | Static | 1.30709 | 35.98% | 94.14% | 3.762 |
| 32 | No | Independent | Last sixteen | 1.60854 | 26.65% | 94.53% | 5.301 |
| 32 | No | Independent | Possible-switch mixture | 1.45864 | 31.95% | 92.97% | 4.457 |
| 32 | No | Duplicated | Static | 1.46752 | 31.95% | 91.02% | 4.340 |
| 32 | No | Duplicated | Last sixteen | 1.63547 | 26.33% | 93.36% | 5.414 |
| 32 | No | Duplicated | Possible-switch mixture | 1.64224 | 27.65% | 91.02% | 5.086 |
| 32 | Yes | Independent | Static | 2.86172 | 15.81% | 58.20% | 4.492 |
| 32 | Yes | Independent | Last sixteen | 1.61957 | 25.61% | 94.92% | 5.238 |
| 32 | Yes | Independent | Possible-switch mixture | 1.34808 | 33.77% | 94.14% | 4.230 |
| 32 | Yes | Duplicated | Static | 2.70194 | 14.70% | 64.84% | 5.141 |
| 32 | Yes | Duplicated | Last sixteen | 2.03524 | 19.60% | 86.33% | 5.891 |
| 32 | Yes | Duplicated | Possible-switch mixture | 1.49320 | 28.99% | 95.31% | 5.016 |
| 128 | No | Independent | Static | 0.56323 | 65.21% | 98.83% | 2.004 |
| 128 | No | Independent | Last sixteen | 1.72157 | 24.79% | 91.80% | 5.402 |
| 128 | No | Independent | Possible-switch mixture | 0.66957 | 62.31% | 97.66% | 2.117 |
| 128 | No | Duplicated | Static | 0.69186 | 58.84% | 98.05% | 2.250 |
| 128 | No | Duplicated | Last sixteen | 1.70291 | 25.37% | 91.80% | 5.281 |
| 128 | No | Duplicated | Possible-switch mixture | 0.81515 | 54.82% | 97.27% | 2.477 |
| 128 | Yes | Independent | Static | 5.12934 | 13.09% | 33.20% | 2.164 |
| 128 | Yes | Independent | Last sixteen | 1.60967 | 27.02% | 92.97% | 5.258 |
| 128 | Yes | Independent | Possible-switch mixture | 0.63962 | 63.60% | 96.48% | 1.953 |
| 128 | Yes | Duplicated | Static | 4.27774 | 14.15% | 42.97% | 2.688 |
| 128 | Yes | Duplicated | Last sixteen | 1.55300 | 27.55% | 94.14% | 5.207 |
| 128 | Yes | Duplicated | Possible-switch mixture | 0.76864 | 56.35% | 96.09% | 2.324 |

At the switched 128-observation independent endpoint, inclusion is 96.48% for the
mixture, 92.97% for reset and 33.20% for static. Mixture-minus-static inclusion is
63.28 percentage points [58.98, 67.97]; mixture-minus-reset is 3.52 [0.39, 6.64].
Their mean set sizes are 1.953, 5.258 and 2.164 respectively. A narrow static set
can exclude truth. These state results do not establish the historical sequence
of local operations. No practical margin was declared for coverage or set size.

Duplicate streams retain 24/96 distinct sources rather than 32/128; every fourth
slot copies the preceding source's context, endpoint and original time. All
methods count each source once, and rotating contexts balance omissions. Source
condition differences include less evidence and cannot isolate pure dependence.
All prefixes, both draws and five metrics remain in 2,112 checked strata and
660 complete paired contrasts; the tables select final checkpoints explicitly.
Raw probabilities remain saved; the log-reporting floor is 1e-300.

Independent reconstruction verifies native paths and probabilities, endpoint laws,
every sampled stream and source assignment, sequential hypothesis products,
current-state remapping, all forecasts/scores and credible-set ties. Its maximum
discrepancy is 5.12e-13. Fourteen controls pass. A separate regroup independently
recombines every mean, paired lineage difference and bootstrap interval. All 35
deterministic original, adjacent and extracted-source outputs agree. All original
and checker sources, inputs, plans, environments and separate timing hashes verify.
Replays did not generate extra scientific observations. Earlier pending numerical
receipts and failed admission tests remain retained as historical records.

READER.zip contains 2,047 distinct anonymous observed-input packets, because two
streams share visible content. It contains only contexts, observed endpoints and
source identity/time. Maker, switch assignment, law, pairing, hypotheses and scores
remain evaluator material. SCIENTIFIC_EVALUATOR.zip preserves the original evidence;
VERIFICATION.zip adds independent reconstruction and complete paired regrouping.
Neither scientific archive is reader input.

**Warrant:** conditional prediction advantage after change, practically small
stationary mixture penalty, and separate current-state coverage improvement.
Exploratory constructed-method evidence, miniature — architecture untested; no
learned competence, human intent, general architecture result or confirmation.
**Pursuit:** remove supplied time and change-type information in the separately
frozen unknown-change comparison. Changed-tool support transfer and the fixed-fit
practice-support diagnostic remain independent prepared alternatives.

[Frozen comparison](TRANSIENT_FILTER_PROTOCOL.md),
[independent checker](TRANSIENT_REVIEW_PROTOCOL.md),
[complete regroup](../../../results/v19/G19-B-transient-filter-1/INDEPENDENT_REGROUP.json),
[next comparison](UNKNOWN_CHANGE_PROTOCOL.md).
