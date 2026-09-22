# Unknown-time/type filtering: conditional prediction and current-state advantage

Can a coherent filter retain current-state information when change time and factor are unknown? After 128 independent observations, the unknown-time/type mixture lowers fresh-endpoint logarithmic loss versus static filtering by 0.05290 nats after a purpose change and 0.08056 after a skill change. Its 90% credible set includes the current maker in 96.48% and 98.05% of cases, respectively. Independent reconstruction, paired regrouping and both complete replays verify this exploratory constructed-method result. The exact law and allowed change family remain supplied; historical process correspondence and human intent are unestablished.

The comparison removes knowledge of the possible switch time and switch factor
in separate arms. Static filtering, a last-sixteen-distinct-source reset, a mixture
with supplied time/type, an unknown-time mixture with supplied type, and an
unknown-time/type mixture receive identical observed endpoint streams. Every arm
receives the true endpoint law. The horizon, allowable times, possible factors
and prior chance of a change remain supplied assumptions. This is an exact
reference for learned readers, not learned change detection.

All 110,592 native three-operation paths remain available across the eight original
admission lineages. Sixteen makers receive equal weight, crossed with two fixed
observation draws, lengths 32/128, purpose/skill switches, no-switch controls and
independent/duplicated-source conditions. A switch complements the declared factor
after observation 16/64. Actual changes occur only at the midpoint; the experiment
does not assess performance across all possible actual times. It uses no new fit,
episode seed or reserved confirmation lineage.

The batch retains 4,096 streams, 327,680 presented observations, 112,640 scored rows
and 7,040 summary strata. No-change streams for purpose and skill are identical
paired controls, never extra independent observations. The exported 3,055 distinct
visible packets deduplicate identical input content. Duplicated streams have 24/96
distinct sources instead of 32/128. Their contrast includes reduced evidence
quantity and cannot isolate a pure dependence effect. Source identity includes
original time; a repeated source does not become fresh post-switch evidence.

Complete hypotheses retain initial maker, change time and type. Half the prior
mass is no change, with the remainder uniform over admitted changed hypotheses.
The unknown-type arm divides change mass equally between purpose and skill;
unknown times range from observation 8 through length minus 8 inclusive. Possible
future changes do not remap present state. Full float64 joint arrays and explicit
hypothesis/row mappings are retained; no pruning or independent-factor approximation
is used.

Endpoint logarithmic loss measures prediction error for a fresh episode under the
current maker, averaged over four contexts. Current-state logarithmic loss, actual
state probability, 90% credible-set inclusion and set size are separate measures.
Type-posterior mass describes the stipulated candidate family. Boundary ties within
1e-14 and the 1e-300 reporting floor are unchanged. Independent numerical reconstruction and paired regrouping now verify all these measures.

The first table shows final predictions after 128 independent observations. Rows
separate actual change and comparator; negative differences favor unknown time/type.
Intervals resample eight paired coefficient lineages, averaging two draws within
lineage with equal weights over sixteen makers. They are conditional exploratory
intervals, not simultaneous confidence statements or training-population uncertainty.

| Actual change | Comparison | Endpoint loss difference [95% interval], nats |
|---|---|---:|
| None (purpose reference) | Unknown time/type minus Static | +0.00417 [+0.00277, +0.00559] |
| None (purpose reference) | Unknown time/type minus Last sixteen sources | -0.03010 [-0.03632, -0.02379] |
| None (purpose reference) | Unknown time/type minus Supplied time/type | +0.00275 [+0.00095, +0.00467] |
| Purpose | Unknown time/type minus Static | -0.05290 [-0.06415, -0.04264] |
| Purpose | Unknown time/type minus Last sixteen sources | -0.02347 [-0.02764, -0.01957] |
| Purpose | Unknown time/type minus Supplied time/type | +0.00183 [+0.00082, +0.00282] |
| None (skill reference) | Unknown time/type minus Supplied time/type | +0.00310 [+0.00169, +0.00444] |
| Skill | Unknown time/type minus Static | -0.08056 [-0.08957, -0.07265] |
| Skill | Unknown time/type minus Last sixteen sources | -0.02210 [-0.02988, -0.01598] |
| Skill | Unknown time/type minus Supplied time/type | +0.00374 [+0.00205, +0.00552] |

The unknown-time/type mixture improves over static filtering after both changes
at both lengths and in both source conditions; every final-checkpoint interval
clears the 0.02-nat practical margin. Its advantages over the fixed reset are smaller;
their intervals do not all establish a benefit larger than that margin. Across
all final length/factor/change/source strata, differences from supplied time/type
have 95% intervals within -0.02 to +0.02 nats (extremes -0.00411 and +0.01912).
This is conditional practical equivalence on this actual-midpoint population,
not equivalence for every prefix, change time, model or prior. Stationary penalties
versus static filtering are positive but remain within the same margin.

The second table retains all five methods at the same final independent checkpoint.
State log loss is the negative log probability of the true current maker, in nats;
true-state probability is its average assigned mass. Set inclusion is the fraction
whose smallest tie-inclusive 90% posterior set contains truth; set size counts its
makers. The no-change row family is shown once, using the purpose reference.

| Actual change | Filter | State log loss | True-state probability | 90% set inclusion | Set size |
|---|---|---:|---:|---:|---:|
| None | Static | 0.56323 | 0.65215 | 98.83% | 2.004 |
| None | Last sixteen sources | 1.72157 | 0.24793 | 91.80% | 5.402 |
| None | Supplied time/type | 0.66957 | 0.62310 | 97.66% | 2.117 |
| None | Unknown time | 0.76011 | 0.57665 | 96.48% | 2.473 |
| None | Unknown time/type | 0.74300 | 0.57018 | 98.83% | 2.586 |
| Purpose | Static | 5.12934 | 0.13086 | 33.20% | 2.164 |
| Purpose | Last sixteen sources | 1.60967 | 0.27021 | 92.97% | 5.258 |
| Purpose | Supplied time/type | 0.63962 | 0.63603 | 96.48% | 1.953 |
| Purpose | Unknown time | 0.69579 | 0.59303 | 97.66% | 2.180 |
| Purpose | Unknown time/type | 0.75504 | 0.57305 | 96.48% | 2.289 |
| Skill | Static | 174.51784 | 0.24322 | 47.66% | 2.168 |
| Skill | Last sixteen sources | 1.53738 | 0.27491 | 96.88% | 5.363 |
| Skill | Supplied time/type | 0.54451 | 0.66137 | 98.44% | 1.938 |
| Skill | Unknown time | 0.57469 | 0.63957 | 98.44% | 2.043 |
| Skill | Unknown time/type | 0.67816 | 0.60461 | 98.05% | 2.246 |

State-set inclusion is separate from exact recovery and endpoint prediction. The
static skill-change log loss is especially sensitive to the 1e-300 reporting floor
when old evidence excludes the actual current state. No practical coverage margin
was declared. The broad family supplied to the mixture contains the real change;
this result does not test outside-family change detection. The full machine-readable
record preserves both no-change reference families, every prefix, all source
conditions and both draws; identical streams never become extra observations.

All 51 deterministic outputs match the original, adjacent and extracted-source
executions. Every source, input, plan, environment and execution-specific timing
binding verifies. The producer has 431 frozen source files and eight native input
blocks. Replay agreement cannot rule out a shared implementation error.

The independent checker reconstructs native path probabilities, endpoint laws,
observation/source assignments, complete hypotheses, scaled likelihood products,
current-state remapping, full binary joint arrays, forecasts, every score and
population denominator. Its scaled products are independent of the producer's
logarithmic accumulation. All saved joint rows must be consumed once and matched
to their mappings. It tests both factor involutions, type/time priors, future
remapping, duplicate/conflicting sources, empty support, uniform nulls, reader
identity rejection and a complete native fixture.

Ten isolated tests pass, including native/portable dispatch and the complete
13,824-path fixture with both draws, lengths and factors. All completed inputs and 435 checker sources verify. Its completed regroup preserves every length,
factor, actual-change, source and checkpoint stratum; two observation draws are
averaged within coefficient lineage. All ten paired arm contrasts retain five
metrics and 10,000 paired lineage resamples. The eight lineages are the uncertainty
units. No new independent population or universal training uncertainty is claimed. The checker agrees within 5.12e-13 over 110,592 native paths, 4,096 streams, 112,640 scores and 7,040 strata. A separate review independently recombines all 4,400 contrasts, lineage values, means and 10,000-resample intervals. A first review-script field-name error is retained and charged; it did not change scientific evidence.

READER.zip contains observed contexts/endpoints/source identities only. Scientific
and evaluator records are split into a core archive and eight lineage archives,
with exact member hashes in EVALUATOR_MANIFEST.json. These include every native
path, stream, joint array, mapping, posterior, forecast, source, plan and timing.
They are reproduction material and never reader input. The independent checker
admission is a separate source-bound record.

**Warrant:** conditional endpoint advantage after change, practical final-checkpoint
equivalence to supplied time/type, and separate current-state coverage improvement.
Exploratory constructed-method evidence, miniature — architecture untested. No
learned change detector, historical local-process identification, human intent or
confirmation follows. **Pursuit:** implement the predeclared changed-tool support
transfer. Fixed-budget practice mixtures and off-midpoint actual changes remain
independent alternatives; none requires a filtering win. The week and protected
reserve remain open. VERIFICATION.zip adds independent reconstruction and full
paired regrouping to the original split scientific/evaluator archives.

[Frozen comparison](UNKNOWN_CHANGE_PROTOCOL.md),
[independent review](UNKNOWN_REVIEW_PROTOCOL.md),
[support-mixture alternative](SUPPORT_MIX_PROTOCOL.md),
[evidence roles](../../../results/v19/G19-B-unknown-change-1/README.md).
