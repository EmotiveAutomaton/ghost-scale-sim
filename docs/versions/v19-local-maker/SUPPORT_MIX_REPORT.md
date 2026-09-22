# Practice support mixtures: coverage benefit and nonmonotonic tradeoff

Can reallocating a fixed practice-evidence budget restore production in an unvisited context? At 512 episodes, a one-quarter matched-context mixture raises average success by 21.14 percentage points for active acquisition and 21.19 for demonstrations. Further reweighting is not uniformly beneficial; fully matched evidence loses some visited-context performance. Independent reconstruction and all paired intervals verify this exploratory constructed-method result, not historical process correspondence or human intent.

The fixed comparison reweights saved transition counts from restricted-start
and matched-start acquisition at weights 0, 1/4, 1/2, 3/4 and 1. It retains all
eight coefficient lineages, two training draws and two acquisition seeds, at
32/128/512 episodes. The one-half count prior appears once, and empirical feedback
mass is exactly three times the episode budget. Fractional sufficient-statistic
mixtures are not new on-policy learning histories. Both endpoints are identities
with the completed parent experiment, not additional independent replications.

Production success is the probability of the functional endpoint condition under
the stipulated high-skill native actor. The table gives equal-context success
in percent; rows are episode budget and acquisition policy, columns are the
fixed weight assigned to matched-start evidence. Restricted-start evidence visits
only the first context; matched-start evidence visits both. Every weight is
reported, without selecting an optimal policy from outcomes.

| Episodes | Acquisition | 0 | 1/4 | 1/2 | 3/4 | 1 |
|---:|---|---:|---:|---:|---:|---:|
| 32 | active | 60.59 | 76.76 | 76.66 | 76.63 | 73.97 |
| 32 | demonstration | 58.81 | 76.53 | 76.40 | 76.68 | 74.64 |
| 128 | active | 62.07 | 81.36 | 81.51 | 81.85 | 80.75 |
| 128 | demonstration | 62.57 | 82.80 | 82.38 | 81.50 | 81.53 |
| 512 | active | 67.25 | 88.39 | 88.58 | 88.58 | 86.15 |
| 512 | demonstration | 68.18 | 89.36 | 89.83 | 88.74 | 87.20 |

The next table reports the first-quarter mixture minus restricted-only evidence, in percentage points. Visited and unvisited columns refer to the original restricted-start context and its previously absent counterpart. Intervals are 95% paired lineage bootstrap intervals, averaging the four draw/acquisition-seed pairs inside each of eight lineages before 10,000 resamples. They condition on saved acquisition and are exploratory, not simultaneous family-wide guarantees.

| Episodes | Acquisition | Equal-context change [interval] | Visited change [interval] | Unvisited change [interval] |
|---:|---|---:|---:|---:|
| 32 | active | +16.18 [+12.55, +19.72] | -0.97 [-3.31, +0.88] | +33.32 [+27.26, +39.48] |
| 32 | demonstration | +17.72 [+14.36, +20.84] | +0.29 [+0.00, +0.87] | +35.16 [+28.33, +41.56] |
| 128 | active | +19.29 [+16.75, +21.86] | +1.08 [+0.00, +2.85] | +37.50 [+32.40, +43.14] |
| 128 | demonstration | +20.23 [+17.47, +22.83] | +0.00 [+0.00, +0.00] | +40.46 [+34.94, +45.66] |
| 512 | active | +21.14 [+18.40, +23.74] | +1.15 [+0.24, +2.29] | +41.12 [+35.64, +46.37] |
| 512 | demonstration | +21.19 [+18.42, +23.81] | -0.54 [-1.23, +0.10] | +42.91 [+37.34, +48.27] |


All six first-quarter equal-context intervals exceed the frozen two-point
practical margin. At 512 episodes the previously unvisited context rises from
40.50% to 81.62% for active acquisition and 83.41% for demonstrations. Its paired
gains are 41.12 [35.64, 46.37] and 42.91 [37.34, 48.27] points. In the original
visited context, active success changes by +1.15 [+0.24, +2.29] points, while
demonstration success changes by -0.54 [-1.23, +0.10]. The demonstration interval
is inside the two-point margin; the active interval does not establish practical
equivalence even though its mean is smaller than that margin.

Further reweighting is not a monotonic improvement. At 512 episodes, moving from
three-quarter to fully matched evidence lowers equal-context success by 2.43
[1.61, 3.06] points for active acquisition and 1.53 [0.67, 2.50] for demonstrations.
These intervals establish a signed decrease but straddle the two-point practical
boundary. The visited-context losses are 5.10 [3.92, 6.13] and 3.36 [1.55, 5.24]
points, while unvisited-context means improve only 0.24 and 0.29 points. At 128
episodes the last demonstration step instead has a small positive mean. These
results reject a universal monotonic account on the stipulated saved mixtures;
they do not identify a generally optimal mixture weight.

None of the 15 active-minus-demonstration equal-context intervals establishes
an advantage for either acquisition arm. At 512 episodes and three-quarter
weight, the interval [-1.93, +1.55] is inside the practical margin; most other
intervals remain inconclusive rather than equivalent. Every adjacent contrast,
all six endpoint-span contrasts and all four fit means remain in REGROUP.json.

Support and policy changes are measured separately from success. Adding any
positive matched weight gives observed transitions to the absent context, while
also changing probabilities in previously visited rows. Thus the experiment
shows a benefit of this evidence allocation; it does not isolate a pure support-
only causal effect. The table below retains mean visited rows, newly visited
rows, policy changes and changed previously-seen probability rows at the
512-episode first-quarter mixture. Each pair is visited/unvisited context;
these counts have no two-point success threshold.

| Acquisition | Visited rows | Newly visited rows | Policy changes | Changed seen probabilities |
|---|---:|---:|---:|---:|
| active | 47.34/41.34 | 1.59/41.34 | 0.78/8.59 | 40.75/0.00 |
| demonstration | 55.75/53.03 | 0.00/53.03 | 0.47/9.41 | 52.66/0.00 |


The independent checker imports no producer kernel, planner, mixture or scorer.
It reconstructs all 576 parent tables from ordered observations and all 960
mixed tables. Independent Bellman reductions check saved values and optimality;
explicit sums over 512 paths per context evaluate the original saved policies.
Maximum discrepancy is 3.34e-16. There are 8,911 different near-tie argmax choices
across the checked tables, but the largest saved-action value gap is 1.12e-16.
Those differences do not replace the frozen first-maximum policy. General
robustness to other tie conventions remains untested.

A separate review independently recombines every one of 900 metric estimates:
30 means and 45 contrasts, each with 12 measures, all lineage values, four fit
means and 10,000-resample intervals. All 1,743 deterministic outputs match across
original, adjacent and extracted-source executions. All 449 checker sources,
1,747 inputs, plans, environments and timing hashes verify. Eleven isolated
controls include a complete native fixture, known/inert laws, pairing and count,
score and ordering corruption. Earlier fixture-mutation failures remain charged.
The earlier 120 Ghost/five Torch regression is not claimed to have been repeated.

OBSERVED_READER.zip is the unchanged deduplicated observed-transition export:
128 content-hash-named arrays with no lineage, seed, arm, score or native law.
No new reader task is claimed. SCIENTIFIC_EVALUATOR.zip retains complete raw
counts, policies, parent inputs, evaluator law and source; REPLAY_RECEIPTS.zip
retains execution receipts; VERIFICATION.zip adds the independent reconstruction
and complete regroup. All members and artifact hashes are public. Evaluator
material never enters reader inputs.

**Warrant:** exploratory constructed-method evidence for useful fixed-budget
evidence reallocation, with a nonmonotonic visited-context tradeoff and no
established acquisition-policy advantage. This is conditional on saved fits,
the frozen planner and a small native world; miniature — architecture untested.
No confirmation, inverse process recovery, embodiment or human intent follows.
**Pursuit:** the independently prepared quarter/three-quarter actual-change
test advances the filtering question. Bounded changed-primitive feedback and
one source-identity omission test remain independent prepared alternatives.
The week, untouched confirmation lineages and protected reserve remain open.

[Frozen experiment](SUPPORT_MIX_PROTOCOL.md),
[independent review](SUPPORT_MIX_REVIEW_PROTOCOL.md),
[evidence roles](../../../results/v19/G19-E-support-mix-1/README.md).
