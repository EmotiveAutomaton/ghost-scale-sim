# V18.3 research results in progress

## A: purpose and paid investigation

Does the purpose of investigation change which evidence is useful? In the uniform-access comparison, choosing demonstrations for learning to act produced 45.31% own-task success, versus 25.94% for uncertainty reduction when both made exactly three attempts. The action-focused reader predicted later behavior less accurately: expected logarithmic loss, where lower is better, was 0.997 versus 0.638. When access depended on hidden maker state, the corresponding success rates were 32.19% and 9.06%. This is descriptive evidence about constructed mechanisms: useful evidence depends on the reader's purpose, and learning to act is not interchangeable with identifying a maker. It does not establish human learning or a universal selector advantage.

The table compares native three-cell task success and future predictive loss. Every row makes exactly three query attempts; missing responses remain charged. Values average 16 architecture cells within each of 20 coefficient draws. Logarithmic loss is measured in nats and lower is better.

| Access mechanism | Selector objective | Own-task success | Future logarithmic loss |
|---|---|---:|---:|
| Uniform | Learn to act | 45.31% | 0.99745 |
| Uniform | Reduce state uncertainty | 25.94% | 0.63825 |
| Query-dependent | Learn to act | 39.38% | 0.95358 |
| Query-dependent | Reduce state uncertainty | 22.50% | 0.64050 |
| Hidden-state-dependent | Learn to act | 32.19% | 1.04767 |
| Hidden-state-dependent | Reduce state uncertainty | 9.06% | 0.79390 |

The stopping prediction selector makes 1.22 attempts under uniform access and reaches loss 0.64096; the three-attempt fixed diagnostic reaches 0.64010. Fewer requests alone do not establish a total-cost advantage. Selector cost counts likelihood-table entries evaluated under an explicit price, not hardware instructions. Primitive observation, practice, planning and final execution costs remain separate. Whole-run CPU is recorded independently. The robust selector tests three access-prior perturbations, not an unrestricted distributionally robust optimum.

### Repair, validity and scope

The first implementation recorded four robust likelihood evaluations but charged only one when deciding whether to stop. The repair charges all four in the stopping utility. A planted regression control passes only if the ordinary selector continues and the robust selector stops at a utility between those costs. Sixteen isolated tests passed. Matched three-attempt task and information arms were also added. The four original packets and their complete proofs remain **superseded**, never an independent replication; the four `-r1` packets replace them. Ordinary existing task outputs are unchanged.

The current 1,088 evaluations comprise three access conditions of 320 and 128 known-answer controls. Means and uncertainty intervals in [A_ROLLUP.json](../../../../results/v18/research-extension/A_ROLLUP.json) average architecture cells within coefficient draw; all intervals are descriptive. Complete raw checks and mean reaggregation passed, with eight fixed whole-unit source-extracted replays per packet, 32 current replays total. This is bounded replay, not a full rerun. No-information, resolved-state, failed-access and identical-evidence controls remain explicit. Native craft uptake reuses actual repeated-fragment learning. Historical state discrimination concerns the declared finite state family, not an arbitrary recovered biography.

The miniature varies decision rule, curriculum interference, available opportunities and repeated sources. Those 16 constructions are tested architectures, not a representative architecture population. Findings remain discovery; fresh third-rule severity and the remaining families are pending.

## B: Changing maker state

What should a reader preserve when a maker changes? After an unannounced goal change, updating goal and belief while retaining the slower roles reduced future logarithmic loss from 3.299 for a static reader to 1.152; lower loss means better probabilistic prediction. The same rule lost when nothing changed, 1.023 versus 0.904, and when acquired skill changed, 1.058 versus 0.942. This is a descriptive constructed-mechanism result: selective updating helps when its timescale assumptions fit the change, and can hurt when they do not. These inference comparisons do not establish better craft uptake, because all readers shared the same separate learning routine.

The table reports future logarithmic loss after the possible change point, averaged within each of 20 coefficient draws across 16 architecture cells. Lower is better. The known-change reader receives the true change time and is a privileged timing comparison.

| What changes | Static history | Selective fast-role update | Update every role | Known change time |
|---|---:|---:|---:|---:|
| stationary | 0.90364 | 1.02325 | 1.04605 | 0.90364 |
| goal | 3.29933 | 1.15235 | 1.18820 | 0.94900 |
| belief | 2.95316 | 1.10247 | 1.15925 | 0.90112 |
| skill | 0.94206 | 1.05753 | 1.04452 | 0.91082 |
| tool | 0.52077 | 0.63100 | 0.64973 | 0.52077 |
| opportunity | 2.85215 | 3.00221 | 2.52897 | 2.85215 |
| gradual | 3.10481 | 1.16896 | 1.21272 | 1.56120 |
| return | 2.04603 | 1.23059 | 1.29380 | 0.95091 |

Eight packets retain 2,560 trajectories, each with 32 time steps. Time steps are not independent samples. Windowed and discounted readers, stationary and return controls, false reset counts, retained skill probability and identical native learning executions remain in the raw record. The selective reader models goal/belief transitions at a fixed rate; it is not a fully fitted Bayesian change-point detector. Its reset diagnostic does not feed a separate undisclosed controller. When opportunities change outside the fitted family, selective updating can lose to broader forgetting. The shared native learning routine prevents attributing execution differences to these inference methods. All retained-unit checks and aggregate means passed; eight source-extracted replays per packet give 64 bounded whole-trajectory replays.

## C: Corroboration and source dependence

When do repeated reports become misleading corroboration? With two underlying sources, six reports per source and a shared error, treating all reports as independent produced logarithmic loss 3.509 and assigned over 95% probability to the wrong answer in 21.88% of networks. A reader allowing uncertain source groups and shared error reduced loss to 0.643 and had no such initial false-confidence cases; after two noisy independent corrections, its rate was 3.13%. This is descriptive evidence about constructed provenance mechanisms and finite inference methods. Dependence modeling helps in this construction; it does not make uncertain provenance or correction infallible.

The table fixes two underlying roots and six descendants per root, except the independent condition, which instead contains twelve independent roots. It reports initial logarithmic loss; lower is better. The known-graph reader also knows the true bias/selection mechanism and is a privileged comparison.

| Source mechanism | Treat reports independently | Known graph/mechanism | Uncertain graph plus shared error |
|---|---:|---:|---:|
| independent | 0.08988 | 0.08988 | 0.29665 |
| copied | 0.73468 | 0.30541 | 0.42109 |
| shared-error | 3.50937 | 0.64559 | 0.64263 |
| selected | 0.60115 | 0.52137 | 0.48024 |
| partial | 1.59801 | 0.53799 | 0.53776 |
| wrong-provenance | 0.32533 | 0.31759 | 0.50003 |

Six packets retain 1,728 source-network evaluations: six mechanisms, root counts one/two/four, one/three/six reports per root, and 32 network draws per condition. The root truth and independent corrections are paired across descendant counts. Reports descending from a root are not independent samples. Under ordinary copied reports, expanding two roots from one to six reports each changes the independence-assuming loss from 0.309 to 0.735, while the known-graph loss stays near 0.305. More reports can still help recover a noisy root; they do not create fresh root evidence.

False confidence means more than 95% posterior probability on the wrong binary claim. Independent corrections themselves have reliability 0.9, so a correction can be wrong. Graph uncertainty is a finite catalog of block and cyclic partitions, not unrestricted provenance discovery. Partial or false source metadata constrains this catalog; inconsistent cases retain explicit invalid denominators. Selection-aware and cautious readers introduce additional assumptions and do not uniformly win. All retained posterior/loss checks and aggregate means passed; 48 fixed source-extracted whole-network replays passed.

## D: Bounded explanation revision

When does revising an explanation improve prediction? When the supplied decision rule was wrong, selecting a replacement after eight observations reduced future logarithmic loss from 3.582 to 0.623; selecting at the initial cue instead gave 2.948. But when the original family was correct, late revision worsened loss from 0.613 to 0.692. These descriptive constructed-method results show a benefit from evidence-informed revision and a cost from unnecessary or premature commitment. Averaging over the candidate families was a strong rival; no unique benefit of committing to one explanation, general open-world repair, or human mechanism is established.

The table reports future logarithmic loss after the same twelve observations. Lower is better. Early revision commits at the initial cue; late revision commits after the first eight observations; both then update states using all twelve. Mixture inference keeps all five candidates.

| True mechanism relative to supplied family | Fixed family | Early revision | Late revision | Candidate mixture |
|---|---:|---:|---:|---:|
| in-family | 0.61338 | 1.31088 | 0.69158 | 0.62797 |
| near-family | 1.01893 | 1.91485 | 1.03314 | 0.97422 |
| missing-rule | 3.58161 | 2.94806 | 0.62347 | 0.62348 |
| missing-acquisition | 0.60999 | 1.51030 | 0.64583 | 0.62788 |
| missing-opportunity | 1.63957 | 1.92454 | 0.62784 | 0.61849 |
| outside-menu | 2.75313 | 2.83774 | 0.71414 | 0.70042 |

Six packets retain 3,840 evaluations crossing two cue orders, sixteen declared factor cells and twenty coefficient draws. Evidence and future outcomes match across order; order-invariant fixed and mixture readers are controls. Source-sharing is inactive in this family: its records are independent, so those sixteen slots contain eight distinct mechanism settings. Averaging redundant slots does not increase the twenty independent coefficient draws. Missing acquisition can be observationally equivalent after relabeling the latent repertoire; a different hidden curriculum alone need not make behavior outside the supplied family.

Candidate choices use prefix evidence, not future outcomes. The fixed catalog includes misleading additions and a truth-outside-menu law; good prediction outside that menu does not prove recovery of the true law. The cautious likelihood, empirical predictor, uniform-abstention rule and cue-only commitment remain explicit. The candidate evaluation count is a logical work measure, not complete method CPU. All retained forecast checks and aggregate means passed, with 48 fixed source-extracted whole-unit replays.
