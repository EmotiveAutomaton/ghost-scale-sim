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

## H: Finite compression and changed questions

A memory optimized for one question need not preserve the best answers to another. With two memory symbols, the exact old-question optimum had logarithmic loss 1.18834 versus 1.18900 for the best observation-bit code, but new-question loss was worse: 1.13447 versus 1.12202. At four symbols the new-question disadvantage was 0.01559. These are descriptive constructed-method results from exhaustive finite codebooks, not evidence that a role-organized neural memory is uniquely necessary.

The table averages within twenty coefficient draws across the sixteen declared cells. Lower logarithmic loss is better. Codes are selected only for the old question; a generator-informed decoder measures the information they retain for new questions. This is not a learned decoder-transfer result.

| Maximum stored bits | Flat old loss | Observation-bit old loss | Flat new loss | Observation-bit new loss |
|---:|---:|---:|---:|---:|
| 0 | 1.47962 | 1.47962 | 1.16829 | 1.16829 |
| 1 | 1.18834 | 1.18900 | 1.13447 | 1.12202 |
| 2 | 1.18206 | 1.18532 | 1.12270 | 1.10710 |
| 3 | 1.18054 | 1.18054 | 1.05147 | 1.05147 |

All 4,140 partitions of eight histories were enumerated, with optima certified separately at cardinalities one, two, four and eight. Equal maximum storage does not mean equal code entropy: at two symbols, entropy was 0.48314 versus 0.52740 nats. At zero and three bits the methods coincide. Across cells, the two-symbol new-question difference has both signs and ties; the pooled advantage is not universal. The acquisition-interference contrast is zero here because the balanced latent-state mixture preserves the relevant observable distribution. Declared factor settings are not sixteen independent architectures.

All 320 retained units and 512 packet means passed checks; eight fixed whole-unit replays from extracted source passed. An earlier discarded pilot had an inactive shared-noise factor and an unwanted constant query-label entropy in the new loss. Both were corrected before this discovery packet. The independent/shared-noise construction now changes the joint while preserving its marginals. Full raw blocks, source, certificate inputs and replay bindings are in `results/v18/research-extension/H-core/`.

## F: Architecture boundaries and a held-out decision law

The purpose and change-of-state findings survive some changes of mechanics, but their boundaries matter. Under a new lexicographic decision law, action-focused inquiry succeeded in 93.75% of uniform-access tasks versus 70.63% for uncertainty-focused inquiry at the same three attempts. Selective state updating still improved goal-change loss, 0.532 versus static 2.976, and still hurt stationary prediction, 0.403 versus 0.296. In the original sixteen-cell design, the pooled action advantage under hidden access concealed four cells with a negative mean difference. These are descriptive results across declared constructions, not universal architectural claims.

The core table counts signs of mean paired method differences in sixteen fixed factor cells. These are descriptive signs, not sixteen significance decisions or independent architectural samples. Repeated cells are averaged within twenty coefficient draws.

| Comparison | Cells favoring first method | Cells tied | Cells favoring second method |
|---|---:|---:|---:|
| Selective vs static, goal | 16 | 0 | 0 |
| Selective vs static, opportunity | 1 | 0 | 15 |
| Selective vs static, skill | 0 | 0 | 16 |
| Selective vs static, stationary | 0 | 0 | 16 |
| Action vs uncertainty, hidden-state | 8 | 4 | 4 |
| Action vs uncertainty, query-dependent | 12 | 4 | 0 |
| Action vs uncertainty, uniform | 8 | 8 | 0 |

All four main factor contrasts and six pairwise interactions are retained for the declared A/B/D/H method contrasts, together with each cell and coefficient draw. A decision-rule change shifts the uniform-access action advantage by −38.75 percentage points. Source dependence reduces the goal-change predictive advantage of selective updating by 0.33871 loss units. These interactions describe this balanced finite design; they are not causal estimates about human architectures. D has an inactive source factor, and H acquisition interference leaves its balanced observable mixture invariant. Zero effects there are limitations of those experimental contrasts, not tests of all possible source or acquisition mechanisms.

The separate held-out-law batch contains 1,920 assigned evaluations: three access conditions, eight state-change conditions and finite compression, each crossing eight other-factor settings and twenty fresh coefficient draws. The fresh draws do not make a new decision law a representative sample of laws. At two memory symbols under that law, flat old-task optimization slightly improves new loss, 0.52707 versus 0.52821; at four symbols it loses, 0.43689 versus 0.35481. Compression transfer therefore also has storage and mechanism boundaries. No untouched-reserve confirmation is claimed.

All twelve held-out-law packets passed complete retained-unit and mean checks plus 96 fixed source-extracted whole-unit replays. `F_ROLLUP.json` retains lineage means, `FACTORIAL.json` retains all contrasts, and `FACTORIAL_points.json.gz` retains the paired observations. The factorial analyzer passed an independently specified additive-and-interaction known-answer test. Neural E/G cells remain in their own records and are incorporated into final synthesis after their separate audits.

## E: Learned memory and a changed reader purpose

Learned recurrent memory improved prediction on familiar kinds of questions but did not transfer reliably to a new question family. On familiar questions, the split reader had logarithmic loss 1.037, the flat reader 1.117 and the direct full-history reader 1.496. On new questions those rankings reversed: 2.399, 2.214 and 1.749 respectively; the exact nominal-family reference had loss 0.656. These are descriptive constructed-method results. Compact maintained state is useful for storage, but this experiment does not establish a representation that is best across purposes or a uniquely necessary role split.

The table reports expected logarithmic loss, averaged over fit seeds and architecture cells within each of thirty-two held-out coefficient draws. Lower is better. Training and development used separate draws, balanced role marginals and pairs, and development-only capacity/checkpoint selection.

| Test condition | Direct history | Flat recurrent | Split recurrent | Exact reference | Passive summary |
|---|---:|---:|---:|---:|---:|
| in-support | 1.49603 | 1.11653 | 1.03746 | 0.68490 | 1.05427 |
| new-combinations | 1.51168 | 1.13405 | 1.07278 | 0.69633 | 1.04807 |
| new-queries | 1.74914 | 2.21365 | 2.39879 | 0.65645 | 0.73488 |
| both-new | 1.74710 | 2.15446 | 2.39064 | 0.65382 | 0.71965 |

All eighteen fits completed their fixed twenty-four epochs and passed their separate learning controls. Widths 24/48, three seeds and equal full-distribution simulator supervision were shared. Parameter counts are retained rather than assumed equal. The intervention predictive bank matched the exact forecasts here; in the sixteen declared closure-test worlds it had ranks 4, 8 or 24 and passed the declared span/update checks. The passive bank had ranks 2, 4, 7 or 13 and failed update closure in every tested cell. These ranks are not a general predictive-state guarantee. The exact reference knows the generator family and uses a 24-state prior, including states absent from the balanced skill-positive test sampling; it is not claimed to be the optimal test-distribution posterior.

The independent audit reconstructed all 24,576 retained method/cell/draw rows into 768 lineage clusters and checked all 72 means. It replayed 128 fixed forecasts in each of 36 retained prediction files from extracted source and selected weights, with zero maximum difference. This is forecast replay, not full retraining. The original E attempt failed during Windows status sharing after five complete fits and part of a sixth. Its source, inputs, selected weights and latest optimizer checkpoints are retained separately; the corrected attempt reused the fixed data and is not a replication.

### New historical-role purpose with frozen memories

The separate diagnostic freezes each selected behavior encoder, then teaches an equal-supervision linear readout to answer curriculum, goal, belief and tradeoff questions. It reuses the same E histories and thirty-two lineages. The raw-history comparator has more features and readout parameters. A failed linear readout can reflect its restricted decoder, not information destroyed by the encoder.

| Test condition | Readout input | Conditional role loss | Mean role accuracy | Whole-state accuracy |
|---|---|---:|---:|---:|
| in-support | raw-history | 0.82074 | 50.73% | 7.03% |
| in-support | direct | 0.75572 | 53.16% | 10.03% |
| in-support | flat | 0.75277 | 53.97% | 9.83% |
| in-support | split | 0.75663 | 53.87% | 9.38% |
| in-support | exact | 0.32057 | 79.09% | 30.83% |
| in-support | passive-summary | 0.60250 | 64.49% | 23.37% |
| in-support | intervention-summary | 0.32057 | 79.09% | 30.83% |
| new-combinations | raw-history | 0.83044 | 48.58% | 4.30% |
| new-combinations | direct | 0.77571 | 49.76% | 6.45% |
| new-combinations | flat | 0.77426 | 51.24% | 5.40% |
| new-combinations | split | 0.77020 | 50.93% | 5.01% |
| new-combinations | exact | 0.32387 | 79.46% | 32.45% |
| new-combinations | passive-summary | 0.59227 | 66.15% | 24.93% |
| new-combinations | intervention-summary | 0.32387 | 79.46% | 32.45% |

The purpose audit checked 13,312 retained rows, 448 lineage clusters and 70 means, with 128 fixed cases replayed in each of 20 prediction files. No encoder was retrained for these labels. Native enactment is measured separately in A and the later source-uptake diagnostic. The review retains fit CPU, cached-query and update costs, model parameters and per-maker storage; storage reduction alone is not a full-cost advantage.

Numerical ties use uniform expected credit among role maxima within 1e-10. Whole-state accuracy is the product of per-role credits, not joint-posterior MAP accuracy. The original purpose packet is retained as superseded for accuracy; all proper losses, inputs, weights and forecasts are unchanged. Independent scalar reconstruction checked all 13,312 corrected accuracy rows before full mean and forecast replay. See E_PURPOSE_REPAIR.json.

## G: Intervention-trained internal representations

Training internal swaps against known interventions made the learned memories more useful for counterfactual prediction. The split reader trained on interventions had logarithmic loss 1.023, compared with 1.810 for behavior-only training, 1.873 for shuffled intervention targets and 1.680 for the direct pair reader. A flat recurrent reader with the same intervention supervision also improved, to 1.063. These are constructed-method results: the supplied intervention mapping became operationally useful, but the experiment does not establish uniquely necessary internal roles.

The table gives expected logarithmic loss, where lower is better. Swaps replace one declared role group using another observed maker history. Ordinary prediction and three swap conditions are separate outcomes. Results average roles, state pairs, fit seeds and sixteen architecture cells within each of twenty held-out coefficient draws; the 737,280 counterfactual probes are not independent samples.

| Reader | Ordinary prediction | Intended swap | Wrong role mapping | Incompatible partition |
|---|---:|---:|---:|---:|
| direct-pair | 1.66108 | 1.68001 | 1.68150 | unavailable |
| exact-public-history | unavailable | 0.63501 | unavailable | unavailable |
| flat-IIT | 0.97878 | 1.06346 | 2.53729 | 1.65442 |
| known-state-ceiling | 0.62509 | 0.62509 | unavailable | unavailable |
| split-IIT | 0.94057 | 1.02282 | 2.45598 | 1.70394 |
| split-behavior | 0.94903 | 1.80967 | 1.84753 | 1.80678 |
| split-shuffled-IIT | 1.13997 | 1.87284 | 1.87208 | 1.73691 |

The role groups are acquired skill plus persistent tradeoff, current goal, and routing belief. Ten fits use two seeds, fixed width 24 and twelve epochs; eight training and two development coefficient draws are separate from test draws. The flat and direct informed rivals receive the same counterfactual distribution labels as the organized reader. The behavior-only and shuffled-target arms are controls for this extra supervision, not equally informed competitors. Parameter counts are recorded rather than presumed equal.

Wrong mappings and an incompatible coordinate partition damage the trained recurrent interventions. A compatible simultaneous coordinate permutation preserves forecasts exactly. Thus alignment to this intervention family is demonstrable, but named coordinates and the chosen ontology are not uniquely identified. State swapping uses the simulator's known surgical rule; this is not evidence about neural anatomy, real-text concepts, human learning, or interventions outside the supplied family. The exact public-history reference uses the supplied law; the known-state ceiling has evaluator information unavailable to learned readers. Unavailable comparisons remain unavailable.

Independent reaggregation checked all 11,520 raw rows, 140 lineage clusters and 42 means. Extracted-source replay checked 128 fixed state-pair probes in each of ten retained forecast files, including ordinary, intended, wrong-mapping and incompatible-partition forecasts, and verified all ten parameter counts. This does not claim replay of the entire training trajectory. All inputs, weights and forecasts are retained in the checksum-bound multipart scientific bundle, which passed unpack-and-hash roundtrip verification. The earlier G plan was superseded before any scientific execution and remains retained.

## C/D follow-ons: Source dependence, native uptake and family revision

Source dependence changes the consequences of accepting a report, but it does not make source inference sufficient or necessary for action. In the two-root, six-copy shared-error cases, cautious and naive recommendation policies both initially succeeded in 53.13% of native tasks. After two noisy independent corrections, cautious success rose to 90.63% versus naive 68.75%; a stronger public-law checker reached 100% by testing both routines. In the separate revision experiment, removing copied evidence improved some comparisons and worsened others. These are descriptive constructed-policy and method results, with correct source identities supplied to the revision diagnostic.

C reuses all 1,728 retained provenance cases; it adds no independent source lineages. Reports select one of two public two-action routines for four practice repetitions, followed by an actual three-cell construction at code budget two. A shared seeded coin resolves numerical posterior ties within 1e-10, independently of truth. The public-law checker can inspect the task and both routines, so source inference is not necessary by construction. Its counterfactual checks cost 67 declared work units in this slice, in addition to actual practice and execution. These logical work units are not CPU instructions.

The table shows native task success before and after two noisy independent corrections, for two roots with six descendants each under shared error. Each row uses the same thirty-two source-network draws.

| Recommendation policy | Initial success | Corrected success |
|---|---:|---:|
| cautious-mixture | 53.12% | 90.62% |
| independent | 53.12% | 68.75% |
| known-graph | 53.12% | 90.62% |
| partial-graph | 53.12% | 90.62% |
| public-law-task-checker | 100.00% | 100.00% |
| selection-aware | 59.38% | 93.75% |
| unknown-graph | 53.12% | 90.62% |

D adds 7,680 fresh assigned evaluations across six truth-family conditions, early/late evidence, naive/source-aware readers, sixteen factor cells and twenty paired coefficient draws. The source factor now varies independent observations versus four roots each repeated three times; it was inactive in original D. Deduplication occurs before likelihood and empirical counting, after the declared evidence prefix is cut. Source identities are accurate here; C carries the uncertain and wrong-identity cases. No original D outcomes are replaced.

The next table reports expected logarithmic loss, lower is better, after late revision with four roots each repeated three times. Fixed retains the initial family; revision selects a candidate; mixture averages candidates. Each pair compares naive and source-aware handling of the same observations.

| Truth family | Fixed naive / aware | Revision naive / aware | Mixture naive / aware |
|---|---:|---:|---:|
| in-family | 0.74346 / 0.71885 | 1.09754 / 0.98746 | 0.77374 / 0.76117 |
| near-family | 1.15339 / 1.13644 | 1.34660 / 1.23510 | 1.21076 / 1.14222 |
| missing-rule | 3.90631 / 3.83934 | 0.69424 / 0.75443 | 0.69429 / 0.70857 |
| missing-acquisition | 0.69667 / 0.69586 | 0.84641 / 0.82612 | 0.73650 / 0.72717 |
| missing-opportunity | 2.11314 / 2.05337 | 1.07395 / 1.08764 | 0.87357 / 0.82391 |
| outside-menu | 2.71859 / 2.70918 | 0.86102 / 0.85400 | 0.82517 / 0.80418 |

With twelve genuinely independent roots, every corresponding reported mean is exactly equal between the source-aware and naive arms. With copies, source-aware late revision improves in-family loss from 1.09754 to 0.98746, but worsens missing-rule loss from 0.69424 to 0.75443. Both remain substantially better than the wrong fixed family in that latter condition. Deduplication is therefore not a blanket finite-sample ranking guarantee. The acquisition-missing condition may remain observationally equivalent under a latent relabeling, and outside-menu predictive success does not identify the true law.

All six source-revision packets passed complete retained-unit and mean checks plus 48 fixed whole-unit replays from extracted source. The native uptake diagnostic independently checks actual execution and code cost; its means were independently reconstructed from all raw cases. The known-answer suite includes independent-source equality, duplicated-root posterior invariance, correct/wrong routine success/failure, and a stronger-checker success control. Figures and unrelated review diagnostics in the same review directory retain separate scope and source bindings.
