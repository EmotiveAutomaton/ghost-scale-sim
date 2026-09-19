# V18.3 G: Intervention-trained internal representations

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
