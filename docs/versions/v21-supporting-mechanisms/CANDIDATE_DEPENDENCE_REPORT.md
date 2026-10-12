# V21 candidate-specific dependence controls

Do the three learned-maker candidates survive a generic dependence correction? The native artifact candidate loses its distinctive advantage, while the two shifted candidates still improve log loss by about 0.053 and 0.051 nats over validation-selected episode weighting. These are verified constructed-method development results; maker-specific causation, copying-specific superiority and confirmation remain unestablished.

All three previously reported candidates and their native/changed companions were fixed before this diagnostic. Artifact evidence uses 8,192 native labels; context and complete evidence use 512. Each has four past episodes of the same evidence tier, without observed inspection, eight training laws, eight reused development worlds and 256 cases per world. Changed execution affects current and historical episodes; this is not the sparse current-only B07 intervention. Two fitted seeds are averaged inside each world before 4,096 paired whole-world bootstrap draws. Cases, fits, copying conditions and repeated worlds are not independent replications. This targeted comparison adds no new candidate-selection worlds.

All 147,456 common zero-copy forecasts reproduce the original B01 readers within 5.56e-17 in probability, including the original exact-table fallback. The original fixed-schedule network and extra-label fresh-law diagnostic remain separately bound; they are not newly fitted copying rivals. Six candidate conditions and one new support diagnostic are accepted here. The full artifact-native scientific replay matches all 578 files, excluding only fit CPU fields in two JSON records. All 995,328 newly checked forecasts pass independent scoring; twelve prior packet score receipts are reused by exact hashes. This includes upstream support/headroom checks, not 995,328 independent observations.

Natural-log loss, in nats, and full multiclass Brier are proper probability losses; lower is better. The next table compares each candidate with the calibration-selected episode rival under zero copying. The difference is learned-maker loss minus episode loss; negative favors the maker. Brackets are descriptive 95% whole-world intervals, without multiplicity correction.

| Candidate | Maker loss | Episode loss | Loss difference [interval] | Maker Brier | Episode Brier |
|---|---:|---:|---|---:|---:|
| Artifact, native, 8,192 labels | 3.43069 | 3.42321 | +0.00747 [-0.01574, +0.03411] | 0.95014 | 0.94946 |
| Context, changed, 512 labels | 3.09418 | 3.14702 | -0.05284 [-0.07776, -0.02877] | 0.94428 | 0.94440 |
| Complete, changed, 512 labels | 1.09972 | 1.15119 | -0.05147 [-0.06203, -0.04204] | 0.64573 | 0.67290 |

Both artifact fits select power 0.5; context and complete fits select 0.25. Selection uses only disjoint validation laws, averaging all three registered copying rates without supplied support, and is independently reconstructed from training counts. No test outcome selects the power. These rates were part of calibration; transfer to unseen copying rates is untested.

The artifact candidate does not show superiority over the generic rival; the interval is not a formal equivalence result. At half and high copying, the generic rival beats it by 0.04580 and 0.09453 nats. Stop a maker-specific attribution for this candidate. Both shifted candidates retain a greater-than-0.02-nat descriptive lower-bound gain over calibrated episodes with no copying. Native companions still lose to the exact table. These retained shifted effects justify a persistence or information-use discriminator, not a necessary-maker interpretation.

The following rows show the two shifted candidates against the calibrated rival as copying increases. Copy probability is the chance each later history episode is replaced by the first; current case and target are unchanged within each execution condition. Loss difference and Brier difference are maker minus calibrated rival.

| Evidence | Copy probability | Loss difference [interval] | Full Brier difference |
|---|---:|---|---:|
| Context, changed, 512 labels | 0.0 | -0.05284 [-0.07776, -0.02877] | -0.00012 |
| Context, changed, 512 labels | 0.5 | -0.07265 [-0.09562, -0.04971] | -0.00450 |
| Context, changed, 512 labels | 0.9 | -0.10166 [-0.13028, -0.07457] | -0.01144 |
| Complete, changed, 512 labels | 0.0 | -0.05147 [-0.06203, -0.04204] | -0.02717 |
| Complete, changed, 512 labels | 0.5 | -0.05270 [-0.06163, -0.04421] | -0.02790 |
| Complete, changed, 512 labels | 0.9 | -0.05347 [-0.06303, -0.04456] | -0.02819 |

The copying cells have no exact law-marginal integration or validation-stopped direct refit to copying. Their gains are descriptive diagnostics, not admitted superiority screens. Every original independent-history gate is exact-law with lower bound above 0.02 and both direct fits validation-stopped. That gate is not transferred to a new copying distribution.

Supplying copy roots improves artifact-native maker loss by 0.10560 nats at high copying, but worsens shifted context loss by 0.02053 and shifted complete loss by 0.00275. Duplicate removal is therefore not a universal improvement. This known-root ablation has evaluator provenance privilege. Removing the duplicated current evidence factor from naive Bayes is a separate control, retained below and in the complete aggregate record.

Each table below reports every new reader’s changed-execution minus native loss, separately at no, half and high copying. Positive means execution change worsened that reader. Pairing is by coefficient world after averaging fits, not by treating cases as independent. All proper-score, accuracy and confident-error intervals remain in the aggregate record.

**Artifact evidence:** the three numeric columns are log-loss changes at copy probabilities 0, 0.5 and 0.9.

| Reader | No copying | Half copying | High copying |
|---|---:|---:|---:|
| Learned maker | +1.09827 | +1.24324 | +1.43619 |
| Maker, supplied copy roots | +1.09827 | +1.32850 | +1.52338 |
| Validation-selected episodes | +1.65961 | +1.81634 | +2.02414 |
| Episodes, power 0.25 | +1.26565 | +1.30363 | +1.35643 |
| Episodes, power 0.5 | +1.65961 | +1.81634 | +2.02414 |
| Episodes, power 1 | +2.73709 | +3.16746 | +3.70054 |
| Native-trained direct | +1.20388 | +1.44456 | +1.70973 |
| Exact chronological table | -1.39310 | -1.30948 | -1.22248 |
| Pooled empirical | -0.86913 | -1.03273 | -1.19449 |
| Naive Bayes | -0.05605 | -0.09296 | -0.13792 |
| Naive Bayes, current duplicate removed | -0.00672 | -0.01843 | -0.02723 |

**Context evidence:** the three numeric columns are log-loss changes at copy probabilities 0, 0.5 and 0.9.

| Reader | No copying | Half copying | High copying |
|---|---:|---:|---:|
| Learned maker | -0.67460 | -0.63012 | -0.61064 |
| Maker, supplied copy roots | -0.67460 | -0.59652 | -0.51835 |
| Validation-selected episodes | -0.36365 | -0.39354 | -0.43601 |
| Episodes, power 0.25 | -0.36365 | -0.39354 | -0.43601 |
| Episodes, power 0.5 | -0.53999 | -0.63630 | -0.75799 |
| Episodes, power 1 | -0.98930 | -1.19260 | -1.41072 |
| Native-trained direct | -0.13608 | -0.12450 | -0.13183 |
| Exact chronological table | -0.07717 | -0.07748 | -0.07717 |
| Pooled empirical | -0.08073 | -0.08149 | -0.07717 |
| Naive Bayes | -0.55895 | -0.54673 | -0.62776 |
| Naive Bayes, current duplicate removed | -0.16164 | -0.12179 | -0.17468 |

**Complete evidence:** the three numeric columns are log-loss changes at copy probabilities 0, 0.5 and 0.9.

| Reader | No copying | Half copying | High copying |
|---|---:|---:|---:|
| Learned maker | -0.17572 | -0.14798 | -0.14917 |
| Maker, supplied copy roots | -0.17572 | -0.16795 | -0.18475 |
| Validation-selected episodes | -0.16529 | -0.17819 | -0.19449 |
| Episodes, power 0.25 | -0.16529 | -0.17819 | -0.19449 |
| Episodes, power 0.5 | -0.20897 | -0.26103 | -0.32573 |
| Episodes, power 1 | -0.35390 | -0.53105 | -0.73380 |
| Native-trained direct | -0.07046 | -0.06584 | -0.06713 |
| Exact chronological table | -0.01566 | -0.01566 | -0.01566 |
| Pooled empirical | -0.01566 | -0.01566 | -0.01566 |
| Naive Bayes | -0.42252 | -0.41414 | -0.43604 |
| Naive Bayes, current duplicate removed | -0.22354 | -0.18589 | -0.15395 |

The bound original diagnostics have no newly evaluated copied cells. Their no-copy changes below are retained to prevent omission of the fixed-schedule and fresh-law rivals. Fresh-law uses additional generated labels and is not the label-matched primary rival. Columns are changed-minus-native loss for artifact, context and complete evidence.

| Original diagnostic | Artifact | Context | Complete |
|---|---:|---:|---:|
| fixed schedule mlp | +0.00275 | -0.14900 | -0.06184 |
| fresh law direct | +1.87910 | +0.17073 | +0.13897 |

Capacity remains unequal: the native direct network has 20,864 parameters in each cell; learned maker models have 37,184 for artifact, 215,104–227,392 for context, and 836,096–852,704 for complete evidence. Effective states range from 16.06 to 30.21. Forty fixed expectation-maximization steps and noncollapse do not demonstrate optimization convergence. The exact table and pooled rival use count tables with empirical-prior fallback; all 73,728 new exact-table forecasts reconstruct independently. Their observed key coverage, original fixed-schedule/fresh-law scores and every fitted record are retained in [the full analysis](../../../results/v21/candidate-dependence/ANALYSIS.json).

All these readers receive supplied exact current support. Learned fit capacity, validation convergence, unknown generator laws, provided copy roots and histories that remain observationally indistinguishable are separate limitations. The learned latent-state sum is not a generator-law information ceiling. No fresh confirmation is inferred from V20 reuse or these repeatedly used V21 worlds. Miniature — architecture untested.

The additional support block uses native context-visible current evidence, sixteen complete past episodes with inspection observed, and 2,048 training labels across eight laws. It is not the 64-history block or a shifted counterpart. Learned pruning excludes 18.4326% of true histories, giving infinite uncapped log loss; supplied exact support reduces finite loss by 0.94773 nats and full Brier by 0.03047. All 32,768 forecasts, every mask and all 2,048 supplied-law posteriors independently reconstruct; maximum reference error is 6.11e-15. The unchanged 129-file support-family replay is reused by source and completion hashes. Each mask’s native-to-shift contrast is unavailable until its exact matching shifted cell is separately accepted. No method-by-support claim follows from one retained direct predictor.

No new scientific repair was used. The earlier initial B02 fixture repair remains historical and consumed; this candidate extension used no additional repair. The unrelated private accounting-path assembly error was retained and charged, with no scientific change. Earlier F03 fixtures remain valid; this group adds no new fixture category.

The registered [A03 answer policy](A03_ANSWER_POLICY_ADMISSION.json) is now implemented and admitted under a fresh frozen capsule. Three known-answer/scalar controls and two complete engineering consumers pass. A full-size engineering world with eight readers and both fits costs 1.8125 CPU seconds; seven complete jobs estimate 152.25 seconds with 50% scaling margin, bounded by 4,200 seconds. It chooses full, partial or withheld answers from unchanged forecasts and includes all fixed-specificity controls. Its scientific replay and result remain pending. Learned candidate-specific persistence and other unimplemented branches remain conditional; the program is not exhausted.

Discovery freeze, confirmation opening, science cutoff and Friday reporting times remain fixed. All 18 CPU hours remain protected. The twelve-hour checkpoint remains October 12 at 21:00 UTC / 14:00 Pacific; these acknowledgements do not move it. [Validity](../../../results/v21/candidate-dependence/VALIDITY.json).
