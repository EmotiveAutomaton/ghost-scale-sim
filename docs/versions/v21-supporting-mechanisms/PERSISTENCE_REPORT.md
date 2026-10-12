# V21 persistence and unrelated histories

Can a reader use personal history without trusting unrelated episodes? With supplied laws, inferred persistence improves log loss by 0.7403–0.8376 nats when all episodes share the current maker, but worsens it by 0.2110–0.2198 nats for unrelated episodes and 0.6296–0.7906 for a coherent foreign-author history. This is a verified constructed-method diagnostic, miniature — architecture untested, not learned contamination detection or confirmation.

We held each current case and four exogenous historical contexts fixed while varying the probability that an episode uses the current maker: 1, 0.9, 0.5 or 0. A separate shuffled-author control takes all four episodes from one independently sampled maker. Original and widened laws each use eight development coefficient worlds and 256 cases per world. These are paired worlds, without trained seeds; the original-law full replay is not another sample. Conditions share random draws and current targets. Independent draws can coincidentally choose the same maker; that is distinct from a same-source assignment.

All readers know the generator law, its finite maker-state vocabulary and exact conditional episode likelihoods. The inferred reader integrates unnormalized joint masses over a uniform prior on the four registered persistence rates before normalization. It uses observations, without current target labels. It is inference inside a supplied model, not learning who authored a corpus. History contexts were deliberately randomized independently of the current maker; this differs from the natural context distribution in the learned-reader study.

Each row gives a reader in one coefficient regime and history condition. Natural-log loss, in nats, and full multiclass Brier are proper probability losses; lower is better. Accuracy is the fraction whose most probable complete target is correct. Contradiction is the fraction of histories the rigid model cannot condition on at all, before the declared current-only decision fallback. Truth exclusion is the fraction assigned exactly zero probability to the actual target, making uncapped log loss infinite.

| Laws | Same-source probability | Reader | Log loss | Full Brier | Accuracy | Contradiction | Truth exclusion |
|---|---|---|---:|---:|---:|---:|---:|
| original | 1.0 | Supplied persistence | 2.5484 | 0.8845 | 15.92% | 0.00% | 0.00% |
| original | 1.0 | Inferred persistence | 2.6499 | 0.8880 | 15.43% | 0.00% | 0.00% |
| original | 1.0 | Rigid same maker | 2.5484 | 0.8845 | 15.92% | 0.00% | 0.00% |
| original | 1.0 | Current only | 3.3902 | 0.9400 | 7.42% | 0.00% | 0.00% |
| original | 0.9 | Supplied persistence | 2.6521 | 0.8915 | 14.84% | 0.00% | 0.00% |
| original | 0.9 | Inferred persistence | 2.7373 | 0.8944 | 14.79% | 0.00% | 0.00% |
| original | 0.9 | Rigid same maker | infinite | 0.8942 | 14.65% | 8.64% | 1.22% |
| original | 0.9 | Current only | 3.3902 | 0.9400 | 7.42% | 0.00% | 0.00% |
| original | 0.5 | Supplied persistence | 3.0921 | 0.9197 | 11.72% | 0.00% | 0.00% |
| original | 0.5 | Inferred persistence | 3.1162 | 0.9210 | 11.43% | 0.00% | 0.00% |
| original | 0.5 | Rigid same maker | infinite | 0.9317 | 9.91% | 36.62% | 8.50% |
| original | 0.5 | Current only | 3.3902 | 0.9400 | 7.42% | 0.00% | 0.00% |
| original | 0.0 | Supplied persistence | 3.3902 | 0.9400 | 7.71% | 0.00% | 0.00% |
| original | 0.0 | Inferred persistence | 3.6101 | 0.9554 | 6.88% | 0.00% | 0.00% |
| original | 0.0 | Rigid same maker | infinite | 0.9691 | 7.03% | 52.10% | 22.17% |
| original | 0.0 | Current only | 3.3902 | 0.9400 | 7.42% | 0.00% | 0.00% |
| original | shuffled-author | Supplied persistence | 3.3902 | 0.9400 | 7.71% | 0.00% | 0.00% |
| original | shuffled-author | Inferred persistence | 4.0198 | 0.9746 | 6.35% | 0.00% | 0.00% |
| original | shuffled-author | Rigid same maker | infinite | 0.9978 | 6.01% | 15.28% | 40.87% |
| original | shuffled-author | Current only | 3.3902 | 0.9400 | 7.42% | 0.00% | 0.00% |
| wide | 1.0 | Supplied persistence | 2.4613 | 0.8646 | 18.90% | 0.00% | 0.00% |
| wide | 1.0 | Inferred persistence | 2.5578 | 0.8689 | 19.14% | 0.00% | 0.00% |
| wide | 1.0 | Rigid same maker | 2.4613 | 0.8646 | 18.90% | 0.00% | 0.00% |
| wide | 1.0 | Current only | 3.3954 | 0.9407 | 7.86% | 0.00% | 0.00% |
| wide | 0.9 | Supplied persistence | 2.5818 | 0.8728 | 18.16% | 0.00% | 0.00% |
| wide | 0.9 | Inferred persistence | 2.6555 | 0.8767 | 18.12% | 0.00% | 0.00% |
| wide | 0.9 | Rigid same maker | infinite | 0.8767 | 17.29% | 9.57% | 1.66% |
| wide | 0.9 | Current only | 3.3954 | 0.9407 | 7.86% | 0.00% | 0.00% |
| wide | 0.5 | Supplied persistence | 3.0336 | 0.9128 | 13.04% | 0.00% | 0.00% |
| wide | 0.5 | Inferred persistence | 3.0635 | 0.9139 | 12.89% | 0.00% | 0.00% |
| wide | 0.5 | Rigid same maker | infinite | 0.9268 | 10.60% | 37.40% | 8.40% |
| wide | 0.5 | Current only | 3.3954 | 0.9407 | 7.86% | 0.00% | 0.00% |
| wide | 0.0 | Supplied persistence | 3.3954 | 0.9407 | 7.52% | 0.00% | 0.00% |
| wide | 0.0 | Inferred persistence | 3.6064 | 0.9586 | 6.01% | 0.00% | 0.00% |
| wide | 0.0 | Rigid same maker | infinite | 0.9756 | 6.45% | 53.32% | 21.04% |
| wide | 0.0 | Current only | 3.3954 | 0.9407 | 7.86% | 0.00% | 0.00% |
| wide | shuffled-author | Supplied persistence | 3.3954 | 0.9407 | 7.52% | 0.00% | 0.00% |
| wide | shuffled-author | Inferred persistence | 4.1859 | 0.9915 | 6.40% | 0.00% | 0.00% |
| wide | shuffled-author | Rigid same maker | infinite | 1.0159 | 6.59% | 15.33% | 41.46% |
| wide | shuffled-author | Current only | 3.3954 | 0.9407 | 7.86% | 0.00% | 0.00% |

The inferred reader uses useful persistence at rates 1, 0.9 and 0.5. At zero persistence it overuses irrelevant history. The coherent foreign-author bundle is worse: its internally consistent history supports the wrong personal explanation. That bundle is outside the independent-contamination mixture. The zero-persistence arm correctly ignores it for the current target, but is not an exact likelihood model of its within-bundle dependence. This separates the value of personal information from detecting contamination under a misspecified history model.

The following paired contrasts subtract current-only loss from inferred-persistence loss. Negative values mean improvement. Intervals are descriptive 95% whole-world bootstrap intervals from 4,096 draws over eight paired worlds; cases and repeated controls are not independent replications.

| Laws | History condition | Loss change | Lower interval | Upper interval |
|---|---|---:|---:|---:|
| original | 1.0 | -0.7403 | -0.8120 | -0.6794 |
| original | 0.9 | -0.6530 | -0.7154 | -0.5986 |
| original | 0.5 | -0.2741 | -0.3095 | -0.2477 |
| original | 0.0 | +0.2198 | +0.1849 | +0.2532 |
| original | shuffled-author | +0.6296 | +0.5358 | +0.7242 |
| wide | 1.0 | -0.8376 | -0.9436 | -0.7561 |
| wide | 0.9 | -0.7399 | -0.8301 | -0.6709 |
| wide | 0.5 | -0.3318 | -0.3695 | -0.2986 |
| wide | 0.0 | +0.2110 | +0.1860 | +0.2360 |
| wide | shuffled-author | +0.7906 | +0.6531 | +0.9486 |

At known zero persistence, supplied-persistence and current-only probabilities agree to numerical precision. Small modal-accuracy differences in their retained records arise from floating-point tie choices among equal-probability targets, not additional information; their proper losses agree. The inferred model has no zero-probability exclusions, yet can still become worse than ignoring history. Rigid personalization has infinite uncapped loss whenever contamination is present in these samples. Empty-condition contradictions and excluding the true target from a nonempty prediction are different failures; the declared fallback resolves only the first. No clipping or smoothing turns these into finite proper losses.

Every reader’s change across persistence conditions, confidence errors, seven-bit accuracy, top-eight coverage, exact zeros and below-floor probabilities is retained in [the aggregate record](../../../results/v21/persistence/ANALYSIS.json), together with evaluator-only readable cases and persistence posteriors. There is no execution shift in B03, so no native-to-shift contrast exists here. B02’s current duplicate-removal and known-copy-root controls remain separate verified diagnostics; a foreign-author history is not a copied-episode manipulation.

Verification binds 200 immutable complete packets, 6,852,608 saved forecasts and 29,066 retained files, reusing 197 exact prior score receipts and checking the three new packets with the complete-score reviewer. The full original-law frozen replay matches all 241 scientific files byte for byte. The same source-bound family supports the widened-law condition. Independent probability-domain references agree within 2.11e-15; every saved score, summary mean, current-case/context match, persistence posterior, contradiction flag and fallback identity verifies. All review and replay CPU is charged. The one prior B03 pre-admission repair remains consumed; scientific source was not changed.

This result establishes a finite supplied-law information ruler, with no learned capacity comparison, optimizer convergence claim, law-marginal ceiling, superiority promotion or fresh confirmation. Exact-law uncertainty and historical aliases remain distinct from these privileged posteriors. No human authorship or ghostwriting inference follows. Learned contamination detection and candidate-specific B01 controls remain conditional; do not scale up personalization on this result alone. No additional F03 category is claimed from evaluator-only examples.

[C02 joint architectures](C02_JOINT_ADMISSION.json) is separately implemented and admitted: direct, autoregressive in two bit orders, pairwise and rank-four joint heads at two approximate parameter budgets, with native and changed-execution forecasts. Both exact-law headroom cells and both direct fits satisfy the frozen gate. Three focused controls and two complete 154-file engineering consumers pass. The full-size one-world cost pilot costs 129.58 CPU seconds; the full group reserves at most 3,600 seconds. These engineering repeats do not replace the queued full scientific replay. One engineering learning-fixture repair used the declared epoch allowance without weakening its threshold; failed source and CPU remain retained. D01/D03/D04/B04/B05 and eligible new-world branches are next, subject to measured resource admission.
