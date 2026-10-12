# V21 copied evidence and generic tempering

Does accounting for copied evidence require a learned maker? In this native context subset, quarter-power episode likelihoods beat the learned shared-maker reader by 0.7443–0.8729 nats across copying rates. Known-root removal helps the latent reader but does not establish maker-specific benefit. This is a verified constructed-method diagnostic, miniature — architecture untested, with supplied support and no confirmation.

We kept current context-visible artifacts and targets fixed while replacing each of the three later historical episodes with the first episode at probability 0, 0.5 or 0.9. All four historical episodes are complete and include witnessed inspection. Training uses 8,192 labels across 64 original-law worlds. Both fitted seeds share each of eight development worlds and 256 current cases; fit means are averaged within worlds before 4,096 whole-world bootstrap draws. Copying rates share uniform draws and current cases. These are eight worlds, not sixteen fitted replications or 122,880 independent observations.

Both fits select episode power 0.25 using only separate validation laws, averaged over the three registered copying rates, without supplied support at calibration. The same power is then frozen for every development condition. New copying rates and wide-law or execution-shift transfer of this calibration remain untested.

Each row is a reader at a declared probability of replacing a later episode. Natural-log loss, measured in nats, and full multiclass Brier are proper probability losses: lower is better. Accuracy is the percentage of cases whose most probable complete target is correct. Confident error is the percentage assigned at least 90% confidence to an incorrect target.

| Copy probability | Reader | Log loss | Full Brier | Accuracy | Confident error |
|---:|---|---:|---:|---:|---:|
| 0.0 | Native-trained direct | 2.9010 | 0.9215 | 12.21% | 0.000% |
| 0.0 | Validation-selected episodes | 2.9152 | 0.9248 | 11.84% | 0.000% |
| 0.0 | Episodes, power 0.25 | 2.9152 | 0.9248 | 11.84% | 0.000% |
| 0.0 | Episodes, power 0.5 | 3.0396 | 0.9461 | 11.67% | 0.049% |
| 0.0 | Episodes, power 1 | 3.5575 | 1.0280 | 11.79% | 0.488% |
| 0.0 | Learned shared maker | 3.7880 | 1.0236 | 10.06% | 1.245% |
| 0.0 | Shared maker, supplied roots | 3.7880 | 1.0236 | 10.06% | 1.245% |
| 0.0 | Naive Bayes | 3.4017 | 0.9800 | 11.38% | 0.049% |
| 0.0 | Naive Bayes, current duplicate removed | 3.2684 | 0.9625 | 11.77% | 0.073% |
| 0.0 | Pooled empirical | 2.9627 | 0.9277 | 9.06% | 0.000% |
| 0.5 | Native-trained direct | 2.9499 | 0.9261 | 11.43% | 0.000% |
| 0.5 | Validation-selected episodes | 2.9736 | 0.9340 | 10.99% | 0.000% |
| 0.5 | Episodes, power 0.25 | 2.9736 | 0.9340 | 10.99% | 0.000% |
| 0.5 | Episodes, power 0.5 | 3.2418 | 0.9748 | 10.42% | 0.073% |
| 0.5 | Episodes, power 1 | 4.1603 | 1.0944 | 9.77% | 1.538% |
| 0.5 | Learned shared maker | 3.7910 | 1.0271 | 10.25% | 1.294% |
| 0.5 | Shared maker, supplied roots | 3.4985 | 0.9925 | 10.03% | 0.537% |
| 0.5 | Naive Bayes | 3.5130 | 0.9944 | 10.57% | 0.195% |
| 0.5 | Naive Bayes, current duplicate removed | 3.3714 | 0.9744 | 10.30% | 0.146% |
| 0.5 | Pooled empirical | 2.9627 | 0.9277 | 9.06% | 0.000% |
| 0.9 | Native-trained direct | 3.0129 | 0.9309 | 11.65% | 0.000% |
| 0.9 | Validation-selected episodes | 3.0425 | 0.9441 | 10.25% | 0.024% |
| 0.9 | Episodes, power 0.25 | 3.0425 | 0.9441 | 10.25% | 0.024% |
| 0.9 | Episodes, power 0.5 | 3.4753 | 1.0031 | 9.30% | 0.171% |
| 0.9 | Episodes, power 1 | 4.8170 | 1.1501 | 8.98% | 2.905% |
| 0.9 | Learned shared maker | 3.7868 | 1.0355 | 9.62% | 1.245% |
| 0.9 | Shared maker, supplied roots | 3.2173 | 0.9569 | 9.81% | 0.146% |
| 0.9 | Naive Bayes | 3.6189 | 1.0040 | 10.69% | 0.146% |
| 0.9 | Naive Bayes, current duplicate removed | 3.4698 | 0.9829 | 10.72% | 0.122% |
| 0.9 | Pooled empirical | 2.9627 | 0.9277 | 9.06% | 0.000% |

Relative to untempered episode likelihoods, quarter-power weighting reduces loss by 0.6423, 1.1867 and 1.7745 nats as copying increases. It also lowers full Brier and confident errors. Relative to the learned shared-maker model, its loss reductions are 0.8729 (descriptive 95% interval 0.8062–0.9382), 0.8174 (0.7560–0.8775) and 0.7443 (0.6828–0.8016). Generic correction therefore defeats a maker-specific interpretation in this tested cell. This native context setting is not either of the two earlier descriptive B01 candidate cells; their mechanism attribution remains unresolved.

The tempered reader does not establish superiority over the frozen direct rival: its excess loss is 0.0142, 0.0238 and 0.0296 nats, with each descriptive interval spanning zero. Its full Brier is worse under both copying conditions. The lowest observed mean loss at 90% copying belongs to the pooled empirical arm, but none of its 12,288 fitted-case keys matches training: it always falls back to the empirical target prior projected onto supplied current support. That is a useful simple rival, not successful matching of a historical maker.

Supplying exact copy roots lowers the latent reader’s loss by 0.2925 nats at 50% copying (0.2647–0.3172) and 0.5695 at 90% (0.5217–0.6145), with lower Brier and fewer confident errors. This arm has evaluator provenance privilege. Without copying it is byte-identical to the full latent reader. Separately, removing the duplicated current evidence factor from naive Bayes lowers loss by 0.1333, 0.1416 and 0.1491 nats; this is distinct from removing copied historical roots.

Rows below show every reader’s change in natural-log loss relative to its no-copy baseline, retaining the two duplicate controls. Positive means copying worsened loss. These are copying changes under native execution, not native-to-shift changes; no execution-shift contrast exists in this initial B02 subset.

| Reader | Half-copy loss change | High-copy loss change |
|---|---:|---:|
| Native-trained direct | +0.0489 | +0.1119 |
| Validation-selected episodes | +0.0585 | +0.1273 |
| Episodes, power 0.25 | +0.0585 | +0.1273 |
| Episodes, power 0.5 | +0.2022 | +0.4356 |
| Episodes, power 1 | +0.6028 | +1.2595 |
| Learned shared maker | +0.0030 | -0.0013 |
| Shared maker, supplied roots | -0.2895 | -0.5707 |
| Naive Bayes | +0.1113 | +0.2172 |
| Naive Bayes, current duplicate removed | +0.1030 | +0.2015 |
| Pooled empirical | +0.0000 | +0.0000 |

The [complete aggregate record](../../../results/v21/dependence/ANALYSIS.json) retains all paired intervals, full-Brier and accuracy changes, seven-bit scores, top-eight coverage, exact zeros and below-floor probabilities. No true target receives zero probability in this block. The source exact-law information gate applies to independent histories only. Copied cells have neither a newly integrated exact-law ceiling nor a direct reader refitted to copying with validation-stopped convergence, so these remain diagnostics, with no superiority admission or promotion.

The finite direct reader has 20,864 parameters and both native fits validation-stopped. The latent reader has 286,208/286,240 parameters, 28.38/28.46 effective states and 40 fixed EM iterations; noncollapse is established, optimization convergence is not. Its full Bayesian calculation sums learned latent states, not generator laws. All arms receive the same exact current support. Supplied support, supplied copy roots, fitted capacity, numerical convergence, law uncertainty and unresolved historical aliases are separate limitations. No fresh confirmation or human mechanism inference follows.

Verification binds 197 immutable complete packets, 6,729,728 saved forecasts and 28,343 retained files. It reuses 195 prior landing receipts and checks both new packets, then reuses those two checks when correcting the replay comparator to ignore only recorded fit CPU timings. The initial review failure and its CPU remain retained. The full frozen scientific replay matches all 603 scientific files, with only two JSON timing fields differing. Separate probability-domain references agree within 4.11e-15. Independent regrouping verifies every summary, copied view, root mask and selected-power identity. The two scientific packets cost 118.05 native CPU seconds together; verification and services are charged separately.

Twelve known-copying fixtures with one, two or four reports from a single root pass the unchanged canonical ProcessCase and offline exporter in both full executions. They are twelve cases, not twenty-four independent replications. Blind inputs and evaluator truth remain separate in [the fixture packet](../../../results/v21/dependence/fixtures/CONSUMER.json). This verifies offline compatibility only. Unknown or corrupt roots remain D03; omission, decisive-selection and retraction fixtures remain conditional.

The initial native B02 diagnostic is accepted and stops a maker-specific interpretation here. Wider B02 and actual B01 candidate-cell dependence tests remain conditional. [B03 persistence](B03_PERSISTENCE_ADMISSION.json) is now implemented and admitted as a supplied-law information ruler under original and wide laws, plus a full original-law replay. Two focused controls and two complete 31-file engineering consumers pass. Full-size one-world original/wide pilots cost 1.4375/1.578125 CPU seconds; the three jobs reserve at most 2,700 seconds. One pre-admission repair records impossible rigid-model evidence and an explicit current-only fallback; failed source and CPU are retained. B03 scientific acceptance awaits its full replay event. Learned contamination detection and C02 architecture comparisons remain separate conditional scope.
