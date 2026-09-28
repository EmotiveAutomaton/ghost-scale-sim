# V20 long-history label-budget review

We tested whether more training labels rescue structured reading of 256 retained episodes across richer evidence tiers. None of the eight new diagnostic conditions beats both direct rivals and the legal template under the declared comparison rule. All 3,670,016 forecasts verify. Exact zero probability reaches 4.22% of truths through numerical underflow; the earlier implementation-family replays remain valid. These are constructed-method results. The context primary remains failed, and confirmation remains untouched.

## Comparison and findings

Eight blocks extend the artifact-only, sparse-record and complete-record curves at 256 retained episodes: complete records at 512 labels, all three tiers at 2,048 and 8,192 labels, and artifact-only at 32,768 labels. Each block uses 32 discovery coefficient worlds, eight separate training worlds, one fixed fit seed, 2,048 queries per discovery world and seven readers. Complete records still withhold private aims and keep inspection latent. Readers share observations, training labels and supplied-law legal support.

Capped joint log loss is the negative natural logarithm of the saved probability on the true seven-part answer, floored at one trillionth; lower is better. It is not an uncapped proper score. The diagnostic rule requires at least 0.02 improvement against both direct rivals and the legal template, a positive lower descriptive whole-world interval, and no increase in confident wrong attribution. These evidence tiers cannot replace the registered context primary.

Table: rows identify evidence tier and training-label count, all at 256 retained episodes. Columns give mean capped joint log loss for the structured account, direct frequency reader, joint neural reader, independent-bit neural reader, legal uniform template and supplied-law oracle, averaged over the same 32 whole worlds.

| Evidence | Labels | Structured | Frequencies | Joint neural | Independent bits | Legal template | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| Complete record | 512 | 12.86871 | 0.75482 | 0.95202 | 1.47994 | 0.69315 | 0.69315 |
| Artifact only | 2,048 | 23.92900 | 3.57173 | 3.56778 | 21.90059 | 3.68868 | 2.76110 |
| Sparse record | 2,048 | 21.28035 | 2.16162 | 2.26167 | 16.15206 | 2.14531 | 2.04099 |
| Complete record | 2,048 | 11.06409 | 0.70668 | 0.78174 | 8.60273 | 0.69315 | 0.69315 |
| Artifact only | 8,192 | 13.99849 | 3.55509 | 3.55862 | 24.34219 | 3.68868 | 2.76110 |
| Sparse record | 8,192 | 12.59452 | 2.15099 | 2.17080 | 12.78309 | 2.14531 | 2.04099 |
| Complete record | 8,192 | 5.44934 | 0.69709 | 0.71718 | 9.16262 | 0.69315 | 0.69315 |
| Artifact only | 32,768 | 9.83143 | 3.54665 | 3.53983 | 21.38315 | 3.68868 | 2.76110 |

Table: rows compare the structured account with each required rival within a condition. Improvement is rival minus structured capped loss; positive favors structured reading. Intervals are descriptive 95% intervals from 4,096 whole-world bootstrap samples. Confident wrong increase is structured minus rival error at confidence of at least 90%.

| Evidence | Labels | Rival | Improvement | World interval | Confident wrong increase |
|---|---:|---|---:|---|---:|
| Complete record | 512 | Frequencies | -12.11388 | [-12.21292, -12.01662] | 0.49007 |
| Complete record | 512 | Joint neural | -11.91669 | [-12.01426, -11.82038] | 0.40831 |
| Complete record | 512 | Legal template | -12.17556 | [-12.27573, -12.07615] | 0.49007 |
| Artifact only | 2,048 | Frequencies | -20.35727 | [-20.55583, -20.14052] | 0.82301 |
| Artifact only | 2,048 | Joint neural | -20.36122 | [-20.55597, -20.14906] | 0.82301 |
| Artifact only | 2,048 | Legal template | -20.24032 | [-20.44240, -20.01997] | 0.82301 |
| Sparse record | 2,048 | Frequencies | -19.11874 | [-19.35241, -18.88376] | 0.77914 |
| Sparse record | 2,048 | Joint neural | -19.01868 | [-19.25186, -18.78606] | 0.77914 |
| Sparse record | 2,048 | Legal template | -19.13504 | [-19.37488, -18.89459] | 0.77914 |
| Complete record | 2,048 | Frequencies | -10.35741 | [-10.45704, -10.25021] | 0.48332 |
| Complete record | 2,048 | Joint neural | -10.28235 | [-10.38039, -10.17563] | 0.47835 |
| Complete record | 2,048 | Legal template | -10.37095 | [-10.47142, -10.26228] | 0.48332 |
| Artifact only | 8,192 | Frequencies | -10.44340 | [-11.15252, -9.78437] | 0.32533 |
| Artifact only | 8,192 | Joint neural | -10.43987 | [-11.14813, -9.78156] | 0.32533 |
| Artifact only | 8,192 | Legal template | -10.30980 | [-11.01740, -9.65114] | 0.32533 |
| Sparse record | 8,192 | Frequencies | -10.44353 | [-10.83110, -10.04607] | 0.53423 |
| Sparse record | 8,192 | Joint neural | -10.42372 | [-10.81069, -10.02619] | 0.53423 |
| Sparse record | 8,192 | Legal template | -10.44920 | [-10.83931, -10.04996] | 0.53423 |
| Complete record | 8,192 | Frequencies | -4.75225 | [-4.83788, -4.67323] | 0.42807 |
| Complete record | 8,192 | Joint neural | -4.73216 | [-4.81801, -4.65280] | 0.42807 |
| Complete record | 8,192 | Legal template | -4.75619 | [-4.84245, -4.67629] | 0.42807 |
| Artifact only | 32,768 | Frequencies | -6.28478 | [-7.21344, -5.43688] | 0.03180 |
| Artifact only | 32,768 | Joint neural | -6.29160 | [-7.21739, -5.44728] | 0.03180 |
| Artifact only | 32,768 | Legal template | -6.14275 | [-7.06874, -5.29526] | 0.03180 |

Whole lineages are paired. Repeated queries and fits are not independent replicates; intervals condition on the shared training worlds and single fit seed. These failures concern the tested structured decoder, not every structured approach. The admitted sparse and complete records at 32,768 labels remain pending.

## Exact zeros, validation and limits

Of 56 method-condition arms, 2 store exact-zero truth probability, all in the structured reader. The largest fraction below the capped-log floor is 77.83508%. Small positive probability below the floor and exact zero remain separate.

Table: rows identify conditions with stored zero truth probability. Columns give the count out of 65,536 queries, its percentage, and the range of independently reconstructed true-to-modal log weights. Every zero truth remains inside common legal support.

| Evidence | Labels | Zero truths | Percent | Log-weight range |
|---|---:|---:|---:|---|
| Complete record | 512 | 2,764 | 4.21753 | [-1403.80348, -745.43422] |
| Artifact only | 2,048 | 194 | 0.29602 | [-810.43592, -745.19574] |

The retained-count audit reconstructs finite positive smoothed likelihoods and verifies that every new exact-zero truth has true-to-modal odds that underflow on exponentiation. This extends the existing numerical-concentration finding, separate from forced candidate omission; saved probabilities and scores are unchanged. Joint squared probability loss, accuracy, goal/tool/order/inspection losses and confident wrong attribution are retained for every arm in the [aggregates](../../../results/v20/budget-tier-wave-1/AGGREGATES.json). No secondary replaces the failed comparison.

All COMPLETE records, frozen plans and source archives, environment identities, fit parameter counts, training/decode costs and blind/evaluator roles verify. The existing independent reviewer reconstructs every stored forecast, secondary score and summary. Prior complete scientific opening-family replays include G1, G4, G5 and G6; subsequent scoring-repair, 64/256-episode, evidence-tier and interpretation replays are reused by immutable hash. There is no new implementation family, repair or changed interpretation requiring another replay. No healthy scientific block was rerun for this checkpoint.

These eight blocks bring the independently verified total to 123. Review used 118.93750 CPU seconds and the zero audit 3.95312. All new science, review, audit and documentation costs plus a conservative 120-second control-plane allowance are charged inside the original 90-hour ceiling with 18 hours protected. Original failed attempts remain retained in the [cost record](../../../results/v20/budget-tier-wave-1/COSTS.json).

Measured dispatch recalibration changes remaining admitted demand from 28.21 to 28.33 hours across 219 discovery blocks. No refill is needed; revisit before measured useful work falls below one day. Twelve diagnostic confirmation blocks remain untouched. The finite conditional backlog contains 669 designs and 76.33 forecast hours, which is not promised execution. Combined finite demand falls short of the requested wall-window backlog margin. No designs, sources, model settings or criteria changed.

Same-host checks do not establish fresh-installation or cross-platform reproduction. The 32 coefficient redraws test sensitivity within this finite miniature, not arbitrary architectures. Execution, numerical acceptance, failed discovery comparisons and untouched confirmation remain separate. No simulator result establishes human intent.

[Validity](../../../results/v20/budget-tier-wave-1/VALIDITY.json), [replay reuse](../../../results/v20/budget-tier-wave-1/REPLAY_REUSE.json), [forecast](../../../results/v20/budget-tier-wave-1/FORECAST.json).
