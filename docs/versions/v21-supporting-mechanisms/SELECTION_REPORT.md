# V21 consequential selection and native-history support

Can selecting a plan change content without rewriting it? One selection changes 4.05 of eight content units on average; sixteen cosmetic edits change none. Character counts assign the selection zero weight. Historical inference and equally informed direct enumeration match exactly. This is a constructed-method result, not a human contribution measure or confirmation.

The [frozen E01 design](E01_SELECTION_ADMISSION.json) generates two binary proposals, then copies one complete proposal into the final artifact. A selection edits no content characters; cosmetic operations toggle a separate style bit. An intervention changes the chosen proposal or removes one cosmetic operation and counts the content units changed. This measures the content target only: stylistic value, effort, ownership and human contribution are different quantities.

Eight development worlds each supply 256 cases, a choice prior and two proposals with 2, 4 or 8 nested units. The same choices/proposals recur at 0, 1, 4 and 16 cosmetic edits and with absent or complete selection records. There are no fitted seeds. A supplied-program historical posterior and an independent direct enumeration receive identical information. Neither is a learned architecture or a law-marginal reader. Miniature — architecture untested.

The table fixes sixteen cosmetic edits and no selection record. Content units are proposal length; changed units count the exact content intervention. Identical proposals leave the choice historically ambiguous. Accuracy is the proportion of actual choices recovered; loss is uncapped natural-log loss; full Brier sums squared probability errors over both choices. World means receive equal weight. Both readers have identical rows.

| Content units | Mean changed units | Identical proposals | Choice correct | Choice loss | Full Brier |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.99854 | 23.68% | 91.85% | 0.14952 | 0.10430 |
| 4 | 2.01562 | 6.10% | 98.24% | 0.03732 | 0.02569 |
| 8 | 4.04932 | 0.29% | 99.95% | 0.00160 | 0.00106 |

For eight units the mean intervention changes 4.04932 units, with a descriptive 95% whole-world interval of 3.98535–4.11035. The selection can change all eight units despite editing no content characters. Character-edit weighting gives it zero; uniform event weighting gives it 1/17, or 5.88%, with sixteen cosmetic events. Target-specific intervention weighting assigns all normalized content effect to the selection when the proposals differ. Identical proposals produce zero content effect; the method leaves that zero explicitly unnormalized. Thus its average selection weight is 99.71%, not an unconditional share of total contribution. At zero cosmetic edits, character weights are all zero and uniform event weight is one, including aliased cases. All weighting rules and absolute weight-error summaries are retained in the [primary record](../../../results/v21/selection/SELECTION.json).

Without a record, identical proposals retain the supplied choice prior exactly. More distinct content reduces their frequency; it does not resolve any still-identical pair. A complete selection record identifies every historical choice even when the content effect is zero. Historical correspondence and intervention magnitude therefore remain distinct. Complete records yield zero choice loss and perfect accuracy; this is supplied evidence, not learned inference. The direct enumeration agrees exactly throughout.

All 73,728 repeated forecasts across cosmetic edit counts are byte-identical. The nested content prefixes and the historical/direct equality are explicit duplicate controls; cases, edits, witnesses and readers are not independent worlds. Intervals use 4,096 whole-world paired resamples. Native-to-shift changes are unavailable for both E01 readers because this consumer has only its native supplied program. It contains no naive-Bayes product arm.

The support result uses an endpoint-only current case and 64 complete historical episodes under native execution. The preceding block uses 16 complete episodes. Their current cases, training/validation labels and nested historical prefixes match exactly. Each condition has its own two fitted models, and all stop on validation. Reconstructed exact-law headroom lower bounds are 0.28846 nats with 16 histories and 0.24012 with 64, above 0.02. Expanded input dimensions also expand direct-model capacity; this is a descriptive history-length comparison, not a matched-capacity causal ablation. This validates admission of the support diagnostic; it does not establish a new structured-reader superiority result.

The next table reports every support arm. Loss is uncapped joint natural-log loss; 64-minus-16-history changes average two fitted seeds inside each of eight worlds. Negative changes improve loss. Full Brier uses all 128 classes. Exclusion is the percentage of truths removed in the 64-history condition. Infinite losses cannot be subtracted into a finite effect; those differences remain undefined. Omission rows remove a registered fraction of exact-support candidate identities.

| Support rule | 16-history loss | 64-history loss | Loss change | Full Brier change | 64-history truth exclusion |
|---|---:|---:|---:|---:|---:|
| none | 3.86656 | 3.90940 | 0.04284 | 0.00271 | 0.00% |
| physical | 3.86656 | 3.90940 | 0.04284 | 0.00271 | 0.00% |
| learned | infinite / undefined difference | infinite / undefined difference | infinite / undefined difference | -0.00585 | 1.59% |
| supplied exact | 3.10221 | 2.96042 | -0.14179 | -0.00926 | 0.00% |
| omission 0 | 3.10221 | 2.96042 | -0.14179 | -0.00926 | 0.00% |
| omission 0.01 | infinite / undefined difference | infinite / undefined difference | infinite / undefined difference | -0.00933 | 0.29% |
| omission 0.05 | infinite / undefined difference | infinite / undefined difference | infinite / undefined difference | -0.00892 | 3.44% |
| omission 0.2 | infinite / undefined difference | infinite / undefined difference | infinite / undefined difference | -0.01023 | 18.26% |

Learned support excludes 1.59% of truths in both conditions, making uncapped loss infinite despite improved Brier with more history. Supplied exact support improves 64-history loss over no support by 0.94897 nats and full Brier by 0.03072. Physical support repeats no mask in this artifact-only cell; zero omission repeats supplied support. These exact duplicate forecasts were checked. Nonzero candidate pruning again creates infinite losses. All mask-specific confidence, accuracy, incompatible mass, bit metrics and paired intervals remain available in the primary records.

The upstream history-length changes use the same matched worlds and nested case/training prefixes. Both naive-Bayes variants are explicit. Every reader has an unavailable native-to-shift contrast: no matching shifted artifact-current, complete-history-64 support block is admitted. A review draft incorrectly labeled these native history conditions as an execution shift; design checks caught and corrected that error before publication or acknowledgement. Numerical receipts remain valid and the failed interpretation is retained locally. They are headroom controls, not newly fitted learned-maker comparisons. The supplied-law ceiling, law-marginal ceiling and fitted rival retain distinct information privileges.

| Upstream reader | 64-minus-16-history loss | 64-minus-16-history full Brier |
|---|---:|---:|
| supplied law | -0.03020 | -0.00239 |
| law marginal | -0.08983 | -0.00910 |
| converged direct | -0.14179 | -0.00926 |
| naive bayes | 23.57034 | 0.20168 |
| naive bayes without evidence | 21.70907 | 0.21834 |

The direct rival improves by 0.14179 nats, interval 0.10184–0.18552, while the law-marginal ceiling improves by 0.08983. Removing the duplicated evidence factor does not prevent severe naive-Bayes deterioration as more native history is supplied. More episodes and increased direct-model input capacity remain separate explanations; no transfer claim follows. This is not the reused V20 audit or fresh confirmation. No method-by-support interaction or learned unknown-law benefit is inferred from the retained direct-predictor mask comparison.

Verification binds 264 immutable complete packets to their actual plans, acceptance, source archives and environment. All 10,074,112 saved forecasts and 64,794 retained files pass, reusing 259 exact prior score receipts. E01 adds 98,304 forecasts, 153,600 intervention checks, 192 blind observation files and its complete 934-file scientific replay, excluding only summary CPU timing. Independent code reconstructs proposal draws, choices, every intervention and weight, exact aliases, posteriors and every reported mean. The new support block adds 32,768 forecasts, independent mask/posterior reconstruction within 2.95e-14 and a hash-reused unchanged 129-file support family replay. Two scientific blocks are accepted, bringing the total to 87; packet counts are not findings. [Validity](../../../results/v21/selection/VALIDITY.json).

Twelve decisive-selection fixtures pass the canonical ProcessCase validator and offline exporter. Evaluator choices, intervention effects and posteriors are separate from reader observations. Canonical source hashes match their retained read-only execution. Placeholder readings establish offline compatibility only; browser/live reader execution, human causal attribution and architecture generality remain untested. E01 used no repair and its scientific source is unchanged.

The next [C05 omission design](C05_OMISSION_ADMISSION.json) uses 100 finite mechanism types, omits 0/10/25/50% by a pre-outcome mechanism rule, and allows 8/16/32/64 expansions. Forced choice, a finite-unknown bucket, uniform expansion, evidence ranking and supplied truth are separate readers. Exact omitted aggregate probability is a supplied-law privilege; the bucket is uniformly distributed across omitted types for full-universe scoring. Coarse candidate-event loss is retained separately and cannot justify expansion by itself. This is finite candidate recovery, not open-world concept invention or a learned superiority screen.

Three source-bound known-answer controls and two complete 144-file engineering consumers pass, including twelve missing-candidate canonical fixtures. One strict no-omission control exposed redundant normalization; the single allowed pre-admission repair preserves exact no-omission parity and the original checker tolerances. Failed source and CPU remain retained; no further family repair is available. A full-grid one-world pilot costs 6.43750 CPU seconds; original plus full replay is estimated at 164.50 seconds with 50% margin and fixture overhead, under a combined 1,200-second cap. Fresh immutable plans bind the tested source. C05 scientific replay and acceptance remain pending.

D05 retraction, candidate-specific dependence/persistence controls and learned/unknown-law extensions remain conditional. No exhaustion, promotion or confirmation is claimed. The fixed twelve-hour checkpoint stays October 12 21:00 UTC; original discovery, confirmation, science-cutoff and Friday report clocks are unchanged.
