# V21 joint heads, marginal controls and complete-current evidence

Does preserving target dependence explain the compact joint reader’s gain? The rank-four head lowers native log loss by 0.1011 nats against the same-budget direct joint head, but the direct head’s own marginal-product control has lower mean loss than rank four in both execution conditions. A dependence-specific advantage is not established. These are verified constructed-method development results, with no promotion or confirmation.

The architecture study uses context-only current evidence and four complete past episodes with inspection observed, under original coefficient laws. Each fit receives 2,048 labels across 64 training laws and 4,096 validation labels across 16 separate laws. Twenty fits cross five heads, two approximate parameter budgets and two seeds. Both seeds are averaged inside each of eight development worlds, with 256 cases per world. Native training and validation are reused unchanged for the execution shift, which changes historical as well as current execution. This is distinct from B07’s sparse current-only adaptation.

Both reference cells pass the exact law-marginal headroom gate: their lower interval bounds exceed 0.02 nats (0.3461 native, 0.4084 changed), and both reference direct fits stopped on validation. All 20 new fits also stopped on validation; no optimizer repair or extra architecture search was performed after outcomes. Parameter matching is approximate, function classes differ, and fit cost remains unequal.

Rows identify a fitted head and nominal parameter budget. Natural-log loss is measured in nats and full multiclass Brier measures squared probability error; lower is better for both. Shift change subtracts native loss from changed-execution loss, so positive means deterioration. Reported probabilities share supplied exact support. CPU is the sum over the two fits; parameters are per fit.

| Budget | Head | Parameters | Fit CPU seconds | Native loss | Changed loss | Shift change | Native Brier | Changed Brier |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4096 | Direct joint | 4016 | 6.12 | 2.9914 | 2.9904 | -0.0010 | 0.9335 | 0.9336 |
| 4096 | Autoregressive forward | 4060 | 24.42 | 2.9414 | 3.0728 | +0.1314 | 0.9235 | 0.9355 |
| 4096 | Autoregressive reverse | 4060 | 26.22 | 2.9657 | 3.1325 | +0.1669 | 0.9267 | 0.9407 |
| 4096 | Pairwise | 4060 | 4.89 | 2.9651 | 3.1969 | +0.2318 | 0.9283 | 0.9480 |
| 4096 | Rank four | 3908 | 18.47 | 2.9392 | 3.0099 | +0.0707 | 0.9222 | 0.9281 |
| 8192 | Direct joint | 7904 | 5.78 | 3.0286 | 3.0538 | +0.0252 | 0.9412 | 0.9426 |
| 8192 | Autoregressive forward | 8092 | 21.19 | 2.9572 | 3.0659 | +0.1087 | 0.9257 | 0.9356 |
| 8192 | Autoregressive reverse | 8092 | 18.86 | 2.9550 | 3.0953 | +0.1404 | 0.9263 | 0.9420 |
| 8192 | Pairwise | 8092 | 4.59 | 2.9570 | 3.1658 | +0.2088 | 0.9272 | 0.9474 |
| 8192 | Rank four | 8012 | 13.72 | 2.9275 | 3.0115 | +0.0840 | 0.9215 | 0.9293 |

At the larger budget, each structured head improves native loss against that budget’s direct joint head. Rank four improves by 0.1011 nats, with a descriptive 95% interval of 0.0628–0.1366. Its shifted difference is unresolved: rank four minus direct is −0.0423 [−0.0810, 0.0013]. Pairwise becomes worse after the shift, and the two autoregressive orders differ. A better native head therefore does not establish general transfer. The smaller direct joint head also outperforms the larger one here; extra parameters alone do not guarantee better fit.

Each following row applies a postprocessing ablation to the very same saved predictor: multiply its exact seven-bit marginal distributions and then apply the common supplied support. There is no retraining. Rows show joint loss and product loss in both conditions; lower is better. A support projection can itself reintroduce dependence.

| Budget | Head | Native joint | Native product | Changed joint | Changed product |
|---:|---|---:|---:|---:|---:|
| 4096 | Direct joint | 2.9914 | 2.9142 | 2.9904 | 2.9149 |
| 4096 | Autoregressive forward | 2.9414 | 2.9299 | 3.0728 | 3.0402 |
| 4096 | Autoregressive reverse | 2.9657 | 2.9565 | 3.1325 | 3.1093 |
| 4096 | Pairwise | 2.9651 | 2.9502 | 3.1969 | 3.1799 |
| 4096 | Rank four | 2.9392 | 2.9493 | 3.0099 | 3.0201 |
| 8192 | Direct joint | 3.0286 | 2.9062 | 3.0538 | 2.9106 |
| 8192 | Autoregressive forward | 2.9572 | 2.9426 | 3.0659 | 3.0187 |
| 8192 | Autoregressive reverse | 2.9550 | 2.9534 | 3.0953 | 3.0749 |
| 8192 | Pairwise | 2.9570 | 2.9519 | 3.1658 | 3.1423 |
| 8192 | Rank four | 2.9275 | 2.9344 | 3.0115 | 3.0189 |

The larger direct head’s marginal product improves native loss by 0.1224 nats and beats rank four by 0.0213 in mean loss; that native contrast is uncertain (rank four minus product: −0.0025 to 0.0456). After the shift, the direct product beats rank four by 0.1009 [0.0722, 0.1271]. Removing rank four’s own learned dependence costs only 0.0069 [−0.0065, 0.0206] native nats. The gain over a raw direct joint head cannot therefore be attributed specifically to preserving target dependence. This is a useful ablation finding, not a claim that all dependencies are harmful.

The retained comparison readers below use the same evidence and worlds. Rows show native-to-shift changes separately for each reader, including the duplicate-removal arm. Shift change is changed loss minus native loss.

| Reader | Native loss | Changed loss | Shift change |
|---|---:|---:|---:|
| Retained 64-unit direct | 3.1091 | 3.1259 | +0.0167 |
| Naive Bayes | 5.2425 | 5.2876 | +0.0450 |
| Naive Bayes without evidence | 4.8862 | 4.9645 | +0.0783 |

[All architecture records](../../../results/v21/joint-architectures/JOINT.json) retain unmasked forecasts, projected forecasts, every reader’s native-to-shift interval for loss/Brier/accuracy/confident errors, parameter counts, widths, validation curves, fit costs, bit scores and top-eight coverage. Unmasked native loss at the larger budget is 3.5990 for direct and 3.3767 for rank four; common support lowers these to 3.0286 and 2.9275. Supplied constraints and learned representation are separate contributions. The [decisive marginal-rival contrasts](../../../results/v21/joint-architectures/MARGINAL_RIVAL.json) include both budgets and both execution conditions.

The same review lands two complete-current learned-reader label curves, including their lower-label prerequisites. Current and four historical episodes expose actions and order but hide inspection. They use eight finite training laws, original coefficients, and label budgets 512/2,048/8,192; this differs from the architecture cell. Each row compares the shared-maker reader with the strongest observed same-label rival, selected descriptively in that cell. A negative loss difference favors the shared-maker reader; intervals are whole-world paired 95% intervals from 4,096 bootstrap draws.

| Labels | Execution | Strongest matched rival | Shared-minus-rival loss | Lower | Upper | Practical screen |
|---:|---|---|---:|---:|---:|---|
| 512 | native | Exact-match table | +0.1055 | +0.0697 | +0.1433 | not met |
| 512 | changed-tool | Exact-match table | -0.0545 | -0.0678 | -0.0426 | descriptive candidate |
| 2048 | native | Exact-match table | +0.4790 | +0.4404 | +0.5218 | not met |
| 2048 | changed-tool | Exact-match table | -0.0028 | -0.0120 | +0.0061 | not met |
| 8192 | native | Grouped empirical | +0.2220 | +0.1559 | +0.2888 | not met |
| 8192 | changed-tool | Exact-match table | +0.0049 | -0.0014 | +0.0118 | not met |

Only the shifted 512-label cell clears the 0.02-nat descriptive margin: 0.0545 [0.0426, 0.0678] improvement against the table. Its full Brier is 0.6457 versus 0.6759, and confident wrong predictions occur in 0.098% versus 0.293%. Higher-label results do not reproduce that practical margin. This adds a third descriptive cell to the earlier two, among 36 accepted learned cells, without freezing a promotion. Candidate-specific dependence, persistence and capacity controls remain required. All finite direct fits validation-stop; fresh-law fits often hit their cap and use extra generated labels, so they remain a separate generator-limit diagnostic. Shared mixtures retain more than two effective states, but use far more parameters than the direct rival.

Every reader’s native-to-shift log-loss change follows; each row is one label budget and reader, and positive means deterioration. The extra-data fresh-law reader is explicitly separate. The complete record also retains uncertainty and metric reversals.

| Labels | Reader | Shift change | Lower | Upper |
|---:|---|---:|---:|---:|
| 512 | Retained 64-unit direct | -0.0705 | -0.1128 | -0.0315 |
| 512 | Exact-match table | -0.0157 | -0.0348 | +0.0017 |
| 512 | Fixed-schedule MLP | -0.0618 | -0.1035 | -0.0278 |
| 512 | Fresh-law direct | +0.1390 | +0.0689 | +0.2034 |
| 512 | Grouped empirical | -0.0157 | -0.0348 | +0.0017 |
| 512 | Naive Bayes | -0.4225 | -0.6759 | -0.2123 |
| 512 | Naive Bayes without evidence | -0.2235 | -0.4605 | -0.0038 |
| 512 | Shared maker | -0.1757 | -0.2021 | -0.1493 |
| 2048 | Retained 64-unit direct | +0.1237 | +0.0446 | +0.1896 |
| 2048 | Exact-match table | +0.0078 | -0.0056 | +0.0202 |
| 2048 | Fixed-schedule MLP | +0.0234 | +0.0003 | +0.0494 |
| 2048 | Fresh-law direct | +0.3669 | +0.2285 | +0.4865 |
| 2048 | Grouped empirical | +0.0078 | -0.0056 | +0.0202 |
| 2048 | Naive Bayes | -0.1383 | -0.2330 | -0.0301 |
| 2048 | Naive Bayes without evidence | +0.0545 | -0.0293 | +0.1543 |
| 2048 | Shared maker | -0.4741 | -0.5142 | -0.4386 |
| 8192 | Retained 64-unit direct | +0.3805 | +0.2404 | +0.5061 |
| 8192 | Exact-match table | +0.0038 | -0.0095 | +0.0167 |
| 8192 | Fixed-schedule MLP | +0.0128 | -0.0070 | +0.0301 |
| 8192 | Fresh-law direct | +0.4404 | +0.2806 | +0.5826 |
| 8192 | Grouped empirical | +0.0038 | -0.0095 | +0.0167 |
| 8192 | Naive Bayes | -0.0951 | -0.1344 | -0.0489 |
| 8192 | Naive Bayes without evidence | -0.0156 | -0.0508 | +0.0224 |
| 8192 | Shared maker | -0.2133 | -0.2789 | -0.1458 |

Two matching support diagnostics apply eight support rules to the same retained direct predictor, without fitting another reader. Rows identify the execution condition and support rule. Truth exclusion is the fraction assigned exactly zero probability to the true history; any such exclusion makes uncapped mean log loss infinite. Full Brier is retained separately.

| Execution | Support rule | Log loss | Full Brier | Truth exclusion |
|---|---|---:|---:|---:|
| native | learned | infinite | 0.8402 | 16.43% |
| native | none | 1.6215 | 0.7577 | 0.00% |
| native | omission 0 | 1.2270 | 0.7222 | 0.00% |
| native | omission 0.01 | infinite | 0.7250 | 0.29% |
| native | omission 0.05 | infinite | 0.7461 | 3.44% |
| native | omission 0.2 | infinite | 0.8310 | 17.19% |
| native | physical | 1.6173 | 0.7579 | 0.00% |
| native | supplied exact | 1.2270 | 0.7222 | 0.00% |
| changed-tool | learned | infinite | 0.8306 | 3.34% |
| changed-tool | none | 1.8269 | 0.8034 | 0.00% |
| changed-tool | omission 0 | 1.3506 | 0.7565 | 0.00% |
| changed-tool | omission 0.01 | infinite | 0.7590 | 0.32% |
| changed-tool | omission 0.05 | infinite | 0.7777 | 3.34% |
| changed-tool | omission 0.2 | infinite | 0.8632 | 17.19% |
| changed-tool | physical | 1.8220 | 0.8037 | 0.00% |
| changed-tool | supplied exact | 1.3506 | 0.7565 | 0.00% |

Learned masks exclude 16.43% of native truths and 3.34% of shifted truths. Supplied exact support lowers log loss from 1.6215 to 1.2270 nats natively and 1.8269 to 1.3506 after the shift, improving full Brier in these four-history cells. Earlier no-history complete-current cells had Brier reversals; that historical result is preserved. Observing actions/order does not reveal hidden inspection or private aims, and it does not repair incorrect learned pruning. None of these support interventions establishes a method-by-support superiority claim or identifies the actual historical route beyond the available evidence.

Verification binds 220 complete immutable packets, 7,802,880 forecasts and 33,362 retained files. Two hundred exact prior score receipts are reused; all 20 new complete packets are checked with the retained-score reviewer. The full C02 frozen replay matches 1,078 scientific files after excluding only fit-time fields in two JSON records. Scalar probability references agree within 1.00e-15; reconstructed support projections and marginal products match exactly. Prior learned/support families retain their unchanged source/replay bindings. All retained summary metrics and meaningful contrasts reconstruct, with fits paired inside worlds. Nine scientific blocks land here, bringing acceptance to 71; packet execution and numerical checks are not independent findings. All costs, including review and service overage, are charged.

[Four readable evaluator cases](../../../results/v21/joint-architectures/CASES.json) show the largest rank-four gain and loss in the first world under each execution condition, with exact observed fields, target-bit meanings and true-target probabilities for direct joint, direct product and rank four. They are explicitly posthoc illustrations and never blind exports or extra replications.

Validation cannot establish new-world confirmation, a universal architecture ranking, unknown-law deployment, human intent or real historical authorship. Intervals are descriptive and unadjusted across settings, which reuse eight development worlds. The constructed generator has no new random-architecture severity pass: miniature — architecture untested beyond this finite world. Source-bound exact-law headroom, convergence, fit capacity, supplied support, law uncertainty and historical aliasing remain separate. C02’s earlier single pre-admission repair remains consumed; its scientific source is unchanged. No F03 category or live Sounding Line inference is added by this review.

The next independent branch is [D01 record acquisition](D01_ACQUISITION_ADMISSION.json): expected target information, report entropy, fixed cheapest, random and no-purchase policies, with a deliberately irrelevant random-action report. Original/wide and native/changed supplied-law rulers plus full replay are frozen and admitted. Two focused controls, two complete 23-file engineering consumers and four full-size one-world cost pilots pass. Pilot consumer costs are 1.20–1.66 CPU seconds; the five-block estimate is 82.5 seconds including 50% margin, against a 3,000-second cap. Engineering results are not scientific acceptance. No family repair was needed. D03/D04/B04/B05 and eligible new-world branches remain next; learned acquisition and omission/decisive-selection/retraction fixtures remain conditional.
