# V21 context retraction

Can an incremental evidence ledger revise conclusions as accurately as recomputing from raw reports? It matches the full calculation, but shows no CPU or storage advantage in this test. True corrections help; false corrections hurt equally. This is a verified constructed-method result, with supplied reliability and no confirmation.

The [registered design](D05_RETRACTION_ADMISSION.json) supplies a positive binary-state prior and independent report reliability. Eight development coefficient worlds each have 256 cases, crossed with 4, 16 or 64 nested reports, reliability 0.6, 0.8 or 1, and five updates. There are no fitted seeds. Cases, history lengths and update conditions share random draws within worlds. Descriptive 95% intervals use 4,096 paired whole-world resamples; repeated cases and fits are not independent worlds.

The incremental ledger retains source-keyed reports, positive log sums and exact zero counts. Replacing a report removes its old factor before adding the new one, at most two factor visits; deleting takes one and an identical repeat takes zero. The independent raw decoder multiplies probabilities for each state from the updated raw records. It is the strongest equally informed rival. Frozen conclusions ignore every update. Append-only summaries retain old factors, add repeats again and ignore retraction; this is an explicitly limited comparator.

True correction and retraction start with source zero deliberately wrong. False correction starts with it true. Irrelevant additions and unchanged repeats use ordinary generated reports. Therefore the five rows below have different starting evidence: they do not estimate a single common-baseline treatment effect. Readers see reports and actual source updates; intervention labels and state truth stay evaluator-only.

The table fixes four reports and reliability 0.8. Each entry is mean uncapped natural-log loss over the two latent states; lower is better. Columns identify the reader's update method. The same cases and worlds are paired within each row.

| Update | Frozen conclusion | Append only | Raw recomputation | Incremental ledger |
|---|---:|---:|---:|---:|
| true correction | 0.60951 | 0.28440 | 0.11640 | 0.11640 |
| false correction | 0.11640 | 0.28440 | 0.60951 | 0.60951 |
| retraction | 0.60951 | 0.60951 | 0.28440 | 0.28440 |
| irrelevant addition | 0.20976 | 0.20976 | 0.20976 | 0.20976 |
| unchanged repeat | 0.20976 | 0.25673 | 0.20976 | 0.20976 |

Replacing a known-wrong report with the true value reduces loss 0.49311 nats versus the frozen conclusion (descriptive paired-world interval 0.46025–0.52841). Replacing a true report with the false value increases loss by the same amount. Retraction reduces loss 0.32510 nats (0.30427–0.34709). Full Brier—the sum of squared probability errors across both states—falls 0.31321 for true correction and rises 0.31321 for false correction. This verifies source-specific updating, not an ability to recognize misinformation. All readers receive the same declared reliability; no learned correction truth detector exists.

Unchanged repeats and irrelevant additions preserve the raw and ledger posteriors in all 36,864 checked cases. At four reports/reliability 0.8, append-only repeat counting worsens loss from 0.20976 to 0.25673 nats. The ledger and raw decoder agree within 1.15e-14 across the entire grid. Their source-aware repeat removal is an exact idempotence control; there is no separate naive-Bayes/product-of-factors ablation in this consumer. All four native-to-shift contrasts are explicitly unavailable because this is a native static-state ruler, not an execution-shift experiment.

Reliability one requires a separate interpretation. Deliberately contradictory certain reports give zero likelihood to every state. The recorded supplied-prior fallback is accompanied by a conflict flag; it is not a defined Bayesian posterior under that impossible evidence. True correction and retraction remove the offending zero and restore valid support. False correction creates a conflict from originally consistent evidence. Scores from those fallback cases remain visible with conflict fractions; they do not establish inference under a coherent observation law.

The next table sums process CPU seconds over 30,720 initialized cases per history length (eight worlds, three reliabilities, five updates and 256 cases). Initial and one-update costs are separate. These are quantized shared-host observations, not isolated wall-clock benchmarks or many-update amortization measurements. Smaller is cheaper.

| Reports | Raw initial CPU | Raw update CPU | Ledger initial CPU | Ledger update CPU |
|---:|---:|---:|---:|---:|
| 4 | 0.015625 | 0.093750 | 0.937500 | 0.734375 |
| 16 | 0.031250 | 0.125000 | 3.859375 | 0.984375 |
| 64 | 0.109375 | 0.406250 | 17.250000 | 0.734375 |

The ledger is slower in these aggregate observations despite touching fewer factors. At 64 reports it also keeps roughly 70 additional serialized bytes: for true correction/reliability 0.8, raw records use 883 bytes and the ledger 953.20 bytes on average. Both payloads count prior and retained source records; the ledger additionally carries aggregates. Common executable code is excluded. One update follows each initialization, so no long-session break-even point was measured. The registered latency/storage continuation condition is unmet; this result does not justify a faster or smaller ledger claim. The current D05 lane stops at this verified initial ruler, leaving learned reliability and dynamic/many-update questions conditional.

Independent review checks COMPLETE/PLAN, source archives, every frozen source file, environment and all 737,280 original-plus-replay scores. The complete 1,870-file scientific replay matches except summary CPU times. Elementary regeneration reproduces priors, states, noise and each condition's starting reports; scalar raw products and separately reconstructed log/zero payloads recover every probability, conflict flag, byte count, factor count and summary. All 368,640 original payload measurements and 360 blind observation files verify. Twelve retraction fixtures exercise the canonical ProcessCase validator and offline exporter read-only, with true state/prior/posteriors evaluator-only; canonical source hashes still match. This establishes offline compatibility, not browser validation or downstream reader performance. Earlier family/score receipts are retained by matching hashes without repeating valid numerical work. No D05 repair was used or scientific source changed during review.

The [primary means, contrasts and timings](../../../results/v21/retraction/RETRACTION.json), [validity](../../../results/v21/retraction/VALIDITY.json) and [bindings](../../../results/v21/retraction/BINDINGS.json) preserve all grid cells and limits. This accepts one scientific block, bringing accepted blocks to 89; 268 numerically verified packets are not 268 findings. No fit-capacity, validation-convergence, unknown-law headroom or historical-alias claim follows from this binary supplied-law ruler. Miniature — architecture untested. There is no human-memory conclusion or fresh confirmation.

D06 is separately [registered](D06_OPPORTUNITIES_ADMISSION.json) and admitted to test fixed preference and skill across contrasting permissions and costs, with random, diversity and target-information selection and fresh contexts held out. A matched direct scalar reader receives all the same evidence. Four focused controls and two complete 206-file engineering consumers pass. The full-grid one-world pilot used 1.140625 CPU seconds; original plus full replay is estimated at 37.375 seconds including 50% margin and a conservative allowance, under two 600-second caps. One pre-admission repair establishes deterministic smallest-index ties within 1e-12 for equivalent information choices; failed source and cost are retained. This is engineering admission, not scientific acceptance. No second D06 repair is allowed.

The [operating receipt](../../../results/v21/retraction/OPERATIONS.json) records measured runway and resource accounting. Candidate-specific B02/B03, A03/A04/C06/E02/E03 and broader learned/unknown-law scope remain conditional and visible. The fixed next routine review remains October 12 14:00 Pacific; discovery freezes October 15 00:00 Pacific, confirmation opens at 06:00, science stops October 16 10:00 and final reporting/closure remains due Friday 18:00. No confirmation worlds were opened.
