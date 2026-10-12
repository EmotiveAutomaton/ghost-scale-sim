# V21 sequential acquisition and long-history support

Two-step lookahead recovers complementary order evidence that greedy acquisition misses, but it does not improve reconstruction over a fixed policy that already buys the useful pair. At a 0.05-nat fee and two-purchase limit, its saving against that fixed policy is only 0.0117–0.0121 nats, entirely from stopping; a practical gain after computation cost is unestablished. This is a verified constructed-method diagnostic, miniature — architecture untested, with supplied law and no confirmation.

The hypothesis was that planning two purchases ahead would reveal complementary historical evidence more efficiently than one-step information or a strong fixed schedule. Eight reused development coefficient worlds supply 64 current episodes each, crossed with native and changed execution, purchase limits 1/2/4/8 and fees 0/.01/.05/.2. All policies within a condition share current cases and potential reports. Native-to-shift comparisons pair whole worlds, not individual episodes. No model fits occur in this acquisition ruler. All intervals resample eight world means 4,096 times and are descriptive; neither multiple fees nor repeated budgets are independent replications.

Two truthful records contain a random pad and operation order XOR that pad. Each alone carries no target information; together they reveal order. Six additional records independently report order with supplied reliability .8. The four policies are one-step information, two-step lookahead with stopping, a fixed full-budget schedule buying the useful pair first, and one-step information with an uncertainty stopping threshold. Exact generator law, the observation channel and the useful pair are supplied. This concerns one episode’s order, not learning a maker or learning to perform a task.

The table fixes the fee at .05 nat-equivalent units per purchase. Rows identify execution and the purchase limit. Each difference is lookahead minus the named rival in uncapped full-target log loss plus purchase fees; negative favors lookahead. The interval is a descriptive paired-world 95% interval. The fixed rival always uses its full budget.

| Execution | Limit | Against greedy | 95% interval | Against fixed pair-first | 95% interval |
|---|---:|---:|---|---:|---|
| native | 1 | +0.00000 | [+0.00000, +0.00000] | -0.21539 | [-0.27058, -0.15079] |
| native | 2 | -0.26285 | [-0.28511, -0.24328] | -0.01172 | [-0.01465, -0.00918] |
| native | 4 | -0.21117 | [-0.24569, -0.17481] | -0.11172 | [-0.11465, -0.10918] |
| native | 8 | -0.18857 | [-0.21929, -0.15781] | -0.31172 | [-0.31465, -0.30918] |
| shifted | 1 | +0.00000 | [+0.00000, +0.00000] | -0.15428 | [-0.19661, -0.11344] |
| shifted | 2 | -0.32350 | [-0.38630, -0.27205] | -0.01211 | [-0.01465, -0.00996] |
| shifted | 4 | -0.25281 | [-0.31446, -0.20842] | -0.11211 | [-0.11465, -0.10996] |
| shifted | 8 | -0.22068 | [-0.26634, -0.18310] | -0.31211 | [-0.31465, -0.30996] |

With one purchase, lookahead is exactly the one-step policy; the fixed schedule wastes that purchase on one uninformative share. With at least two purchases and fees up to .05, lookahead and pair-first have identical reconstruction probabilities. At the two-purchase, .05-fee setting, lookahead buys 1.7656 native or 1.7578 shifted records on average, while the fixed policy buys two. The current evidence already determines order in the skipped cases. At limits four and eight, the larger fixed-policy loss mainly charges additional records after the pair has already resolved order. It is not stronger reconstruction or evidence of a need for sophisticated planning. A simple pair-first policy with appropriate stopping remains a substantive untested rival.

Greedy acquisition leaves positive order uncertainty in 88.28% of native and 87.89% of shifted cases even at limit eight; it never starts the individually useless pair. At fee .05, lookahead resolves order in every case. At fee .2 it sometimes stops with order unresolved: 1.95% native and 2.15% shifted. It then has slightly worse reconstruction than pair-first, despite lower declared net loss. Full multiclass Brier, the sum of squared probability errors, reverses in the same direction: lookahead loses .9068/.8962 versus pair-first .9051/.8954 in native/shifted conditions. Reduced expense and improved inference remain separate.

The next table reports every policy’s changed-execution minus native uncapped log loss at the two-purchase, .05-fee setting. Negative means lower loss after the shift, not learned transfer. The supplied-law channel changes with execution. The complete record retains all 16 budget/fee combinations for all four policies, with full Brier, accuracy, confidence errors, purchases and missed identification.

| Policy | Change in log loss | 95% interval |
|---|---:|---|
| greedy-information | -0.04778 | [-0.13559, +0.02458] |
| two-step-lookahead | -0.10882 | [-0.16254, -0.06383] |
| fixed-pair-first | -0.10882 | [-0.16254, -0.06383] |
| uncertainty-threshold | -0.04778 | [-0.13559, +0.02458] |

[All sequential contrasts and world means](../../../results/v21/sequential/SEQUENTIAL.json) preserve every reader. Each query is purchased at most once; the six noisy reports are distinct conditionally independent observations, and the complementary pair is integrated jointly. There is no duplicated product-of-factors reader in this consumer to remove. The prior learned-reader duplicate-removal controls remain separately bound. These facts do not establish behavior under copied acquisition records.

The supplied native support event concerns 64 complete past episodes and a complete current observation with inspection hidden, under the original law. It reuses two 2,048-label direct fits inside each of eight worlds. Its exact-law headroom lower bound is .27899 nats, and both fits validation-stop. This is a retained-predictor support intervention, not a method-by-support architecture comparison.

The support table gives uncapped log loss, full multiclass Brier and the fraction of true histories removed. Lower loss/Brier is better; any true-history removal gives infinite uncapped loss. Rows change only the mask applied to the same direct probabilities.

| Support | Log loss | Full Brier | Truth removed |
|---|---:|---:|---:|
| none | 3.19926 | 0.94654 | 0.00% |
| physical | 3.12259 | 0.94902 | 0.00% |
| learned | infinite | 0.86155 | 16.43% |
| supplied-exact | 1.00887 | 0.67428 | 0.00% |

Supplied support lowers log loss by 2.19039 nats, with descriptive interval [2.10473, 2.27763], and lowers full Brier by .27226. Learned support removes 16.43% of true histories; better modal accuracy does not repair infinite log loss. Random fixed candidate omissions of 1%, 5% and 20% also exclude truth. Physical support slightly improves log loss while worsening Brier. [All support masks and world means](../../../results/v21/sequential/SUPPORT.json) and [upstream exact headroom with duplicate removal](../../../results/v21/sequential/SUPPORT_GATE.json) retain the qualifications. No matching shifted long-history support block is complete at this review, so a native-to-shift contrast is unavailable for every support arm. A shorter history is not a matched substitute.

All 240 immutable complete packets bind to their plans, source files, archives, acceptance and environment: 8,708,096 saved forecasts and 47,354 retained files verify, reusing 227 exact score receipts. The two sequential conditions contribute 65,536 forecasts. Their complete native scientific reference matches 529 files, excluding only CPU timing in the summary. Independent elementary-row current probabilities, scalar planning, 172,002 decisions, purchases, net losses and summary means reconstruct; maximum probability discrepancy is 2.23e-16. All 1,024 complementary-share cases satisfy the single-share/pair controls. The long-history support block contributes 32,768 forecasts; independently grouped supplied-law probabilities agree within 2.84e-14, and the unchanged 129-file C03 full replay is reused by source and receipt hashes. [Validity](../../../results/v21/sequential/VALIDITY.json). Three scientific blocks are newly accepted, bringing the total to 79; numerical packet counts are not counts of findings. No confirmation worlds are opened.

The two original sequential consumers record 66.5 CPU seconds including generation, checks, memoization and export. Shared cache state and loop ordering prevent an isolated per-policy latency comparison. The stipulated purchase fees are not measured computation prices. The prior zero-prior repair remains D04’s only repair; no scientific source or score changed during this review. Supplied support, fit capacity, convergence, law uncertainty and historical aliasing remain separate. No new F03 category, learned policy, realistic computation advantage, or human-process conclusion is accepted.

The next independent B04 change-point ruler is admitted in a fresh frozen source after three focused controls, two complete 205-file engineering consumers, and a full-grid one-world pilot costing 5.078125 CPU seconds. The pilot forecasts 121.875 CPU seconds for original plus full replay with 50% margin; immutable packet caps total 3,600 seconds. A single pre-dispatch export repair removes intervention labels from reader filenames and saves only each permitted prefix. The initial capsule and two unexecuted plans are retained as withdrawn; all 200 pilot forecast files remain byte-identical after the export repair. No second B04 repair is available. Its 32/96-episode horizons use paired prefixes with explicitly retained absolute change positions. Static, fast, slow, one-role-mixture and unrestricted chronological state inference share the same known channel; this is not a learned architecture comparison. Scientific acceptance awaits its own complete replay review. [Frozen B04 admission](B04_DRIFT_ADMISSION.json). Learned chronological models, B05, eligible new-world work and omission/decisive-selection/retraction fixtures remain visible conditional scope.
