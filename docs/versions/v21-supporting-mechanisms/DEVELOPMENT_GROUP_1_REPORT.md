# V21 first learned-reader and shifted-support review

Subsequent disposition: the [adaptation control review](ADAPTATION_REPORT.md) completes the B07 instrumentation and full replay described as pending in this dated batch. Its fixed target is not met. The original batch counts and findings below remain the record of that earlier review.

Does sharing a learned maker state improve reconstruction beyond the strongest matched reader? Two of 30 development comparisons clear the descriptive 0.02-nat margin, while the two primary native settings lose at 8,192 labels and 64 training laws. Learned hard support excludes truth in 79.5% to 95.3% of changed-execution forecasts and gives infinite uncapped log loss. These are constructed-method findings, miniature - architecture untested, not confirmation or a general mechanism claim.

Natural-log loss penalizes assigning low probability to the actual seven-bit history; lower is better and units are nats. Full multiclass Brier is the sum of squared probability errors over all 128 histories; lower is better. Modal accuracy is the fraction of highest-probability histories that are correct. Confident error is the fraction of wrong answers assigned at least 90% probability. Exact zeros remain infinite log loss, not a capped finite score.

Thirty B01 cells use eight development coefficient worlds, 256 cases per world and two fits averaged inside each world. The same eight worlds recur across cells. Descriptive 95% intervals use 4,096 whole-world paired bootstrap draws; they are not adjusted for the 30 inspected comparisons or selection of the strongest observed rival. No promotion is frozen and no confirmation world is exposed. All cells pass an independently reconstructed exact-law headroom gate with both finite direct fits validation-stopped.

## What the comparisons establish

With four artifact-only past episodes and an artifact-only current case under native execution, 8,192 labels from eight training laws give the shared reader a 0.06893-nat loss reduction against the strongest matched rival, the converged direct network (interval 0.04390 to 0.08838). Full Brier falls by 0.00788 and modal accuracy rises by 1.15 percentage points. The fresh-law diagnostic is better still, at 3.3750 versus 3.4307 nats, but consumes 909,312 and 983,040 extra generated labels; one fit reaches its epoch cap. This is a dependence/persistence candidate for B02/B03, not proof of maker inference.

With four context-visible past episodes and a context-visible current case under changed execution, 512 native labels give a 0.10653-nat reduction against the strongest matched rival, the direct table (0.06181 to 0.15004). Full Brier instead rises by 0.00411, with its interval crossing zero, while confident errors rise by 0.34 percentage points (0.10 to 0.63). Both fresh-law fits reach the epoch cap. This is a log-loss finding with metric reversals, not an unqualified superiority result.

In the two primary native settings with complete, inspection-observed histories, 8,192 labels from 64 training laws yield shared-reader losses 0.43023 and 0.88704 nats worse than the converged direct reader for artifact-only and context-visible current evidence. Their intervals are 0.36064 to 0.49341 and 0.81615 to 0.97072. A sparse shifted 512-label comparison improves by 0.04805 nats, but its interval of 0.00085 to 0.09242 does not clear the practical margin; it remains unresolved for that screen.

These finite implementations do not match parameter capacity. The artifact candidate uses 37,184 shared-model parameters versus 20,864 direct-network parameters; the context candidate uses about 215,000 to 227,000. Effective shared states stay above two, so neither result is a collapsed-state artifact. Finite direct validation stopping, fresh-law extra labels and cap failures, fitted capacity, and supplied exact support are separate qualifications. Per-cell fit costs, capacities, all secondary metrics and every matched-rival contrast are retained in the public aggregate.

The table below lists every learned cell. Cell numbers name fixed evidence designs: 00 artifact current/complete observed history; 01 context current/complete observed history; 05/06 artifact current and history, native/changed; 11/12 context current and history, native/changed; 17/18 sparse current and history, native/changed. Every history has four episodes. Labels are native training labels; laws are training coefficient worlds. Difference is shared-reader minus strongest matched-rival log loss, so negative favors sharing. Intervals are descriptive, not confirmation.

| Cell | Laws | Labels | Shared loss | Strongest matched rival | Rival loss | Difference [95% interval] |
|---|---:|---:|---:|---|---:|---|
| 00 | 8 | 512 | 4.2246 | direct-table | 3.4081 | +0.8165 [+0.7677, +0.8606] |
| 00 | 8 | 2048 | 4.2888 | direct-table | 3.3172 | +0.9715 [+0.8988, +1.0503] |
| 00 | 8 | 8192 | 3.7535 | converged-direct | 3.2866 | +0.4669 [+0.3684, +0.5534] |
| 00 | 64 | 512 | 4.2032 | direct-table | 3.4130 | +0.7902 [+0.7584, +0.8223] |
| 00 | 64 | 2048 | 4.3199 | grouped-empirical | 3.3210 | +0.9990 [+0.9377, +1.0627] |
| 00 | 64 | 8192 | 3.6938 | converged-direct | 3.2635 | +0.4302 [+0.3606, +0.4934] |
| 01 | 8 | 512 | 3.3713 | direct-table | 3.0654 | +0.3059 [+0.2548, +0.3562] |
| 01 | 8 | 2048 | 3.8041 | direct-table | 2.9837 | +0.8205 [+0.7549, +0.8836] |
| 01 | 8 | 8192 | 3.8013 | converged-direct | 2.9246 | +0.8767 [+0.8198, +0.9442] |
| 01 | 64 | 512 | 3.3308 | direct-table | 3.0695 | +0.2612 [+0.2087, +0.3135] |
| 01 | 64 | 2048 | 3.8174 | direct-table | 2.9882 | +0.8291 [+0.7787, +0.8751] |
| 01 | 64 | 8192 | 3.7880 | converged-direct | 2.9010 | +0.8870 [+0.8162, +0.9707] |
| 05 | 8 | 512 | 4.3323 | converged-direct | 3.9290 | +0.4032 [+0.3419, +0.4622] |
| 05 | 8 | 2048 | 3.6788 | fixed-schedule-mlp | 3.6438 | +0.0349 [-0.0240, +0.0939] |
| 05 | 8 | 8192 | 3.4307 | converged-direct | 3.4996 | -0.0689 [-0.0884, -0.0439] |
| 06 | 8 | 512 | 3.9320 | direct-table | 3.7344 | +0.1976 [+0.1487, +0.2505] |
| 06 | 8 | 2048 | 4.1360 | fixed-schedule-mlp | 3.6244 | +0.5116 [+0.4424, +0.5843] |
| 06 | 8 | 8192 | 4.5290 | fixed-schedule-mlp | 3.5887 | +0.9402 [+0.8604, +1.0217] |
| 11 | 8 | 512 | 3.7688 | direct-table | 3.2779 | +0.4909 [+0.4167, +0.5543] |
| 11 | 8 | 2048 | 3.9200 | fixed-schedule-mlp | 3.1588 | +0.7612 [+0.6880, +0.8428] |
| 11 | 8 | 8192 | 3.3297 | fixed-schedule-mlp | 3.0722 | +0.2575 [+0.1718, +0.3465] |
| 12 | 8 | 512 | 3.0942 | direct-table | 3.2007 | -0.1065 [-0.1500, -0.0618] |
| 12 | 8 | 2048 | 3.2494 | fixed-schedule-mlp | 3.1134 | +0.1360 [+0.0631, +0.2118] |
| 12 | 8 | 8192 | 3.5238 | fixed-schedule-mlp | 3.0461 | +0.4777 [+0.4107, +0.5481] |
| 17 | 8 | 512 | 3.3429 | grouped-empirical | 2.8874 | +0.4554 [+0.3918, +0.5194] |
| 17 | 8 | 2048 | 3.5714 | fixed-schedule-mlp | 2.7825 | +0.7889 [+0.7335, +0.8543] |
| 17 | 8 | 8192 | 2.9864 | fixed-schedule-mlp | 2.6931 | +0.2934 [+0.2006, +0.3834] |
| 18 | 8 | 512 | 2.7568 | direct-table | 2.8049 | -0.0480 [-0.0924, -0.0008] |
| 18 | 8 | 2048 | 2.8967 | direct-table | 2.7257 | +0.1711 [+0.1210, +0.2300] |
| 18 | 8 | 8192 | 3.1917 | fixed-schedule-mlp | 2.6665 | +0.5253 [+0.4666, +0.5870] |

## Every reader's change under shifted execution

The following table reports shifted minus native natural-log loss at 8,192 native labels and eight training laws. Positive means the reader deteriorates. Rows name the reader; columns name the common current/history evidence tier. Full results, intervals, Brier, accuracy and confidence changes at all three label budgets are in `LEARNED.json`. Training bytes and fit metadata match across each pair except CPU timing; coefficient worlds are paired, but individual test draws differ. These comparisons change historical as well as current execution. B07's current-only change is separate.

| Reader | Artifact evidence | Context evidence | Sparse evidence |
|---|---:|---:|---:|
| converged-direct | +1.20388 | +0.94700 | +0.97365 |
| direct-table | -1.39310 | -0.05873 | -0.06434 |
| fixed-schedule-mlp | +0.00275 | -0.02610 | -0.02661 |
| fresh-law-direct | +1.87910 | +1.54351 | +1.86312 |
| grouped-empirical | -0.86913 | -0.11313 | -0.07841 |
| naive-bayes | -0.05605 | -0.15190 | -0.16414 |
| naive-bayes-without-evidence | -0.00672 | -0.05383 | -0.06351 |
| shared-maker | +1.09827 | +0.19409 | +0.20530 |

Removing the evidence feature changes naive Bayes's apparent shift benefit from -0.05605 to -0.00672 nats for artifact evidence, -0.15190 to -0.05383 for context, and -0.16414 to -0.06351 for sparse evidence. The artifact and context duplicate-removed intervals cross zero; the sparse residual is negative. This is not a universal disappearance claim. The shared reader itself deteriorates in all three 8,192-label shifts. Relative robustness to a worsening neural rival is distinct from improving on its own native behavior.

## Hard support and unreliable exclusions

Seventeen support cells have actual forecasts; two artifact-only, zero-history cells admit no support screen because the prerequisite direct fit has unresolved convergence. Those two are gate dispositions, not negative support experiments. Their completion events are resolved by recording that distinction. A separate 8,192-label native zero-history gate has unresolved precision; no extension is silently admitted.

The table lists each observed support cell. The evidence design follows the fixed cell registry; the full design is in `SUPPORT.json`. No-support and exact-support columns give log loss from the same retained direct predictor. Exclusion is the percentage of forecasts whose true history the learned hard mask removes. Every learned-mask cell therefore has infinite uncapped log loss.

| Support cell | No-support loss | Supplied exact-support loss | Learned-mask truth exclusion |
|---|---:|---:|---:|
| v21-c03-00 | 3.8643 | 3.4505 | 1.59% |
| v21-c03-01 | 3.7004 | 3.1093 | 18.43% |
| v21-c03-02 | 3.7293 | 3.0884 | 95.26% |
| v21-c03-05 | 3.7646 | 3.6671 | 1.59% |
| v21-c03-06 | 4.2080 | 4.0106 | 79.54% |
| v21-c03-07 | 3.4486 | 3.1988 | 1.59% |
| v21-c03-08 | 4.2904 | 3.9885 | 79.54% |
| v21-c03-09 | 3.8247 | 3.7720 | 18.43% |
| v21-c03-10 | 5.3062 | 4.7451 | 95.26% |
| v21-c03-11 | 3.6462 | 3.1972 | 18.43% |
| v21-c03-12 | 3.8331 | 3.2528 | 95.26% |
| v21-c03-13 | 3.7105 | 2.7412 | 18.43% |
| v21-c03-14 | 4.0310 | 2.7393 | 95.26% |
| v21-c03-15 | 3.4692 | 3.4235 | 18.41% |
| v21-c03-16 | 5.1741 | 4.5808 | 93.77% |
| v21-c03-17 | 3.3154 | 2.8299 | 18.41% |
| v21-c03-18 | 3.5128 | 2.8939 | 93.77% |

Native exclusion ranges from 1.59% to 18.43%; shifted exclusion ranges from 79.54% to 95.26%. Supplied exact support always lowers log loss in these retained-predictor comparisons, but Brier reverses in four cells. All physical, learned, exact and intentionally corrupted masks, omissions, per-world metrics and matched native-to-shift contrasts are preserved. Independent sequential law/maker grouping reconstructs every supplied posterior used by these support diagnostics. This is a support-privilege result, not a method-by-support interaction or proof of deployment-available constraints. C04 soft reliability is the registered next test.

## Adaptation and transport disposition

No B07 arm reaches the registered mean excess target of 0.1 nat in the six executed cells. The best modular mean is 0.1838 at 8,192 native labels and 64 training laws; it flattens after 16 shifted labels. This numerical diagnostic is retained, but scientific transfer acceptance remains pending because original packets omit native modular forecasts and duplicate-removal adaptation controls. A single bounded instrumentation completion is source-frozen under new identities, checks reproduction of every original shifted forecast, and adds a complete family replay. It neither changes the original target nor supplies a new confirmation result. See [the frozen control specification](B07_CONTROL_COMPLETION.json).

The 24 F03 opening fixtures cover all four evidence tiers. All pass the current canonical ProcessCase validator and offline exporter read-only, reproducing retained packets and sidecars. Reader inputs and evaluator truth are exported separately. The full frozen transport family replay matches 122 scientific files. Copying, false-context, omission, decisive-selection and retraction interventions remain to be implemented with their respective families. Offline export does not establish browser behavior or live reader performance.

## Verification and remaining work

All 173 immutable completed packets in the frozen review input pass source, environment, PLAN, COMPLETE, retained-file and numerical score checks: 5,320,704 forecasts and 21,668 retained files. The latest pass reuses 156 matching score receipts; those include the previous review's 108 reused receipts. Five complete frozen-source family replay bindings and the separately accepted complete C03 replay match. Sixty-six exact-law headroom intervals and admission decisions are independently reconstructed. The family implementation hashes match the reviewed frozen sources. No suite is rerun merely because an event arrived.

This lands 47 new scientific blocks: 30 learned cells, 16 additional support diagnostics and the opening transport subset. With the two previously accepted blocks, 49 scientific blocks are accepted. Two further support cells receive explicit non-admission dispositions. Numerical dependencies, repeated source-bound ceilings, replays and verification jobs do not count as additional independent worlds or 173 separate findings. No B07 acceptance or formal promotion follows from this report.

The remaining frontier includes B02/B03 dependence and persistence, C02/C04 representation and reliability, longer and widened asymmetric histories, historical aliasing, acquisition and new-world branches. They are conditional implementations until tested and admitted. The two descriptive candidates require those discriminators and registered precision/diversity review before any promotion. Running the same valid grids again would not satisfy that obligation. The retained resource failure and its five blocked descendants remain unresolved scope; this report does not restart them.

Confirmation remains sealed until October 15 at 13:00 UTC, after discovery freezes at 07:00 UTC. Science stops October 16 at 17:00 UTC; final report and owned-service closure remain due October 17 at 01:00 UTC. This is an interim batch report, not campaign closure. The next fixed twelve-hour inspection remains October 12 at 09:00 UTC.

[Public aggregate and validity](../../../results/v21/development-group-1/VALIDITY.json). This simulation validates constructed methods; it contains no human or language-model evidence.
