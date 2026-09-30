# V20 maximum-label evidence paired shifts

We tested whether the largest training budget preserves a structured-reader advantage across repeated fits with one, four or sixteen past episodes under changed rules. Two artifact-only execution diagnostics pass at four and sixteen episodes; the other twelve comparisons fail. All 6,422,528 new forecasts and both positive full-block replays verify, with no exact-zero truth probabilities. These are constructed-method results; confirmation remains untouched.

The exact result event covered fourteen new second-fit packets at 131,072 training labels: complete records with one past episode, and artifact-only, sparse and complete records with four or sixteen episodes, each under execution and presentation changes. First fits were already verified. The supervisor held at the boundary with no scientist active; review and replay acquired the exclusive scientific-worker lock. There are now 511 verified packets.

Both fits are averaged within each of 32 coefficient worlds before pairing readers and using 4,096 fixed-seed descriptive world-bootstrap resamples. Neither rows nor fits are independent worlds; uncertainty is conditional on shared training worlds. Capped joint log loss is minus the natural logarithm of the probability assigned to the true seven-part answer, floored at one trillionth; lower is better. The all-rival rule requires at least 0.02 natural-log units improvement against both direct rivals and the legal template, a positive lower world interval, and no increase in incorrect attribution at 90% confidence. These tiers remain nonprimary diagnostics.

Table: each row averages both fits within worlds. Evidence names the available record, history counts past episodes, and change names the altered law. The loss columns compare structured reading, direct frequencies, joint neural reading and the legal template; lower is better. All rows use 131,072 labels.

| Evidence | History | Change | Structured | Frequencies | Joint neural | Template | Diagnostic |
|---|---:|---|---:|---:|---:|---:|---|
| Complete | 1 | Execution | 1.39383 | 1.41036 | 1.38823 | 1.39801 | Fail |
| Complete | 1 | Presentation | 1.43389 | 1.80177 | 1.38326 | 1.39729 | Fail |
| Artifact | 4 | Execution | 3.53931 | 3.79032 | 3.56701 | 3.83271 | Pass |
| Artifact | 4 | Presentation | 6.10836 | 5.10003 | 4.10684 | 3.82909 | Fail |
| Sparse | 4 | Execution | 2.65940 | 2.74173 | 2.64463 | 2.74603 | Fail |
| Sparse | 4 | Presentation | 4.75256 | 2.74635 | 3.11429 | 2.74022 | Fail |
| Complete | 4 | Execution | 1.09443 | 1.08387 | 1.07360 | 1.08350 | Fail |
| Complete | 4 | Presentation | 1.15953 | 1.08077 | 1.06871 | 1.08038 | Fail |
| Artifact | 16 | Execution | 3.34090 | 3.58791 | 3.47093 | 3.73141 | Pass |
| Artifact | 16 | Presentation | 8.78167 | 3.59812 | 3.79727 | 3.72836 | Fail |
| Sparse | 16 | Execution | 2.42662 | 2.38897 | 2.33896 | 2.39389 | Fail |
| Sparse | 16 | Presentation | 4.94152 | 2.38474 | 2.64315 | 2.38982 | Fail |
| Complete | 16 | Execution | 0.83887 | 0.80156 | 0.80865 | 0.80124 | Fail |
| Complete | 16 | Presentation | 0.90928 | 0.80125 | 0.80771 | 0.80088 | Fail |

Table: structured loss improvement over each required rival, with descriptive whole-world intervals in natural-log units. Positive values favor structured reading. These comparisons keep the twelve failures and unresolved differences visible.

| Evidence / history / change | Rival | Improvement | World interval |
|---|---|---:|---|
| Complete / 1 / execution | Frequencies | 0.01653 | 0.00151 to 0.03150 |
| Complete / 1 / execution | Joint neural | -0.00560 | -0.01799 to 0.00673 |
| Complete / 1 / execution | Legal template | 0.00418 | -0.01174 to 0.01988 |
| Complete / 1 / presentation | Frequencies | 0.36788 | 0.35186 to 0.38422 |
| Complete / 1 / presentation | Joint neural | -0.05062 | -0.06367 to -0.03744 |
| Complete / 1 / presentation | Legal template | -0.03659 | -0.05322 to -0.02003 |
| Artifact / 4 / execution | Frequencies | 0.25101 | 0.23515 to 0.26641 |
| Artifact / 4 / execution | Joint neural | 0.02769 | 0.02414 to 0.03111 |
| Artifact / 4 / execution | Legal template | 0.29339 | 0.27741 to 0.30996 |
| Artifact / 4 / presentation | Frequencies | -1.00834 | -1.05030 to -0.96856 |
| Artifact / 4 / presentation | Joint neural | -2.00152 | -2.03812 to -1.96454 |
| Artifact / 4 / presentation | Legal template | -2.27927 | -2.32893 to -2.23064 |
| Sparse / 4 / execution | Frequencies | 0.08233 | 0.05001 to 0.11370 |
| Sparse / 4 / execution | Joint neural | -0.01477 | -0.03125 to 0.00145 |
| Sparse / 4 / execution | Legal template | 0.08664 | 0.05228 to 0.11943 |
| Sparse / 4 / presentation | Frequencies | -2.00621 | -2.05592 to -1.95629 |
| Sparse / 4 / presentation | Joint neural | -1.63827 | -1.67291 to -1.60283 |
| Sparse / 4 / presentation | Legal template | -2.01234 | -2.05775 to -1.96705 |
| Complete / 4 / execution | Frequencies | -0.01056 | -0.03594 to 0.01474 |
| Complete / 4 / execution | Joint neural | -0.02082 | -0.03814 to -0.00388 |
| Complete / 4 / execution | Legal template | -0.01093 | -0.03628 to 0.01432 |
| Complete / 4 / presentation | Frequencies | -0.07876 | -0.10611 to -0.05049 |
| Complete / 4 / presentation | Joint neural | -0.09082 | -0.10991 to -0.07115 |
| Complete / 4 / presentation | Legal template | -0.07915 | -0.10651 to -0.05092 |
| Artifact / 16 / execution | Frequencies | 0.24701 | 0.20584 to 0.28539 |
| Artifact / 16 / execution | Joint neural | 0.13004 | 0.10851 to 0.14954 |
| Artifact / 16 / execution | Legal template | 0.39051 | 0.35115 to 0.42807 |
| Artifact / 16 / presentation | Frequencies | -5.18354 | -5.36956 to -5.00436 |
| Artifact / 16 / presentation | Joint neural | -4.98440 | -5.13331 to -4.83768 |
| Artifact / 16 / presentation | Legal template | -5.05331 | -5.23320 to -4.88092 |
| Sparse / 16 / execution | Frequencies | -0.03765 | -0.07258 to -0.00347 |
| Sparse / 16 / execution | Joint neural | -0.08767 | -0.11184 to -0.06455 |
| Sparse / 16 / execution | Legal template | -0.03273 | -0.06890 to 0.00236 |
| Sparse / 16 / presentation | Frequencies | -2.55678 | -2.69106 to -2.41777 |
| Sparse / 16 / presentation | Joint neural | -2.29838 | -2.41034 to -2.18350 |
| Sparse / 16 / presentation | Legal template | -2.55170 | -2.68071 to -2.41819 |
| Complete / 16 / execution | Frequencies | -0.03731 | -0.05638 to -0.01924 |
| Complete / 16 / execution | Joint neural | -0.03022 | -0.04476 to -0.01669 |
| Complete / 16 / execution | Legal template | -0.03763 | -0.05667 to -0.01956 |
| Complete / 16 / presentation | Frequencies | -0.10803 | -0.13555 to -0.08111 |
| Complete / 16 / presentation | Joint neural | -0.10157 | -0.12293 to -0.08090 |
| Complete / 16 / presentation | Legal template | -0.10839 | -0.13592 to -0.08151 |

Under changed execution, the four-episode artifact diagnostic improves on joint neural reading by 0.02769 natural-log units (0.02414 to 0.03111); the sixteen-episode diagnostic improves by 0.13004 (0.10851 to 0.14954). Both also exceed the frequency reader and legal template by the declared margin without increased confident wrong attribution. Both second fits independently pass the same diagnostic rule. These retain two earlier first-fit positives, not two additional positive configurations. The distinct third artifact-positive configuration at 64 episodes awaits its second-fit review.

All seven presentation-shift comparisons fail. All complete- and sparse-record comparisons fail, including unresolved joint-neural differences with one complete-record episode and four sparse-record episodes under changed execution. At sixteen episodes, sparse and complete records also increase confident wrong attribution. Favorable comparisons against a single rival do not satisfy the all-rival rule. No nonprimary tier replaces the context primary, and no additional confirmation extension is admitted.

Table: secondary outcomes for the two positive artifact diagnostics, averaged over fits within worlds. Squared probability loss is the summed squared error across all joint probabilities and is proper; lower is better. Accuracy is the fraction whose most probable joint answer is correct; higher is better. Confident wrong attribution is the fraction wrong at at least 90% confidence.

| History | Reader | Squared loss | Accuracy | Confident wrong attribution |
|---:|---|---:|---:|---:|
| 4 | Structured | 0.95987 | 0.06049 | 0.0000000 |
| 4 | Frequencies | 0.99138 | 0.04881 | 0.0020523 |
| 4 | Joint neural | 0.95952 | 0.06109 | 0.0000000 |
| 4 | Legal template | 0.97294 | 0.04771 | 0.0000000 |
| 16 | Structured | 0.95339 | 0.07107 | 0.0000000 |
| 16 | Frequencies | 0.96359 | 0.04980 | 0.0000000 |
| 16 | Joint neural | 0.95742 | 0.06528 | 0.0000000 |
| 16 | Legal template | 0.97154 | 0.04829 | 0.0000000 |

At four episodes, the log-loss gain does not extend to squared probability loss or joint-answer accuracy: structured reading is slightly worse than joint neural reading on both. At sixteen episodes, structured reading improves all three. Component losses remain separate; the diagnostic rule does not require universal improvement across secondary outcomes.

Matching native second-fit maximum-label references are queued but unexecuted, so no native-to-shift effect is estimated here. First-to-second shifted-fit pairs are verified. The execution change affects both proposal routes; current and retained episodes both use the shifted law. Every learned reader receives exact shifted-law legal support. This does not test recovery of an unknown law.

Table: percentages of new second-fit truth probabilities below the log-score floor. All are positive probabilities; exact-zero truth mass is zero in every new arm. History counts past episodes.

| Evidence | History | Change | Structured below floor | Independent-bit below floor |
|---|---:|---|---:|---:|
| Complete | 1 | Execution | 0.00000% | 13.50250% |
| Complete | 1 | Presentation | 0.00000% | 15.26184% |
| Artifact | 4 | Execution | 0.00000% | 89.90784% |
| Artifact | 4 | Presentation | 0.00000% | 90.18707% |
| Sparse | 4 | Execution | 0.00000% | 71.37756% |
| Sparse | 4 | Presentation | 0.00000% | 70.86792% |
| Complete | 4 | Execution | 0.00000% | 1.22070% |
| Complete | 4 | Presentation | 0.00000% | 1.15204% |
| Artifact | 16 | Execution | 0.00000% | 68.10760% |
| Artifact | 16 | Presentation | 0.05951% | 66.10718% |
| Sparse | 16 | Execution | 0.00000% | 42.55676% |
| Sparse | 16 | Presentation | 0.01831% | 38.18665% |
| Complete | 16 | Execution | 0.00000% | 0.00000% |
| Complete | 16 | Presentation | 0.00000% | 0.00000% |

Capping hides the full penalty of tiny positive probabilities; capped log loss is not an uncapped proper score. Every stored secondary and summary reconstructs, including squared loss, component losses, accuracy and confident wrong attribution. Only reader/ is blind input; casebooks and summaries are evaluator artifacts. Bindings, fit/decode costs and attempts reconcile. Execution, numerical acceptance, discovery criteria and confirmation remain separate.

Both positive second-fit blocks replay completely from their frozen source, each matching 548 scientific files and its complete summary. Opening G1/G4/G5/G6, repaired scoring, history and shift references are reused by immutable hashes. These replays reproduce numbers; they are not independent confirmation. Producer source, scoring rules and interpretation of shifted-law support are unchanged. Validation does not cover fresh installation, cross-platform replay, universal structured-reader performance, real-text history or human intent.

The fourteen original packets consumed 3899.53125 native CPU seconds. Independent review used 143.17188, the supplement 0.70312, and both complete replays 543.90625. Forecasting, documents, publication and operating overhead are charged separately. Earlier failed sources, outputs and costs remain retained. Total/discovery ceilings remain 90/72 CPU hours with 18 protected.

Measured useful admitted work was 21.71 hours before refill. Forty-eight remaining frozen comparisons at 8,192 and 32,768 labels were admitted across zero, one, four and sixteen episodes, artifact/sparse/complete records, both shifts and registered fits. They complete lower-budget paired-fit comparisons for the retained positives and neighboring failures. The queue now has 24.64 measured hours across 191 discovery blocks. No new model setting, lineage or design outside the forest was added.

Useful admitted plus conditional demand totals 52.00 hours; the 1.25 planning cushion shortfall is 6.82 hours. Eighteen redundant descriptions remain excluded. Forecasts are estimates, not execution or occupancy. At October 1 08:00 UTC the twelve frozen diagnostic confirmation blocks retain first priority, then the original two-fit context extension within its inclusive 4,000-second cap, reserve and cutoff. Confirmation and challenge lineages remain untouched.

[Validity](../../../results/v20/shift-wave-19/VALIDITY.json), [paired fits](../../../results/v20/shift-wave-19/PAIRED_FITS.json), [aggregates](../../../results/v20/shift-wave-19/AGGREGATES.json), [zero audit](../../../results/v20/shift-wave-19/ZERO_MASS_AUDIT.json), [full replays](../../../results/v20/shift-wave-19/REPLAY.json), [replay reuse](../../../results/v20/shift-wave-19/REPLAY_REUSE.json), [costs](../../../results/v20/shift-wave-19/COSTS.json), [forecast](../../../results/v20/shift-wave-19/FORECAST.json).
