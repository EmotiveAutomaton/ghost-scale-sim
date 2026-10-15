# V21 final diagnostic: information limits and remaining reader error

Does the trained direct reader reach the information available in its observations?
No. Across 32 reserved worlds, its excess log loss is **0.16681 nats**, with a
95% paired whole-world interval of **0.14575–0.18847**. Both unique fits stopped
on validation. The fixed 0.02-nat criterion therefore returns **EXCESS READER
ERROR**. This is a verified constructed-method diagnostic, not confirmation of
a learned maker mechanism or evidence about human intent.

## Fixed comparison

The [pre-exposure manifest](FINAL_DIAGNOSTIC_MANIFEST.json) fixes 32 worlds,
256 cases per world, four complete observed historical episodes, the original
finite law menu and two training seeds. Training has 8,192 labels from 64
development laws. Exact inference integrates the 12,500 possible coefficient
laws. The supplied-law oracle is a separate privilege. The final test uses
8,192 cases; the two fits stay nested within each world.

Three execution partitions cover 11, 11 and 10 worlds. Their six reported fits
are repetitions of **two identical fitted models**, with training arrays and
weights checked against the frozen development reference. They are not six
independent training replicates. The final consumer pools all 32 equally weighted
worlds before 4,096 paired resamples; partition-specific admission labels and
intervals do not determine the result.

The table compares all readers on the same cases. Log loss is negative natural-log
probability assigned to the true seven-bit history; full Brier is total squared
probability error. Lower is better for both. Accuracy is the fraction whose most
probable complete history is correct; top-eight coverage is the fraction where
truth appears among the eight highest-probability histories. Excess subtracts
the exact reader that integrates unknown laws.

| Reader | Log loss | Excess loss, 95% interval | Full Brier | Accuracy | Top-eight coverage |
|---|---:|---:|---:|---:|---:|
| Supplied-law oracle | 3.00056 | Separate privilege | 0.92717 | 12.02% | 62.16% |
| Exact law-marginal | 3.13842 | Reference | 0.93785 | 10.75% | 56.77% |
| Validation-stopped direct | 3.30523 | 0.16681 [0.14575, 0.18847] | 0.95270 | 9.46% | 51.20% |
| Naive Bayes | 3.78262 | 0.64420 [0.60179, 0.68806] | 0.99916 | 8.62% | 49.38% |
| Naive Bayes, duplicate evidence removed | 3.65705 | 0.51863 [0.47863, 0.55913] | 0.98538 | 8.48% | 49.49% |

Removing duplicate evidence improves naive Bayes log loss and full Brier but
slightly lowers modal accuracy. The direct reader outperforms both naive Bayes
variants while still leaving information unused. No reader assigns zero mass to
truth in this diagnostic. Confident complete-history errors are absent for the
direct reader and occur at 0.0061% for each naive Bayes variant; all metrics and
intervals remain in [the full diagnostic](../../../results/v21/final-closeout/DIAGNOSTIC.json).

## Which uncertainty remains

The expected loss decomposition is independently reconstructed under the true
supplied-law predictive distribution. Mean irreducible uncertainty is 2.99942
nats; integrating unknown laws adds 0.14043 nats; the direct reader adds a further
0.16803 nats. Those expected quantities differ from realized sample losses above.
Their sum agrees with the independently computed expected direct loss. Expected
full-Brier terms are 0.92671, 0.01110 and 0.01511, respectively.

The exact posterior still supports 37.69 complete histories per case on average.
Thus historical ambiguity and reader error coexist. A nonzero reader gap does
not imply that all of the history can be recovered, identify the best replacement
architecture, or establish that the maker model supplies the missing capability.
Zero maker-specific candidates survived the earlier historical-content test.

## Validity and disposition

The independent consumer checks every new completion/file binding, score,
truth alignment, expected uncertainty/error telescope, training array and fitted
weight. All checks pass. The unchanged source reuses its already-complete exact
ceiling and headroom family replays by their frozen completion hashes. This is
the project's declared complete-family replay policy, not a claim that each
new reserved world was separately replayed. The
[validity receipt](../../../results/v21/final-closeout/FINAL_VALIDITY.json) binds
the review and the [full retained raw-file index](../../../results/v21/final-closeout/RAW_HASH_INDEX.json.gz).

The October 14 owner amendment advances the frozen diagnostic and closes V21
without calendar waiting. The original acceptance, scientific criteria and
source capsule remain unchanged. This single diagnostic resolves F01; it does
not create a maker-mechanism promotion or trigger F02. No exploratory refill is
admitted. Miniature — architecture untested beyond this finite construction.
