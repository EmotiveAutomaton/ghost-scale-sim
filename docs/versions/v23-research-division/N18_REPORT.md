# N18: play, constraint and detector accuracy

N18 puts the mean play-detector break-even accuracy at 63.96% (95% whole-law interval 63.54% to 64.43%) in its constructed symmetric-error channel. Thresholds across the 16 laws range from 62.89% to 65.96%. A 60%-accurate detector leaves play less informative than the strongly constrained record; 80% reverses that comparison. Full replay and independent reconstruction pass. The weak-situation advantage is built into the generator, not established for human play.

The question was how accurately an inversion detector must work before a playful record contains more information about the maker than a strength-0.8 record. Sixteen independent norm/noise laws were drawn. Every law is evaluated by exact finite-channel sums. The inherited 64 enumeration records per law are duplicates, retained for the transport contract; they create no additional sample size.

Each table entry is mean information about the binary maker parameter, in nats per record, across the 16 laws. Strength is the norm weight; the remaining weight belongs to the maker. Only strength 0.2 has a fair inversion. Detector accuracy has symmetric binary errors, and the reader knows the norm/noise law.

| Strength | Detector accuracy | Ignores frames | Uses detector | Oracle frames |
|---:|---:|---:|---:|---:|
| 0.2 | 60% | 0.000000 | 0.003733 | 0.099314 |
| 0.2 | 80% | 0.000000 | 0.034224 | 0.099314 |
| 0.2 | 95% | 0.000000 | 0.079307 | 0.099314 |
| 0.2 | 100% | 0.000000 | 0.099314 | 0.099314 |
| 0.5 | 60% | 0.041369 | 0.041369 | 0.041369 |
| 0.5 | 80% | 0.041369 | 0.041369 | 0.041369 |
| 0.5 | 95% | 0.041369 | 0.041369 | 0.041369 |
| 0.5 | 100% | 0.041369 | 0.041369 | 0.041369 |
| 0.8 | 60% | 0.007849 | 0.007849 | 0.007849 |
| 0.8 | 80% | 0.007849 | 0.007849 | 0.007849 |
| 0.8 | 95% | 0.007849 | 0.007849 | 0.007849 |
| 0.8 | 100% | 0.007849 | 0.007849 | 0.007849 |

At 60% accuracy the play channel has 0.003733 nats per record, below 0.007849 for strength 0.8. At 80% it has 0.034224, above the strongly constrained channel but still below the medium-strength channel's 0.041369. Ignoring play frames loses all maker information under the stipulated fair inversion; perfect detection equals the oracle. Do not generalize the strong-constraint comparison to every context.

The primary 63.96% is the average of law-specific equality thresholds, not a universal safe cutoff. Strictly above a law's threshold gives the advantage in that law. The largest threshold among these 16 draws is 65.9623%; this is a descriptive maximum, not a proved bound over untested laws. The engine route requires an independently measured detector and defensible calibration of the actual error channel before increasing play weights. N8's reversal warning remains.

Controls verify the entropy/joint-sum identity, zero information at full constraint and chance detection, perfect-detector/oracle equality, monotonicity and an independent bisection solution. Every raw channel, score and whole-law interval reconstructs; complete fresh-process replay passes. The batch plus independent review charged 17.015625 CPU-seconds. No human-play mechanism or engine admission follows.

[Protocol](N18_PROTOCOL.md), [summary](../../../results/v23/n18/SUMMARY.json), [proof](../../../results/v23/n18/PROOF.json), [raw hashes](../../../results/v23/n18/RAW_INDEX.json).
