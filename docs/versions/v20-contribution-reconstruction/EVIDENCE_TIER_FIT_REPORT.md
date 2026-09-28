# V20 evidence-tier repeated fits

We tested whether repeating the fits changes the remaining long-history context comparison or the evidence-tier diagnostics. The context comparison and all 36 diagnostics fail after averaging both fits within each generator world. All 16,973,824 new forecasts verify. These are constructed-method results; confirmation remains untouched.

The final context comparison uses 256 retained episodes and 32,768 labels. Combined with the preceding 29 context conditions, all thirty native context conditions through 32,768 labels fail after averaging their two registered fits. This does not add thirty independent conditions or double the number of independent worlds. The 36 nonprimary diagnostics cover artifact, sparse-record and complete-record evidence with zero or one past episode across five budgets, plus four past episodes at 128 and 512 labels. The remaining evidence-tier repetitions and maximum-budget repetitions are pending.

Both fits use the same eight training worlds, with separate sampled examples and initialization. Their reader and evaluator cases match by immutable hashes. Each score is averaged over the two fits inside each of 32 discovery worlds before a fixed 4,096-resample world bootstrap. Intervals describe finite coefficient sensitivity conditional on the training worlds. Rows and fits are not independent worlds.

The declared comparison requires at least 0.02 natural-log units improvement against direct frequencies, the joint neural reader and the legal template, a positive lower descriptive 95% world interval, and no increase in incorrect attribution at confidence of at least 90%. Capped joint log loss is minus the natural logarithm of the true seven-part answer probability after a floor of one trillionth; lower is better. It is not an uncapped proper score. Diagnostics cannot replace the context primary.

Table: rows identify evidence access, retained episode count and training-label budget. Each loss averages the two fits within each world, then the 32 worlds. Frequencies are the direct table, joint neural is the joint-output MLP, and the oracle uses supplied law.

| Evidence | Episodes | Labels | Structured | Frequencies | Joint neural | Independent bits | Legal template | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context | 256 | 32768 | 9.91703 | 2.55646 | 2.57391 | 15.35451 | 2.49182 | 2.38750 |
| artifact | 0 | 128 | 7.93139 | 6.54316 | 5.48523 | 14.62456 | 4.14532 | 3.84758 |
| sparse | 0 | 128 | 6.74698 | 5.63247 | 5.39001 | 3.55738 | 3.23565 | 3.05794 |
| complete | 0 | 128 | 4.53296 | 3.11595 | 4.04847 | 1.79694 | 1.64786 | 1.56031 |
| artifact | 0 | 512 | 5.38573 | 4.99854 | 4.26143 | 14.63101 | 4.14532 | 3.84758 |
| sparse | 0 | 512 | 4.62874 | 5.58573 | 3.65367 | 3.38177 | 3.23565 | 3.05794 |
| complete | 0 | 512 | 2.84107 | 3.57205 | 2.14870 | 1.73234 | 1.64786 | 1.56031 |
| artifact | 0 | 2048 | 4.18782 | 4.03387 | 4.02493 | 4.14796 | 4.14532 | 3.84758 |
| sparse | 0 | 2048 | 3.49997 | 4.12831 | 3.27623 | 3.28673 | 3.23565 | 3.05794 |
| complete | 0 | 2048 | 1.87229 | 2.48052 | 1.71545 | 1.69011 | 1.64786 | 1.56031 |
| artifact | 0 | 8192 | 3.99203 | 3.93728 | 3.97569 | 4.12612 | 4.14532 | 3.84758 |
| sparse | 0 | 8192 | 3.28931 | 3.27188 | 3.22617 | 3.29677 | 3.23565 | 3.05794 |
| complete | 0 | 8192 | 1.69051 | 1.71106 | 1.65075 | 1.68087 | 1.64786 | 1.56031 |
| artifact | 0 | 32768 | 3.96782 | 3.92484 | 3.96564 | 4.00185 | 4.14532 | 3.84758 |
| sparse | 0 | 32768 | 3.25209 | 3.17794 | 3.20997 | 3.23725 | 3.23565 | 3.05794 |
| complete | 0 | 32768 | 1.66011 | 1.62823 | 1.63595 | 1.68593 | 1.64786 | 1.56031 |
| artifact | 1 | 128 | 12.11512 | 6.34252 | 5.72254 | 25.14752 | 3.99116 | 3.62277 |
| sparse | 1 | 128 | 9.76644 | 3.43664 | 5.40368 | 19.78973 | 3.02941 | 2.78890 |
| complete | 1 | 128 | 5.91508 | 1.57907 | 3.38160 | 12.03397 | 1.39966 | 1.26393 |
| artifact | 1 | 512 | 6.79892 | 5.91326 | 4.17354 | 25.75585 | 3.99116 | 3.62277 |
| sparse | 1 | 512 | 5.82216 | 3.79996 | 3.53455 | 20.43001 | 3.02941 | 2.78890 |
| complete | 1 | 512 | 3.62385 | 1.53525 | 1.94367 | 9.18209 | 1.39966 | 1.26393 |
| artifact | 1 | 2048 | 4.30461 | 4.54333 | 3.86679 | 22.21827 | 3.99116 | 3.62277 |
| sparse | 1 | 2048 | 3.57666 | 4.49147 | 3.06413 | 22.64772 | 3.02941 | 2.78890 |
| complete | 1 | 2048 | 1.89893 | 1.59465 | 1.47031 | 14.80176 | 1.39966 | 1.26393 |
| artifact | 1 | 8192 | 3.87109 | 3.87314 | 3.80857 | 25.04985 | 3.99116 | 3.62277 |
| sparse | 1 | 8192 | 3.14964 | 4.99478 | 3.00405 | 13.88132 | 3.02941 | 2.78890 |
| complete | 1 | 8192 | 1.53985 | 1.97131 | 1.40431 | 13.29911 | 1.39966 | 1.26393 |
| artifact | 1 | 32768 | 3.81049 | 3.73888 | 3.79779 | 26.38524 | 3.99116 | 3.62277 |
| sparse | 1 | 32768 | 3.05331 | 4.49317 | 2.98810 | 12.88310 | 3.02941 | 2.78890 |
| complete | 1 | 32768 | 1.45407 | 2.65922 | 1.38829 | 13.57100 | 1.39966 | 1.26393 |
| artifact | 4 | 128 | 17.42283 | 4.31139 | 5.87392 | 20.15173 | 3.83026 | 3.26405 |
| sparse | 4 | 128 | 15.51889 | 2.92571 | 4.60588 | 18.00417 | 2.74228 | 2.37875 |
| complete | 4 | 128 | 8.80047 | 1.23584 | 2.18899 | 13.36786 | 1.08387 | 0.91450 |
| artifact | 4 | 512 | 10.31841 | 4.69479 | 4.08565 | 25.00921 | 3.83026 | 3.26405 |
| sparse | 4 | 512 | 8.87884 | 2.84249 | 3.31696 | 16.81606 | 2.74228 | 2.37875 |
| complete | 4 | 512 | 5.41668 | 1.16606 | 1.63194 | 4.85044 | 1.08387 | 0.91450 |

Table: the final context condition at 256 episodes and 32,768 labels compared with each required rival. Improvement is rival minus structured capped loss; positive favors structured. Confident wrong increase is structured minus rival. The interval resamples whole worlds after within-world fit averaging.

| Rival | Improvement | World interval | Confident wrong increase |
|---|---:|---|---:|
| Frequencies | -7.36057 | [-7.89600, -6.84202] | 0.19196 |
| Joint neural | -7.34312 | [-7.87820, -6.82425] | 0.19196 |
| Legal template | -7.42521 | [-7.95829, -6.90611] | 0.19196 |

No new forecast assigns exactly zero probability to its truth. Positive probabilities below the log-score floor are counted separately in the zero-mass audit. This does not remove previously retained underflow failures.

All secondary scores, including proper squared probability loss, accuracy, component losses and confidence, were independently checked. Every new immutable completion, plan, source archive, environment, fit/decode cost and blind/evaluator role verifies. Only reader/ is blind input. Prior first-fit summaries are reused by immutable completion and summary hashes; matching reader/evaluator hashes establish pairing. Full opening G1/G4/G5/G6, scoring-repair, evidence-history and interpretation replays remain valid and are reused by hash. No new implementation, repair or interpretation requires a repeated replay. These checks do not establish cross-platform or fresh-installation reproducibility, universal failure of structured methods, or human intent.

The first pairing helper stopped on the complete-record, zero-history, 8,192-label counterpart because it required the same branch label. That earlier packet is the opening joint-head diagnostic, which calls the identical core consumer. Frozen core, training, input and posterior functions match, all evaluation cases match by hash, and the retained scoring-repair audit records no discrete changes. The corrected lookup permits only this named pair; it remains nonprimary. The failed helper and its five-second conservative CPU charge are retained. Its attempted write to a prior immutable cost receipt was refused, leaving that receipt unchanged.

There are now 211 verified packets. Score review used 463.15625 CPU seconds and paired-fit/zero-mass review used 1.84375; documentation and a conservative 180-second control-plane allowance are charged separately. All failures remain retained. The 90-hour total ceiling, 72-hour discovery ceiling and 18-hour protected reserve are unchanged.

Measured dispatch recalibration leaves 29.09 hours across 171 useful discovery blocks, from 28.68 hours under the prior estimates. No designs are added. No refill needed: measured useful admitted demand remains above one day. Complete remaining registered evidence-tier repetitions, then distribution-shift, acquisition, omission and correction challenges. All thirty native context conditions through 32768 labels fail after within-world fit averaging; no positive confirmation candidate is declared. Reassess the immutable forest before useful runway falls below one day; no extra seeds for occupancy. The 629 conditional designs total 69.46 forecast hours; combined finite demand meets the requested backlog margin. Forecast demand is not promised execution. Twelve frozen diagnostic confirmation blocks remain untouched.

[Validity](../../../results/v20/tier-fit-wave-1/VALIDITY.json), [paired comparisons and secondary means](../../../results/v20/tier-fit-wave-1/PAIRED_FITS.json), [zero-mass audit](../../../results/v20/tier-fit-wave-1/ZERO_MASS_AUDIT.json), [replay reuse](../../../results/v20/tier-fit-wave-1/REPLAY_REUSE.json), [costs](../../../results/v20/tier-fit-wave-1/COSTS.json), [forecast](../../../results/v20/tier-fit-wave-1/FORECAST.json).
