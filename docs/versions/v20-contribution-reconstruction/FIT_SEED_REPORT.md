# V20 repeated-fit review

We tested whether repeating the fits changes the structured reader advantage. None of the 29 context conditions passes after averaging both fit seeds within each generator world; the final large-budget complete-record diagnostic also fails. All 13,762,560 new forecasts verify. These are constructed-method results; confirmation remains untouched.

The 29 context comparisons cover 0, 1, 4, 16 and 64 retained episodes at 128 through 32,768 labels, plus 256 episodes at 128 through 8,192 labels. The second fit at 256 episodes and 32,768 labels remains pending. Both fits use the same eight training worlds, with separate training samples and initializations; their 32 discovery worlds and evaluator cases match. Seeds are averaged within each discovery world before the 4,096-resample world bootstrap. This remains 32 worlds, not 64; intervals describe coefficient sensitivity conditional on these training worlds.

The declared rule requires at least 0.02 natural-log units improvement over both direct rivals and the legal template, a positive lower descriptive 95% world interval, and no increase in wrong attribution made at confidence of at least 90%. Capped joint log loss is minus the natural logarithm of the true seven-part answer probability, floored at one trillionth. Lower is better; it is not an uncapped proper score.

Table: each row identifies a context condition by retained episode count and training-label budget. Columns give capped joint log loss averaged first over two fits within each world, then over the 32 worlds.

| Episodes | Labels | Structured | Frequencies | Joint neural | Independent bits | Legal template | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 7.28113 | 6.10427 | 5.69618 | 3.84168 | 3.58216 | 3.40445 |
| 0 | 512 | 5.09266 | 6.00349 | 4.00512 | 3.77533 | 3.58216 | 3.40445 |
| 0 | 2048 | 3.88361 | 4.48726 | 3.65087 | 3.67443 | 3.58216 | 3.40445 |
| 0 | 8192 | 3.63795 | 3.62048 | 3.59515 | 3.69495 | 3.58216 | 3.40445 |
| 0 | 32768 | 3.59887 | 3.52491 | 3.58141 | 3.66775 | 3.58216 | 3.40445 |
| 1 | 128 | 10.71069 | 3.93081 | 5.80240 | 22.32449 | 3.37592 | 3.13541 |
| 1 | 512 | 6.49065 | 4.46815 | 3.88777 | 21.84260 | 3.37592 | 3.13541 |
| 1 | 2048 | 4.00263 | 5.21052 | 3.44843 | 24.10838 | 3.37592 | 3.13541 |
| 1 | 8192 | 3.49326 | 5.47146 | 3.38197 | 12.38630 | 3.37592 | 3.13541 |
| 1 | 32768 | 3.39799 | 4.63025 | 3.36694 | 22.90627 | 3.37592 | 3.13541 |
| 4 | 128 | 16.50570 | 3.30362 | 5.08399 | 21.41026 | 3.08879 | 2.72526 |
| 4 | 512 | 10.08586 | 3.24243 | 3.67626 | 17.59890 | 3.08879 | 2.72526 |
| 4 | 2048 | 4.79309 | 3.17259 | 3.12453 | 22.89621 | 3.08879 | 2.72526 |
| 4 | 8192 | 3.36394 | 3.16283 | 3.04623 | 24.61812 | 3.08879 | 2.72526 |
| 4 | 32768 | 3.09814 | 3.17419 | 3.03072 | 24.23451 | 3.08879 | 2.72526 |
| 16 | 128 | 22.21027 | 2.95700 | 4.26055 | 20.07731 | 2.73679 | 2.46121 |
| 16 | 512 | 17.08505 | 2.88736 | 3.48671 | 20.02434 | 2.73679 | 2.46121 |
| 16 | 2048 | 8.81905 | 2.81829 | 2.84391 | 18.80645 | 2.73679 | 2.46121 |
| 16 | 8192 | 4.15810 | 2.80677 | 2.77921 | 18.21450 | 2.73679 | 2.46121 |
| 16 | 32768 | 3.06383 | 2.80207 | 2.76087 | 18.64611 | 2.73679 | 2.46121 |
| 64 | 128 | 23.86502 | 2.74385 | 3.68036 | 16.66392 | 2.52141 | 2.39208 |
| 64 | 512 | 22.16124 | 2.67106 | 3.00121 | 18.94514 | 2.52141 | 2.39208 |
| 64 | 2048 | 15.40469 | 2.60244 | 2.68976 | 19.06007 | 2.52141 | 2.39208 |
| 64 | 8192 | 8.47378 | 2.59152 | 2.61162 | 13.27694 | 2.52141 | 2.39208 |
| 64 | 32768 | 4.17281 | 2.58604 | 2.59933 | 14.10832 | 2.52141 | 2.39208 |
| 256 | 128 | 24.37863 | 2.71396 | 3.51449 | 13.39117 | 2.49182 | 2.38750 |
| 256 | 512 | 23.95454 | 2.64138 | 2.85347 | 9.70401 | 2.49182 | 2.38750 |
| 256 | 2048 | 21.86334 | 2.57288 | 2.65664 | 17.63723 | 2.49182 | 2.38750 |
| 256 | 8192 | 14.37568 | 2.56198 | 2.58188 | 18.25667 | 2.49182 | 2.38750 |

Table: rows show the largest completed two-fit label budget for each history and each required rival. Improvement is rival minus structured capped loss, so positive favors structured reading. The interval resamples whole worlds after averaging the fits. Confident wrong increase is structured minus rival. All remaining intervals and all secondary means are retained in the [paired-fit evidence](../../../results/v20/fit-seed-wave-1/PAIRED_FITS.json).

| Episodes | Labels | Rival | Improvement | World interval | Confident wrong increase |
|---:|---:|---|---:|---|---:|
| 0 | 32768 | Frequencies | -0.07397 | [-0.08127, -0.06641] | 0.00000 |
| 0 | 32768 | Joint neural | -0.01746 | [-0.02502, -0.00984] | 0.00000 |
| 0 | 32768 | Legal template | -0.01671 | [-0.03028, -0.00321] | 0.00000 |
| 1 | 32768 | Frequencies | 1.23226 | [1.20768, 1.25661] | -0.01996 |
| 1 | 32768 | Joint neural | -0.03105 | [-0.04050, -0.02126] | 0.00000 |
| 1 | 32768 | Legal template | -0.02207 | [-0.03951, -0.00526] | 0.00000 |
| 4 | 32768 | Frequencies | 0.07605 | [0.04421, 0.10885] | -0.00049 |
| 4 | 32768 | Joint neural | -0.06742 | [-0.08494, -0.04935] | 0.00000 |
| 4 | 32768 | Legal template | -0.00935 | [-0.04229, 0.02346] | 0.00000 |
| 16 | 32768 | Frequencies | -0.26176 | [-0.30346, -0.21911] | 0.00010 |
| 16 | 32768 | Joint neural | -0.30296 | [-0.33419, -0.27140] | 0.00010 |
| 16 | 32768 | Legal template | -0.32704 | [-0.37027, -0.28405] | 0.00010 |
| 64 | 32768 | Frequencies | -1.58677 | [-1.68900, -1.48745] | 0.01166 |
| 64 | 32768 | Joint neural | -1.57349 | [-1.67334, -1.47563] | 0.01166 |
| 64 | 32768 | Legal template | -1.65140 | [-1.75190, -1.55039] | 0.01166 |
| 256 | 8192 | Frequencies | -11.81370 | [-12.17448, -11.45859] | 0.47298 |
| 256 | 8192 | Joint neural | -11.79379 | [-12.15420, -11.43982] | 0.47298 |
| 256 | 8192 | Legal template | -11.88385 | [-12.24739, -11.52541] | 0.47298 |

The separate complete-record diagnostic at 64 episodes and 131,072 labels uses one fit seed. Structured capped loss is 0.89361, versus 0.69765 for direct frequencies and 0.69748 for the legal template. It fails the comparison rule and cannot replace the context primary. Complete observation still withholds private aims and leaves inspection latent.

Exact zero truth probability reaches 43.08% in a new structured arm. All 38,133 such rows retain supplied legal support and have finite learned log weights; their true-to-largest-class odds underflow on exponentiation. This repeats the previously replayed numerical-concentration failure, separately from forced candidate omission. Positive values below the score floor are also retained. Saved probabilities and criteria are unchanged. Squared probability loss, accuracy, component scores and confidence secondaries independently verify.

New score reconstruction covers every forecast and secondary; plan, frozen source, environment, fit and decoding costs, attempted failures and blind/evaluator roles verify. Only reader/ is blind input. Prior verified first-seed summaries are reused by immutable completion hashes, with matching reader/evaluator hashes across fits. Full opening G1/G4/G5/G6 replays and subsequent scoring-repair, history, evidence-tier and interpretation replays remain valid and are reused by hash. No new family, repair or changed scientific interpretation requires another replay. These checks do not establish fresh-installation or cross-platform reproducibility, or that every structured method fails.

There are now 174 verified packets. Independent review used 457.06250 CPU seconds and paired-fit/zero-mass reconstruction used 10.56250; documentation and a conservative 180-second control-plane allowance are also charged within the unchanged 90-hour ceiling. All failed sources and prior resource incidents remain retained; 18 hours stay protected.

Dispatch recalibration changes the remaining useful forecast from 30.54 to 30.83 hours across 208 discovery blocks. No designs are added. No refill needed at this boundary: recalibrated useful admitted demand remains above one day. Complete the last second-seed context comparison, then the admitted distribution-shift, acquisition, omission and correction challenges. Second-seed replication does not rescue the observed native comparison, so no positive confirmation candidate is declared. Retain the frozen finite forest and reassess before useful runway falls below one day; no extra seeds for occupancy. The 629 conditional designs total 68.75 forecast hours; combined finite demand meets the requested backlog margin. This is not promised execution. Twelve diagnostic confirmation blocks remain untouched. Execution, numerical acceptance, failed discovery criteria and confirmation stay separate. No simulator result establishes human intent.

[Validity](../../../results/v20/fit-seed-wave-1/VALIDITY.json), [costs](../../../results/v20/fit-seed-wave-1/COSTS.json), [zero-mass audit](../../../results/v20/fit-seed-wave-1/ZERO_MASS_AUDIT.json), [replay reuse](../../../results/v20/fit-seed-wave-1/REPLAY_REUSE.json), [forecast](../../../results/v20/fit-seed-wave-1/FORECAST.json).
