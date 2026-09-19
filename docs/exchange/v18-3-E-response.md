# V18.3 E: Learned memory and a changed reader purpose

Learned recurrent memory improved prediction on familiar kinds of questions but did not transfer reliably to a new question family. On familiar questions, the split reader had logarithmic loss 1.037, the flat reader 1.117 and the direct full-history reader 1.496. On new questions those rankings reversed: 2.399, 2.214 and 1.749 respectively; the exact nominal-family reference had loss 0.656. These are descriptive constructed-method results. Compact maintained state is useful for storage, but this experiment does not establish a representation that is best across purposes or a uniquely necessary role split.

The table reports expected logarithmic loss, averaged over fit seeds and architecture cells within each of thirty-two held-out coefficient draws. Lower is better. Training and development used separate draws, balanced role marginals and pairs, and development-only capacity/checkpoint selection.

| Test condition | Direct history | Flat recurrent | Split recurrent | Exact reference | Passive summary |
|---|---:|---:|---:|---:|---:|
| in-support | 1.49603 | 1.11653 | 1.03746 | 0.68490 | 1.05427 |
| new-combinations | 1.51168 | 1.13405 | 1.07278 | 0.69633 | 1.04807 |
| new-queries | 1.74914 | 2.21365 | 2.39879 | 0.65645 | 0.73488 |
| both-new | 1.74710 | 2.15446 | 2.39064 | 0.65382 | 0.71965 |

All eighteen fits completed their fixed twenty-four epochs and passed their separate learning controls. Widths 24/48, three seeds and equal full-distribution simulator supervision were shared. Parameter counts are retained rather than assumed equal. The intervention predictive bank matched the exact forecasts here; in the sixteen declared closure-test worlds it had ranks 4, 8 or 24 and passed the declared span/update checks. The passive bank had ranks 2, 4, 7 or 13 and failed update closure in every tested cell. These ranks are not a general predictive-state guarantee. The exact reference knows the generator family and uses a 24-state prior, including states absent from the balanced skill-positive test sampling; it is not claimed to be the optimal test-distribution posterior.

The independent audit reconstructed all 24,576 retained method/cell/draw rows into 768 lineage clusters and checked all 72 means. It replayed 128 fixed forecasts in each of 36 retained prediction files from extracted source and selected weights, with zero maximum difference. This is forecast replay, not full retraining. The original E attempt failed during Windows status sharing after five complete fits and part of a sixth. Its source, inputs, selected weights and latest optimizer checkpoints are retained separately; the corrected attempt reused the fixed data and is not a replication.

### New historical-role purpose with frozen memories

The separate diagnostic freezes each selected behavior encoder, then teaches an equal-supervision linear readout to answer curriculum, goal, belief and tradeoff questions. It reuses the same E histories and thirty-two lineages. The raw-history comparator has more features and readout parameters. A failed linear readout can reflect its restricted decoder, not information destroyed by the encoder.

| Test condition | Readout input | Conditional role loss | Mean role accuracy | Whole-state accuracy |
|---|---|---:|---:|---:|
| in-support | raw-history | 0.82074 | 50.73% | 7.03% |
| in-support | direct | 0.75572 | 53.16% | 10.03% |
| in-support | flat | 0.75277 | 53.97% | 9.83% |
| in-support | split | 0.75663 | 53.87% | 9.38% |
| in-support | exact | 0.32057 | 79.09% | 30.83% |
| in-support | passive-summary | 0.60250 | 64.49% | 23.37% |
| in-support | intervention-summary | 0.32057 | 79.09% | 30.83% |
| new-combinations | raw-history | 0.83044 | 48.58% | 4.30% |
| new-combinations | direct | 0.77571 | 49.76% | 6.45% |
| new-combinations | flat | 0.77426 | 51.24% | 5.40% |
| new-combinations | split | 0.77020 | 50.93% | 5.01% |
| new-combinations | exact | 0.32387 | 79.46% | 32.45% |
| new-combinations | passive-summary | 0.59227 | 66.15% | 24.93% |
| new-combinations | intervention-summary | 0.32387 | 79.46% | 32.45% |

The purpose audit checked 13,312 retained rows, 448 lineage clusters and 70 means, with 128 fixed cases replayed in each of 20 prediction files. No encoder was retrained for these labels. Native enactment is measured separately in A and the later source-uptake diagnostic. The review retains fit CPU, cached-query and update costs, model parameters and per-maker storage; storage reduction alone is not a full-cost advantage.

Numerical ties use uniform expected credit among role maxima within 1e-10. Whole-state accuracy is the product of per-role credits, not joint-posterior MAP accuracy. The original purpose packet is retained as superseded for accuracy; all proper losses, inputs, weights and forecasts are unchanged. Independent scalar reconstruction checked all 13,312 corrected accuracy rows before full mean and forecast replay. See E_PURPOSE_REPAIR.json.
