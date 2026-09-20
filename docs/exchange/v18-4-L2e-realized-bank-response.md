# V18.4 realized-target bank control

Does the sampled realized-target control retain useful history when new answers are decoded from its old-answer bank? Yes under the supplied law: farther mean logarithmic loss, where lower is better, is 0.78259 for flat memory and 0.77849 for split memory, versus 0.92558 without history and 0.92214 for the same-world prior. This establishes the control arm, not the effect of conditional training. This is a descriptive constructed-method result, miniature — architecture untested.

The diagnostic reuses the verified realized-target models without refitting or
selecting weights again. Loss-independent indices select 1,536 histories across
sixteen cells and both maker halves, with twelve banks per history. The unchanged
feasible decoder projects eighty original-answer probabilities onto mixtures of
twenty-four supplied-law state banks, then answers composed and farther questions.

The table reports mean logarithmic loss in nats; lower is better. Rows identify
the input reader and decoder; columns separate the two new-question groups.
Fit seeds average within histories. Reused histories and cells are paired data,
not independent replications of parent training.

| Reader and decoder | Composed questions | Farther forms |
|---|---:|---:|
| Question/world bank, feasible | 0.86289 | 0.92558 |
| Direct-history bank, feasible | 0.76199 | 0.83614 |
| Flat-memory bank, feasible | 0.69815 | 0.78259 |
| Split-memory bank, feasible | 0.69055 | 0.77849 |
| Question/world bank, clipped inverse | 1.21165 | 0.92747 |
| Direct-history bank, clipped inverse | 1.40782 | 0.87192 |
| Flat-memory bank, clipped inverse | 1.33166 | 0.81458 |
| Split-memory bank, clipped inverse | 1.25758 | 0.79722 |
| Same-world prior | 0.86199 | 0.92214 |
| History posterior | 0.62823 | 0.74313 |

All three feasible history banks beat both no-history references on both question
groups. The matched conditional bank is pending. Supplied-law access does not
establish learned law access, uniquely identified state or recursive predictive
closure. The decoder's uniform twenty-four-state prior differs from the teacher's
sixteen-state prior and is a stipulated comparator. The direct farther test has
a different state allocation; its full means are not paired decoder effects.

All source/archive, plan, verified-parent forecast and portable-file hashes match.
All 1,536 units pass retained gates and 18,432 bank certificates pass, with maximum
simplex gap 2.00586076e-14 against the frozen 1e-7 threshold. All 3,200 means and
eight extracted-source whole-unit replays agree. The previously passed 21 isolated
controls apply to byte-identical instrument and test sources; no repeat suite is
represented as newly run. Verification is bounded replay, not full retraining or
architecture severity. Historical instrument failures, including V15 C11/M01,
remain retained. No human-intent claim follows.

Science charged 261.843750 CPU seconds and verification 116.390625, totaling
378.234375 CPU seconds and 6.47 wall minutes. All raw bank evidence, source,
plan, exact verification and references are portable.

[Portable evidence](../../results/v18/exploratory-loop/L2e-realized-bank-1/SCIENTIFIC_MANIFEST.json).
