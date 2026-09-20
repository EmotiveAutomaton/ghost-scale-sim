# V18.4 conditional-target bank comparison

Do conditional training targets improve useful history that a common supplied-law decoder can recover from the old-answer bank? Yes, slightly in these paired fits: farther mean logarithmic loss, where lower is better, falls from 0.78259 to 0.77983 for flat memory and from 0.77849 to 0.77717 for split memory; direct-history banks improve too. All three remain better than both no-history references under this decoder, despite their earlier deterioration in direct farther answers. This is a descriptive constructed-method result, miniature — architecture untested.

The two completed training arms differ only in original-question targets: realized
hidden-state response distributions versus the exact mixture conditional on public
history under their sampled sixteen-state training prior. Public arrays, fitted
capacity, initializations, updates and realized-state development selection match.
This diagnostic adds the identical supplied-law feasible decoder to both old-answer
banks without refitting. Every selected world, history and evaluation state matches
across the 1,536 histories and twelve banks per history in each arm.

The table reports mean logarithmic prediction loss in nats; lower is better.
Rows name reader, decoder and question group. Columns compare realized-state
targets with conditional targets and their paired difference; negative favors
conditional targets. Fit seeds average within histories; these fixed-fit descriptive
comparisons do not include independent training-dataset replications.

| Reader and decoder | Questions | Realized targets | Conditional targets | Conditional minus realized |
|---|---|---:|---:|---:|
| Question/world bank, feasible | Composition | 0.86289 | 0.86182 | -0.00107 |
| Question/world bank, feasible | Farther | 0.92558 | 0.92501 | -0.00057 |
| Direct-history bank, feasible | Composition | 0.76199 | 0.75633 | -0.00566 |
| Direct-history bank, feasible | Farther | 0.83614 | 0.83271 | -0.00343 |
| Flat-memory bank, feasible | Composition | 0.69815 | 0.69557 | -0.00258 |
| Flat-memory bank, feasible | Farther | 0.78259 | 0.77983 | -0.00276 |
| Split-memory bank, feasible | Composition | 0.69055 | 0.68320 | -0.00735 |
| Split-memory bank, feasible | Farther | 0.77849 | 0.77717 | -0.00133 |
| Question/world bank, clipped inverse | Composition | 1.21165 | 1.20638 | -0.00528 |
| Question/world bank, clipped inverse | Farther | 0.92747 | 0.92619 | -0.00128 |
| Direct-history bank, clipped inverse | Composition | 1.40782 | 1.35969 | -0.04813 |
| Direct-history bank, clipped inverse | Farther | 0.87192 | 0.85454 | -0.01738 |
| Flat-memory bank, clipped inverse | Composition | 1.33166 | 1.30129 | -0.03037 |
| Flat-memory bank, clipped inverse | Farther | 0.81458 | 0.79997 | -0.01461 |
| Split-memory bank, clipped inverse | Composition | 1.25758 | 1.21349 | -0.04408 |
| Split-memory bank, clipped inverse | Farther | 0.79722 | 0.79034 | -0.00688 |
| Same-world prior | Composition | 0.86199 | 0.86199 | +0.00000 |
| Same-world prior | Farther | 0.92214 | 0.92214 | +0.00000 |
| History posterior | Composition | 0.62823 | 0.62823 | +0.00000 |
| History posterior | Farther | 0.74313 | 0.74313 | +0.00000 |

All three feasible history banks improve on both new-question groups. The gains
are small and all readers already retain useful history under this decoder.
Conditional targets therefore do not make the old bank worse in this access test;
the direct-answer deterioration does not exhaust what those banks contain.
The no-history bank also improves slightly. Optimization, learned decoding and
selection remain competing contributors, not uniquely diagnosed mechanisms.

The supplied finite law is privileged information. Its uniform twenty-four-state
decoder prior differs from the sixteen-state training teacher and is stipulated,
not optimal for deterministic test allocation. Direct farther tests use a different
state allocation; their full means are contextual, not paired decoder-effect
estimates. The two bank arms do share every selected history and state. Neither
forecast improvement nor feasible probabilities identify unique maker roles,
learned law access or recursive predictive-state closure.

All source, plan, verified-parent input and forecast, raw-block and portable-file
hashes match. Every unit passes its gates; all 18,432 bank certificates pass with
maximum simplex gap 1.88607810e-14 below 1e-7. All 3,200 independent means and eight
extracted-source whole-unit replays agree. The earlier 21 isolated controls apply
to byte-identical frozen instrument and test sources. This is bounded replay,
not full retraining or architecture severity. Retained failed instruments and
V15 C11/M01 remain visible. No inference about human intent follows.

This arm charged 258.468750 science and 111.843750 verification CPU seconds,
totaling 370.312500 CPU seconds and 6.35 wall minutes. Its original learning cost
remains separately charged in the parent; reusing weights does not erase it.

[Portable evidence](../../results/v18/exploratory-loop/L2e-conditional-bank-1/SCIENTIFIC_MANIFEST.json).
