# V18.4 partition and forecast equivalence

Why do some incorrect history-purpose pairings nearly match correct alignment? With four memory symbols, shifting the supplemental targets by four history positions preserves the same future forecasts to numerical tolerance in 252 of 320 original-rule worlds and 240 of 320 alternative-rule worlds; the actual history partitions match in only 244 and 166, respectively. Near ties partly reflect equivalence for the tested future questions: different memory partitions need not change those answers. This is a descriptive constructed-method result with supplied decoding, miniature — architecture untested.

This new diagnosis reuses all 640 verified purpose-alignment worlds and their
previously selected codes. It adds no draws, fits, target permutations or code
selection. All parent payloads match their source blocks, and every used future
score reproduces before new interpretation. At each memory size, aligned codes
are compared with all seven cyclic shifts and the marginal-target control.
Symbol names are removed before comparing partitions. Future forecasts are
uniform-history averages within each code class under the common supplied decoder.

The table reports all memory sizes and comparisons. Each row covers 320 worlds
under the named rule group. Original rules include the parent's softmax and
satisficing cells; the alternative is lexicographic. Symbols means available
memory codes. Shift numbers are history-row rotations; marginal removes the
supplemental targets' history dependence. Same partition counts identical history
groupings. Same forecasts counts largest probability differences at most 1e-12
over all five future tasks. Loss difference is comparison minus aligned mean
logarithmic loss in nats; positive favors alignment. The final column restricts
that descriptive difference to worlds with different partitions. A dash means
no such worlds. Reused cells and shifts are paired, not independent replications.

| Rule group | Symbols | Comparison | Same partition | Same forecasts | Mean loss difference | Difference among changed partitions |
|---|---:|---|---:|---:|---:|---:|
| Original | 1 | marginal | 320 | 320 | +0.000000 | — |
| Original | 1 | shift1 | 320 | 320 | +0.000000 | — |
| Original | 1 | shift2 | 320 | 320 | +0.000000 | — |
| Original | 1 | shift3 | 320 | 320 | +0.000000 | — |
| Original | 1 | shift4 | 320 | 320 | +0.000000 | — |
| Original | 1 | shift5 | 320 | 320 | +0.000000 | — |
| Original | 1 | shift6 | 320 | 320 | +0.000000 | — |
| Original | 1 | shift7 | 320 | 320 | +0.000000 | — |
| Original | 2 | marginal | 244 | 244 | +0.016615 | +0.069957 |
| Original | 2 | shift1 | 182 | 182 | +0.020652 | +0.047888 |
| Original | 2 | shift2 | 220 | 220 | +0.014705 | +0.047056 |
| Original | 2 | shift3 | 208 | 208 | +0.020608 | +0.058880 |
| Original | 2 | shift4 | 228 | 228 | +0.008047 | +0.027990 |
| Original | 2 | shift5 | 182 | 182 | +0.023689 | +0.054931 |
| Original | 2 | shift6 | 216 | 216 | +0.015291 | +0.047050 |
| Original | 2 | shift7 | 196 | 196 | +0.021466 | +0.055396 |
| Original | 4 | marginal | 176 | 198 | +0.030747 | +0.068326 |
| Original | 4 | shift1 | 108 | 120 | +0.037804 | +0.057063 |
| Original | 4 | shift2 | 200 | 212 | +0.002158 | +0.005755 |
| Original | 4 | shift3 | 96 | 108 | +0.037399 | +0.053427 |
| Original | 4 | shift4 | 244 | 252 | +0.000651 | +0.002740 |
| Original | 4 | shift5 | 96 | 108 | +0.037310 | +0.053300 |
| Original | 4 | shift6 | 200 | 212 | +0.002241 | +0.005975 |
| Original | 4 | shift7 | 96 | 108 | +0.037821 | +0.054030 |
| Original | 8 | marginal | 320 | 320 | +0.000000 | — |
| Original | 8 | shift1 | 320 | 320 | +0.000000 | — |
| Original | 8 | shift2 | 320 | 320 | +0.000000 | — |
| Original | 8 | shift3 | 320 | 320 | +0.000000 | — |
| Original | 8 | shift4 | 320 | 320 | +0.000000 | — |
| Original | 8 | shift5 | 320 | 320 | +0.000000 | — |
| Original | 8 | shift6 | 320 | 320 | +0.000000 | — |
| Original | 8 | shift7 | 320 | 320 | +0.000000 | — |
| Alternative | 1 | marginal | 320 | 320 | +0.000000 | — |
| Alternative | 1 | shift1 | 320 | 320 | +0.000000 | — |
| Alternative | 1 | shift2 | 320 | 320 | +0.000000 | — |
| Alternative | 1 | shift3 | 320 | 320 | +0.000000 | — |
| Alternative | 1 | shift4 | 320 | 320 | +0.000000 | — |
| Alternative | 1 | shift5 | 320 | 320 | +0.000000 | — |
| Alternative | 1 | shift6 | 320 | 320 | +0.000000 | — |
| Alternative | 1 | shift7 | 320 | 320 | +0.000000 | — |
| Alternative | 2 | marginal | 272 | 272 | +0.018664 | +0.124426 |
| Alternative | 2 | shift1 | 192 | 192 | +0.024521 | +0.061302 |
| Alternative | 2 | shift2 | 272 | 272 | +0.018664 | +0.124426 |
| Alternative | 2 | shift3 | 272 | 272 | +0.018664 | +0.124426 |
| Alternative | 2 | shift4 | 240 | 240 | +0.005857 | +0.023429 |
| Alternative | 2 | shift5 | 192 | 192 | +0.031545 | +0.078863 |
| Alternative | 2 | shift6 | 272 | 272 | +0.018664 | +0.124426 |
| Alternative | 2 | shift7 | 272 | 272 | +0.018664 | +0.124426 |
| Alternative | 4 | marginal | 82 | 160 | +0.056359 | +0.075777 |
| Alternative | 4 | shift1 | 0 | 80 | +0.121454 | +0.121454 |
| Alternative | 4 | shift2 | 160 | 240 | +0.000603 | +0.001205 |
| Alternative | 4 | shift3 | 6 | 80 | +0.106733 | +0.108772 |
| Alternative | 4 | shift4 | 166 | 240 | +0.000032 | +0.000067 |
| Alternative | 4 | shift5 | 0 | 80 | +0.121463 | +0.121463 |
| Alternative | 4 | shift6 | 160 | 240 | +0.000834 | +0.001668 |
| Alternative | 4 | shift7 | 6 | 80 | +0.107616 | +0.109673 |
| Alternative | 8 | marginal | 320 | 320 | +0.000000 | — |
| Alternative | 8 | shift1 | 320 | 320 | +0.000000 | — |
| Alternative | 8 | shift2 | 320 | 320 | +0.000000 | — |
| Alternative | 8 | shift3 | 320 | 320 | +0.000000 | — |
| Alternative | 8 | shift4 | 320 | 320 | +0.000000 | — |
| Alternative | 8 | shift5 | 320 | 320 | +0.000000 | — |
| Alternative | 8 | shift6 | 320 | 320 | +0.000000 | — |
| Alternative | 8 | shift7 | 320 | 320 | +0.000000 | — |

For the four-position shift, mean loss differences are only 0.000651 under original
rules and 0.000032 under the alternative. Among changed partitions, the differences
are 0.002740 and 0.000067. Equality of partitions is therefore not the only source
of near ties: some changed partitions still give the same forecasts. Conversely,
the remaining changed forecasts are not all uniformly close: maximum probability
differences reach 0.23994 and 0.00760 in those two groups. Aggregate closeness can
hide individual differences. These are descriptions of the retained cases,
not an intervention establishing which code features are necessary.

One- and eight-symbol partitions are fixed up to labels. Two- and four-symbol
results retain both equivalence and harmful misalignment. Across the retained
comparisons there are no exact loss ties at the declared tolerance with different
forecasts; the independent fixture nevertheless establishes that this is possible
in general. The observed absence is an outcome, not a validity gate.

All 640 units, 7,168 independent means and eight whole-unit extracted-source
replays pass. Eleven isolated known-answer, placebo, label-invariance, score,
corruption and runtime controls pass. All source/plan/parent/raw/portable hashes
match; raw scientific cases and every comparison remain available. Independent
scalar membership sets, conditional probabilities and proper scores reconstruct
the diagnostic. The validation does not establish learned access, unrestricted
future sufficiency, architecture severity, population effects or human intent.
Historical V15 C11/M01 and earlier retained failures are unchanged.

Science charged 143.671875 CPU seconds and replay 26.156250, totaling 169.828125; combined wall time was 2.91 minutes. This supersedes the 150–400-second forecast with a measured cost.

[Portable evidence](../../results/v18/exploratory-loop/P3-partition-access-1/SCIENTIFIC_MANIFEST.json).
