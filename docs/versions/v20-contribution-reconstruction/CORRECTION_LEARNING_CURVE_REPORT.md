# V20 context-correction learning curve and candidate completion

We tested whether more training makes context revision and candidate expansion reliable. At 128 labels, correcting false context helps less than an irrelevant update, and retracting it makes predictions worse. Correction exceeds the controls at the five larger budgets. Larger candidate lists improve coverage; one lower-label capped-loss comparison is unresolved. All 10,420,224 forecasts and three full replays verify. These are constructed-method findings; confirmation remains untouched.

Fifteen new packets finish the admitted candidate and correction sensitivity panels. Six candidate packets add 16, 32 or 64 alternatives at 2,048 labels in the first registered fit and 32,768 labels in the second. Nine correction packets complete both registered fits at six budgets from 128 to 131,072 labels, reusing the three already verified first-fit references. Each condition uses 32 discovery coefficient worlds and 2,048 evaluation cases per world. Repeated fits are averaged within each world before the fixed-seed 4,096-resample world bootstrap. Rows, budgets and fits are not independent worlds.

## Context revision

Capped log loss is negative log truth probability with a floor of 1e-12, in natural-log units; lower is better. A gain is saved-answer loss minus recomputed-answer loss, so positive values favor recomputation. Exact-zero truth probability is recorded separately. Squared probability loss and all secondary scores also verify.

Table: each row is a training-label budget, with both registered fits averaged within each whole world. Correction replaces a false context cue with the true cue; retraction removes the false cue; an irrelevant update adds an independent random cue. Values are capped-loss gains with descriptive 95% whole-world intervals. Specific correction is the correction gain minus the irrelevant-update gain.

| Labels | Correction gain | Retraction gain | Irrelevant-update gain | Specific correction |
|---:|---:|---:|---:|---:|
| 128 | 0.15648 [0.14924, 0.16352] | -0.41243 [-0.42015, -0.40440] | 0.44457 [0.43672, 0.45268] | -0.28810 [-0.30066, -0.27577] |
| 512 | 0.45357 [0.44006, 0.46723] | 0.25319 [0.24665, 0.25942] | 0.07282 [0.05986, 0.08617] | 0.38075 [0.35487, 0.40650] |
| 2,048 | 1.05074 [1.03892, 1.06290] | 0.80599 [0.79925, 0.81277] | -0.15795 [-0.16594, -0.15022] | 1.20869 [1.19477, 1.22220] |
| 8,192 | 1.15922 [1.14543, 1.17255] | 0.81069 [0.80323, 0.81788] | -0.16950 [-0.17395, -0.16498] | 1.32872 [1.31223, 1.34460] |
| 32,768 | 1.19780 [1.18349, 1.21181] | 0.80034 [0.79258, 0.80814] | -0.13536 [-0.14016, -0.13027] | 1.33316 [1.31538, 1.34981] |
| 131,072 | 1.17733 [1.16291, 1.19088] | 0.77296 [0.76529, 0.78024] | -0.13092 [-0.13552, -0.12610] | 1.30826 [1.29079, 1.32472] |

The 128-label condition fails specificity: a positive correction gain alone would misidentify an improvement that is smaller than the irrelevant-update control. Retraction is harmful at this budget. At 512 labels, correction exceeds both unchanged and irrelevant controls, while the irrelevant update itself still improves the score. From 2,048 labels onward, irrelevant updates harm prediction. These reversals concern a finite learned decoder, not a general claim that random evidence is informative or that false context should be retained.

All unchanged and already-correct cue controls reproduce the saved answer exactly. Ledger and bounded-raw recomputation are byte-identical throughout: they are the same learned readout on the same revised evidence, a reconstruction control rather than independently learned methods. No context fact is inferred from human material. This family is separate from the structured-versus-direct primary comparison and does not create a new confirmation candidate.

## Candidate coverage and forecast quality

The starting list contains 64 of the 128 supplied process alternatives. Expansion ranks omitted alternatives with the learned decoder and a fixed stable tie rule. Preserved unknown mass is distributed uniformly over omitted alternatives only for full-universe scoring. The candidate-plus-unknown event score changes its target partition as lists expand and must not rank budgets on a common target.

Table: rows identify the training panel and added alternatives. Coverage is the percentage of truths explicitly listed. Loss gain compares expansion with retaining unknown mass without expansion; intervals use paired whole worlds. The lower-label panel contains only its first registered fit.

| Training panel | Added alternatives | Truth coverage | Capped loss | Loss gain [interval] |
|---|---:|---:|---:|---:|
| 2,048 labels, first fit | 16 | 90.98% | 3.90334 | 0.10235 [0.09376, 0.11109] |
| 2,048 labels, first fit | 32 | 95.97% | 3.90673 | 0.09895 [0.08922, 0.10861] |
| 2,048 labels, first fit | 64 | 100.00% | 3.88832 | 0.11737 [0.10800, 0.12688] |
| 32,768 labels, both fits | 16 | 91.36% | 3.64345 | 0.17578 [0.16789, 0.18423] |
| 32,768 labels, both fits | 32 | 96.63% | 3.61532 | 0.20391 [0.19606, 0.21189] |
| 32,768 labels, both fits | 64 | 100.00% | 3.59353 | 0.22570 [0.21918, 0.23252] |

At 2,048 labels, the point estimate for increasing additions from 16 to 32 raises capped loss by 0.00340 [-0.00025, 0.00724], even though truth coverage rises. The full 32-addition block replays exactly. The point estimate worsens but its interval spans zero, so the capped-loss difference is unresolved. Squared probability loss improves, with a wholly negative interval for the larger-minus-smaller difference. Coverage and the two forecast scores are distinct outcomes. Full-universe expansion reaches 100% coverage by construction and retains learned decoding error. The matched first-fit label-budget comparisons and squared-loss differences are retained in the evidence.

Forced choice still assigns exact zero to the same 16,372 of 65,536 truths in each candidate packet (24.98169%). The six new packets contain 98,232 stored zero rows, repeating the same omitted truths rather than supplying independent replications. Uncapped loss is infinite for that arm. All other new candidate and correction forecasts have nonzero truth mass. Capped loss and exact zeros are distinct results. The supplied-hypothesis control uses evaluator truth to insert the answer and remains privileged. Finite expansion is not autonomous open-world invention.

## Verification, costs and remaining scope

All fifteen COMPLETE, plan, source/archive, environment, fit, cost and evidence-role bindings pass. The independent analysis reviewer reconstructs every stored forecast score, secondary measure and summary. Separate audits reconstruct candidate ranks, masks and mass transport; check every correction/retraction/irrelevance view; and verify exact control identities. Prior matching summaries and complete references are reused by immutable hashes.

Two full correction blocks replay from frozen source: the 128-label first fit and 131,072-label second fit, with all five updates, three readouts and 32 worlds. Each matches 1,346 scientific files and its full summary. The lower-label 32-addition candidate reference matches all 450 scientific files and its summary. Earlier full opening G1/G4/G5/G6, repaired scorer, full-universe candidate and lower-label/repeated-fit candidate references remain verified by immutable hashes. No scientific producer, criterion, model setting or environment changed. Earlier failures and attempted costs remain retained; no new scientific failure occurred.

The verified total is 453 packets. The queue retains 28.74 measured eligible hours across 108 discovery blocks, so no refill is required at this boundary. Matched native/shifted fits and lower-label acquisition second fits are already admitted. The useful combined forecast misses the 1.25 planning cushion by 0.36 hours at this snapshot; forecasts are not inflated.

All science and overhead share the 90-hour CPU ceiling, with discovery capped at 72 and 18 protected. The twelve frozen confirmation diagnostics remain first priority on October 1 at 08:00 UTC, followed by the already frozen context extension within its existing inclusive cap. No confirmation or challenge world was used here. Validation does not establish fresh-installation or cross-platform reproduction, universal architecture, human intent or real-text process correspondence.

[Validity](../../../results/v20/correction-wave-2/VALIDITY.json), [correction paired fits](../../../results/v20/correction-wave-2/CORRECTION_PAIRED_FITS.json), [candidate paired fits](../../../results/v20/correction-wave-2/PAIRED_FITS.json), [candidate increment](../../../results/v20/correction-wave-2/CANDIDATE_INCREMENT.json), [full correction replays](../../../results/v20/correction-wave-2/REPLAY.json), [candidate replay](../../../results/v20/correction-wave-2/REPLAY_SUPPLEMENT.json), [costs](../../../results/v20/correction-wave-2/COSTS.json), [forecast](../../../results/v20/correction-wave-2/FORECAST.json).
