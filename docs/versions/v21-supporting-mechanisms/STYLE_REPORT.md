# V21 mechanism family versus surface style

Does compatibility follow mechanism or surface style? With the supplied parser, mechanism-matched training gives 82.86% accuracy versus 17.14% for style-only matching. Two inversions also let the raw surface reader succeed when neither matches. Reversing style correlations leaves every forecast unchanged by construction, so this does not establish learned robustness. These are verified constructed-method results, with no confirmation or claim about actual model families.

Eight independently drawn binary-noise worlds each supply two fitted seeds, 512 labels per archive and 256 paired queries. Each reader fits the same four mechanism/style archives and selects one per query. Mechanism flips the target before noise; style flips its rendered token. Mechanism archive selection and the invertible parser are supplied privileges. Archive names are independently permuted; this tests name invariance, not shuffled training targets.

The rows show which training archive matches the query and whether the reader uses the raw token or the supplied parser. Log loss is negative natural-log probability of the true target; full Brier is total squared probability error; lower is better for both. Modal accuracy is the fraction of correct most-probable targets. The final columns show changes in log loss from native style agreement of 90% to independent agreement of 50%, or reversed agreement of 10%. Every other saved metric and every per-case forecast is also unchanged.

| Matching archive | Decoder | Log loss | Full Brier | Accuracy | Independent minus native | Reversed minus native |
|---|---|---:|---:|---:|---:|---:|
| Mechanism and style | Raw surface | 0.44977 | 0.27996 | 82.86% | 0 | 0 |
| Mechanism and style | Supplied parser | 0.44977 | 0.27996 | 82.86% | 0 | 0 |
| Mechanism only | Raw surface | 1.51058 | 1.14007 | 17.14% | 0 | 0 |
| Mechanism only | Supplied parser | 0.44977 | 0.27996 | 82.86% | 0 | 0 |
| Style only | Raw surface | 1.51058 | 1.14007 | 17.14% | 0 | 0 |
| Style only | Supplied parser | 1.51058 | 1.14007 | 17.14% | 0 | 0 |
| Neither | Raw surface | 0.44977 | 0.27996 | 82.86% | 0 | 0 |
| Neither | Supplied parser | 1.51058 | 1.14007 | 17.14% | 0 | 0 |

With the parser, matching the mechanism reduces mean log loss by 1.06081 nats relative to matching style alone (descriptive 95% paired-world interval 0.79627–1.35978). With raw tokens, changing both mechanism and style cancels two binary inversions and exactly matches the both-matched reader. Surface success therefore cannot identify mechanism compatibility. The apparent survival of correlation reversal is an identity of this construction: shared archive draws and supplied selection remove any sensitivity to the style distribution. It provides no learned family-recognition or learned transfer result.

All 196,608 original/replay forecast scores pass the independent score consumer. Complete frozen-source scientific replay matches 498 files byte for byte, including 496 arrays. Independent scalar counting and selection reconstruct all 98,304 original forecasts, 12,288 repeated query records, every training table/count, every summary and arbitrary-name invariance. The maximum probability discrepancy is 2.78e-17. All 65,536 non-native forecasts exactly equal their paired native forecasts. Two fits are averaged before uncertainty across eight worlds; neither fits, cases nor correlation regimes count as additional worlds.

Fit capacity is equal and solved by counts, with no early stopping or convergence comparison. Reliability is estimated through matched labels; mechanism locality and parsing are supplied. The binary target occupies two states in the common 128-state scorer, so unused states and top-eight coverage add no identifying evidence. There is no support-pruning or law-marginal superiority test, historical alias claim, execution-noise shift or feature-product duplicate-removal arm. These contrasts are unavailable, not silently passed. The purpose-built miniature remains architecture untested. Unknown parsers, learned archive selection and actual language-model families remain untested. No new promotion or confirmation follows; the correlation-reversal claim stops at the construction identity.

[Validity](../../../results/v21/style/VALIDITY.json), [all reader scores](../../../results/v21/style/SUMMARY.json), [paired contrasts](../../../results/v21/style/CONTRASTS.json), [world means](../../../results/v21/style/WORLD_ROWS.json), [contrast availability](../../../results/v21/style/CONTRAST_AVAILABILITY.json), [frozen admission](E03_STYLE_ADMISSION.json).
