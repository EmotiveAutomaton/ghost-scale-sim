# V18.3 D: Bounded explanation revision

When does revising an explanation improve prediction? When the supplied decision rule was wrong, selecting a replacement after eight observations reduced future logarithmic loss from 3.582 to 0.623; selecting at the initial cue instead gave 2.948. But when the original family was correct, late revision worsened loss from 0.613 to 0.692. These descriptive constructed-method results show a benefit from evidence-informed revision and a cost from unnecessary or premature commitment. Averaging over the candidate families was a strong rival; no unique benefit of committing to one explanation, general open-world repair, or human mechanism is established.

The table reports future logarithmic loss after the same twelve observations. Lower is better. Early revision commits at the initial cue; late revision commits after the first eight observations; both then update states using all twelve. Mixture inference keeps all five candidates.

| True mechanism relative to supplied family | Fixed family | Early revision | Late revision | Candidate mixture |
|---|---:|---:|---:|---:|
| in-family | 0.61338 | 1.31088 | 0.69158 | 0.62797 |
| near-family | 1.01893 | 1.91485 | 1.03314 | 0.97422 |
| missing-rule | 3.58161 | 2.94806 | 0.62347 | 0.62348 |
| missing-acquisition | 0.60999 | 1.51030 | 0.64583 | 0.62788 |
| missing-opportunity | 1.63957 | 1.92454 | 0.62784 | 0.61849 |
| outside-menu | 2.75313 | 2.83774 | 0.71414 | 0.70042 |

Six packets retain 3,840 evaluations crossing two cue orders, sixteen declared factor cells and twenty coefficient draws. Evidence and future outcomes match across order; order-invariant fixed and mixture readers are controls. Source-sharing is inactive in this family: its records are independent, so those sixteen slots contain eight distinct mechanism settings. Averaging redundant slots does not increase the twenty independent coefficient draws. Missing acquisition can be observationally equivalent after relabeling the latent repertoire; a different hidden curriculum alone need not make behavior outside the supplied family.

Candidate choices use prefix evidence, not future outcomes. The fixed catalog includes misleading additions and a truth-outside-menu law; good prediction outside that menu does not prove recovery of the true law. The cautious likelihood, empirical predictor, uniform-abstention rule and cue-only commitment remain explicit. The candidate evaluation count is a logical work measure, not complete method CPU. All retained forecast checks and aggregate means passed, with 48 fixed source-extracted whole-unit replays.

[Complete current record](../versions/v18-selective-acquisition/research-extension/RESULTS.md). V18.3 remains active.
