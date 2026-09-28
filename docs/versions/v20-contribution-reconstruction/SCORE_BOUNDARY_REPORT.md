# V20 learning budgets and saved-score repair

We tested whether more labels or one retained episode gives the structured reader a primary advantage. None of the six new conditions passes: at 32,768 labels without history, its capped log loss is 3.60082, versus 3.52468 for direct frequencies and 3.58216 for the legal template. A saved-probability scoring defect was repaired and fully replayed; earlier findings are unchanged. These are constructed-method results; confirmation remains untouched.

## Comparison

Each condition uses 32 whole discovery lineages, 2,048 test cases per lineage, the same eight training lineages and one fixed fit seed. All seven readers receive the same evidence and supplied legal support. Rows and repeated fits are not independent worlds. Capped log loss uses a probability floor of one trillionth; lower is better. It is not an uncapped proper score. Joint squared probability loss and all declared secondary scores are retained in the aggregate receipt.

Table: rows are training-label budgets and retained past-episode counts. Columns give mean capped joint log loss for the structured account, direct frequency table, joint neural reader, independent-bit neural reader, legal uniform template and supplied-law oracle. The last column applies the frozen primary criterion.

| Labels | Past episodes | Structured | Frequencies | Joint neural | Independent bits | Legal template | Oracle | Primary |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 0 | 7.18410 | 6.08045 | 5.59482 | 3.78849 | 3.58216 | 3.40445 | Failed |
| 512 | 0 | 5.20407 | 6.07084 | 4.05750 | 3.76610 | 3.58216 | 3.40445 | Failed |
| 2048 | 0 | 3.88374 | 4.48706 | 3.65535 | 3.75250 | 3.58216 | 3.40445 | Failed |
| 32768 | 0 | 3.60082 | 3.52468 | 3.58328 | 3.68700 | 3.58216 | 3.40445 | Failed |
| 128 | 1 | 10.51648 | 3.92254 | 5.74223 | 23.60363 | 3.37592 | 3.13541 | Failed |
| 512 | 1 | 6.69183 | 4.50814 | 3.92593 | 23.20341 | 3.37592 | 3.13541 | Failed |

Table: direct-table means direct frequencies; direct-mlp means the joint neural reader; template means legal uniform prediction. Improvement is rival loss minus structured loss, so positive favors structure. Intervals are descriptive 95% intervals from 4,096 whole-world bootstrap resamples.

| Labels | Past episodes | Rival | Improvement | World interval |
|---:|---:|---|---:|---|
| 128 | 0 | direct-table | -1.10365 | [-1.14292, -1.06450] |
| 128 | 0 | direct-mlp | -1.58928 | [-1.62073, -1.55803] |
| 128 | 0 | template | -3.60194 | [-3.65716, -3.54891] |
| 512 | 0 | direct-table | 0.86677 | [0.83450, 0.89863] |
| 512 | 0 | direct-mlp | -1.14658 | [-1.17089, -1.12176] |
| 512 | 0 | template | -1.62191 | [-1.66425, -1.57768] |
| 2048 | 0 | direct-table | 0.60332 | [0.58416, 0.62389] |
| 2048 | 0 | direct-mlp | -0.22839 | [-0.23680, -0.22034] |
| 2048 | 0 | template | -0.30158 | [-0.32083, -0.28261] |
| 32768 | 0 | direct-table | -0.07614 | [-0.08368, -0.06839] |
| 32768 | 0 | direct-mlp | -0.01754 | [-0.02494, -0.00999] |
| 32768 | 0 | template | -0.01866 | [-0.03167, -0.00558] |
| 128 | 1 | direct-table | -6.59394 | [-6.67186, -6.51778] |
| 128 | 1 | direct-mlp | -4.77426 | [-4.84246, -4.70787] |
| 128 | 1 | template | -7.14057 | [-7.23521, -7.04850] |
| 512 | 1 | direct-table | -2.18369 | [-2.23903, -2.12603] |
| 512 | 1 | direct-mlp | -2.76591 | [-2.81782, -2.71078] |
| 512 | 1 | template | -3.31592 | [-3.38824, -3.23981] |

More labels reduce structured loss without producing a primary advantage. At both paired small budgets, one past episode raises structured loss even though the supplied-law oracle and legal template improve. That distinction separates learned decoding from information supplied through legal support. The independent-bit reader with one episode has capped losses 23.60363 and 23.20341; its confident wrong attribution rates are 47.316% and 69.499%. These failures remain outcomes, not numerical rejections.

At 128 labels, 78.25775% of independent-bit truth probabilities lie below the log-loss floor. At 512 labels, 77.55585% of independent-bit truth probabilities lie below the log-loss floor. Exact zero truth mass is zero for every reader in these six conditions. Small positive mass below the cap is therefore separate from candidate omission and exact zero mass. All history contrasts and secondary scores are in [AGGREGATES](../../../results/v20/score-boundary-1/AGGREGATES.json).

## Failure and repair

The original 512-label, one-episode block failed independent checking in the prior reader on lineage 27. Three forecasts reconstructed from the retained inputs were 0.8999999999999999; a second normalization inside the score producer changed them to 0.9 and flipped the fixed confident-error indicator. The checker correctly rejected disagreement with saved probabilities. Scoring now validates and directly uses the saved forecast. The 90% threshold, models, training settings, designs and criteria are unchanged.

The failed namespace, partial data, source and 6.21875 native CPU seconds remain retained. Its repair has a new identity. All 43,008 previously written forecast rows are identical in the repaired block. A targeted impact scan over the twelve completed packets found zero discrete-score changes and at most 3.56e-15 continuous-score drift; their earlier interpretation and replay evidence remain valid.

All 50 V20 controls pass, including values immediately below, at and above the threshold. The repaired complete scientific block reproduces all 548 scientific files and its summary from frozen source. Five earlier completed packets and the repaired packet have 2,752,512 newly verified forecast rows. Plan/source/environment bindings, costs, fit metadata, failures and reader/evaluator separation verify. Future consumers use an admitted source-only repair; 318 original failed or unexecuted identities remain preserved. The immutable forest and twelve untouched confirmation designs are unchanged.

## Continuation and limits

The queue retains 305 eligible discovery blocks, twelve held diagnostics and 33.77 measured forecast hours including review. No extra designs or seeds are needed to maintain a day of useful admitted work. Source repair admission uses the existing allocation; all attempts and replay are charged, with the 18-hour reserve protected. The common repair was completed within its two-hour active allowance.

Execution, numerical acceptance, failed primary criteria and untouched confirmation are distinct. The miniature and fixed optimizer do not establish general architectural behavior. Optimization remains a possible explanation for independent-bit failure; no new hyperparameters were searched. Validation covers this host and installed environment, not fresh installation or cross-platform replay. No simulator result establishes human intent.

[Validity](../../../results/v20/score-boundary-1/VALIDITY.json), [failure manifest](../../../results/v20/score-boundary-1/FAILURE_MANIFEST.json), [source repair](../../../results/v20/score-boundary-1/REPAIR.json), [costs](../../../results/v20/score-boundary-1/COSTS.json), [forecast](../../../results/v20/score-boundary-1/FORECAST.json).
