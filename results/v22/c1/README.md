# V22 C1: Joint goal-tree inference

Does joint inference improve recovery across a goal hierarchy? Joint reading improves internal-node probability forecasts over the one-direction ablations, but bottom-up inference already matches it at the root. Removing distant evidence can hurt more than removing immediate-child evidence, and root accuracy is not monotone in every coverage series. These are finite constructed-tree results: marginal node scores do not establish a joint-prediction advantage or mathematical smoothness.

[Report](../../../docs/versions/v22-mechanism-tests/C1_REPORT.md).

- `PROVENANCE.json`: source/plan/environment and exact independent/full-replay receipt bindings.
- `ANALYSIS.json`: paired-world mechanism estimates, separate targets and readers.
- `SUPPLEMENT.json`: paired high-label metric differences and additional controlled contrasts.
- `STRATA_summary.csv`: all original target/control strata, each reader and both proper scores; rows are aggregates, not cases. Empty numerical fields indicate nonfinite or missing-world estimates, not zeros.
- `HEADROOM.json`: original immutable matched gates for both label budgets.
- `FITS.json`: validation-stop metadata and weight hashes.

Log loss uses natural logarithms; lower is better. Full multiclass Brier is the sum of squared probability errors; lower is better. Accuracy is the modal correct fraction; higher is better. Confident error is an incorrect modal answer with probability at least 0.9; lower is better. Differences name their direction. Bounds are descriptive whole-world 95% percentile intervals, never case-level standard errors. C1 targets are marginal nodes, not a joint score. The compact public view omits per-world arrays; exact retained aggregate hashes are in `SUMMARY.json`.
