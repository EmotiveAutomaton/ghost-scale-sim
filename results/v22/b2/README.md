# V22 B2: Acquisition routes and devaluation

Can acquisition route predict a choice after need satisfaction? The two-route reader beats both single-route assumptions on post-devaluation choice loss in every tested condition. Under broken contingency its loss is 0.456 nats, versus 0.926 for the model-based-only reader and 1.212 for association-only. Yet observational aliases keep route probability exactly at one half. This is a constructed-mechanism prediction result, not proof that the historical route is always identifiable.

[Report](../../../docs/versions/v22-mechanism-tests/B2_REPORT.md).

- `PROVENANCE.json`: source/plan/environment and exact independent/full-replay receipt bindings.
- `ANALYSIS.json`: paired-world mechanism estimates, separate targets and readers.
- `SUPPLEMENT.json`: paired high-label metric differences and additional controlled contrasts.
- `STRATA_summary.csv`: all original target/control strata, each reader and both proper scores; rows are aggregates, not cases. Empty numerical fields indicate nonfinite or missing-world estimates, not zeros.
- `HEADROOM.json`: original immutable matched gates for both label budgets.
- `FITS.json`: validation-stop metadata and weight hashes.

Log loss uses natural logarithms; lower is better. Full multiclass Brier is the sum of squared probability errors; lower is better. Accuracy is the modal correct fraction; higher is better. Confident error is an incorrect modal answer with probability at least 0.9; lower is better. Differences name their direction. Bounds are descriptive whole-world 95% percentile intervals, never case-level standard errors. C1 targets are marginal nodes, not a joint score. The compact public view omits per-world arrays; exact retained aggregate hashes are in `SUMMARY.json`.
