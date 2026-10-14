# V22 A1: Source inference and fixed trust

Does inferring a source explain uptake better than one fixed trust? In this constructed mechanism, source inference improves native property log loss by 0.102, 0.115 and 0.249 nats against trusts 0.2, 0.54 and 0.9. Steering detection rises from 13.4% without records to 100% with 32 records. The proposed competence ordering of discrepancy-curve peaks fails: with long records, competent informers peak earlier than weak informers on the tested grid. No universal trust crossover or broad architecture claim follows.

[Report](../../../docs/versions/v22-mechanism-tests/A1_REPORT.md).

- `PROVENANCE.json`: source/plan/environment and exact independent/full-replay receipt bindings.
- `ANALYSIS.json`: paired-world mechanism estimates, separate targets and readers.
- `SUPPLEMENT.json`: paired high-label metric differences and additional controlled contrasts.
- `STRATA_summary.csv`: all original target/control strata, each reader and both proper scores; rows are aggregates, not cases. Empty numerical fields indicate nonfinite or missing-world estimates, not zeros.
- `HEADROOM.json`: original immutable matched gates for both label budgets.
- `FITS.json`: validation-stop metadata and weight hashes.

Log loss uses natural logarithms; lower is better. Full multiclass Brier is the sum of squared probability errors; lower is better. Accuracy is the modal correct fraction; higher is better. Confident error is an incorrect modal answer with probability at least 0.9; lower is better. Differences name their direction. Bounds are descriptive whole-world 95% percentile intervals, never case-level standard errors. C1 targets are marginal nodes, not a joint score. The compact public view omits per-world arrays; exact retained aggregate hashes are in `SUMMARY.json`.
