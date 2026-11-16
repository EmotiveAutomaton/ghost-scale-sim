# Optional secondary diary estimator

Can the diary estimator be made more precise without claiming that its observation model is exact? At the ES2-shaped, low-event-rate setting, the random-slope moment estimator has 35.6%–49.3% less between-world spread than the split-day estimator in five planted populations. This meets the packet’s optional-recipe trigger; bias remains and the method is offered only as a secondary sensitivity analysis.

All rows use 2,272 people, ten days, six prompts per day and mean event rate 0.08. Spread is the standard deviation across 200 independent simulated worlds, not a standard error. Bias is mean estimate minus the planted reactivity correlation.

| Population | Split-day spread | Moment spread | Spread ratio | Moment bias |
|---|---:|---:|---:|---:|
| TWO80 | 0.0260 | 0.0152 | 0.584 | -0.0175 |
| TWO50 | 0.0471 | 0.0239 | 0.507 | +0.0230 |
| TWO0 | 0.0482 | 0.0285 | 0.591 | +0.0944 |
| RIVAL30 | 0.0200 | 0.0129 | 0.644 | -0.0228 |
| RIVAL60 | 0.0378 | 0.0201 | 0.532 | +0.0114 |

Recipe: fit a separate intercept and event slope for each feeling family within each person. Form the between-person covariance matrix of the full-record slopes. For each person estimate the two slope variances and their covariance from the paired regression residuals, divided by residual degrees of freedom and the centred event sum of squares. Subtract the mean sampling-error matrix from the observed slope covariance matrix, then normalize its off-diagonal by the square root of its corrected diagonal product. If either corrected variance is nonpositive, report unavailable. Do not clip the correlation. Use person-level deletion groups for the within-dataset interval.

This is method-of-moments correction for correlated random slopes. The rating floor violates its linear residual model, so lower spread does not establish lower total error or a correct floor-aware latent correlation. Use beside the registered primary, never as a retrospective replacement. Keep event-poor people and availability explicit; real missingness and event reporting are not calibrated here.

The five cells are already included in the 216-cell map, with full seed replay and independent regression reconstruction. No new draw or registered claim. Miniature architecture untested. Route: Sounding Line ES2, optional secondary estimator.

[Trigger and exact source bindings](../../results/v31/D_OPTIONAL_RECIPE_TRIGGER.json).
