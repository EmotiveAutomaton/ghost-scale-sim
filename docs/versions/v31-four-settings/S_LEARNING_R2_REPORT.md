# S.b numerical repair: verified and admitted

Can the independent learning calculation preserve very small posterior weights
without changing the Bayesian model? The owner-approved second repair passes:
all 22 controls, reconstruction of the retained failed fixture, full seed replay
and independent reconstruction agree at the unchanged tolerances. Its production
outputs are byte-identical to the valid original pass from R1.

The original preparation and R1 both failed because converting tiny posterior
weights to ordinary probabilities lost finite support. Windows `longdouble`
did not extend the available exponent range. The new independent calculation
retains log weights and normalizes with scalar arithmetic and compensated
summation. Both earlier failures remain retained and charged.

The two failed preparations contain the same eight-world fixture, verified by
raw-data and summary hashes; it was reconstructed once and attributed to both.
The complete new eight-world, 400-trial cost profile, replay and reconstruction
took 9.59375 CPU-seconds. The whole validation job used 18.4375 CPU-seconds,
charged to healing. It occupied one slot in the existing Gear 2 controller.

All 64 original S.b settings are admitted, each with forty excluded pilots,
200 scientific worlds in eight blocks, full seed replay and independent
reconstruction. Their measured reservation is 518.0625 CPU-seconds per cell
within the unchanged S allocation. Admission is not completed science; no
learning-mechanism result follows from these numerical checks. The COST-learning
follow-on still requires its declared trigger and separate admission.

Shared priors, grid and model assumptions remain limitations of this check;
numerical agreement does not establish human learning or an engine reader.
No further automatic S repair is authorized.

[Frozen protocol](protocols/S_LEARNING_R2.md),
[owner authorization](../../../results/v31/S_LEARNING_R2_AUTHORIZATION.json),
[validation and admission evidence](../../../results/v31/S_LEARNING_R2_VALIDATION.json).
