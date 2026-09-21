# R3: calibrate the whole purchase opportunity before evaluating missing laws

Does subtracting intrinsic predictive randomness improve the targeting of paid
model expansion when both sensors have the same finite calibration purchase rate?
This implements the calibration stage of the retained [R3 design](../v18-selective-acquisition/exploratory-loop/NEXT_DESIGN_2026-09-20.md).
Evaluation is a separate admission after the immutable threshold artifact and its
independent reconstruction are reviewed. Calibration success is not a scientific
claim that the sensor targets missing laws or controls a future error rate.

Use the existing distinct-law world and four cells 0, 2, 8 and 10, with 128 new
calibration indices 196000–196127, three paired observation orders and both
equal-class and doubled-supplied-law priors: 3,072 stream evaluations. Each has
32 unique observations and supplied-law truth. Reuse the existing deterministic
world/state/observation namespaces with these previously unused indices. Freeze
evaluation indices 197000–197019 in advance: four cells, four truth conditions,
three orders and two priors, giving 1,920 later evaluations. These are distinct
from calibration, fixtures, prior R2 indices and the local-world reserved rosters.
Indices are pairing metadata, never reader inputs. No evaluation stream is
generated in the calibration job.

Both sensors use the initial two supplied candidate laws and their native public
program observations. This is an explicitly privileged supplied-law/program task,
not artifact-only local-goal inference. The third catalog law remains unavailable
to the sensor. Before each observation, form the two-law posterior predictive
distribution over the complete executable-program alphabet. Raw surprise is
minus the logarithm of that observation's probability. Centered surprise subtracts
the entropy of the same pre-observation program distribution. Centering must use
the alphabet actually scored; artifact entropy is not interchangeable here.

At each purchase opportunity before unique observations 9 through 32, take the
mean sensor value of the preceding four observations. This is the inherited
minimum-eight-observation timing. The final observation cannot trigger a purchase
after the stream ends. Calibration uses the maximum over those 24 opportunities.
Within each cell/order/prior stratum and each sensor, sort 128 maxima and choose
the 116th value (one-based). Purchase requires a strict greater-than comparison,
so at most 12/128 streams purchase. Report ties, the attained rate and all maxima.
This is finite empirical whole-stream calibration, not an optional-stopping theorem.

Retain every public stream separately from evaluator truth, all predicted program
probabilities, surprises, entropies, rolling values and maxima. Independently
reconstruct every prefix using scalar batch likelihood products over the two
candidate laws and all persistent states. Validate probability mass, observed
surprise, entropy and maxima, strict threshold ties, source-copy invariance and
future-outcome noninterference. Known informative and no-information fixtures
must pass before dispatch; conflicting source copies are rejected. Full adjacent
and extracted-source replays must match deterministic evidence. No calibration
rate or missing-law advantage is used as an instrument gate.

The complete calibration-plus-evaluation branch retains its three-CPU-hour cap,
including original work, failures, validation and both replays. One CPU worker,
one numerical thread, no additional tiny-model setting. Source snapshots and
thresholds are immutable. Calibration completion alone does not finish R3.

The evaluation successor must bind the reviewed threshold artifact before any
evaluation outcome. Both sensors then drive the same paid-mixture expansion;
retain fixed, select, initial-mixture/no-purchase, expanded-from-start references
and the old raw thresholds 1.5 and 3 as contextual baselines. Report ever-purchase
rates by truth, purchase time, sequential/final logarithmic loss, native action
match, stipulated price sensitivities, likelihood evaluations and CPU. Independently
verify mixture forecasts/actions, behaviorally inert purchases with exact costs,
and corruption rejection before admitting evaluation. Neither equal calibration
rate nor entropy subtraction guarantees equal timing, work or held-out targeting.

Two independent alternatives remain prepared. A2 keeps the original/random/
diversity/training-residual portfolios at equal size and the fixed ridge grid in
the [bounded reset protocol](BOUNDED_RESET_PROTOCOL.md), with no test-law labels
in selection and a five-hour cap. C1 now has capable tiny predictors, but needs
frozen donor/recipient belief and governing-purpose changes, unchanged-factor
controls, native change/stay targets and fixture replay before any alignment fit.
Use the same saved architectures and weights; no third sequence-model setting.
C1 is prepared, not implemented or admitted. Static factor interchange and
ordered dynamic updates remain distinct, and no local goal is assigned afterward.
