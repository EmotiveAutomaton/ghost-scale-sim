# Joint local goal and operation readout

Does a learned bank help recover a complete three-step goal-and-operation account
at matched evidence and target labels? This successor addresses the historical
joint target omitted by the earlier marginal readout. It preserves the exact
three-unit world and all four evidence tiers. It is a new exploratory comparison,
not a retrospective replacement of the original G-P1 screen.

Freeze 32 original training lineages 193000–193031, the 16 original development
lineages 190000–190015, two draws 190201/190202, and five feature seeds
190101–190105. Each draw samples 64 full paths per training lineage under its
native mass and interleaves lineages to make nested budgets 32/128/512/2048.
Evaluate the complete native development population grouped by identical public
packets. Test and confirmation lineages remain untouched. Random-feature seeds
are not five independent sequence-model training initializations.

Every learner receives identical training artifact evidence, complete local
goal/operation labels at its current budget, and a fixed shared auxiliary pool
of 2,048 old observable-outcome distribution labels. The latter are teacher
information computed from training makers; this cost is explicit, and no such
label or maker identity enters an evaluation feature. Old-outcome labels train
the existing 128-tanh-feature ridge head, penalty 0.01 with unpenalized intercept.
Its probability-repaired four eight-outcome predictions form the learned bank.
The frozen latent state is the projection of those 129 features onto the old
head's left singular vectors. Raw evidence is the existing public serialization.
All three target readers have 128 fixed tanh features plus an intercept, from the
same paired seed offset 17, padded input width 512. Feature geometry differs.
Auxiliary acquisition/fitting, target fitting and query costs are recorded.

Target labels are complete goal-sequence/operation-sequence pairs. The public
syntactic universe has 3^3 times 6^3 = 5,832 tuples; it does not assert that every
tuple is executable or supported by a packet. Each budget's fitted alphabet uses
only labels in that training prefix. A reserved unknown mass 1/(budget+1) is
distributed uniformly over the remaining syntactic tuples. This preserves one
fixed, fully normalized outcome space for proper joint loss and learning curves;
an unknown bin alone would change the scored question as the alphabet grows.
Unknown probability is reported explicitly and may assign mass to impossible
processes. No evaluation truth selects an alphabet, mask, or probability repair.

On the observed alphabet, fit multinomial logistic target heads from zero using
L-BFGS-B, at most 200 iterations/400 function calls, gradient tolerance 1e-7,
function tolerance 1e-12, maximum 30 line-search steps, and ridge penalty 0.01
excluding intercepts. Record iterations, status, objective and maximum gradient;
failure to converge is a limitation, not a hidden pass. Use no tuning or warm
starts. A softmax gives valid probabilities; the known alphabet receives the
remaining probability after the declared unknown allocation. Same-budget
smoothed empirical frequencies and the full exact posterior are separate
references. The latter has supplied-law privilege. There is no new tiny setting.

Measure expected complete-process logarithmic loss under each exact posterior,
squared probability error over the fixed universe, probability mass compatible
with the evidence, exact-posterior mass of the predicted 90% candidate set,
top-process incompatibility, and expected correctly localized goals and operations
at each of the three positions. Abstain from a single process assertion when its
maximum probability is below 0.5. These are separate outputs, never an empathy
score. Naturally weight packets within lineage; average lineages equally. Store
full forecasts, training selections, source identities and references separately
from anonymous reader packets. Prediction scores cannot certify true history.

Primary comparison is the normalized trapezoid area of complete-process loss
over log training budget, paired bank-minus-raw and bank-minus-latent. Report
every budget and evidence tier as well. Bootstrap the 16 development lineages
with 10,000 resamples and seed 190501, averaging feature seeds and
draws inside each lineage. Report draw and feature-seed contrasts separately.
No precise universal training-population uncertainty follows from two draws.
The practical margin remains 0.02 nats; promotion requires independent precision
and validity review, followed by untouched confirmation.

Before admission require known-answer and constant-label head controls, an
independent finite-difference gradient check, fixed-universe probability and
score checks, unknown-label loss, training-only alphabet, hidden-field rejection,
and a complete isolated native fixture. Freeze source, schedule, environment,
protocol and input manifest. Charge six CPU hours initially to this G19-05
successor within the unchanged shared allocation, including replays and failures.
If incomplete, preserve outputs; do not silently relax the optimizer schedule.

Two independent alternatives remain prepared, not implemented jobs. The D-family
composition crosses the existing presentation-tool rule with an inverted-evidence
repair rule: for a skilled maker repair sets evidence to the complement of its
perceived claim; unskilled repair remains a toggle. Enumerate original, tool-only,
repair-only and combined rules, compare original/singly/jointly supplied families,
and distinguish incompatible evidence from compatible missing processes. Freeze
the 8-lineage roster and controls before dispatch; supplying combinations is not
inventing a cause. The F-family support diagnostic will keep the existing two
training draws and 2,048 paths, compare direct and exact learned propagation by
training query/transition visitation, then use one predeclared compositional holdout
of action sequences with seen primitive transitions. Equal common smoothing and
separate per-arm operation/time accounting distinguish support generalization
from sampling effects. Both need their own complete manifest and admission; neither
requires the joint readout or tiny causal instrument to win.
