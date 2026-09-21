# First learned readout and independent local alternatives

This protocol precedes new fits. G19-04 uses the first eight development lineages,
two paired feature seeds and budgets 32/128. G19-05 uses all sixteen development
lineages, two paired feature seeds, two independent training draws and budgets
32/128/512/2048. The 128 training, 32 test and 32 confirmation lineage namespaces
are the frozen opening roster; test and confirmation remain untouched in these
development screens. No new outcome selects a lineage, budget or feature setting.

## Readout architecture and target access

The first CPU readout is a fixed random-feature ridge model, not the tiny causal
sequence reader. It consumes only allowlisted evidence. There is no Torch fit or
tiny-setting consumption in these two screens. The shared tiny-reader ownership
must be resolved separately before G19-06; these screens do not admit G19-C.

A common 128-coordinate tanh basis, with Gaussian weights scaled by input width
and seed-paired bias, encodes padded public features. A ridge least-squares fit
with penalty 0.01 learns old observable response probabilities from 2,048
training histories per draw. Its full coefficient singular basis supplies an
80-coordinate frozen internal state (zero padding if fewer outputs); its repaired
old forecasts supply the learned bank. No singular component is selected by test
results. The artifact-history, learned-bank and frozen-state inputs each receive
the same 128-coordinate tanh readout basis and 129 fitted coefficients per target
coordinate. Inputs are padded to 512, with explicit masks and no silent truncation.
The exact-bank input is a separately labelled privileged ceiling condition.

All arms receive identical old-outcome and new-target training records. The raw
arm retains a separately fitted old-task auxiliary head on its shared fixed
history basis, but its new head uses the original history features. This matches
supervision availability and fitted target-head capacity, not total representation
capacity or optimization geometry. The latent basis is learned through the old
head; no claim of a task-trained transformer or recurrent hidden state is made.
Report raw input width, basis dimension, fitted parameters and old fit cost.
These limitations constrain G-P1 interpretation and are not grounds to silently
declare the primary complete.

Fit target heads once per nested budget with the same ridge penalty; no tuning
grid, early stopping or selected checkpoint. Probability output uses nonnegative
clipping, addition of 1e-6 to each category and normalization within each query.
Record invalid uncorrected rows. Preserve old and target coefficients, random
weights, input schemas, training/evaluator arrays and forecasts. Exact replay uses
the same frozen sources; wall-clock timings are separate from scientific hashes.

Old-world inputs contain eight artifact observations, their public query contexts
and source-duplication indicators. They omit executed programs, rule labels, policy
matrices, hidden state and unknown numerical maker-policy parameters. The exact
artifact-history posterior uses the uniform 24-state prior; the generated state
is sampled from that same prior. All sixteen old cells occur at equal weight.
Old targets are the five old-query artifact distributions; new targets are the
two farther-query artifact distributions. These generator-distribution teacher
labels are available only on training rows. This is newly sampled artifact evidence,
not a reanalysis of privileged program histories.

Local inputs are the unchanged E0, E1, sparse E2 and full E2 projections. All arms
use the same trajectories per evidence condition. Old observable targets are
final-artifact distributions from four independent fresh episodes with the same
persistent maker under the four admitted contexts. The training teacher supplies
those expected distributions. New targets are the three actually selected local
goals and the three actually executed operations, as separate categorical heads.
All target arms receive the same evaluator-labelled training examples. Goal
labels are privileged supervision, not observed reader inputs. Test labels and
exact posterior answers remain evaluator-only. Forecast accuracy is separate from
joint process correspondence; D2 supplies a distinct jointness diagnostic.

Generate 16 histories per training lineage (2,048 total), using a fixed shuffled
nested order per draw. Evaluate 32 histories per development lineage; old cells
are balanced by index modulo sixteen. Conditional exact targets, no-history
forecasts and a shuffled-training-target control accompany every representation.
Score each category query, each evidence condition and each old class separately,
plus equal-class averages and normalized area over log budget. Pair lineage means;
fit-seed and training-draw variation are separate descriptive records. These
development estimates are not confirmation or a precise universal training interval.

## D1 frame and D2 jointness: independent executable alternatives

Both leaves use the first eight local development lineages and the two frozen
initial sampling seeds. They require no trained reader. Thirty-two trajectories
per lineage/seed are sampled from the complete weighted support before scoring.
The complete enumeration is retained, source-bound and unchanged.

D1 compares the same E1 evidence under a neutral and a self-like purpose prior
(3:1 prior weight for functional purpose). Literal rearrangement, equal-length
reread and an irrelevant frame are exact identity controls. A supplied relation
asserts that the first and last local goal agree or disagree; a truthful and a
wrong version are classified against the sampled evaluator record, and correction
retracts the wrong assertion. These relation cues contain new teacher information;
their gains cannot count as reorganization of identical evidence. Record atomic
content, entailment, selection access, goal/process correspondence, contradictions,
coverage and unknown detection. An outside-family artifact tests abstention.

D2 compares exact complete-hypothesis mixtures, a single best account and the
product of goal-sequence and operation-sequence marginals under E0/E1/sparse/full
E2. Score true joint log loss, marginal loss and mass assigned outside the supplied
compatible support. Construct a deterministic two-by-two reachable hypothesis
rectangle when one exists: diagonal and off-diagonal mixtures have identical
goal and process marginals but opposite association. Their disagreement concerns
a joint question; marginal prediction cannot resolve it. Do not mislabel a
crossed but executable hypothesis as physically impossible. Retain both the
constructed matched-marginal fixture and natural-support compatibility results.
No assertion here extends the world to simultaneous aims or multiple contributors.

D1 and D2 each retain a four-CPU-hour ceiling. Readout scout retains four hours;
local primary screen retains six. All verification and failed attempts are charged
within the accepted exploration ceiling, leaving sixteen CPU hours protected.
The independent leaves do not depend on a positive learned result.
