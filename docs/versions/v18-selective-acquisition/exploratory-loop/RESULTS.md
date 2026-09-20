# V18.4 results

Commissioned adaptive exploration, still active. A queue drain triggers review and
refill; it is not campaign completion. Earlier V18.3 results remain unchanged.

## P1: prospective purpose portfolios

Can a memory prepare for new questions without choosing its code using their answers? In this finite comparison, training for a declared portfolio of purposes improves transfer. With four memory symbols under the core rules, mean logarithmic prediction loss on held-out questions falls from 0.92126 for the canonical old-task optimum to 0.88764 for the portfolio code, while old-task loss rises from 1.14444 to 1.14757. Merely choosing the highest-entropy old optimum is not a general repair: under the lexicographic rule it worsens future loss from 0.30243 to 0.33243, whereas the portfolio reaches 0.25098. These are descriptive constructed-method results with supplied-law decoders and additional declared training-purpose access, not evidence of task-independent or learned memory.

The packet crosses 20 fresh coefficient draws, 16 fixed architecture cells, two
rule boards and three deployment-frequency tilts: 1,920 assigned evaluations.
Repeated tilts, cells and selector comparisons are not independent draws. Each
enumerates all 4,140 deterministic partitions of eight histories at cardinalities
1/2/4/8; the selectors receive no held-out question loss. Logarithmic loss is in
nats and lower is better. Maximum symbol capacity and actual entropy are distinct.

The table reports mean future loss at unchanged deployment frequencies. Each row
is a predeclared selector; columns distinguish maximum stored symbols and rule board.

| Selector | Core, 2 symbols | Core, 4 symbols | Lexicographic, 2 symbols | Lexicographic, 4 symbols |
|---|---:|---:|---:|---:|
| Canonical old optimum | 0.93042 | 0.92126 | 0.37242 | 0.30243 |
| Random old optimum | 0.93041 | 0.91905 | 0.37242 | 0.31419 |
| Highest-entropy old optimum | 0.92610 | 0.91488 | 0.37242 | 0.33243 |
| Observation-bit product | 0.92602 | 0.91758 | 0.37310 | 0.26906 |
| Mean training-purpose portfolio | 0.92575 | 0.88764 | 0.37242 | 0.25098 |
| Worst training-purpose regret | 0.91448 | 0.88780 | 0.33371 | 0.25098 |

The stronger two-symbol minimax transfer also pays an old-task price: core old
loss 1.22428 versus 1.15063 for the canonical optimum; lexicographic 0.80187 versus
0.58987. A favorable future score does not erase that tradeoff. Full cell/tilt
results and all selected codes are retained in the
[scientific bundle](../../../../results/v18/exploratory-loop/P-prospective-1/SCIENTIFIC_MANIFEST.json).

Validation: known-answer uniform ties, one/full-symbol equivalence, future-label
noninterference and corruption controls passed. All 1,920 retained units were
checked; a separate extracted-source pass reconstructed 9,216 means and replayed
eight fixed evenly spaced whole units exactly. This is bounded replay, not a
second independent sample or a claim about arbitrary future purposes.

Next: distinguish a portfolio's aligned extra purpose information from generic
diversity by replacing it with equally sized irrelevant or misleading portfolios;
test decoder learning separately from the present supplied-law reconstruction.

## Neural timing pilot and initial validity

The pilot completed all three eight-epoch width 24 fits. Their measured fitting CPU
times were 0.719 seconds for direct history, 4.125 for flat recurrence and 9.328 for
split recurrence. It is a limited timing/control packet, not a main support result.
All 90 retained means independently reproduce; 15 source-extracted forecast files
replay 64 fixed rows each with maximum absolute difference zero.

The current main slate has six count-matched support/query regimes, three models
and three seeds: 54 fits, width 48, 64 epochs, 512 training histories/cell, 128 development
histories/cell and 96 test histories/cell. Initial timing projects roughly 5.9–9.1
CPU hours for fitting after an estimated width adjustment; data, development and
verification add work. This forecast will be replaced with full-packet timings.
Pre-execution coverage and checkpoint-storage amendments retain original plans.
Twelve new Ghost controls and one actual PyTorch exact-resume control pass.

Raw condition names retain the predecessor's even-support convention: `in-support`
means even combinations/old questions, `new-combinations` means odd/old,
`new-queries` means even/composed questions and `both-new` means odd/composed.
These labels do not establish novelty in the new arms. For odd-support training,
the combination novelty reverses; full-support training covers both. Diverse
supervision covers the composed questions. Only the separate far-query family is
absent from every training/development question menu. Main reports must use those
actual access relationships rather than quote the legacy names as conclusions.

## U1: uncertainty over the changing role, 32-step streams

Averaging possible kinds and rates of change reduces the cost of adaptive memory in this constructed comparison. With independent observations and no change, logarithmic prediction loss is 0.704 for the mixture, 0.770 for fixed fast-role updating and 0.695 for static inference. After a goal change, the mixture retains the adaptive benefit: 0.814 versus 1.574 for static inference, with expected artifact-match rates of 76.72% versus 65.86%. It does not solve every change: static inference remains slightly better under skill changes and an unrepresented decision-rule change. These are descriptive constructed-method results, not an acquired-learning advantage or a general solution to model revision.

The completed packet contains 8,640 assigned stream evaluations: nine change
conditions, three copy spans, 20 coefficient draws and 16 fixed cells. The explicit
copy-span manipulation overrides the world's nominal sharing flag, so that cell
factor is inactive here; duplicate settings are not independent architectures.
All methods receive the same evidence trace. Forecasts precede the current
observation; repeated sources add no likelihood but time still advances state.

The table reports mean logarithmic prediction loss in nats for independent
observations; lower is better. Rows name the planted change. Columns compare a
static reader, fixed fast-role updating and the role/rate mixture.

| Planted change | Static | Fixed fast-role update | Role/rate mixture |
|---|---:|---:|---:|
| None | 0.69496 | 0.77018 | 0.70390 |
| Goal | 1.57443 | 0.81458 | 0.81398 |
| Skill | 0.71968 | 0.79609 | 0.72389 |
| Decision rule outside the supplied family | 2.06500 | 2.13490 | 2.07376 |

Across all three copy spans, the mixture's mean prediction loss beats static
inference after goal, belief, joint, gradual, returning-goal and opportunity
changes, but loses under stationarity, skill change and decision-rule change.
Better prediction need not change the constructed artifact: under independent
opportunity-change streams, expected matching is 63.82% for the mixture versus
63.86% for static inference. Every method's forecasts, native programs and
observed/expected matches are retained, including stronger fixed-role variants,
half-unit abstention and the separately privileged known-state forecast.

Independent verification reconstructs 15,120 means and exactly replays eight
fixed whole streams from extracted source. The two-state filter has an independently
enumerated known answer; uniform-likelihood updates, source identity, scalar scores
and native execution checks pass. The 96-step follow-on is reported separately below.

Next: the live boundary is a changing law or opportunity process rather than only
a changing role. Compare uncertainty over rule families, paid family revision and
task-relevant source audits, keeping predictive loss separate from action value.


## U96: longer change-adaptation streams

Does uncertainty over changing roles remain useful in longer streams? With independent observations in the 96-step comparison, the mixture improves prediction after skill changes: logarithmic loss is 0.67629 versus 0.68863 for static inference, reversing the shorter packet's ordering. After goal changes, its loss is 0.71549 versus 1.62893 for static inference and 0.75456 for fixed fast-role updating; expected artifact matching is 79.56%, 67.11% and 79.48%, respectively. The skill benefit disappears with copied observations. A small stationary cost remains, and the mixture still does not repair an omitted decision rule. These are descriptive constructed-method results; the horizons use separate generated traces, so their difference is not a paired causal estimate of extra observation time.

The second adaptation packet contains 8,640 assigned 96-step streams over the same
nine conditions, three copy spans, 20 coefficient draws and 16 nominal cells.
The source-sharing cell flag remains inactive because explicit copy span replaces
it. Horizon enters the trace seed, and change time scales with horizon; the two
horizon packets are not nested observations of one event. Their within-packet
method contrasts share traces, but the cross-horizon contrast also changes the
generated trace and change schedule. No confidence interval or independent-world
sample-size claim is made from the repeated cells.

The table shows mean logarithmic prediction loss in nats for independent sources;
lower is better. Each row names the planted change, and each column a reader.

| Planted change | Static | Fixed fast-role update | Role/rate mixture |
|---|---:|---:|---:|
| None | 0.65235 | 0.73115 | 0.65631 |
| Goal | 1.62893 | 0.75456 | 0.71549 |
| Skill | 0.68863 | 0.76660 | 0.67629 |
| Decision rule outside the supplied family | 2.01365 | 2.09008 | 2.01981 |

All raw forecasts, native programs and outcomes are retained. Independent
source streams carry the skill benefit: with copy spans three and six the mixture
instead has excess loss 0.00487 and 0.01393 over static inference. Other beneficial
change conditions retain a prediction advantage over static across copy spans;
stationary and omitted-rule streams retain a penalty. Independent
extracted-source verification reconstructed 15,120 means and exactly replayed
eight fixed whole streams. Export reassembly checked every scientific file hash.
This is bounded replay, not independent retraining or acquired craft.

The skill reversal motivates a matched-prefix/change-time diagnostic before
attributing it specifically to longer evidence. The persisting rule failure
keeps uncertainty over explanatory families and paid model revision high on the
backlog. Existing 32-step results remain intact.


## L1 first regime: combination versus question transfer

Does learned memory fail on new maker combinations or on new questions? With old-question training on half the maker combinations, both recurrent readers retain their advantage on the other combinations: logarithmic prediction loss, where lower is better, is 0.88207 for flat memory and 0.84053 for split memory versus 1.35942 for direct history. On new questions about familiar combinations, the ordering reverses: 2.97557 and 3.04533 versus 1.71226. This isolates a question-transfer failure in the tested regime; it does not yet distinguish missing supervision from a deficient memory or decoder. These are descriptive constructed-method results, miniature — architecture untested.

This is the first main L1 regime: even-combination support, five old questions,
three models and three fitting seeds, width 48, 64 epochs. There are 512 training,
128 development and 96 test histories per fixed cell; the 16 cells are paired
within coefficient-draw lineage. Positive-skill states only are used. Fits and
repeated queries are not independent observations. Development selects the saved
weights; test results did not choose capacities or checkpoints.

The table reports mean logarithmic prediction loss in nats; lower is better.
Rows distinguish actual combination and question exposure for this even/old
training regime. Columns are direct history, one recurrent state, three recurrent
slots, and the separately supplied-law exact reference.

| Test exposure | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|
| Familiar combinations, old questions | 1.36277 | 0.89072 | 0.84726 | 0.67637 |
| Other combinations, old questions | 1.35942 | 0.88207 | 0.84053 | 0.67039 |
| Familiar combinations, new composed questions | 1.71226 | 2.97557 | 3.04533 | 0.64611 |
| Other combinations, new composed questions | 1.69962 | 2.96531 | 3.00888 | 0.64068 |
| Farther composition absent from all training menus | 1.72506 | 3.02999 | 2.92077 | 0.75339 |

The larger trained regime preserves the predecessor's reversal. Missing latent
combinations alone does not explain it: the same old questions transfer across
the held-out combinations. This does not establish that the recurrent states
discarded the needed information. Diverse-question supervision is already queued;
an observable predictive-bank objective and frozen-state decoder comparison remain
distinct next tests. The exact reference and reconstructed intervention summary
use the finite law, so their advantage does not establish learned access.

Validation found no learning-control failures. It reconstructed 180 scalar scores
and 288 target/exact references. Independent extracted-source verification
reconstructed all 90 retained means and replayed 64 fixed forecast rows in each of
45 model-seed/test files (2,880 forecasts), with maximum absolute error zero.
Input, source, raw-data and selected-weight hashes match; export reassembly checks
every portable scientific file. This is bounded forecast replay, not independent
retraining or a random-architecture severity test. Historical V15 C11/M01 failed
instruments remain unchanged.

The science attempt used 2,428.28125 charged CPU seconds (219.859375 native parent
plus 2,208.421875 child) and 2,492.94 wall seconds; queued verification used
4.515625 charged CPU seconds. Individual direct/flat/split fits cost about
38/259/436 CPU seconds. Five remaining matched packets imply approximately 3.4
CPU hours or 3.5 wall hours at this measured rate, with a 25% planning margin;
this is a forecast, not fulfillment of the exploration window.

[Portable scientific evidence](../../../../results/v18/exploratory-loop/L-even-old-1-r4/SCIENTIFIC_MANIFEST.json).


## L1 support swap: the reversal persists

Does swapping the trained maker combinations remove the failure on new questions? It does not. With old-question training on the complementary half, logarithmic prediction loss, where lower is better, is 0.88296 for flat memory and 0.83120 for split memory versus 1.36712 for direct history on unseen combinations. On new questions about trained combinations, the losses reverse to 2.86223 and 2.99240 versus 1.67857. The failure therefore survives this support swap; missing question supervision, the learned state and its decoder remain competing explanations. These are descriptive constructed-method results, miniature — architecture untested.

The second main L1 regime swaps even for odd training support while retaining
512 training, 128 development and 96 test histories per cell, width 48, 64 epochs,
three models and three fitting seeds. The five old training questions are unchanged.
The test worlds and histories are shared with the first regime, so this is a paired
support manipulation, not a new independent replication. The legacy raw condition
names have their exposure meaning reversed for odd-support training.

The table reports mean logarithmic prediction loss in nats; lower is better.
Rows identify actual combination and question exposure. Columns compare direct
history, one recurrent state, three recurrent slots and the supplied-law reference.

| Test exposure | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|
| Trained odd combinations, old questions | 1.36726 | 0.88050 | 0.83518 | 0.67039 |
| Unseen even combinations, old questions | 1.36712 | 0.88296 | 0.83120 | 0.67637 |
| Trained odd combinations, new composed questions | 1.67857 | 2.86223 | 2.99240 | 0.64068 |
| Unseen even combinations, new composed questions | 1.67388 | 2.86232 | 3.02345 | 0.64611 |
| Farther question family, both combination halves | 1.76186 | 2.89984 | 2.93177 | 0.75339 |

Both support halves retain the old-question advantage and the new-question reversal.
This weakens a one-sided missing-combination explanation without choosing between
objective, question supervision and decoder limitations. The queued full-support
and diverse-question regimes address coverage. A frozen-state decoder ladder is
the next distinct diagnostic; additional labels for its readouts must be declared,
and successful decoding does not establish that the original model used that information.

Validation found no learning-control failures. The evaluator reconstructed 180
scalar scores and 288 target/exact references. The independent extracted-source
pass reconstructed all 90 means and replayed 2,880 fixed forecasts with maximum
absolute difference zero. Source, input, raw-data and selected-weight hashes match;
the portable export was reassembled and every scientific file checked. This is
bounded forecast replay, not independent retraining, a random-architecture severity
test or evidence about human mechanisms. V15 C11/M01 failures remain unchanged.

Science cost was 2,351.984375 charged CPU seconds and 2,416.56 wall seconds;
adjacent verification added 4.421875 CPU seconds. The two completed main packets
average 2,390.13 science CPU seconds and 40.91 wall minutes. Four remaining main
packets project about 2.73 wall hours from this completion, plus the admitted
matched-prefix diagnostic; allow a 25% planning margin. This forecast does not
fulfill the minimum exploration window.

[Portable scientific evidence](../../../../results/v18/exploratory-loop/L-odd-old-1-r4/SCIENTIFIC_MANIFEST.json).


## L1 full maker support: new-question failure remains

Does training on every maker combination remove the failure on new questions? It does not. Across the two combination halves, mean logarithmic prediction loss, where lower is better, is 0.88336 for flat memory and 0.83404 for split memory versus 1.35010 for direct history on old questions. On new composed questions, losses reverse to 2.84959 and 2.98549 versus 1.71639. Missing maker support alone therefore cannot explain this failure at the tested training budget; question supervision, objectives and decoding remain unresolved. These are descriptive constructed-method results, miniature — architecture untested.

This third main L1 regime trains on all positive-skill maker combinations while
retaining 512 training, 128 development and 96 test histories per cell, width 48,
64 epochs, three reader kinds and three fit seeds. Total histories and label rows
match the half-support regimes; each individual combination consequently receives
fewer examples. The five training question forms are unchanged. Test worlds and
histories are shared across these regimes: the support comparison is paired,
not another independent replication. Both test combination halves are now trained.

The table reports mean logarithmic prediction loss in nats; lower is better.
Rows identify actual exposure. Columns compare direct history, a flat recurrent
state, three recurrent slots and the supplied-law exact reference. The opening
paragraph averages the equally sized combination halves, not independent studies.

| Test exposure | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|
| Trained even combinations, old questions | 1.35632 | 0.88693 | 0.83583 | 0.67637 |
| Trained odd combinations, old questions | 1.34388 | 0.87979 | 0.83226 | 0.67039 |
| Trained even combinations, new composed questions | 1.72093 | 2.85837 | 2.99784 | 0.64611 |
| Trained odd combinations, new composed questions | 1.71184 | 2.84082 | 2.97315 | 0.64068 |
| Farther question family, both trained halves | 1.80569 | 2.98964 | 3.06669 | 0.75339 |

Full maker support preserves both recurrent old-question gains and new-question
failure. A matched per-combination sample increase could test an interaction with
data density, but missing question supervision and decoder access have more direct
pending comparisons. Three diverse-question regimes and the frozen-state readout
ladder remain queued. A supplied-law predictive-bank diagnostic is the next
distinct access test; it must report inversion conditioning and invalid raw
probabilities rather than let clipping disguise failure.

No learning-control failures were recorded. The evaluator reconstructed 180 scalar
scores and 288 target/exact references. The independent extracted-source verifier
reconstructed all 90 means and replayed 2,880 fixed forecasts with maximum absolute
error zero. Source, input, raw-data and selected-weight hashes match. The portable
export was reassembled and every scientific file checked. This is bounded forecast
replay, not independent retraining, random-architecture severity or human evidence.
The retained V15 C11/M01 instrument failures remain unchanged.

Science used 2,653.3125 charged CPU seconds and 2,725.0692 wall seconds; adjacent
verification added 5.265625 charged CPU seconds. The three completed main regimes
average about 2,477.86 science CPU seconds and 42.41 wall minutes. Three remaining
main regimes project 2.12 wall hours from the 02:25 UTC boundary, with a 25% planning
margin, before the already admitted U2 and L3 diagnostics. These forecasts do not
fulfill the minimum exploration window.

[Portable scientific evidence](../../../../results/v18/exploratory-loop/L-all-old-1-r4/SCIENTIFIC_MANIFEST.json).


## L1 even support with diverse questions: covered repair, farther deterioration

Does adding composed questions to training repair learned-memory transfer? It repairs those now-covered questions, but farther-question prediction worsens. Across the two maker-combination halves, mean logarithmic prediction loss, where lower is better, is 0.85285 for flat memory and 0.83334 for split memory versus 0.87369 for direct history on the added compositions. On the untouched farther questions, losses are 3.82158, 3.45505 and 2.66348, all worse than their matched old-question-training counterparts. Question coverage explains a repair within the trained menu; it does not establish systematic transfer beyond that menu. These are descriptive constructed-method results, miniature — architecture untested.

The first diverse-question regime holds maker support, 512 training histories per
cell, 128 development histories, 96 test histories, width 48, 64 epochs and three
fit seeds fixed against the even-support old-question regime. Five of eight
question forms rotate per history, so the total label count stays fixed while
exposure to each original question falls. The three added forms are covered in
both training and development. Two farther forms remain untouched. Test worlds
and histories are shared: this is a paired supervision intervention, not an
independent replication or proof of a general composition mechanism.

The table gives mean logarithmic prediction loss in nats; lower is better. Rows
describe actual training exposure. Columns compare direct history, flat recurrent
memory, three recurrent slots and the supplied-law exact reference. The opening
paragraph averages the equally sized combination halves.

| Actual test exposure | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|
| Trained combinations, original questions | 1.36652 | 0.90364 | 0.86402 | 0.67637 |
| Untrained combinations, original questions | 1.35963 | 0.89848 | 0.86069 | 0.67039 |
| Trained combinations, now-covered compositions | 0.87305 | 0.85589 | 0.83841 | 0.64611 |
| Untrained combinations, now-covered compositions | 0.87432 | 0.84982 | 0.82827 | 0.64068 |
| Both combination halves, untouched farther questions | 2.66348 | 3.82158 | 3.45505 | 0.75339 |

The old-question-trained counterparts' losses on the now-covered compositions
were 1.70594 direct,
2.97044 flat and
3.02711 split. Their farther-question losses
were 1.72506, 3.02999
and 2.92077. Repair also occurs for untrained maker
combinations, but farther questions worsen for all three architectures. Lower
per-question exposure, finite optimization and decoder extrapolation remain
competing explanations. Complementary/full maker-support diversity packets,
the frozen-state readout ladder and the supplied-law bank are already queued.

No learning-control failures were recorded. Independent extracted-source
verification reconstructed all 90 means and replayed 2,880 fixed forecasts with
zero maximum absolute discrepancy. Source, input, raw-data and selected-weight
hashes match; portable export reassembly verified every exported scientific file.
This is bounded forecast replay, not independent retraining or random-architecture
severity. V15 C11/M01 instrument failures remain unchanged.

Science used 2,877.1875 charged CPU seconds and 2,946.8097 wall seconds;
adjacent verification added 5.21875 charged CPU seconds. This first diversity fit
took 49.11 wall minutes, longer than the 42.41-minute mean of the preceding three
main regimes. The remaining two diversity fits therefore project about 98 minutes
from 03:15 UTC, before U2, L3 and L2a; timing remains a forecast.

[Portable scientific evidence](../../../../results/v18/exploratory-loop/L-even-diverse-1-r4/SCIENTIFIC_MANIFEST.json).


## L1 odd support with diverse questions: the coverage boundary survives the swap

Does the repair from teaching composed questions survive swapping the maker combinations used for training? It does: on the now-covered compositions, mean logarithmic prediction loss, where lower is better, is 0.85764 for flat memory and 0.83488 for split memory versus 0.87301 for direct history. Untouched farther-question losses are 3.61553, 3.45373 and 2.75825, again worse for every reader than with old-question training. The paired support swap strengthens the specific-supervision explanation; it does not establish systematic transfer beyond the taught menu. These are descriptive constructed-method results, miniature — architecture untested.

The complementary maker-support regime matches its old-question counterpart in
512 training histories per cell, 128 development histories, 96 test histories,
width 48, 64 epochs and three fit seeds. Five of eight questions rotate per
history, preserving total labels but reducing exposure per question. The added
compositions are covered in training and development; the two farther forms are
untouched. Shared world draws and test histories make this a paired supervision
intervention and support swap, not independent test-world replication.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows identify actual training exposure; columns compare direct history, flat
recurrent memory, three recurrent slots and the supplied-law exact reference.
The opening paragraph averages the equally sized maker-combination halves.

| Actual test exposure | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|
| Untrained even combinations, original questions | 1.36816 | 0.90644 | 0.85836 | 0.67637 |
| Trained odd combinations, original questions | 1.36839 | 0.89472 | 0.85840 | 0.67039 |
| Untrained even combinations, now-covered compositions | 0.86912 | 0.85767 | 0.83407 | 0.64611 |
| Trained odd combinations, now-covered compositions | 0.87691 | 0.85761 | 0.83569 | 0.64068 |
| Both combination halves, untouched farther questions | 2.75825 | 3.61553 | 3.45373 | 0.75339 |

On these same compositions, old-question-trained losses were
1.67623 direct,
2.86228 flat and
3.00793 split. Farther-question counterparts
were 1.76186, 2.89984
and 2.93177. Both trained and untrained maker
combinations benefit on the newly covered forms. Original-question losses rise
slightly; supervision density, optimization and decoder extrapolation remain
competing explanations. Full-support diversity, frozen-state readouts and finite
predictive-bank access are admitted and still pending.

No learning-control failures occurred. Independent extracted-source verification
reconstructed all 90 means and replayed 2,880 fixed forecasts with zero maximum
absolute discrepancy. Source, input, raw-data and selected-weight hashes match;
portable reassembly checked every exported scientific file. This is bounded
forecast replay, not retraining or random-architecture severity. V15 C11/M01
remain recorded instrument failures.

Science cost 2,591.484375 charged CPU seconds and 2,657.9505615 wall seconds
(44.30 minutes). Adjacent verification cost 4.765625 charged CPU seconds.
The first two diverse fits average 46.71 wall minutes. This updates the final
diversity fit's forecast to about 04:46 UTC, with a 25% margin to about 04:58 UTC;
neither estimate is a completion receipt. Further useful designs remain open.

[Portable scientific evidence](../../../../results/v18/exploratory-loop/L-odd-diverse-1-r4/SCIENTIFIC_MANIFEST.json).


## L1 full support with diverse questions: taught repair does not extend farther

Does teaching composed questions generalize beyond the taught menu when all maker combinations are represented? It does not in this comparison. On the now-covered compositions, mean logarithmic prediction loss, where lower is better, is 0.84787 for flat memory and 0.83017 for split memory versus 0.87602 for direct history. Untouched farther-question losses are 3.73355, 3.37085 and 2.81020, all worse than their paired old-question-trained counterparts. Full maker support preserves the specific-question repair and farther deterioration seen on both support halves. These are descriptive constructed-method results, miniature — architecture untested.

The full-support regime matches its old-question counterpart: 512 training
histories per cell, 128 development histories, 96 test histories, width 48,
64 epochs and three fit seeds. Five of eight questions rotate per history,
preserving total labels while lowering exposure per original question. All maker
combinations and the added compositions occur in training and development; only
the farther forms remain question-family holdouts. The same world draws and test
histories are reused. This is a paired supervision intervention, not independent
test-world replication. Full support has fewer examples per combination than
the half-support regimes at this fixed total sample budget.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows state actual training exposure; columns compare direct history, flat recurrent
memory, three recurrent slots and the supplied-law exact reference. The opening
paragraph averages the equally sized maker-combination halves.

| Actual test exposure | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|
| Trained even combinations, original questions | 1.36571 | 0.91002 | 0.85893 | 0.67637 |
| Trained odd combinations, original questions | 1.34674 | 0.90130 | 0.85340 | 0.67039 |
| Trained even combinations, now-covered compositions | 0.87771 | 0.85058 | 0.83463 | 0.64611 |
| Trained odd combinations, now-covered compositions | 0.87433 | 0.84516 | 0.82571 | 0.64068 |
| Both trained halves, untouched farther questions | 2.81020 | 3.73355 | 3.37085 | 0.75339 |

Old-question-trained composition losses were 1.71639
for direct history, 2.84959 for flat memory and
2.98549 for split memory. Their farther losses were
1.80569, 2.98964 and
3.06669. The specific coverage intervention repairs the
added compositions in all three support regimes. It does not isolate reduced
supervision density, optimization, or the original decoder's extrapolation as the
cause of farther deterioration. The frozen-state readout and supplied-law bank
diagnostics remain admitted to distinguish access from information loss.

No learning-control failures occurred. Independent extracted-source verification
reconstructed all 90 means and replayed 2,880 fixed forecasts with zero maximum
absolute discrepancy. Source, input, raw-data and selected-weight hashes match;
portable reassembly checked every exported scientific file. This is bounded
forecast replay, not retraining or random-architecture severity. V15 C11/M01
remain recorded instrument failures.

Science cost 2,555.90625 charged CPU seconds and 2,622.591923 wall seconds
(43.71 minutes). Adjacent verification cost 4.5625 charged CPU seconds. The three
diversity fits average 45.71 wall minutes. The initial six main learning regimes
are now verified; the adaptive campaign remains open, with further diagnostic
science and replay already queued. Completion does not fulfill the minimum window.

[Portable scientific evidence](../../../../results/v18/exploratory-loop/L-all-diverse-1-r4/SCIENTIFIC_MANIFEST.json).


## U2: matched evidence prefixes separate timing from a guaranteed horizon benefit

Does the skill-update reversal persist when shorter and longer scores use the same trajectory? It does for the later skill switch with independent evidence: mean logarithmic prediction loss, where lower is better, changes from 0.70717 for the role/rate mixture versus 0.70292 for static inference in the first 32 steps to 0.66483 versus 0.68067 over 96 steps. With the earlier switch, the mixture already leads at 32 steps. Copied evidence usually removes the skill benefit, while goal-change benefits and omitted-rule failures remain. Evidence age and switch timing therefore matter within matched trajectories; longer observation alone does not guarantee an adaptive advantage. These are descriptive constructed-method results, miniature — architecture untested.

The packet contains 3,360 assigned 96-step trajectories: twenty fresh coefficient
draws crossed with eight active architecture cells, three source-copy spans and
seven change/time conditions. Stationarity uses one window anchor; goal, skill
and omitted-rule changes occur after 12 or 28 observations. Shared seeds exclude
condition, switch time, copy span and horizon. Each trajectory supplies nested
32/64/96-step scores; readers and prefixes are not independent replications.
The inactive nominal source-sharing bit is omitted. Every other reader, cell,
matching outcome and scoring window remains in the portable raw record.

The table reports differences in mean logarithmic prediction loss in nats.
Negative values favor the role/rate mixture; positive values favor static
inference. Rows cross switch timing with the number of repeated reports per
unique source. Columns score nested prefixes of the same trajectory.

| Skill switch after this many observations | Reports per source | Mixture minus static, first 32 | First 64 | First 96 |
|---|---:|---:|---:|---:|
| 12 | 1 | -0.00369 | -0.00528 | -0.00364 |
| 12 | 3 | +0.01899 | +0.01264 | +0.00950 |
| 12 | 6 | +0.03295 | +0.02000 | +0.01469 |
| 28 | 1 | +0.00426 | -0.01481 | -0.01584 |
| 28 | 3 | +0.02082 | +0.00489 | -0.00056 |
| 28 | 6 | +0.03562 | +0.01791 | +0.01021 |

The later independent-source switch reproduces a within-trajectory sign reversal,
while the earlier switch already favors the mixture at the shortest prefix. With
three reports per source and the later switch, the 96-step advantage is only
0.00055 nats; no significance or practical-superiority conclusion is attached to
it. The other copied-evidence skill comparisons retain a mixture penalty at 96
steps. No extra seeds are used to inflate this descriptive comparison.

Equal post-change windows separate evidence age from the proportion of the prefix
that occurred before a switch. The table gives mean logarithmic prediction loss
with independent sources; rows name the switch time, and paired columns compare
static inference with the mixture in two fixed 16-step windows.

| Skill switch | Static, first 16 after change | Mixture, first 16 | Static, steps 48–63 after change | Mixture, steps 48–63 |
|---|---:|---:|---:|---:|
| After 12 observations | 0.69129 | 0.67417 | 0.63062 | 0.62784 |
| After 28 observations | 0.72556 | 0.68607 | 0.64757 | 0.63036 |

The mixture's early post-skill-change losses are lower under both switch times;
the later switch's first-32-step penalty includes 28 pre-change steps. Thus the
prefix reversal combines changing evidence age with changing pre/post-change
weights. It is not an isolated causal effect of elapsed time. Static inference
also improves after the switch, and the contrasts differ with initial evidence.

At 96 steps, goal-change loss favors the mixture over static inference at both
switch times and every copy span. With independent observations and the later
goal switch, losses are 0.66609 versus 1.39073. Yet 48–63 steps after that change,
static loss is 0.62659 versus mixture 0.63801: a retained cumulative advantage
does not imply permanent local superiority. Stationary and omitted-rule losses
remain worse for the mixture in all copy conditions. At the later independent
rule switch, 96-step loss is 1.99542 versus static 1.98841. The already admitted
family-selection/averaging/expansion experiment addresses that distinct failure.

All recorded instrument gates passed. Independent extracted-source verification
reconstructed 16,464 means and exactly replayed eight fixed whole trajectories;
all 3,360 units had independent scientific score/native-execution checks during
aggregation. Source, plan and raw hashes match; portable reassembly verified every
scientific file. This is bounded replay, not replay of every whole trajectory or
a new confirmatory campaign. Native artifact matching is not learned uptake.
Historical V15 C11/M01 instrument failures remain unchanged.

Science used 584.890625 charged CPU seconds and 603.165643 wall seconds; adjacent
verification added 29.28125 charged CPU seconds and 30.572842 wall seconds.
The combined observed duration was 10.56 minutes. Future tests should separate
pre-change learning from post-change information arrival, or replace the omitted
law family, instead of merely extending already completed horizons.

[Portable scientific evidence](../../../../results/v18/exploratory-loop/U2-matched-prefix-1/SCIENTIFIC_MANIFEST.json).


## L3: Frozen-state readouts: recoverable signal without a demonstrated composition advantage

Can an added decoder recover useful answers from a frozen memory that failed on new questions? The tested readouts recover aligned predictive signal, but they do not establish a memory advantage on taught compositions: at the larger label budget, the question-and-world-only nonlinear rival has lower logarithmic prediction loss than every frozen-memory readout. Untouched farther questions still resist reliable transfer. This is a descriptive constructed-method result, miniature — architecture untested; failure of this bounded decoder family does not prove information loss.

The encoders remain frozen. Fresh equal training labels teach the five original
and three composed questions to linear ridge and fixed-random-feature nonlinear
readouts. The two farther question forms remain absent. The two nested budgets
are 384 and 768 histories per architecture cell; all 160 readout fits, both
parent support regimes, all fit seeds and context-stratified shuffled controls
remain in the evidence. No test result selected a decoder or penalty.

The table reports mean logarithmic prediction loss in nats, lower being better,
at the larger label budget. Rows specify available representation; columns cross
decoder family with taught compositions or untouched farther questions. Frozen
state rows average the two equally sized parent-support regimes and covered
columns average both maker halves. These averages are descriptive paired summaries.

| Representation | Linear, covered | Nonlinear, covered | Linear, farther | Nonlinear, farther |
|---|---:|---:|---:|---:|
| Direct frozen state | 1.76211 | 1.60010 | 2.11660 | 2.19937 |
| Flat frozen state | 1.77644 | 1.66325 | 2.13945 | 2.23210 |
| Split frozen state | 1.78788 | 1.68076 | 2.15315 | 2.26525 |
| Full raw history | 1.91958 | 2.04079 | 2.29169 | 2.16762 |
| Question and world only | 1.95580 | 1.21823 | 2.32466 | 6.96048 |

Aligned frozen-state readouts improve on their shuffled-target versions and on
the linear question-only comparison. The nonlinear question-only result shows
that supplied world and question features can explain much of the taught-task
score without a history. Its farther-query failure also shows why a good covered
score cannot be treated as systematic extrapolation. Doubling label count does
not supply a general decoder-capacity theorem. Linear input dimensions differ;
nonlinear fitted widths match while fixed projections have different sizes.

Exact-reference farther loss is 0.75827. The remaining gap does not identify
whether a better objective, decoder or optimization procedure would close it.
This fresh readout lineage differs from the original training test lineage;
before/after losses from those two lineages are not a paired causal effect.
Neither extra-supervision recovery nor a shuffled-label difference establishes
that the original predictor used the recoverable information.

All 1,005 means and 51,200 fixed forecast rows replay independently with zero
discrepancy. The evaluator independently checks 288 targets/exact forecasts and
2,445 scalar scores. Source, inputs, copied encoders, fitted readouts and raw
hashes match; every portable file reassembled. This is bounded forecast replay,
not full refitting or architecture severity. No learning-control failures occur.
Science used 524.6875 charged CPU seconds and 545.893751 wall seconds;
verification added 51.34375 charged CPU seconds and 58.830537 wall seconds.
Historical V15 C11/M01 instrument failures remain unchanged.

[Portable scientific evidence](../../../../results/v18/exploratory-loop/L3-frozen-decoders-1/SCIENTIFIC_MANIFEST.json).


## L2a: Frozen prediction banks: useful access with unstable raw inversion

Can old-question predictions answer new questions when given the finite world law? They can improve substantially under the fixed truncated bank map: untouched farther-question logarithmic losses are 0.84141 for flat memory and 0.82760 for split memory, versus their original decoders at 2.98964 and 3.06669. But the raw maps frequently violate probability constraints, and truncation trades exact span for stability. This is a descriptive constructed-method result with extra supplied-law access, miniature — architecture untested; it is neither learned law discovery nor evidence that the unrepaired map is a valid probability model.

This paired decoder intervention reuses the verified full-maker-support,
old-question weights and test cases. It performs no training or new label access.
The supplied world law maps the five old-query forecasts to each requested
question. Fixed relative singular-value cutoffs of 1e-10 and 1e-3 define full
and truncated maps; a passive-only bank remains a reduced-coverage comparator.

The table gives mean logarithmic prediction loss in nats, lower being better.
Rows specify the learned reader; columns cross decoder with composed or farther
question families. Composed columns average the two equally sized maker halves.

| Reader | Original, composed | Full bank, composed | Truncated, composed | Original, farther | Full bank, farther | Truncated, farther |
|---|---:|---:|---:|---:|---:|---:|
| Direct history | 1.71639 | 2.32156 | 1.46050 | 1.80569 | 1.43142 | 0.91709 |
| Flat memory | 2.84959 | 2.14881 | 1.40731 | 2.98964 | 1.20910 | 0.84141 |
| Split memory | 2.98549 | 2.04212 | 1.31057 | 3.06669 | 1.20324 | 0.82760 |

The full bank spans the composed and farther target laws to about 1.2e-12 maximum
absolute residual, but its condition number reaches 83,259.54. The maximum map
norm is 642.39 for compositions. Truncation reduces the maximum condition number
to 994.50 while leaving target-span residuals as large as 0.18251 for compositions
and 0.01396 for farther questions. The passive bank has residuals up to 0.65333.
Finite mathematical span is not a guarantee of stable access from learned inputs.

The scored bank forecasts floor entries at 1e-8 and normalize. Across all three
fit seeds, the following diagnostic table retains invalidity and repair, rather
than treating the repaired score as a valid raw probability construction. Invalid
means any negative/out-of-range entry or normalization failure under the frozen
tolerances; repair is mean absolute change summed over the 16 outcomes.

| Reader and map | Invalid composed rows (%) | Mean composed repair | Invalid farther rows (%) | Mean farther repair |
|---|---:|---:|---:|---:|
| direct full | 99.96021 | 2.14324 | 99.94575 | 0.32138 |
| direct truncated | 99.94575 | 0.42010 | 99.91319 | 0.02752 |
| direct passive | 99.61661 | 0.89739 | 98.33984 | 0.59107 |
| flat full | 99.94936 | 1.62777 | 99.97830 | 0.23458 |
| flat truncated | 99.94936 | 0.40865 | 99.97830 | 0.02132 |
| flat passive | 99.69256 | 1.33829 | 98.84983 | 0.91828 |
| split full | 99.99277 | 1.45481 | 100.00000 | 0.21406 |
| split truncated | 99.99277 | 0.31808 | 99.95660 | 0.01633 |
| split passive | 99.79384 | 1.23044 | 98.62196 | 0.82872 |

The truncation benefit does not remove the constraint violations. A supplied-law
simplex-constrained bank fit and an exact-bank reference would separate clipping
effects, approximation bias and forecast error. A prior-only supplied-law rival
is also needed before attributing the entire gain to retained history.

All 225 score means, 75 law-map diagnostic summaries, 11,520 fixed forecast rows
and every raw mapped probability replay independently with zero discrepancy.
The evaluator also checks 288 targets/exact forecasts and 585 scalar scores.
Source, parent inputs, selected weights and portable reassembly verify. This is
paired test reuse with bounded scored-forecast replay, not independent replication,
retraining or unrestricted recursive closure. No learning-control failures occur.
The initial bank plan was superseded before execution to preserve whole-file
batch arithmetic during replay; no failed scientific bank run is hidden.
Science used 165.96875 charged CPU seconds and 170.984152 wall seconds;
verification added 81.921875 charged CPU seconds and 84.835655 wall seconds.
Historical V15 C11/M01 instrument failures remain unchanged.

[Portable scientific evidence](../../../../results/v18/exploratory-loop/L2a-frozen-bank-1-r2/SCIENTIFIC_MANIFEST.json).


## P2: matched target content does not fix transfer

Does useful compression depend on matching supplemental purposes to the right histories, even when their target content is unchanged? In this finite test, the aligned four-symbol portfolio has lower future logarithmic loss than all seven shifted portfolios: 0.92620 under the original rules and 0.23258 under the alternative rule. Several shifts nearly tie it, while others are worse than training only for the original purpose. Target content alone therefore does not determine transfer; its pairing with history and other tasks matters. This is a descriptive constructed-method result with uniform history allocation and supplied-law decoding, miniature — architecture untested.

Twenty fresh coefficient draws cross 16 fixed architecture cells and the
alternative lexicographic rule, giving 640 assigned finite worlds. Every world
enumerates all 4,140 partitions of eight histories. Uniform history allocation
makes all seven cyclic shifts preserve supplemental target content and its
information measures exactly. The original passive purpose stays fixed; each
portfolio shifts both supplemental target vectors together. Future answers never
select a code. The marginal-only control removes history information.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows name the training selector. Columns cross the rule family with future-purpose
or original-purpose loss. Each value averages the architecture cells and twenty
paired coefficient draws; shifts and cardinalities are not independent replications.

| Training selector, four symbols | Original rules: future | Original rules: old | Alternative rule: future | Alternative rule: old |
|---|---:|---:|---:|---:|
| Original purpose only | 0.95617 | 1.19961 | 0.28445 | 0.51167 |
| Aligned portfolio | 0.92620 | 1.20360 | 0.23258 | 0.51191 |
| Marginal-only supplement | 0.95695 | 1.19961 | 0.28894 | 0.51167 |
| Cyclic shift 1 | 0.96401 | 1.20122 | 0.35403 | 0.51209 |
| Cyclic shift 2 | 0.92836 | 1.20817 | 0.23318 | 0.51940 |
| Cyclic shift 3 | 0.96360 | 1.20195 | 0.33931 | 0.58808 |
| Cyclic shift 4 | 0.92685 | 1.20396 | 0.23261 | 0.51202 |
| Cyclic shift 5 | 0.96351 | 1.20364 | 0.35404 | 0.51212 |
| Cyclic shift 6 | 0.92844 | 1.20999 | 0.23341 | 0.52354 |
| Cyclic shift 7 | 0.96402 | 1.20968 | 0.34019 | 0.60220 |

Original rules: the seven shifted future losses average 0.94840, ranging from 0.92685 to 0.96402.
Alternative rule: the seven shifted future losses average 0.29811, ranging from 0.23261 to 0.35404.

The near ties under several shifts are retained without a significance claim.
These rotations can preserve useful structure; they are not all possible
permutations or guaranteed complete misalignment. Their compatibility with the
unchanged passive purpose and future questions can differ. Thus the comparison
does not isolate a task-independent relevance scalar or match partition difficulty.
Compared with original-purpose selection, the aligned portfolio's future gain
also comes with a small original-purpose cost in both rule groups.

The next table exposes other storage budgets. Rows are the maximum number of code
symbols; columns compare aligned selection, original-purpose selection and the
mean of seven shifts, first under original rules and then the alternative rule.
All figures are future logarithmic loss in nats. Full capacity removes the code
selection difference because every history has its own symbol.

| Symbols | Original: aligned | Original: old only | Original: mean shift | Alternative: aligned | Alternative: old only | Alternative: mean shift |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.99122 | 0.99122 | 0.99122 | 0.40333 | 0.40333 | 0.40333 |
| 2 | 0.95329 | 0.96995 | 0.97107 | 0.33636 | 0.35502 | 0.35587 |
| 4 | 0.92620 | 0.95617 | 0.94840 | 0.23258 | 0.28445 | 0.29811 |
| 8 | 0.92100 | 0.92100 | 0.92100 | 0.23249 | 0.23249 | 0.23249 |

Independent verification reconstructs all 8,320 means and exactly replays eight
fixed complete worlds. Aggregation independently scores every selected code in
all 640 worlds; source, plans, raw blocks and portable reassembly verify. The
protocol's known-answer, zero-information, matched-content, noninterference and
corruption controls pass. This is bounded whole-world replay, not independent
new-world confirmation. It is exact finite compression, not neural learning or
native construction. V15 C11/M01 failures remain unchanged.

Science cost 686.890625 charged CPU seconds and 696.372329 wall seconds; replay
added 6.703125 CPU seconds and 7.294509 wall seconds, 11.73 minutes combined.
A useful follow-on would separate content-preserving transformations that retain
the same optimal partitions from those that change them. More random shifts alone
would not resolve that structural explanation. The new matched-exposure training
round addresses the related learning question separately.

[Portable evidence](../../../../results/v18/exploratory-loop/P2-purpose-alignment-1/SCIENTIFIC_MANIFEST.json).


## R1: useful paid expansion, aliased initial candidates

Can paying to add a missing explanation improve prediction without wasting resources when the original explanation is correct? In these finite worlds, paid expansion reduces sequential logarithmic loss from 3.94154 to 1.64570 when the added rule is needed, but buys unnecessarily in 67.08% of correct-rule streams at the lower surprise threshold. The initial two candidates are predictively equivalent under a skill-label swap, so their selection–averaging tie does not test competing explanations. This is a descriptive constructed-method result about supplied catalog access and its costs, miniature — architecture untested.

Twenty fresh coefficient draws cross eight active architecture cells, four truth
conditions, three orders of the same 32 unique observations, and one or three
marked reports per source: 3,840 assigned streams. Orders, copies and methods are
paired, not independent replications. All raw predictions and all nine readers
are retained, including both paid-selection arms omitted from the compact table.

The table reports mean sequential logarithmic prediction loss in nats; lower is
better. Rows name the reader and columns name the actual generating law. Each
entry averages 20 draws, eight cells and three orders for single reports. The
three-report values are identical because correct source identities are supplied.

| Reader | Supplied rule | Coupling alternative | Purchasable rule | Outside catalog |
|---|---:|---:|---:|---:|
| Fixed supplied law | 0.76002 | 0.75400 | 3.94154 | 2.78834 |
| Initial selection | 0.76002 | 0.75400 | 3.94154 | 2.78834 |
| Initial average | 0.76002 | 0.75400 | 3.94154 | 2.78834 |
| Paid average, threshold 1.5 | 0.76003 | 0.75400 | 1.64570 | 1.18207 |
| Paid average, threshold 3 | 0.76002 | 0.75400 | 2.10548 | 1.21515 |
| Expanded selection from start | 0.78142 | 0.77610 | 0.87950 | 0.83875 |
| Expanded average from start | 0.77421 | 0.76823 | 0.79621 | 0.71598 |

The coupling alternative exchanges the acquired repertoires of the two nonzero
skill states. The uniform prior is invariant under that swap, and all public
context likelihoods map by the same permutation. The two original candidates
therefore express one predictive law. A post-result mathematical control checks
the repertoire and full probability-matrix identity across the eight architecture
cells and twelve contexts. Direct comparison of every retained forecast confirms
the equivalence to numerical precision. The instrument computes correctly; the
initial catalog lacks the intended scientific contrast. This limitation does not
erase the paid access comparison with a different decision rule.

With two equivalent labels and one new law, the expanded equal-label prior gives
two-thirds of its mass to the original predictive class. R1 retains that declared
prior; it is not a class-balanced comparison. The next experiment must distinguish
law access, genuine finite-evidence averaging and sensitivity to duplicate labels.

The next table gives mean expected native-artifact match probability and the net
score after each stipulated cost sensitivity. Rows cross actual law with reader.
Low price charges one match unit for access plus 0.00001 per state-likelihood
evaluation; high price charges four plus 0.0001. Total costs are divided by 32
opportunities. Values are expected matching, not observed success or learned uptake.

| Actual law / reader | Match probability | Low-price net | High-price net | Purchase fraction |
|---|---:|---:|---:|---:|
| Supplied rule / Initial average | 0.74863 | 0.74815 | 0.74383 | 0.00000 |
| Supplied rule / Paid average 1.5 | 0.74831 | 0.72671 | 0.65805 | 0.67083 |
| Supplied rule / Paid average 3 | 0.74863 | 0.74336 | 0.72445 | 0.15208 |
| Supplied rule / Expanded average | 0.74512 | 0.71315 | 0.61292 | 1.00000 |
| Purchasable rule / Initial average | 0.23322 | 0.23274 | 0.22842 | 0.00000 |
| Purchasable rule / Paid average 1.5 | 0.61427 | 0.58388 | 0.48844 | 0.95000 |
| Purchasable rule / Paid average 3 | 0.48053 | 0.45741 | 0.38416 | 0.71875 |
| Purchasable rule / Expanded average | 0.73023 | 0.69826 | 0.59803 | 1.00000 |
| Outside catalog / Initial average | 0.55402 | 0.55354 | 0.54922 | 0.00000 |
| Outside catalog / Paid average 1.5 | 0.81337 | 0.79209 | 0.72443 | 0.66042 |
| Outside catalog / Paid average 3 | 0.80814 | 0.79224 | 0.74096 | 0.48958 |
| Outside catalog / Expanded average | 0.86489 | 0.83292 | 0.73269 | 1.00000 |

When the new rule is needed, the lower-threshold reader purchases in 95.00% of
streams; its final loss is 0.65764, versus 3.76415 without expansion and 0.60396
with access from the start. The higher threshold purchases in 71.88% and ends at
1.12660. Correct-rule final losses tie at 0.60432, while unnecessary access reduces
priced matching. Outside-catalog improvements show useful approximation, not
recovery or identification of the absent lexicographic law.

Ordering the same observations changes sequential loss and the causal purchase
trigger. For missing-rule truth, lower-threshold paid-average losses are 1.50206
with original contexts first, 1.93179 with composed contexts first and 1.50325
interleaved. Their purchase fractions are 95.63%, 93.13% and 96.25%. The query
order changes with its associated observation, so this is not an isolated effect
of evidence order on a fixed sequence of scored questions. Static final evidence
is order invariant; adaptive final access need not be. All order cells are public.

All 13,824 aggregate means were independently reconstructed; eight evenly spaced
whole streams replay exactly from extracted source. Every unit also passed scalar
score, native action, cost and batch-posterior reconstruction. Source, plan, raw
hashes and every reassembled portable file match. Ten isolated controls pass,
including the new alias control. Replay is bounded, not fresh-world confirmation.
No favored ranking is a validity gate. V15 C11/M01 remain failed instruments.

Science charged 777.062500 CPU seconds; replay charged 22.187500 including its
waiting parent and child. Combined wall time was 823.848938 seconds (13.73 minutes).
Both failed and superseded earlier work remain retained; no campaign clocks reset.

[Portable evidence](../../../../results/v18/exploratory-loop/R1-paid-family-expansion-1/SCIENTIFIC_MANIFEST.json).


## S1: source information has little priced decision value here

Does paying to investigate copied sources improve truth decisions? Audit-only decision policies buy nothing throughout this finite comparison. Two-step planning sometimes combines audits with evidence: in one calibrated condition its net utility is 0.86359 versus 0.86342 for evidence alone, a small advantage that reverses when audit reliability is overstated. A graph-information policy buys almost two audits yet reaches only 0.75898 net utility there. This is a descriptive constructed-method result: learning source structure and improving a truth decision are different objectives, miniature — architecture untested.

Twenty fresh parameter draws cross two initial shared-error rates, five actual/
assumed audit channels, three decision stakes and two audit prices: 1,200 exact
models. Every six-report string and every terminal path of at most two purchases
is enumerated. Reports, graph conditions and policy paths are not independent
observations. Initial dependence includes four supplied graphs with equal prior
mass; the two-audit menu leaves the alternating and fully independent graphs
aliased. Purchased evidence and audit errors are conditionally independent.

The table shows the condition with no shared initial flip, true and assumed audit
accuracy 95%, unit decision stake, audit price 0.025 and evidence price 0.1.
Rows name inquiry policies; columns give truth logarithmic loss in nats (lower
better), correct-choice probability, net utility after purchase costs, and expected
audit/evidence counts. Values average the twenty parameter draws and all exact
report paths under equal graph mass. A probability score and a decision score
need not favor the same policy.

| Policy | Truth loss | Correct choice | Net utility | Audits | Evidence |
|---|---:|---:|---:|---:|---:|
| Stop | 0.42211 | 0.80752 | 0.80752 | 0.00000 | 0.00000 |
| Assume independence; stop | 0.62975 | 0.80752 | 0.80752 | 0.00000 | 0.00000 |
| Fixed audit then evidence | 0.22416 | 0.91175 | 0.78675 | 1.00000 | 1.00000 |
| Same trace; assume independence | 0.41125 | 0.88227 | 0.75727 | 1.00000 | 1.00000 |
| Evidence only | 0.27669 | 0.91372 | 0.86341 | 0.00000 | 0.50310 |
| Audits only | 0.42211 | 0.80752 | 0.80752 | 0.00000 | 0.00000 |
| One-step decision planning | 0.31169 | 0.89721 | 0.86176 | 0.00000 | 0.35450 |
| Two-step decision planning | 0.27870 | 0.91301 | 0.86359 | 0.01726 | 0.48993 |
| Graph information per price | 0.39764 | 0.81060 | 0.75898 | 1.97832 | 0.02168 |

The two-step benefit is 0.000172 utility units in this slice, not a significance
claim or a broad audit advantage. Its mean audit count is 0.01726; most decisions
use no audit. Its truth loss is slightly worse than evidence-only loss, retaining
the difference between optimizing a binary action and a proper probability score.
Audit-only and one-step decision policies choose no audits in every model. Across
the 60 parameter settings averaged over draws, two-step planning buys any audits
in six; that count includes calibration stress tests, not six positive findings.

The next table fixes no initial shared error, unit stake and the cheaper audit.
Rows change the actual audit accuracy while the reader assumes 95%; columns
compare two-step decision planning with evidence-only planning and the graph
information heuristic. Values are net utility after all purchase costs.

| Actual / assumed audit accuracy | Two-step | Evidence only | Graph information |
|---|---:|---:|---:|
| 0.95 / 0.95 | 0.86359 | 0.86341 | 0.75898 |
| 0.75 / 0.95 | 0.86306 | 0.86341 | 0.75885 |
| 0.50 / 0.95 | 0.86240 | 0.86341 | 0.75869 |

With a 20% initial shared flip, calibrated 95% audits and the same prices/stake,
two-step and evidence-only policies coincide: net utility 0.79350, expected
correct choice 0.88959 and mean evidence count 0.96086, with no audits. The graph
heuristic buys 1.99595 audits and achieves net utility 0.63489. It reduces graph
squared-probability error from 0.60767 at stopping to 0.27093; this statistic sums
four squared differences from the actual graph indicator, with zero best. That
structural improvement does not establish decision value. At high audit prices
and low stakes, the heuristic can spend more than its expected decision reward.

Common traces isolate the effect of the initial dependence model. The table uses
the first table's calibrated condition and the same fixed audit-then-evidence
trace for both readers. Rows identify the actual source graph; columns report
truth logarithmic loss and probability of assigning over 95% to the wrong truth.
Graph-conditional values average twenty draws and exact paths within that graph.

| Actual graph | Dependence-model loss | Naive loss | Dependence false confidence | Naive false confidence |
|---|---:|---:|---:|---:|
| One root | 0.28643 | 0.85127 | 0.02119 | 0.12412 |
| Two contiguous roots | 0.23300 | 0.33055 | 0.00576 | 0.03095 |
| Two alternating roots | 0.22767 | 0.33055 | 0.00935 | 0.03095 |
| Six independent roots | 0.14955 | 0.13264 | 0.00388 | 0.00521 |

Assuming independence hurts most under a single copied root; it has lower loss
on the actual independent graph. Thus the mixture's average protection is not
uniform conditional superiority. All remaining stakes, prices, channels and
graph-conditional scores are retained in the public rollups and raw policy paths.

Independent verification reconstructed 4,320 means and replayed eight complete
models exactly from extracted source. Every unit's likelihood table was checked
by independent latent-root enumeration and every terminal posterior/mass by scalar
path products; budget, coverage, costs and conditional graph scores reconstruct.
Brute-force contingent policies check decision recursion; a separate joint-only
fixture proves it can detect complementary inquiries. Source/input/raw hashes and
portable reassembly match. The scoped combined suite passes 23 controls, including
seven S1 controls. V15 C11/M01 remain failed instruments. This is not a test of
larger policy trees, graph misspecification, shared audit/evidence errors, native
construction, learned uptake, human source trust or general provenance recovery.

Science charged 339.718750 CPU seconds, verification 17.593750 including its
parent and child: 357.312500 total. Combined wall time was 366.156719 seconds,
6.10 minutes. Useful follow-ons should test whether the audit's limited decision
value changes under shared purchased-channel error, graph omission or a third
audit that separates the retained graph alias; repeating this roster is unnecessary.

[Portable evidence](../../../../results/v18/exploratory-loop/S1-paid-noisy-audits-1/SCIENTIFIC_MANIFEST.json).


## L4 original-menu exposure control: history helps only within the old menu

Does learned history help beyond a reader given only the question and public world law? In the first matched-exposure control, recurrent memory improves original-question prediction, but the no-history reader has lower loss on compositions and untouched farther questions. Farther mean logarithmic prediction loss, where lower is better, is 1.61451 without history, 1.67326 for direct history, 2.29268 for flat memory and 2.01584 for split memory. This establishes a history-use boundary in this control; whether restored question exposure changes it awaits the paired training arms. This is a descriptive constructed-method result, miniature — architecture untested.

This first L4 arm uses 480 full-support training histories per architecture cell,
five original questions per history, 384 common diverse development histories,
96 test histories and three fit seeds. Width 48, 64 epochs and all selection
rules were frozen. Every maker combination is familiar. Compositions enter
development selection in every arm; only farther questions remain absent from
both training and selection. The no-history rival zeros all history inputs while
retaining the direct reader's architecture and public query/world information.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows name actual question exposure. Columns compare the four learned readers and
the supplied-law exact posterior reference. Equal-size maker halves are averaged;
queries, fit seeds and architecture cells are paired within coefficient lineage.

| Question exposure | Question/world only | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|---:|
| Original questions, both trained maker halves | 1.49986 | 1.37708 | 1.02764 | 1.19240 | 0.68016 |
| Compositions, development selection only | 1.51787 | 1.58302 | 2.17733 | 1.89814 | 0.64746 |
| Farther forms, neither training nor development | 1.61451 | 1.67326 | 2.29268 | 2.01584 | 0.75365 |

The question/world-only rival beats all history-bearing learned readers on the
compositions and farther questions. On original questions, flat and split memory
beat the no-history rival. Thus no-history superiority is not universal and
recurrent storage does carry usable information for its training menu. The exact
reference retains a large advantage; no necessity or optimality claim follows
for these learned architectures. The two remaining exposure arms are already
admitted. Their unfinished effects are not interpreted here, and this packet
alone does not resolve the exposure-dilution hypothesis.

All learning controls pass. Independent extracted-source verification reconstructed
105 means and replayed 3,840 fixed forecasts across 60 reader/test combinations,
with zero maximum discrepancy. Source, input, selected-weight and retained-file
hashes match; portable reassembly checked every scientific file. This verifies
bounded forecasts, not full retraining or architecture-wide severity. No human
intent conclusion follows. V15 C11/M01 remain failed instruments.

Science charged 3,049.781250 CPU seconds and took 3,140.194064 wall seconds.
Adjacent verification charged 6.062500 CPU seconds and took 8.301326 wall seconds.
The combined charge is 3,055.843750 CPU seconds; the 52.47-minute wall duration
is within the admitted forecast. The adaptive campaign and minimum exploration
window remain open.

[Portable evidence](../../../../results/v18/exploratory-loop/L4-old-exposure-1/SCIENTIFIC_MANIFEST.json).


## L4 restored exposure: covered repair and farther deterioration persist

Does restoring original-question exposure prevent diverse training from harming farther predictions? It does not in this comparison: farther mean logarithmic loss, where lower is better, rises from 1.67326 to 2.70708 for direct history, 2.29268 to 3.55791 for flat memory, and 2.01584 to 3.77260 for split memory. The no-history reader also worsens, from 1.61451 to 3.08139, while now-trained compositions improve. Reduced original-question exposure alone therefore cannot explain the boundary; the matched-total control remains pending. This is a descriptive constructed-method result, miniature — architecture untested.

The paired arms share nested history draws and exactly 480 labels per original
question in each architecture cell. Diverse training restores that exposure by
using 768 histories and 3,840 labels per cell, compared with 480 histories and
2,400 labels in the original-menu control. Every state/length/original-question
cell has ten labels in both arms. The added compositions each receive 480 labels.
Three seeds, width 48 and 64 epochs are fixed. Both arms select using the same
diverse development set and score the same 96 coefficient lineages. Compositions
are absent from original-menu training but present in development for both arms;
only farther forms are absent from both training and selection.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows identify the training allocation and scored question set. Columns compare
four learned readers and the supplied-law exact posterior. Equal-size maker
halves are averaged; queries, seeds and architecture cells are paired within lineage.

| Training arm and question set | Question/world only | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|---:|
| Original menu, 480 histories; original | 1.49986 | 1.37708 | 1.02764 | 1.19240 | 0.68016 |
| Original menu, 480 histories; composed | 1.51787 | 1.58302 | 2.17733 | 1.89814 | 0.64746 |
| Original menu, 480 histories; farther | 1.61451 | 1.67326 | 2.29268 | 2.01584 | 0.75365 |
| Diverse menu, 768 histories; original | 1.49335 | 1.31951 | 0.86544 | 0.82689 | 0.68016 |
| Diverse menu, 768 histories; composed | 0.88942 | 0.85576 | 0.82872 | 0.81183 | 0.64746 |
| Diverse menu, 768 histories; farther | 3.08139 | 2.70708 | 3.55791 | 3.77260 | 0.75365 |

Restored exposure retains covered improvement and farther deterioration in all
four learned readers. In the diverse arm, direct history now beats its no-history
rival on farther forms; both recurrent readers remain worse than that rival.
The no-history deterioration means this failure cannot be attributed solely to
recurrent memory. It remains compatible with fitting the new query distribution,
decoder extrapolation and optimization or selection effects. The two arms change
history count and total updates, so this is not an isolated question-diversity
effect. The queued 768-history original-menu arm supplies the matched-total
contrast. No partial running effects enter this report.

All learning controls pass. Independent extracted-source verification reconstructs
105 means and 3,840 fixed forecasts across 60 reader/test combinations with zero
maximum discrepancy. All retained-file, source, plan and selected-weight hashes
match; portable reassembly checks every scientific file. Verification covers
bounded forecast replay, not full retraining or architecture-wide severity.
No human-intent conclusion follows. V15 C11/M01 remain failed instruments.

Science charged 4,795.796875 CPU seconds and took 4,945.508667 wall seconds;
adjacent verification charged 6.062500 CPU seconds and took 8.294156 wall seconds.
Together they used 4,801.859375 CPU seconds and 82.56 wall minutes, within the
admitted 4,150–5,650 CPU-second science estimate. The campaign remains active.

[Portable evidence](../../../../results/v18/exploratory-loop/L4-diverse-restored-1/SCIENTIFIC_MANIFEST.json).


## L4 matched total: query allocation retains the farther boundary

Does diverse training still harm farther prediction when histories and update counts match? Yes: farther mean logarithmic loss, where lower is better, rises from 1.68038 to 2.70708 for direct history, 2.10836 to 3.55791 for flat memory, and 2.15324 to 3.77260 for split memory. The no-history reader also worsens, from 1.58653 to 3.08139, while taught compositions improve. Together with the matched-original-exposure comparison, this rules out either reduced original-question exposure or total training volume alone as an explanation. Query allocation, decoder extrapolation and the resulting optimization path remain unresolved. This is a descriptive constructed-method result, miniature — architecture untested.

The matched-total pair uses the same 768 histories and 3,840 target rows per cell,
initialization seeds, batch size and 64 epochs. Only question allocation changes:
the original-menu arm has 768 labels per original question, whereas the diverse
arm has 480 per original or composed question. The earlier 480-history arm matches
that original-question count separately. No single comparison holds both original
exposure and total labels fixed. Both 768-history arms select on the same diverse
development set; compositions are exposed to selection even in the old-only arm.
Farther forms remain absent from fitting and selection. The tests share 96
coefficient lineages across sixteen cells, with seeds and queries paired within them.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows identify training allocation and scored question set; columns identify four
learned readers and the supplied-law exact posterior. The original and composition
rows average equally sized maker halves. Farther tests use their retained allocation.

| Training arm and question set | Question/world only | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|---:|
| Original menu, 480 histories; original | 1.49986 | 1.37708 | 1.02764 | 1.19240 | 0.68016 |
| Original menu, 480 histories; composed | 1.51787 | 1.58302 | 2.17733 | 1.89814 | 0.64746 |
| Original menu, 480 histories; farther | 1.61451 | 1.67326 | 2.29268 | 2.01584 | 0.75365 |
| Original menu, 768 histories; original | 1.49299 | 1.33843 | 1.10288 | 1.08275 | 0.68016 |
| Original menu, 768 histories; composed | 1.53564 | 1.60926 | 2.04502 | 2.02509 | 0.64746 |
| Original menu, 768 histories; farther | 1.58653 | 1.68038 | 2.10836 | 2.15324 | 0.75365 |
| Diverse menu, 768 histories; original | 1.49335 | 1.31951 | 0.86544 | 0.82689 | 0.68016 |
| Diverse menu, 768 histories; composed | 0.88942 | 0.85576 | 0.82872 | 0.81183 | 0.64746 |
| Diverse menu, 768 histories; farther | 3.08139 | 2.70708 | 3.55791 | 3.77260 | 0.75365 |

Increasing original-menu training from 480 to 768 histories does not reproduce
the diverse arm's farther deterioration: the no-history and flat readers improve
slightly, while direct and split readers worsen much less. At matched total
histories, diverse training improves taught compositions but substantially raises
farther loss for every reader. This triangulates against two single-factor
explanations; it does not isolate a universal causal effect of diversity or exclude
interactions with optimizer dynamics and development selection. The no-history
reversal again shows that recurrent memory is not required for the failure.

Ten isolated publication controls and all learning controls pass. Extracted-source verification independently rebuilds
105 means and 3,840 fixed forecasts across 60 reader/test combinations with zero
maximum discrepancy. Source, plan, retained input and selected-weight hashes match;
portable reassembly validates every exported scientific file. This is bounded
forecast replay, not full retraining or architecture-wide severity. Fit seeds,
queries and cells are not independent training-dataset replications. V15 C11/M01
remain failed instruments; this result makes no human-intent claim.

Science charged 4,461.421875 CPU seconds and took 4,604.257853 wall seconds;
adjacent replay charged 5.656250 CPU seconds and took 7.262147 wall seconds.
Together they used 4,467.078125 CPU seconds and 76.86 wall minutes. This lies within
the admitted science estimate of 4,150–5,650 CPU seconds. Campaign clocks remain
unchanged. Conditional-target learning, feasible bank decoding and distinct-law
mixtures are separate admitted experiments, not established results here.

[Portable evidence](../../../../results/v18/exploratory-loop/L4-old-total-1/SCIENTIFIC_MANIFEST.json).


## R2: distinct-law averaging and a poorly selective purchase trigger

Does averaging distinct explanations avoid premature commitment, and does surprise identify when to buy a missing law? With equal class priors and alternative-rule truth, averaging lowers sequential logarithmic loss, where lower is better, from 0.89372 to 0.79964 versus selection. The lower surprise threshold buys in only 37.50% of missing-law streams but unnecessarily in 65.00% of correct-law streams. At that threshold, expansion improves missing-law forecasts, yet its costs erase the average native-action benefit at both stated prices. This is a descriptive constructed-method result about finite catalog inference, miniature — architecture untested.

Twenty fresh coefficient draws cross four rule/opportunity cells, four true laws,
three orders of the same 32 unique observations and two class priors: 1,920 assigned
streams. The initial pair is softmax versus satisficing; purchase adds lexicographic
choice. The outside-catalog truth is a colder softmax law. Orders, priors and readers
are paired comparisons, not independent replications. Exact source identities
are supplied; copy invariance remains a control rather than repeated science.

The table reports mean sequential logarithmic prediction loss in nats under equal
class priors, lower being better. Rows identify readers and columns identify the
true law. Entries average four cells, three orders and twenty coefficient draws.
All nine readers, final scores, priors and cell/order means remain in the evidence.

| Reader | Supplied rule | Other initial rule | Purchasable rule | Outside catalog |
|---|---:|---:|---:|---:|
| Fixed supplied | 0.77770 | 3.84827 | 2.79109 | 2.90573 |
| Initial selection | 0.80532 | 0.89372 | 0.87611 | 0.80079 |
| Initial average | 0.80468 | 0.79964 | 0.74896 | 0.71490 |
| Paid average, threshold 1.5 | 0.80479 | 0.79999 | 0.66265 | 0.70937 |
| Paid average, threshold 3 | 0.80468 | 0.79968 | 0.73942 | 0.71413 |
| Expanded selection from start | 0.81513 | 0.90714 | 0.47570 | 0.77051 |
| Expanded average from start | 0.81381 | 0.81007 | 0.31509 | 0.65671 |

The distinct-law comparison removes R1's forced alias tie. Averaging improves
sequential loss over selecting one initial law under the alternative, purchasable
and outside-menu truths. With supplied-rule truth, fixed correct inference is
still better than either. Their final losses can agree after distinct transient
costs; averaging does not guarantee the best native action or recover missing truth.

The next table reports the initial mixture's sequential loss for equal class mass
and for twice the prior mass on the supplied class. Rows name the true law; this
is an intentional prior change, not a relabeling invariant. Splitting one class's
fixed total mass among identical labels is separately checked as an exact control.

| True law | Equal class mass | Doubled supplied-class mass |
|---|---:|---:|
| Supplied rule | 0.80468 | 0.79450 |
| Other initial rule | 0.79964 | 0.81494 |
| Purchasable rule | 0.74896 | 0.74992 |
| Outside catalog | 0.71490 | 0.71696 |

The changed prior helps when the supplied class is correct and harms when the
other initial class is correct. It changes transient predictions without changing
those two final losses here. Missing-law and outside-menu effects remain descriptive
and small; equal priors over labels are not neutral to predictive-class duplication.

For equal-prior missing-law streams, the lower-threshold paid mixture improves
sequential loss from 0.74896 to 0.66265 and final loss from 0.55557 to 0.39845.
It buys in 37.50% of streams. Correct-rule streams trigger purchase in 65.00%,
with unchanged final loss and worse priced action. Raising the threshold buys
in only 4.17% of missing-law streams and 16.25% of correct-rule streams. Raw
surprise therefore fails as a selective missing-law detector in this comparison.
The purchase-rate contrast alone does not establish whether intrinsic randomness,
predictive calibration or the particular omitted law explains the failure.

The next table gives expected native-artifact match probability and net match
after the two stipulated prices, under equal priors and missing-law truth.
Low price charges one match unit for catalog access plus 0.00001 per state-likelihood
evaluation; high price charges four plus 0.0001. Total cost is divided by the 32
opportunities. Rows name readers; columns are expected scores, not observed
success, learned uptake, CPU cost or an empirical economic valuation.

| Reader | Match probability | Low-price net | High-price net | Purchase fraction |
|---|---:|---:|---:|---:|
| Initial average | 0.84759 | 0.84711 | 0.84279 | 0.00000 |
| Paid average 1.5 | 0.85422 | 0.84194 | 0.80165 | 0.37500 |
| Paid average 3 | 0.84937 | 0.84758 | 0.83926 | 0.04167 |
| Expanded average | 0.87798 | 0.84601 | 0.74578 | 1.00000 |

At the lower trigger threshold, the gain in match probability is insufficient
to repay either price. The higher trigger gives a tiny low-price benefit but
loses at high price. These are threshold-specific results, not a claim that every
expansion policy is dominated. Access from the beginning predicts much better
yet also carries costs. Improved approximation outside the catalog does not
identify the true law. Query order changes with evidence order, so order effects
are not a pure evidence-order intervention on fixed scored questions.

All 7,776 means were independently reconstructed, with eight exact whole-stream
replays from extracted source. Every unit passed scalar prefix posterior, score,
native-action and cost reconstruction. All source, plan, raw and portable-file
hashes match. Ten isolated controls pass, including distinct-law arithmetic,
split-mass duplication, copy invariance, causal triggers, corrupted predictions
and exact replay. No desired ranking gates validity. Bounded replay is not fresh
confirmation or architecture-wide severity; V15 C11/M01 remain failed instruments.

Science charged 600.671875 CPU seconds and adjacent replay 10.546875, totaling
611.218750 CPU seconds. Combined wall time was 623.617782 seconds (10.39 minutes).
Earlier failed work and fixed campaign clocks remain unchanged.

[Portable evidence](../../../../results/v18/exploratory-loop/R2-distinct-class-priors-1/SCIENTIFIC_MANIFEST.json).


## L2c: valid state-mixture decoding retains useful learned history

Do learned old-question predictions retain useful history when decoded as valid mixtures of supplied finite states? Yes in this comparison: farther mean logarithmic loss, where lower is better, is 0.81764 for flat memory and 0.83576 for split memory, versus 0.93884 for the no-history bank and 0.93101 for the same-world prior. Both memory readers improve over clipped inversion, which gives 0.91168 and 0.97634. Useful history therefore survives this feasible decoder, but the decoder is given the world law and does not identify unique maker roles. This is a descriptive constructed-method result, miniature — architecture untested.

Reuse 1,536 fixed histories from the verified 480-history original-menu L4 fits:
48 indices from each of sixteen cells and both maker halves, balanced across
8/16/32-step lengths. Each supplies twelve selected learned banks: four readers
and three fits. Concatenate five old-question distributions over sixteen artifacts
into eighty coordinates. Fit nonnegative weights summing to one over the supplied
24 state-specific banks, then apply those weights to composition and farther
answers. This uses supplied law access, with no new training or test-driven
model selection. Seeds and histories are paired reused evidence, not independent
replication of the parent learning experiment.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows identify learned inputs and decoding method. Columns separate composed
questions from farther forms. Both reference rows retain the same public world;
the history posterior additionally conditions on observed history. All fit seeds
are averaged within each retained history before aggregating equally sized cells.

| Reader and decoder | Composed questions | Farther forms |
|---|---:|---:|
| Question/world bank, feasible | 0.87989 | 0.93884 |
| Direct-history bank, feasible | 0.79682 | 0.86033 |
| Flat-memory bank, feasible | 0.75050 | 0.81764 |
| Split-memory bank, feasible | 0.77099 | 0.83576 |
| Question/world bank, clipped inverse | 1.34289 | 0.94636 |
| Direct-history bank, clipped inverse | 1.48912 | 0.92748 |
| Flat-memory bank, clipped inverse | 1.47163 | 0.91168 |
| Split-memory bank, clipped inverse | 1.51861 | 0.97634 |
| Same-world prior | 0.87227 | 0.93101 |
| History posterior | 0.64775 | 0.75805 |

All three history readers beat the no-history learned bank and same-world prior
on both question groups after feasible decoding. The exact history posterior
remains better. Projection also improves all four readers relative to the fixed
clipped inverse in this aggregate. This establishes useful history in the retained
old-answer predictions under this supplied-law decoder; it does not establish a
learned law, unique roles, recursive state closure or superiority across architectures.
The inverse's raw probability violations affect 99.90–100% of mapped question
vectors, pooled over all five decoded questions. Clipping makes scored outputs
valid, but does not make their bank attainable by the declared finite state family.

The question/world-only bank has a smaller Euclidean projection residual than the
history banks but worse prediction. Residual is therefore a geometric diagnostic,
not a scientific success criterion. A feasible bank also need not determine unique
state weights; the alias control retains this limitation. Direct farther forecasts
from the original fit use a different test-state allocation and are not compared
as a within-history decoder intervention. The prior and posterior references use
a stipulated uniform 24-state prior, not the deterministic allocation's exact prior.

All 18,432 learned banks pass independent feasibility certificates; the maximum
simplex duality gap is 1.75658099e-14, below the unchanged 1e-7 bound. The gap
bounds remaining convex objective improvement; it is not prediction loss. The
measured affine ranks span 3–23 across the retained worlds.

Every unit passed scalar mixture, probability, score, prior/posterior and gap
reconstruction. All 3,200 aggregate means and eight whole-unit extracted-source
replays agree. Source, plan, parent input/forecast and portable-file hashes match;
eleven isolated bank/runtime controls pass. Verification is bounded replay,
not fresh-world confirmation or architecture-wide severity. The earlier numerical
fixture failure remains retained; its repair preceded admission and did not relax
the gap threshold. V15 C11/M01 remain failed instruments.

Science charged 261.765625 CPU seconds and adjacent verification 115.062500,
totaling 376.828125. Combined wall time was 385.151485 seconds (6.42 minutes),
consistent with the 180–500 CPU-second packet forecast. The already admitted
L2d pair applies this same decoder to matched-total old versus diverse training;
conditional-target fits independently examine how the old-answer bank is learned.
These follow-ons remain queued or running, not established by this result.

[Portable evidence](../../../../results/v18/exploratory-loop/L2c-feasible-bank-1/SCIENTIFIC_MANIFEST.json).


## L2b realized targets: original-question gains without transfer

Does learned history transfer beyond the original questions when training states are sampled independently and selection sees only those questions? No in this realized-target control: farther mean logarithmic loss, where lower is better, is 1.70223 for direct history, 2.91068 for flat memory and 3.02246 for split memory, versus 1.57940 without history. Recurrent memory still helps on the original questions. The conditional-target comparison remains pending, so this arm does not establish the effect of changing training targets. This is a descriptive constructed-method result, miniature — architecture untested.

This first L2b arm fits twelve selected readers: four architectures and three
initialization seeds, width 48, 64 epochs and batch size 128. Each of sixteen
cells supplies 768 training histories with five original-question targets per
history and 384 development histories. Hidden states are sampled independently
from the sixteen permitted nonzero-skill states. Targets are exact response
distributions at the sampled state, not single sampled artifacts. Development
uses the same realized-state target and original menu. Both compositions and
farther forms are absent from fitting and selection.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows distinguish the original, composed and farther question sets. Columns show
the no-history neural rival, three learned history readers and supplied-law exact
inference. The first two rows average equally sized maker halves. The test set
contains 96 coefficient lineages with paired cells, queries and fit seeds; those
repeated observations are not independent training-dataset replications.

| Question set | Question/world only | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|---:|
| Original | 1.48711 | 1.31306 | 0.85567 | 0.80949 | 0.67447 |
| Untaught compositions | 1.55192 | 1.64328 | 2.95921 | 3.07101 | 0.64288 |
| Farther | 1.57940 | 1.70223 | 2.91068 | 3.02246 | 0.75870 |

The recurrent gains remain specific to the original questions. On compositions
and farther questions, every learned history reader is worse than the no-history
rival. Farther paired excess losses over that rival are 0.12282 for direct,
1.33128 for flat and 1.44306 for split; their descriptive lineage-bootstrap
intervals are respectively [0.09327, 0.15506], [1.18192, 1.47814] and
[1.23814, 1.64622]. These are descriptive, within-dataset comparisons, not
architecture-wide confidence statements or independent fit replications.

This is the baseline for the already running conditional-target arm. Its new
state sampling, lineage and old-only development differ from L4, so cross-study
score differences do not isolate any of those changes. Only the two L2b arms
form the controlled target comparison. The exact reference still uses its
stipulated 24-state prior and is not claimed Bayes-optimal for the balanced test
allocation. A supplied-law bank decoder is a separate diagnostic of access.

All learning controls pass. Independent extracted-source verification reconstructs
all 105 means and 3,840 fixed forecasts across 60 reader/test combinations exactly.
The separate scalar target audit checks 160 training/development rows; maximum
teacher discrepancy is 3.33e-16. The same audit records state counts without
requiring random balance. Plan, source, input, raw forecast and selected-weight
hashes match, and portable reassembly validates every scientific file. Replay is
bounded forecast reconstruction, not full retraining or architecture severity.
Historical V15 C11/M01 remain failed instruments; no human-intent claim follows.

Science and adjacent verification charged 4305.359375 CPU seconds and took 75.84 wall minutes. Science alone charged 4285.937500 seconds; verification charged 19.421875. Native CPU receipts take precedence when larger than self-reported parent time. The fixed campaign budget and clocks are unchanged.

[Portable evidence](../../../../results/v18/exploratory-loop/L2b-realized-targets-1/SCIENTIFIC_MANIFEST.json).
