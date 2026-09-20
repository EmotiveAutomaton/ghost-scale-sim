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
