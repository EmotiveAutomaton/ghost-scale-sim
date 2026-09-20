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
