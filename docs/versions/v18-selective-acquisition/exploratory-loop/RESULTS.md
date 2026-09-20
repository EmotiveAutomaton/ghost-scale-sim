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
