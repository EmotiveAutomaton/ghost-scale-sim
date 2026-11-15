# Pre-flight summary, 14 November 2026

What the analyst computed before writing the three packets of 14 November, and what each result changed. Scripts and outputs: `PREFLIGHT_2026-11-14.zip`. One pass in one evening, no replay; the receiving repositories reproduce every number before relying on it.

## 1. The public corpus of event reports (Crowd-enVENT)

Downloaded from the authors' site on 14 November (SHA-256 `61c22527795669a0556df8d7441d3a7bf13ef8b2266b0433881645e92bf885e6`; CC BY 4.0 by its own licence file). 6,600 reports by 2,379 writers, each with the writer's label, her 21 ratings, durations, a short personality inventory and twelve "right now" ratings (asked before the recall in the one questionnaire printout the analyst read, round 5); 6,000 readers' judgements on 1,200 of the reports; and the authors' model predictions for those 1,200.

| Finding | Number | What it changed |
|---|---|---|
| One reader, five readers, the authors' model, a bag of words (macro-F1 on 13 labels) | 0.492, 0.565, 0.594, 0.393 | The ladder and bars of the engine's TP10 |
| Reports where four or five of five readers agree; where all five do | 49.7%, right about the writer 74.2%; 21.8%, right 80.2% | No absolute bar of 0.85 is reachable; confidence bars set against these |
| Error on the 21 ratings: the constant, one reader, five readers, bag-of-words ridge, the label's mean row | 1.499, 1.573, 1.262, 1.374, 1.336 | Round 18's bar of 1.57 sits below the floor. The engine's bar is the constant |
| Ratings readable from text (five readers against the writer) | eight at 0.58 or more; four at 0.41 or less | The engine reads eight features, not 21 |
| No-emotion reports found: one reader, five readers, the authors' model | 0.208, 0.200, 0.490 | Round 18's 0.70 on these reports replaced by 0.40, with 0.90 on plainly neutral text |
| Near-twin labels by their 21-rating rows | pride and joy 0.99; guilt and shame 0.95; trust and joy 0.94; fear and sadness 0.92; anger and disgust 0.91 | Sounding Line's IN13; the engine's TP11 |
| Telling the twins apart, writers held out: her 21 ratings against the words of her report | 0.613, 0.627, 0.728, 0.749, 0.798 against 0.757, 0.700, 0.863, 0.787, 0.891 | The engine's main instrument is a reader of what the event was about; the 21 ratings are a second layer |
| Signs predicted from the theory's channel distinctions, written before the numbers | 23 of 30 right, most gaps under 0.4 standard deviations | The distinctions are there in the ratings and they are small |
| A row for a label never seen in training (bag-of-words ridge) | median correlation with the true row 0.93; nearest row is its own for 9 of 13; size 0.13 to 0.32 of true; trust fails at 0.10 | The engine's TP13 (rows for the channels nobody rated), with a gate |
| A bag's mix from raw and from corrected counts (peaked mixes, 60 reports) | one reader 0.061 raw, 0.040 corrected; the authors' model 0.052 raw, 0.033 corrected | The engine never shows a raw count (TS2, TP12) |
| Consecutive surveys by one writer | 4,221 pairs, median gap 4.8 minutes; a rating changed in 64.1% | Sounding Line's IN14 is supported. Its outcome was not looked at |

## 2. Ghost worlds

| Family | Pre-flight result | What it changed |
|---|---|---|
| P, pages until a maker's largest label is known | With the measured tables, raw counts never reach 0.80 right; with 80% neutral sentences they stay at 0.08 to 0.22. Corrected counts reach 0.80 in 16 to 64 pages where a perfect reader needs 8 to 32. When makers are alike nobody gets there in 64 pages | The family maps estimators and table errors, not only reader accuracy |
| L, exposure, reaction or carry-over | Three makers with the same show rate. Shows alone through a reader with a hit rate of 0.6: 3 to 20% separated in 1,500 steps. With an event detector that hits 0.7: 58 to 98% | "Emotional" is three settings, and the engine needs an event reader beside its channel reader |
| R, asked or watched | The two engine questions the miniature models (purpose and audience) move a rule 0.61 of the 3 units needed against a habit; a price question separates them by definition; the third engine question was not modelled. Watching settles a rule in 20 to 68 steps with need and audience unseen. Asking settles it at once only when answers are right nine times in ten | A fourth probe kind in the engine's log; the family's grid |

Families K, V, N, E and F were not pre-flighted.

## 3. Fixtures

Every planted value in the Sounding Line cards IN13, IN14, IN15, DC1 and LA6 and in the engine's TS2 was computed by `fixtures.py` or `fix_in15.py` before it was written.

## 4. What this pre-flight is not

- It is not a result. No controls beyond holding writers out, no replay, default settings.
- Section 1's twins rows were seen by the analyst, so Sounding Line's IN13 Part B confirms a computation and is not a blind test.
- The human rows come from crowd readers of 13-word reports. A reader of essays faces different text.
