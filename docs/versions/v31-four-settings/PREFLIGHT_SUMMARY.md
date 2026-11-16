# Pre-flight summary, 15 November 2026

What the analyst computed before writing the three packets of 15 November, and what each result changed. Scripts and outputs: `PREFLIGHT_2026-11-15.zip`. One pass in one evening. Every script was replayed once from the bundle and reproduced its saved output byte for byte on the analyst's machine; the receiving repositories reproduce every number before relying on it.

No row of any restricted or newly collected set was read: not openESM's data, MIDUS, IPIP-NEO, Enron, the chess table or the typing fit. Two public sources were read: Crowd-enVENT (as on 14 November) and Story Commonsense.

## 1. A reader of needs, tried cold on event reports (engine TP23; Sounding Line IN13 Part D)

Story Commonsense ([Rashkin and others 2018](https://aclanthology.org/P18-1213/); SHA-256 `989a8294fee4b825b2761ab0a201a0e316b74a297efe2be467b1ee7490907f72`; no licence stated in its readme): annotators marked why story characters act, in nineteen need categories. A bag-of-words reader trained only on those stories (mean area under the curve 0.723, stories held out) was applied to the 6,600 Crowd-enVENT reports.

| Finding | Number | What it changed |
|---|---|---|
| Mean need profile of each label | pride: status, competition, approval; FEAR: tranquility (safety); DISGUST: food; trust: contact; sadness: family; relief: savings, rest, health | the engine has an answer key for which need an event touches, which its record said it lacked |
| Telling near-twins apart, the writer's 21 ratings against ratings with need tags | pride and joy 0.613 to 0.708; guilt and shame 0.627 to 0.632; fear and sadness 0.728 to 0.785; anger and disgust 0.749 to 0.785; trust and joy 0.798 to 0.838 | the trigger reader is the engine's centre this week; Sounding Line's blocked Part C is replaced by Part D |
| All 13 labels, macro-F1 | 0.375 to 0.421 | same |
| Predictions written first | 32 of 39 signs right; a gain of 0.03 on four pairs where three were predicted | |
| One prediction wrong | fear and sadness were expected to gain least; guilt and shame did | which need was touched does not separate guilt from shame |

Part D is therefore a replication under Sounding Line's controls, and is labelled so.

## 2. One dial or two (Sounding Line ES2, MI2; Ghost family D)

Constructed diaries with a rating floor. The worlds plant a correlation between the logs of two reactivities; the estimator targets the correlation between the reactivities themselves.

| Planted between logs | Target | Cross-half estimate (the cards') | Same-prompt estimate (the trap) | Eight daily reports |
|---|---|---|---|---|
| 1.0 | 1.00 | 1.00 | 1.13 | 1.01 |
| 0.8 | 0.78 | 0.79 | 0.91 | 0.80 |
| 0.5 | 0.47 | 0.49 | 0.61 | 0.50 |
| the thread's rival (one dial plus a second input into anger, spread 0.3) | 0.84 | 0.845 | 0.95 | not run |

What it changed: the cards fix the cross-half estimator and a reading rule (lower end at 0.90 for one dial, upper end at 0.85 for more). Slopes from the same prompts would pass a 0.8 world off as one dial. The rival reads 0.845, with three constructed worlds in ten above 0.85: a result near the line is what it predicts, so the trait check decides.

## 3. Why a maker stays (Ghost family S)

Exact, no sampling. Four ways to wait longer, matched at odds 0.75.

| Added wait at odds | 0.25 | 0.5 | 0.75 | 0.9 |
|---|---|---|---|---|
| the cost weighs less | 1.76 | 3.92 | 4.0 | 4.0 |
| a higher prior (optimism) | 1.55 | 3.92 | 4.0 | 4.0 |
| a sharper prior (confidence) | -0.47 | 0.0 | 4.0 | 8.0 |
| slower use of evidence | 0.07 | 1.42 | 4.0 | 6.56 |

What it changed: the handoff's two makers became four. A lighter cost and a flat optimism cannot be told apart by watching waits at any odds (310 to 799 trials); changing the cost tells them in 39 to 65. Confidence is told from either in 5 to 18 trials once even or lower odds are shown. The published pattern in mice (an effect at 0.75, none at 0.5 or 0.25) is the confidence row.

## 4. Four settings (Ghost family W)

A minimal maker with a full-view reader and unmatched moves. Every pair separated in 2 to 219 steps. The slow pair was swap temperature against the weight of Stress on a plan (16 to 219), and slowest where few drops could be acted on. Moving one setting raised switches sevenfold where doubling another barely moved them.

What it changed: the family matches each pair on the number an onlooker would compute first, and gives the reader four views.

## 5. Fixtures (Sounding Line)

Every planted value in cards KP1, KP2, CH13, CH14, ES1, ES3, MI1, IP1 and EN1 was computed by `fixtures31.py` or `fixtures31b.py`. Three taught something:

| Card | What the fixture showed | What it changed |
|---|---|---|
| CH14 | with fixed preferences and no effect of use, the count coefficient reads 0.87; with no effect of recency, the recency coefficient reads -0.05 | the primary is the held-out gain, and count's coefficient is not read as evidence |
| CH13 | when touchier players return sooner after a loss, the pooled short-wait shift grows from -0.030 to -0.036, because those players crowd the short waits | the primary is within player, and a second wait measure is built from other players |
| EN1 | a daily rhythm with no carry gives a correlation near 0.10 that is flat and then gone | the rhythm is removed first, and the card reads the fade |

## 6. The openESM inventory

From the public metadata only (`sl/ESM_CANDIDATES.md`, `sl/esm_inventory.json`). It moved two of the handoff's cards (no variable is typed as a trait measure and static data were removed from 29 members pending review, so Extraversion as a level and the trait checks go to the sealed MIDUS run; see `sl/esm_checks.txt`) and added one (member 0057's status events and feelings, 2,272 people: card ES1).

## 7. What this pre-flight is not

- It is not a result. Default settings, one pass, no controls beyond holding writers or stories out.
- The need reader is a bag of words trained on invented stories. Its transfer is one number per pair on one corpus.
- The constructed worlds are the analyst's. A fixture says an estimator recovers what was planted in that world, and nothing about people.
- Families T and Q of the Ghost packet, the engine's gate and the sealed MIDUS script have no pre-flight.
