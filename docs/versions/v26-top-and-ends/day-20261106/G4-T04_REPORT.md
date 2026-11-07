# G4-T04 — precision-6

How many well-priced offers distinguish a strong terminal weight, and how many records do offers save? The separate texture variant retains 3 settings with sixteen independent whole-world blocks per setting. Primary bar conjunction: met. records needed (A9): 171.5, 95% interval 133 to 221; offers needed (A9): 12, 95% interval 9 to 15. Constructed-method evidence; miniature, architecture untested.

Claim `G-V26-DAY-G4-T04-1`. No human mechanism or engine display is validated.

This table identifies the setting index, observation stratum, measure, point estimate, pointwise 95% whole-block interval, registered bar and its disposition. Records-needed estimates are censored; unavailable limits are not imputed to the horizon. Asterisks mark results fixed by definitions or expected from the case construction, rather than independent tests of the theory.

| Setting | Stratum | Measure | Estimate | 95% interval | Bar | Assessment |
|---:|---|---|---:|---|---|---|
| 0 | {"reader": "contrasts", "truth": "A9"} | nothing saved | 0.59375 | 0.475846 to 0.711654 | — | descriptive |
| 0 | {"reader": "contrasts", "truth": "A9"} | records saved | 43.5 | 20 to 60.5 | — | descriptive |
| 1 | {"reader": "design", "truth": "A1"} | off target share | 0.00438169 | 0.00181984 to 0.00789344 | — | descriptive |
| 0 | {"reader": "design", "truth": "A9"} | off target share | 0.00983768 | 0.00445324 to 0.0161212 | — | descriptive |
| 2 | {"reader": "design", "truth": "C"} | off target share | 0.0511364 | 0.0401681 to 0.0623243 | — | descriptive |
| 0 | {"reader": "full", "truth": "A9"} | class records | 70.5 | 59 to 89 | — | descriptive |
| 0 | {"reader": "full", "truth": "A9"} | offers needed | 4 | 3 to 5 | — | descriptive |
| 2 | {"reader": "full", "truth": "C"} | lock delay * | 8 | 6.5 to 9 | — | descriptive |
| 0 | {"reader": "offers", "truth": "A9"} | class records * | 171.5 | 133 to 221 | at most 256 | met |
| 0 | {"reader": "offers", "truth": "A9"} | offers needed | 12 | 9 to 15 | — | descriptive |
| 2 | {"reader": "offers", "truth": "C"} | lock delay * | 36 | 26 to 52.5 | — | descriptive |
| 0 | {"reader": "ordinary", "truth": "A9"} | class records | 103 | 91 to 125 | — | descriptive |
| 0 | {"reader": "ordinary", "truth": "A9"} | offers needed | 6 | 5 to 9 | — | descriptive |
| 2 | {"reader": "ordinary", "truth": "C"} | lock delay * | 11 | 9 to 13 | — | descriptive |

Pre-check scale, not saved scientific results: Pre-checks: about 12 offers and about 199–222 records for A9 at one offer per 16 records; records saved about 29.5, with about 58% saving nothing. This finite ladder does not test all prices.

For conjunctive primaries, every required component must be met; one not-met component makes the conjunction not met, otherwise it remains unresolved. No headline verdict is read from a point estimate. The one-nat sustained margin is finite-horizon descriptive evidence, not an error-controlled or anytime-valid decision.

By construction (retained from the prospective card):

**By construction** (report for scale only):
1. On offers aimed at the maker's top value, the thresholds are m > w/(1 − w). A75 is bought at m 4 and 8. A9 never prefers the trade, since its price of 9 sits just past the ladder. A1 and locked C are never bought. The target rule aims some offers elsewhere: in the pre-check, 11.1% of A75's offers, 5.3% of C's, 2.4 to 4.3% of A9's and 1.7% of A1's. Every type takes those, because they pay m on its top value. Report the off-target share.
2. On offers, A1 and locked C differ only through choice noise. A1 accepts with probability 0.98/(1 + e³) + 0.01 = 0.057 at every m; locked C accepts with probability 0.01. That is about 0.05 nats per offer.
3. A9's evidence on offers comes mostly from m 8, about two-thirds of it: about 1.29 nats per on-target offer at m 8, 0.39 at m 4, 0.19 at m 2 and 0.12 at m 1. Against that sits a within-class prior handicap of about 1.6 nats: 24 locked-on-k hypotheses among C's 90, against 1 for A9 among A's 18.
4. **The A9 verdict follows.** Items 1 to 3 set the evidence each offer carries, so the offers needed to separate A9 from C hardly depend on ρ. The pre-checks' sustained crossings took about 12 offers; a drift estimate from item 3 (1 nat plus the 1.6-nat handicap, at about 0.5 nats per on-target offer) gives about 5 offers to a first crossing. Records needed are offers divided by ρ, so the expected value of the records bar is fixed: met when about 12/ρ ≤ 256. ρ only rescales; whether the interval clears the bar depends on the sample size.
5. With τ unknown, the τ-256 C hypotheses copy every on-grid compensatory type until record 256. So an on-grid terminal maker's A-against-C margin plateaus near ln 5 = 1.61 nats until then: a soft ceiling, which nearby hypotheses can lift to about 4. A9 is off the C grid and escapes it.
6. Before τ, a C maker is compensatory, so its records before τ carry no evidence of lock-in.
7. After lock-in, the reader names C within about 6 to 7 records. That is E1's sharpness gap: a locked maker takes a k-at-1 option with probability at least 0.985, while A at precision 3 gives it at most about 0.86. Check 1 requires the gap. On the locked-by-t contrast the delay was 7 to 8 records for lock times 32 to 128, and 15 for lock time 0, so the detection delay is set by the gap too.
8. **On-grid lock-state figures are set by the lock-time prior.** At each lock-time grid point, the C hypothesis with an on-grid maker's own weights joins "locked", carrying a fifth of the maker's prior mass, so the locked-by-t margin drops to about ln 5 and crossings bunch just after the grid points. In the pre-check, 31% of A1's crossings fell between records 257 and 288, and offers saved no records at all for 72% of A1 and 62% of A75 makers. A lock-time grid every 8 records removed the bunching and slowed the read (A1's median 405 records). Off-grid A9 is far less affected.



Declared modelling choices and assumed choice model: Provisional adaptive target uses the last sixteen chosen outcome vectors, counting exact component value one, with uniformly broken ties; observed menu-selection factors are common to all hypotheses and cancel. Lock applies at record t >= tau; tau=0 is immediate. Tau-known cells disclose a uniformly sampled counterfactual lock time also for A makers, with C hypotheses restricted to that time. Equal A/C priors; uniform within each class. Four independent makers per block per type/cell. The finite price ladder cannot establish all-price refusal.

Case classification: passed the excluded case checks.

Complete original/replay raw bytes match. Independent inference, complete block-bootstrap reaggregation and separate decimal mean aggregation passed. Some likelihood kernels and the declared generative model are shared; this validates finite calculations, not model realism. Raw makers remain nested within independent blocks. Intervals are pointwise and descriptive, not multiplicity-adjusted.

Sounding Line: signature-commitment refusals need priced alternatives and ordinary choices; this finite ladder cannot establish refusal at every price.
ToMpathy: distinguish lexicographic choice rules from terminal weights only under a tested choice-noise model.

Recipient delivery awaits the owner’s uptake; this report does not claim a new cross-repository delivery.

[Protocol](protocols/G4-T04.json). [All measures and frozen settings](../../../../results/v26/day-20261106/v26-day-20261106-g4-t04-study/SUMMARY.json). [Independent review](../../../../results/v26/day-20261106/v26-day-20261106-g4-t04-study/INDEPENDENT_REVIEW.json).
