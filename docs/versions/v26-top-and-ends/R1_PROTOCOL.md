# R1: frozen V26 protocol

W1 REPAIR. Which intervention breaks an alias between ends that share every available piece? (ceiling 3)

**Why.** V25's W1 asked this, but its libraries couldn't produce the case in sampled worlds: the full-menu training situation always offered single-end pieces, so no two ends could share every available piece. That was an error in the V25 packet's design, not a finding.

**Worlds.** W1's design (6 pieces, 3 ends, β, records and the remove, satiate and switch interventions), with new libraries and situations: for one pair of ends, declared per world, pieces 0 to 4 have equal outcome values on those two ends, and piece 5's values on them differ. The four training situations draw only on pieces 0 to 4 (three hold four of them each, one holds all five), so hypotheses that differ only in the split of weight between the pair are exactly aliased on every passive record. Remove and satiate act on the five-piece training situation; each of the two switch situations holds piece 5 and two of pieces 0 to 4.

**Case and check.** Exact aliases of that kind in at least 25% of worlds (the rule makes it 100%; the check verifies the implementation).

**By construction (amendment E).** Remove never separates such a pair; switch always does, through piece 5; satiating one end of the pair separates it exactly when the other end's values vary across pieces 0 to 4, and satiating the third end never does. A habit on piece 5, which no training situation offers, is also exactly aliased with no habit until a switch shows piece 5. These shares are reported for scale only.

**Primary.** For each intervention, among the split-alias pairs at 32 passive records: the true-hypothesis log-score gain in nats from one intervention chosen by expected information, over one more passive record, and the interventions needed to reach 1 nat. Under 0.05 nats counts as no practical advantage.

**Known-answer checks.** W1's own fixtures (the tie, and the tie that breaks under satiation where the shared ends' outcome values differ), plus a new fixture for the case where it doesn't (the unsatiated end's values equal across the probe situation's pieces), which W1's controls lack.

**Texture grid:** overlap structures, β.

**Shed order:** β = 1; half the training situations.

**Route.**
- To Sounding Line: none this week.
- To ToMpathy: the owner-question order its packet planned to take from W1's alias map.


## Declared implementation choices

Eight sampled split-alias pairs per world, alternating their true endpoints. Full W1 210-hypothesis reader retained. Five training pieces share values on ends0/1; sixth differs. Separate adaptive remove/satiate/switch tracks up to16 interventions, each compared with equal-count passive records. Ties in expected information choose the first declared probe.

All choices are provisional scientific modelling choices, frozen before the excluded cost probe. No observed study result selected them. Sixteen independently seeded whole-world blocks per setting; check-only and cost-probe identities are excluded. Full byte replay and separate scalar posterior/decimal aggregation are required; some likelihood kernels remain shared. Known-answer fixtures independently verify the critical equations.

Texture variants are finite one-factor settings in the order below. Each receives its own excluded measured probe, complete replay and reconstruction. No texture result upgrades the primary.

```json
[
  {
    "beta": 2
  },
  {
    "beta": 4
  },
  {
    "beta": 3,
    "structure": "constant"
  },
  {
    "beta": 3,
    "structure": "varying"
  },
  {
    "beta": 3,
    "structure": "graded"
  }
]
```
