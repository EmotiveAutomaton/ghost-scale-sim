# H2: frozen V26 protocol

PAUSES_DIFFICULTY_DELIBERATION. Do pauses reveal goals when some pauses come from difficulty rather than deliberation? (ceiling 4)

**Why.** This replaces W7, whose vetoes let the final text reveal the goals. It adds the strongest rival from the response-time literature: pause length can track how hard a choice is, not which system made it (Krajbich et al. 2015).

**Worlds.**
- **Units:** 8 types, a document of 32 units.
- **Autopilot:** a habit distribution over the 8 types, from a declared grid of three habits, each with every type between 5% and 25%.
- **Goals:** the maker holds one of 6 top goals. A goal reweights types multiplicatively, with weights drawn log-uniformly in [0.5, 2] and no zero weights; every goal has at least two types with weight below 1 (redraw otherwise), so every goal can deliberate.
- **Deliberation:** each unit is proposed from the habit. When the proposal's goal weight is below 1, the maker deliberates with probability d (0.3, 0.6): she pauses and redraws once from the habit times the goal weights, renormalized; the redraw is final.
- **Difficulty:** two of the 8 types are hard. When the final unit is hard and was not deliberated, it emits a pause with probability q (0, 0.2, 0.5), whatever the goal. Each unit carries one pause flag, set by deliberation or by difficulty.

**Observations:** (a) the final text; (b) plus where the pauses fell; (c) plus what each redrawn unit replaced.

**Readers:** exact posterior over goals and the habit, (i) with the correct pause model, and (ii) misread: every pause treated as deliberation.

**Case and check.** The case: goals the final text alone leaves uncertain.
- Check: at 32 units, reader (a)'s posterior on the true goal is below 0.8 in at least 50% of worlds at the primary setting; otherwise reduce goal weights' range within [0.67, 1.5] and recheck.

**Primary.** Evidence (b) adds over (a), in nats per deliberation pause, at d = 0.6, q = 0.2, 32 units. Under 0.05 nats per deliberation pause counts as none.

**Falsifiable arm.** The misread reader's gain of (b) over (a) at d = 0.6, q = 0.5, 32 units, pooled per block as total gain over total pauses of either kind. Prediction: difficulty pauses mislead (below −0.05 nats per pause).

**Known-answer checks.**
- d = 0 and q = 0: (b) equals (a) exactly.
- All goal weights equal to 1: every goal posterior equals the prior.

**Also report:** (c) over (b); the gain by d and q; which goals stay aliased under (a).

**Texture grid:** units per document (16, 64), hard types (1, 3), goal-weight ranges, habits with a dominant type.

**Shed order:** d = 0.3; (c).

**Route.**
- To Sounding Line's S5 (planning labels after a stop) and S6 (choice conflict against deliberation in pauses): whether pauses can carry goal evidence once difficulty is modelled, and how much a misread pause model costs.
- To ToMpathy: whether current-attention marks drawn from pauses may cite constructed support, and on what condition (a difficulty model).


## Declared implementation choices

Six makers per world, each goal once with cyclic habits. Goal range .5-2, after the full-range check; narrower engineering check retained separately. Observed replacements identify a redraw even if the replacement equals the old unit. Primary is total gain divided by total deliberations within a world, not mean case ratios. Misread reader uses q=0.

All choices are provisional scientific modelling choices, frozen before the excluded cost probe. No observed study result selected them. Sixteen independently seeded whole-world blocks per setting; check-only and cost-probe identities are excluded. Full byte replay and separate scalar posterior/decimal aggregation are required; some likelihood kernels remain shared. Known-answer fixtures independently verify the critical equations.

Texture variants are finite one-factor settings in the order below. Each receives its own excluded measured probe, complete replay and reconstruction. No texture result upgrades the primary.

```json
[
  {
    "d": 0.6,
    "q": 0.2,
    "units": 16
  },
  {
    "d": 0.6,
    "q": 0.2,
    "units": 64
  },
  {
    "d": 0.6,
    "q": 0.2,
    "hard": 1
  },
  {
    "d": 0.6,
    "q": 0.2,
    "hard": 3
  },
  {
    "d": 0.6,
    "q": 0.2,
    "low": 0.8,
    "high": 1.25
  },
  {
    "d": 0.6,
    "q": 0.2,
    "low": 0.5,
    "high": 2.0
  },
  {
    "d": 0.6,
    "q": 0.2,
    "dominant": true
  }
]
```
