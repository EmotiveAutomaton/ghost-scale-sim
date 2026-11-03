# E3: frozen V26 protocol

FEELINGS_AS_CACHED_SOLUTIONS. What do emotion-like signatures reveal if feelings are cached solutions rather than amplifiers of current wants? (ceiling 3)

**Why.** The curator: emotion-like signals "would just be expertise again. Right? Isn't that what we're treating feelings as?" W6 found RAGE-like signals helped only with a logged trigger and misled under a wrong link.

**Worlds.** W6's choice worlds with logged blocks (W1's pieces at partial and heavy overlap, β = 3). Each record also carries a cue pattern, 4 binary features each present with probability 0.5, drawn without reference to the agent; the record is blocked with a probability set by its cue pattern through a declared map averaging 0.1. Three generators for a RAGE-like signature, each firing only on a blocked record, with probability 0.02 + 0.96 · (1 − exp(−3x)), so no reader gives the true ends probability 0:
- **(i) amplifier:** x is the value the blocked piece was serving, w·e (W6's value-linked generator);
- **(ii) cached appraisal:** x is the mean, over the features present, of the agent's learned block association a(f), which starts at 0 and after every record moves for each feature present by a(f) ← a(f) + λ(B − a(f)), with B equal to 1 on a blocked record and 0 otherwise, λ (0.1, 0.3), whatever her current wants (learned, like expertise);
- **(iii) innate cue:** x is 1 when a declared feature is present and 0 otherwise, whatever her history or wants (phylogenetic).

**Readers:** exact posterior over ends, assuming (i), (ii) or (iii), and a reader that infers which generator holds.

**Case and check.** The case: worlds where (i) and (ii) would fire on different records.
- Check: at least 25% of signature-eligible records differ in firing probability by more than 0.2 between (i) and (ii).

**Primary.** Under truth (ii), the gain on ends from signatures for a reader assuming (i), at 32 records. Prediction: below −0.05 nats per block (the amplifier assumption misleads when feelings are cached solutions).

**Also report:** what a reader assuming (ii) recovers instead: the agent's cue map (which situations she has learned to treat as blocks), scored against the true map; and the generator-inferring reader's accuracy.

**Known-answer checks.**
- In a fixture where (ii)'s x equals w·e on every blocked record, (i) and (ii) give identical likelihoods.
- (iii) with no declared feature fires with probability exactly 0.02 on every blocked record.

**Texture grid:** learning rates for (ii), the share of innate cues, a FEAR-like signature on threatened outcomes.

**Shed order:** (iii); the generator-inferring reader.

**Route.**
- To Sounding Line's V3: drive-channel cues in text may index a writer's cue map, what she has learned to meet with a given system, rather than her ends.
- To ToMpathy: the core values panel's drive channels should say which reading the evidence supports.


## Declared implementation choices

Eight makers per world. Independent fair binary cues; block probability .02+.16*feature0, mean .1. Signature precedes cache update; cache updates after every observed record. Cache-map inference marginalizes the finite learning rates using generator evidence; it is not an independently identifiable hidden map when rate and full history are supplied. End posterior assumes beta3 and no habit. Primary is mean total end-score difference per whole world; per-outcome-block rates also retained.

All choices are provisional scientific modelling choices, frozen before the excluded cost probe. No observed study result selected them. Sixteen independently seeded whole-world blocks per setting; check-only and cost-probe identities are excluded. Full byte replay and separate scalar posterior/decimal aggregation are required; some likelihood kernels remain shared. Known-answer fixtures independently verify the critical equations.

Texture variants are finite one-factor settings in the order below. Each receives its own excluded measured probe, complete replay and reconstruction. No texture result upgrades the primary.

```json
[
  {
    "overlap": "partial",
    "rate": 0.05,
    "truth": "cached"
  },
  {
    "overlap": "partial",
    "rate": 0.2,
    "truth": "cached"
  },
  {
    "overlap": "partial",
    "rate": 0.5,
    "truth": "cached"
  },
  {
    "overlap": "partial",
    "rate": 0.1,
    "truth": "cached",
    "innate_share": 0.25
  },
  {
    "overlap": "partial",
    "rate": 0.1,
    "truth": "cached",
    "innate_share": 0.5
  },
  {
    "overlap": "partial",
    "rate": 0.1,
    "truth": "cached",
    "innate_share": 0.75
  },
  {
    "overlap": "partial",
    "rate": 0.1,
    "truth": "cached",
    "fear": true
  }
]
```
