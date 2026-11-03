# U1: frozen V26 protocol

OPACITY_GATE. Does a source the learner can't reverse-engineer get taken in like an authority, whatever its label? (ceiling 3)

**Why.** Labelling a message "AI", "human expert" or nothing changed neither persuasion nor perceived accuracy in a 2025 study, although almost everyone believed the label (Gallegos et al.). The curator's account: neither an expert's nor an AI's text can be reverse-engineered, so both read as equally accurate. In another study, a bulleted list of similar one-sided arguments from the same model moved attitudes less than inline autocomplete did, and its authors credit writing with the suggestions rather than seeing the information (Williams-Ceci et al. 2026).

**Worlds.**
- Two hypotheses, h1 and h0, with prior 0.5 each. On each record the world holds a pool of 8 evidence items, each pointing to h1 with probability 0.7 if h1 is true and 0.3 if h0 is true.
- The source presents one item per record. Unbiased: an item drawn uniformly from the pool. Biased toward h1 with strength s (0.5, 1): with probability s, an item pointing to h1 if the pool holds one; otherwise an item drawn uniformly.
- **Transparent:** the learner knows the source's policy and s, sees only the presented items, and updates by their exact probability under each hypothesis given that policy, summing over the unseen pool: a rational learner treating hypothesis-conditioned samples correctly (Batista & Griffiths 2026).
- **Opaque:** the learner sees only the presented items, treats each as a uniform draw from the pool, and multiplies its log likelihood ratio by a default trust τ (0.5, 0.8, 1).
- **Labels:** "AI" or "expert" map to the same τ unless the learner has a track record of that source type: earlier records (16 by default; the primary varies the number) whose pools were revealed afterwards, from which it infers the type's s.

**Case and check.** The case: biased sources whose bias is visible in their sampling.
- Check: at least 25% of biased sources choose claims whose likelihood ratio differs from the unbiased policy's by more than a factor of 2.

**Primary (records needed).** Capture is the learner's mean posterior on h1 in worlds where h0 is true, at 32 claims. A learner given a track record of the source's type infers s from it (prior uniform on 0, 0.5 and 1), then reads new claims as the transparent learner does, under that posterior. The primary is the track-record length, among 1, 2, 4, 8, 16, 32 and 64 records, at which its capture from a source with s = 0.5 comes within 0.05 of the transparent learner's; the result can be "not within 64 records".

**By construction (amendment E); no falsifiable arm.** In expectation the opaque learner's log odds move by τ · ln(7/3) · (2p − 1) per claim, where p = 0.3 + 0.642 · s is the chance that a presented item points to h1 under h0, so it is captured exactly when s exceeds 0.31, at a speed set by τ; the transparent learner is calibrated. Report opaque minus transparent capture, by s and τ, for scale only. Without track records the label effect (AI minus expert) is exactly zero, the Gallegos pattern.

**Known-answer checks.**
- An unbiased source and τ = 1: the opaque and transparent posteriors are equal on every record (tolerance 1e-12).
- s = 0 (a fixture): the biased source's presentation probabilities equal the unbiased source's, item by item.

**Also report:** a menu condition, where the learner sees several offered claims and which one the source chose (like CoAuthor's menus), between opaque and transparent.

**Texture grid:** τ, bias strength, claims per record.

**Shed order:** the menu condition.

**Route.**
- To Sounding Line's U1 (whether modelling the source protects, on public co-writing logs): the observable a real-record test would need, such as evidence that a writer modelled the source (edits to accepted text) against verbatim acceptance.
- To ToMpathy: none this week.


## Declared implementation choices

Thirty-two makers, balanced true hypotheses; source strength fixed across track and claims. Track records reveal all eight pool items. Posterior on source strength integrated jointly across all new claims, never resampled per claim. Menu arm reveals four random pool items plus selected sign, using exact hypergeometric marginal. Records-needed uses h0-world mean capture curve, not individual hits. Design-check share averages the two biased source strengths equally.

All choices are provisional scientific modelling choices, frozen before the excluded cost probe. No observed study result selected them. Sixteen independently seeded whole-world blocks per setting; check-only and cost-probe identities are excluded. Full byte replay and separate scalar posterior/decimal aggregation are required; some likelihood kernels remain shared. Known-answer fixtures independently verify the critical equations.

Texture variants are finite one-factor settings in the order below. Each receives its own excluded measured probe, complete replay and reconstruction. No texture result upgrades the primary.

```json
[
  {
    "s": 0.5,
    "tau": 0.25
  },
  {
    "s": 0.5,
    "tau": 0.65
  },
  {
    "s": 0.25,
    "tau": 1.0
  },
  {
    "s": 0.75,
    "tau": 1.0
  },
  {
    "s": 0.5,
    "tau": 1.0,
    "claims": 16
  },
  {
    "s": 0.5,
    "tau": 1.0,
    "claims": 64
  },
  {
    "s": 0.5,
    "tau": 1.0,
    "claims": 128
  }
]
```
