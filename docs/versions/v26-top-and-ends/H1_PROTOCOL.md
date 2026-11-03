# H1: frozen V26 protocol

HIERARCHY_TOP. Can the top of a goal hierarchy be read from steps, and does the reader's own expertise help or mislead? (ceiling 6)

**Why.** The curator: "you're able to rigidly reverse engineer the bottom... unable to reverse engineer the top... It is the critical issue", and humans get there by "slotting in your existing expertise into bits and pieces of the larger puzzle."

**Worlds.**
- **A three-level hierarchy over steps.** Top goals: a library of 6. Subgoals: a library of 8. Pieces: a library of 12. Steps: an alphabet of 10.
  - Each top goal is a distribution over subgoals; each subgoal a distribution over pieces; each piece emits a fixed short step string (2 to 4 steps).
  - **Overlap** between top goals' subgoal distributions: none (disjoint supports), partial (two shared subgoals), heavy (four shared subgoals with different weights).
- **A document:** 6 subgoal episodes drawn from the maker's top goal; each episode, 1 to 3 pieces from the subgoal; pieces' steps concatenated. Episode and piece boundaries are hidden; a variant shows episode boundaries (like paragraph breaks).
- **Maker:** one top goal per maker; documents independent given it.
- **Records:** 1, 2, 4, 8 and 16 documents.

**Readers** (all exact, with marginal likelihoods by dynamic programming over hidden segmentations). Every piece-based reader (R-exact, R-expert-k, R-pieces) also holds one unknown piece, used at any piece slot with probability 0.05 whatever the subgoal, which emits 2 to 4 steps (length uniform), each step uniform over the alphabet; every other piece's probability is scaled by 0.95. So no document has zero likelihood under any reader.
- **R-exact (the ceiling):** the true libraries at every level. It is the generator's own model up to the unknown-piece allowance, so it is reported as the ceiling, never as a contrast.
- **R-novice:** the true subgoal and top libraries with an empty piece library, so every piece is unknown; every top goal then has the same likelihood and its gain is exactly 0, the reference for "misleads".
- **R-expert-k:** the reader's own piece library, which agrees with the maker's on k = 100%, 75% or 50% of pieces; its remaining pieces emit different step strings. Subgoal and top libraries are true.
- **R-flat:** top goal to step bigrams (no hierarchy): each top goal's bigram table and first-step distribution are estimated once from 10,000 documents simulated from the true generator under a declared seed, with add-one smoothing, and frozen before any draw. It is a projection of the true model, reported as a baseline.
- **R-pieces:** pieces known; each top goal is read as a distribution over pieces (its subgoal mixture summed out), pieces independent given the top goal, segmentation summed over (no subgoal or episode level).

**Case and check (amendment E).** The case: top goals that a single document can't tell apart, because they share subgoals and no step is near-exclusive to one top goal.
- Check: in every sampled world, no top goal holds more than half of any step type's expected occurrences summed over the six top goals; redraw worlds that fail, up to 1,000 draws per world, after which the grid is revised within its ranges.
- And in at least 25% of partial- and heavy-overlap worlds, R-exact's posterior on the true top goal after one document is below 0.8.

**Primary.** At 4 documents, partial overlap: R-expert-75's true-top log-score minus R-flat's. Under 0.05 nats counts as no practical advantage. Also the documents needed to reach 1 nat, by reader and overlap.

**Falsifiable arm.** R-expert-50's true-top log-score gain over the prior at 4 documents, heavy overlap. Prediction: a half-mismatched expert misleads (a gain more than 0.05 nats below R-novice's exact 0). If it doesn't, expertise mismatch at the piece level doesn't mislead at the top in these worlds. Report R-expert-50 minus R-flat beside it.

**Known-answer checks.**
- R-expert-100 equals R-exact (posteriors within 1e-12).
- With all top goals given identical subgoal distributions (a fixture), every reader's top-goal posterior equals the prior.
- The dynamic-programming likelihood equals brute-force enumeration over segmentations on a fixture of two short documents.

**Also report:** the gain from visible episode boundaries; how the top-goal posterior's error concentrates (which top goals alias).

**Texture grid:** library sizes (top 4/6/10, pieces 8/12/20), steps per piece (2 to 6), documents with 3 or 12 episodes, mixtures of two top goals per maker.

**Shed order:** the 16-document level; R-pieces; heavy overlap except at 4 documents, which the falsifiable arm needs.

**Route.**
- To Sounding Line's T3 (a writer's personal chunk library) and T5 (plans inferred by a language model, checked against her later choices): whether a reader's own library speeds inference at the top, and how much mismatch costs. That sets expectations for reading real writers with a language model's expertise.
- To ToMpathy: the top tier of the expertise-and-habit panel may cite H1 as constructed-world support, with its records-needed figure; its marks stay "untested" on real records.


## Declared implementation choices

Six makers, one per top goal per world. Dirichlet piece mixtures with concentration2; common-overlap concentration8, 90% shared and10% extra support. Seeded strings2-4 steps, no exclusive-step acceptance. Fixed10,000 training documents per top goal per world, separate training seed, before observations; add-one bigrams. Hidden piece-count is uniform1-3. Four/eight/sixteen document prefixes share trajectories. Piece-reader retains the total-count distribution but removes shared-subgoal dependence. Expert libraries replace a fixed last quarter/half of piece strings. Novel strings have the stipulated .05 unknown channel.

All choices are provisional scientific modelling choices, frozen before the excluded cost probe. No observed study result selected them. Sixteen independently seeded whole-world blocks per setting; check-only and cost-probe identities are excluded. Full byte replay and separate scalar posterior/decimal aggregation are required; some likelihood kernels remain shared. Known-answer fixtures independently verify the critical equations.

Texture variants are finite one-factor settings in the order below. Each receives its own excluded measured probe, complete replay and reconstruction. No texture result upgrades the primary.

```json
[
  {
    "overlap": "partial",
    "tops": 4
  },
  {
    "overlap": "partial",
    "tops": 10
  },
  {
    "overlap": "partial",
    "pieces": 8
  },
  {
    "overlap": "partial",
    "pieces": 20
  },
  {
    "overlap": "partial",
    "max_piece_length": 2
  },
  {
    "overlap": "partial",
    "max_piece_length": 6
  },
  {
    "overlap": "partial",
    "episodes": 3
  },
  {
    "overlap": "partial",
    "episodes": 12
  },
  {
    "overlap": "partial",
    "mixed_tops": true
  }
]
```
