# E2: frozen V26 protocol

ONE_MODEL_TWO_DIRECTIONS. When does reading another through one's own model beat reading through the population's? (ceiling 4)

**Why.** The curator: "the reader in their moment is using their own autopilot, but having been auto-corrected for the general population if they want to... generally they just start with what they're able to do." And: "we can use fear both to reverse engineer someone's painting, or to forward engineer our own survival. Not survival, like persistence." The literature's term for the first is anchoring and adjustment in perspective taking (Epley et al. 2004).

**Worlds.** W1-style piece worlds (6 pieces, 3 ends). Makers' weights are drawn uniformly from the 15 grid points (step 1/4), so the population mean is (1/3, 1/3, 1/3) and the true prior is uniform. For each maker, a reader's weights are drawn uniformly from the grid points in a distance bin (below) chosen in equal shares. Each reader has its own piece library.

**Readers:**
- **S, simulation:** prior on grid weights w proportional to exp(−κ · TV(w, reader's weights)), κ (2, 8), and the reader's own library as the forward model;
- **P, population:** prior proportional to exp(−κ · TV(w, population mean)), the same κ, true library;
- **I, ideal (the ceiling):** the true uniform prior and the true library. It wins by construction and is reported as the ceiling.

**Similarity.** Reader-to-maker distance in weights, total variation in the bins 0, 0.25, 0.5 and 0.75 or more; library overlap: all 6, 4 or 3 of the maker's 6 pieces.

**Case and check.** The case: readers who differ from the maker.
- Check: each distance bin holds at least 20% of reader and maker pairs at each overlap level (the stratified drawing makes it so; the check verifies the implementation).

**Primary.** For each κ, the distance bin at which S's log-score on the maker's ends first falls below P's, at 8 records, with the reader's own library matching 4 of the maker's 6 pieces: the crossover. At full overlap S and P share the likelihood, so where S falls behind is set by where the two priors sit relative to the maker; report that crossover for scale only.

**Falsifiable arm.** Prediction: for near makers (distance 0.25, one grid step), with the reader's library matching 4 of the maker's 6 pieces, S beats P at 8 records by at least 0.05 nats, at each κ. If it doesn't, starting from one's own model doesn't pay even for similar makers once its library is partly wrong, in these worlds. At distance 0 S's prior peaks at the truth by construction; report that bin for scale only.

**Known-answer checks.**
- With S's prior replaced by the true population prior and full library overlap, S equals I (posteriors within 1e-12).
- With the reader's weights set to the population mean (a fixture) and full library overlap, S equals P.
- κ = 0: S's and P's priors both equal the flat prior.

**Also report:** the egocentric error at 32 records (S's posterior mean pulled toward the reader's own weights) by library overlap; a corrected reader that mixes S and P priors 50:50, the "every man kind of filter".

**Texture grid:** κ, number of ends, mismatched libraries in which the reader lacks one of the maker's pieces.

**Shed order:** κ = 8; the 50% overlap level.

**Route.**
- To Sounding Line's T5 and to the engine: when a reader should anchor on its own expertise (a language model's) and when on the population, and how far to correct. W4 already found borrowed population priors didn't help deviation detection.
- To ToMpathy: whether the engine's prior for a new maker should be its own model, the population's, or a mixture.


## Declared implementation choices

Four pairs per distance bin per world, uniform maker grid and uniform reader within bin. Known beta3, no habit in this W1-style choice variant. Partial mismatch replaces effect rows of final pieces with independent grid outcomes. S/P contrast uses posterior log scores, not gains relative to different priors. Mixture uses half of each prior with the own library.

All choices are provisional scientific modelling choices, frozen before the excluded cost probe. No observed study result selected them. Sixteen independently seeded whole-world blocks per setting; check-only and cost-probe identities are excluded. Full byte replay and separate scalar posterior/decimal aggregation are required; some likelihood kernels remain shared. Known-answer fixtures independently verify the critical equations.

Texture variants are finite one-factor settings in the order below. Each receives its own excluded measured probe, complete replay and reconstruction. No texture result upgrades the primary.

```json
[
  {
    "kappa": 0.5,
    "overlap": 4
  },
  {
    "kappa": 1.0,
    "overlap": 4
  },
  {
    "kappa": 4.0,
    "overlap": 4
  },
  {
    "kappa": 16.0,
    "overlap": 4
  },
  {
    "kappa": 2.0,
    "overlap": 4,
    "ends": 4
  },
  {
    "kappa": 2.0,
    "overlap": 4,
    "missing_piece": true
  }
]
```
