# E1: frozen V26 protocol

TERMINAL_OR_CIRCULATING. Which observations tell terminal values from circulating ones, and from values that lock into identity? (ceiling 6)

**Why.** The curator's two worlds: "there are terminal values that are set in some way. The other option is that they're all bleeding into each other in a kind of circuitous fashion, and it is instead the ratio of how often they fire that we care about over time, both of which I think are extractable from the data that we have." And: "when you have set a value such that it is a terminal output of a utility function, then that is when it becomes an identity."

**Two corrections from the literature, built into the design.**
- A value that has become identity isn't a large utility weight. In the sacred-values literature it refuses trade-offs at any price, and adding a material incentive to a deal can backfire (Tetlock et al. 2000; Ginges et al. 2007). So type C locks in **lexicographically**, and the environments include trade offers.
- One snapshot can't separate a network of mutually reinforcing values from a fixed common cause; the two are equivalent cross-sectionally, and only repeated measures over time tell them apart (van der Maas et al. 2006; Kruis & Maris 2016; Kievit et al. 2017). This card is longitudinal by construction. A snapshot here would be one record, or choice shares with their order discarded (the usage-ratio predictor below); the stationary-only arm keeps the order.

**Worlds.**
- Each choice is among 4 options in a situation; each option serves 3 value dimensions with outcome vectors on {0, 0.5, 1}.
- **Agent types:**
  - **A, terminal:** fixed value weights w on the simplex grid with step 1/4.
  - **B, circulating:** after each choice, w moves toward the outcome vector of the chosen option: w ← normalize((1 − η) w + η · e(chosen)), with η in {0.005, 0.02, 0.1}. Use begets value.
  - **B2, preference learning by counts:** w is proportional to the agent's counts of experienced outcomes plus a prior count of 2 per dimension, the count-based preference learning of Sajid et al. 2021's active-inference agent, here with softmax choice rather than expected free energy. It is B with a learning rate that falls as experience accumulates.
  - **C, identity lock-in:** as B until any weight passes θ (0.6, 0.8), starting from grid weights whose largest weight is below θ, so that lock-in comes only from use. From then on, whenever some option serves the locked value at 1, the agent takes one of those options, choosing among them by its frozen weights, whatever the other options pay (a lexicographic rule, with the 0.02 lapse); otherwise it chooses by its frozen weights.
- **Choice:** softmax with β = 3 and a lapse rate of 0.02.
- **Environments,** mixed in declared proportions. No environment's construction uses the agent's type, weights or history; every random choice below is uniform and independent of the agent, so the menus carry no information about her type:
  - stationary: menus drawn from a fixed distribution in which a declared share of options (0.5 to 1) are specialists, serving one value dimension at 1 and the others at 0, and the rest are drawn uniformly on {0, 0.5, 1}^3;
  - forced: on one record in eight the agent must take an option drawn uniformly from the menu (a constraint, not a choice);
  - devaluation: for 8 consecutive records, a value dimension drawn uniformly at random has every option's outcome on it halved;
  - shift: from a declared record onward, a value dimension drawn uniformly at random becomes costly: options serving it at 1 pay 0 on every other dimension;
  - trade offer: a two-option menu; one option serves a value dimension drawn uniformly at random at 1 and pays 0 elsewhere, and the other pays 0 on that dimension and 0.5, then 1, on both other dimensions across a declared ladder of records.

**Readers:** exact posterior over type, w (grid, and for C the starting points below θ that its generator uses), η and θ, given choices and environments. B and C's weights are deterministic functions of the observed history, so their likelihoods are exact.

**Case and check.** Two cases, each checked on check-only draws (amendment E) before the probe:
- B close to A on stationary records: at the smallest η, at least 25% of B agents hold under 1 nat of evidence against A after 64 stationary records.
- Lock-in by use: at η = 0.1 and θ = 0.6, at least 25% of C agents pass θ between records 8 and 128 on stationary records, from starting weights below θ.
- If a check fails, revise the specialist share of stationary options within 0.5 to 1 and recheck. C cells other than η = 0.1 with θ = 0.6 report their lock-in rate and aren't required to pass.

**By construction (amendment E).** Whenever exactly one offered option serves C's locked value at 1, C takes it with probability at least 0.985, while A at β = 3 puts at most e³/(e³ + 3) ≈ 0.87 on any one of four options; so C separates from A after lock-in on ordinary records, with or without trade offers. Report C against A for scale only. On these outcome levels an A agent with weight 0.5 or more on the offered value never prefers the trade, and C, offered a trade on its locked value, refuses it except by lapse (probability 0.01); offered one on another value, it takes the option that serves its locked value at 1 when there is one, and otherwise follows its frozen weights. η = 0 nests A in B, and θ above 1 nests B in C.

**Primary (records needed).** Records to reach 1 nat between A and B, by environment mix: stationary only, plus forced records, plus devaluation, plus shift. The result can be "not within 256 records".

**Falsifiable arm (stated predictions).**
- Use begets value shows in stationary records alone: at η = 0.1, B separates from A within 256 records in at least half of agents. If it doesn't, the drift is invisible without designed variation, in these worlds.
- Forced records add a trace beyond the drift: at the smallest η, one forced record in eight cuts the median records needed to separate B from A by at least a quarter. If it doesn't, forced use leaves no trace beyond B's own drift, in these worlds.

**Also report:**
- records needed to separate B from B2, and B from C, and which environment does each;
- the posterior on θ after lock-in;
- how often a C agent's locked value is the one it was forced into early (path dependence);
- a usage-ratio predictor: softmax at β = 3 over the offered options, each scored by its outcome vector times the value shares of the agent's last 16 chosen outcome vectors. Report its held-out next-choice log loss minus the exact reader's posterior-predictive log loss, per record, at 64 and 256 records, by type and environment mix, beside the exact reader's type posterior at the same records; under 0.01 nats per record counts as predicting as well. In experience sampling, complex affect-dynamic measures added little beyond the mean and variance in predicting well-being (Dejonckheere et al. 2019); the question here is whether simple usage shares predict as well while the type stays unresolved.

**Known-answer checks.**
- η = 0: B's likelihood equals A's on every record.
- θ > 1: C's likelihood equals B's.
- B2 with an infinite prior count equals a fixed-weight agent at (1/3, 1/3, 1/3), a fixture outside A's grid.
- A forced record contributes no choice likelihood for any type (it isn't a choice).

**Texture grid:** η and θ ranges, 5 options or 4 value dimensions, forced-record rates (1/16, 1/4), devaluation strength.

**Shed order:** θ = 0.8; the shift environment; η = 0.02.

**Route.**
- To Sounding Line's V3 (whether a writer's mix of drive cues recurs) and V4 (paying to say it herself): the real-record analogue of a forced record, such as accepted AI text on a topic followed by her own later writing on it, and how many such events a test would need.
- To ToMpathy: whether the core values panel should show values as fixed weights or as usage ratios that drift, and what evidence would license either.


## Declared implementation choices

One maker of each type per world, uniformly sampled from eligible start grid; equal type priors and uniform within-type finite states. B2 begins at prior counts2 in every dimension. Four environment mixes separately compare stationary, forced every8, eight-record devaluation from64, shift from128, and a separate trade ladder. Forced choices update state with unit likelihood. Exact source histories, not future outcomes, update the predictor. Usage predictor uses last16 outcomes and uniform fallback on zero total.

All choices are provisional scientific modelling choices, frozen before the excluded cost probe. No observed study result selected them. Sixteen independently seeded whole-world blocks per setting; check-only and cost-probe identities are excluded. Full byte replay and separate scalar posterior/decimal aggregation are required; some likelihood kernels remain shared. Known-answer fixtures independently verify the critical equations.

Texture variants are finite one-factor settings in the order below. Each receives its own excluded measured probe, complete replay and reconstruction. No texture result upgrades the primary.

```json
[
  {
    "eta": 0.0025,
    "theta": 0.6,
    "environment": "stationary"
  },
  {
    "eta": 0.05,
    "theta": 0.6,
    "environment": "stationary"
  },
  {
    "eta": 0.1,
    "theta": 0.7,
    "environment": "stationary"
  },
  {
    "eta": 0.1,
    "theta": 0.6,
    "environment": "stationary",
    "options": 5
  },
  {
    "eta": 0.1,
    "theta": 0.6,
    "environment": "stationary",
    "dimensions": 4
  },
  {
    "eta": 0.1,
    "theta": 0.6,
    "environment": "forced",
    "forced_every": 16
  },
  {
    "eta": 0.1,
    "theta": 0.6,
    "environment": "forced",
    "forced_every": 4
  },
  {
    "eta": 0.1,
    "theta": 0.6,
    "environment": "devaluation",
    "devaluation": 0.25
  }
]
```

## Pre-admission gate refinement

The small-learning-rate check uses the full production type posterior, marginalizing starting weights and learning rates. The earlier matched-start-pair engineering check is preserved but is not the admission gate. No scientific draw preceded this refinement.
