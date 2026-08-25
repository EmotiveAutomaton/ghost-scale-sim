# V12 — The Other Model. Results

**Discovery lane, 2026-08-24.** Sixty-two mandatory cards and the twelve-attack matrix, all
resolved: seventy-three landed, one (Q02) closed as an instrument failure. Every number below is read from a verdict file under
`results/validation/soundingline/v12/`; the generated table of every card is
[`RESULTS_PACKET.md`](RESULTS_PACKET.md). The program, its floors and its criteria were locked
before any result was seen (`results/v12/prereg_v12_lock.json`); the two deviations made after
seeing results are listed under *Controls, repairs, and failures* and nowhere else.

Everything here is a property of a constructed world. The claim ceiling on every card is
**method** or **constructed mechanism**; nothing below is a claim about people.

## 1. Where the project world moved

**A reader that starts from itself wins only when the maker is like it, and loses otherwise.**
The program's central question was whether reusing one's own generative organization, the
self-first prior, buys anything a matched generic prior does not. It does, and the shape of the
gain is the finding. For the fifth of makers nearest the reader's own measured profile, the
self-first reader scores 0.60 nats better on the maker's profile after one artifact than a prior
with the same entropy centred on the population. For the middle three fifths it scores 0.09 to
0.20 nats worse. For the farthest fifth there is no difference. Averaged over all makers the
self-first reader is worse than a plain population prior (−1.59 against −1.49 nats at first
evidence). A prior that keeps the self prior's entropy but permutes its correspondence loses the
whole near-maker gain, so what the self prior carries is correspondence, not confidence. The gain
transfers to fresh worlds and to a second surface convention, and it turns into a loss on a family
built to be anti-similar to the reader. Projection is real, priced, and self-correcting: the mass
a self-first reader wrongly places on its own profile falls from 0.25 after one artifact to 0.01
after fifty; counter-evidence corrects it with a half-life of four artifacts, and the order of
evidence leaves no trace on the final posterior.

**Values live in opportunities, not in counts.** A reader that scores what a maker chose against
what it could have chosen and at what cost predicts held-out choices 0.42 nats better than one
that ignores the costs, and the cost-blind reader learns nothing beyond the population average.
The same choice moves the posterior eight times further when it was made against a large cost
than under a near tie. A profile recovered in one domain predicts choices in a second domain with
a different payoff generator, cost scale and habit, 0.9 nats above a baseline that carries only
the maker's identity; it predicts changed costs, commissions and new goals prospectively. When
a habit stores a previous profile's values, partialling the habit out deletes 1.5 nats of profile
signal; modelling it costs nothing.

**Regime is readable, and the cooperative assumption is a bet, not a default.** Three regimes,
bard, neutral, and concealer, are matched on every surface statistic to machine precision and
differ only in what the tail cue means. A reader that assumes cooperation gains 0.8 nats on bards
and loses 10 nats on concealers relative to a neutral reader. A reader that lets the regime switch
recovers a source's new regime within five artifacts on average (92% within the twelve the
criterion set); a reader that keys the regime to the source identity takes fifteen. Accurately
modelling a concealer raises the reader's own payoff on a task that depends on the maker's next
move (0.73 against 0.64 for a reader fooled into cooperation); understanding an adversary and
cooperating with one come apart.

**Reconstruction accuracy and uptake weight are separate levers.** With the posterior-to-preference
bridge tested on planted posteriors, the weight moves the policy and the accuracy decides whether
the movement helps: the wrong-and-confident posterior moves the policy in the wrong direction
56% of the time, the accurate posteriors at most 23%. Source reliability drives movement
(+0.62 on the linear scale) where maker competence does not (+0.04), because competence never
enters the bridge and nothing leaked it in. After exposure, belief moves twice as far as
standing preference (0.49 against 0.19). Reliable counter-evidence reverses harmful movement
85% of the time while a quarter of the mass stays on the false source.

**Supply gains are symmetric when the reader is exact.** The trunk that asks what supplying one
latent buys for another found that the gain is a conditional mutual information, and conditional
mutual information is symmetric: supplying the goal buys 0.085 nats on the profile and supplying
the profile buys 0.085 nats on the goal, in every off-ceiling cell, to the fourth decimal. Any
apparent directionality in this record's earlier temporal results (purpose first, then method)
therefore comes from evidence order, ceilings, or non-exact readers, never from the information
structure. The process latent of this construction (the realization slot) is nearly inert:
supplying it buys nothing for the goal or the profile at the neutral regime.

**Upstream reach survives rewriting; local hands do not.** A director's goal is attributed at
0.92 after every feature of every part has been rewritten by someone else, where attribution of
the local contributor falls to chance (0.26 against 0.25). The exact structure reader separates a
central director from a shared brief at 0.81 with coherence matched so that a coherence baseline
sits at 0.50. Identical artifacts with different histories (ratified against merely unnoticed)
are abstained on by the artifact-only reader 97% of the time and separated by the record reader
every time; six later artifacts separate them for the artifact-only reader 87% of the time.

**Layered intent is recoverable in principle and underdetermined at the spec's floor.** The
dependency ruler (mutual information between inferred upstream and downstream choices) reaches
AUC 1.00 against flattened worlds when blocks are long and numerous (128 steps, 60 blocks), 0.96
at 32 steps, and 0.51 at the spec floor of 4 steps and 12 blocks, where every ruler is at chance.
Three-way separation of fewer goals, erased dependency and low effort is 0.95 once the blocks are
readable.

**Active reading buys little where the artifacts already speak.** The PyMDP agent chooses the
informative probe (89% agreement with the exact ranking), but the probes hardly differ: the
expected-information spread across commissions is 0.05 nats, commissioning a goal changes the
profile information an artifact carries by 0.001 nats against a free look, and choosing which
two of six episodes to inspect captures 0.02 of the 0.40 nats that perfect selection would. The
card that asked whether commissions probe better than free looks closed as an instrument
failure after its one repair. Active choice appeared to matter in one place, challenges chosen
for information about the regime, and that margin (0.16 nats over uncertainty sampling) did not
survive the confirmation lineage. Against a concealer planting evidence, modelling the planting
was worth 3.4 nats and choosing probes 0.3.

**The self-first gain survives the attacks it should and dies of the ones it should.** It is
bit-identical under shuffled source labels and a changed surface convention, survives a false
biography, reversed evidence and fresh worlds, keeps a third of itself under the PyMDP solver and
two thirds under a cheap histogram reader, is destroyed by permuting the prior's correspondence,
is diluted to nothing by wrong labels or equifinal twins, and an adaptive maker that shapes its
work toward the reader costs the self-first reader 0.37 nats on far makers while leaving the
near gain intact.

## 2. Which competing explanations gained or lost probability

| explanation | direction | what decided it |
|---|---|---|
| The self prior's advantage is confidence (a sharper prior), not correspondence | **lost** | the entropy-matched permuted prior loses the near-maker gain (0.55 nats gap at the nearest fifth); the entropy-matched generic prior is worse than the population prior everywhere |
| The self prior's advantage is a similarity-in-general effect that would appear for any near prior | **partly survived** | a random local prior of the same entropy was run as a control and is reported, not gated; the decomposition (S10) puts almost all pair-level variance outside the measured similarity axes (R² 0.07), with profile similarity the only axis that registers (0.02) |
| Projection is a stable bias | **lost** | it decays with target evidence (0.25 → 0.01) and corrects with a four-artifact half-life |
| Expertise does not matter for profile recovery (V11's C2 failure read literally) | **lost** | off ceiling, the expert reads goals 0.09 better and profiles 0.28 nats better than a half-corrupted reader |
| Choice frequency is a sufficient record of values | **lost** | the cost-blind and count readers learn nothing beyond the population prior; opportunity strength moves the posterior eightfold |
| Coherence explains what a director explains | **lost** | matched coherence, structure reader 0.81 against baseline 0.50 |
| A cooperative reading assumption is free | **lost** | 10 nats against concealers |
| Understanding a maker implies cooperating with it | **lost** | own payoff rises with model accuracy; adoption is a separate lever |
| Current goal and standing profile are jointly recoverable better than separately | **not supported** | joint reader 0.51 / 0.71 against 0.48 / 0.74 for the single-latent readers (R06) |
| Self-first improves hidden-continuation prediction | **not supported** | −0.007 nats at first evidence, within 0.01 of zero at every evidence dose (S07) |
| Policy and process similarity carry independent predictive value for the self-first gain | **not supported** | partial R² 0.004 and 0.002 (S10) |
| Layered intent is readable from short artifacts | **lost at the floor** | chance at 4 steps × 12 blocks; recoverable at 32 × 60 |
| Choosing what to look at is what makes a maker-reader good | **lost, in this construction** | probes differ by 0.05 nats; commissions inert (Q02 closed); selection captures 0.02 of 0.40 available (Q03); the apparent exception (B04, +0.16 over uncertainty sampling on the regime) did not replicate on the confirmation lineage |
| The self-first gain is a solver or likelihood artifact | **lost** | a third survives the PyMDP solver, two thirds a histogram reader; the identity attacks are bit-identical |
| The self-first gain is a labelling artifact | **survived as a warning** | half-wrong labels and equifinal twins dissolve it to the same average, as they must; the far bin rises to 0.17 |

## 3. What remained unmeasured

- **Real makers.** Every self-model here is measured from the maker's own artifacts under a
  known convention. Nothing in the program says whether a human reader has such a record of
  itself; self-report is not that record.
- **The process latent.** The realization slot carries so little surface mass by construction
  that supplying it buys nothing. The T trunk's answer about process is an answer about a weak
  cue, not about process.
- **Non-exact readers under supply.** Directionality is zero for exact readers by identity. The
  interesting cases, readers with the V-series' order effects, were not the ones run here.
- **B06's adoption arm.** With a uniform reader preference, adopting anyone's profile costs
  nothing on the reader's own task, so the "adoption not required" half of B06 is a construction
  artifact; only the own-payoff half stands.
- **Layered intent at the floor.** The program can say the ruler works and that the floor is
  below where it works; it cannot say where between 4 and 32 steps real artifacts sit.
- **Cards that were not promoted.** The confirmation lineage was run only on the twenty-two
  promoted cards (see *Frozen confirmation cards*); the failed criteria (S07, R06, R08), the
  closed card (Q02) and the descriptive T results have one lineage behind them, not two.

## 4. Questions for the curator

1. The self-first result is a bet with a known payoff shape: large gains near self, moderate losses
   in the middle distances, break-even far away, and a net loss against a population prior. Which of
   these is the claim the theory wants: that readers *do* start from themselves (a mechanism to
   detect), or that they *should* (which this says is false on average)?
2. Symmetric supply gains for exact readers make "purpose first, then method" a statement about
   readers, not about information. Is the temporal form of E36 still the form worth defending?
3. The layered-intent floor was set at 4 steps × 12 blocks. The ruler needs about eight times
   that. Is the floor a property of real artifacts, or a number to revise?

## 5. Recommendation by trunk

| trunk | pursuit | warrant | recommendation |
|---|---|---|---|
| I instruments | PROMOTE | CONFIRMATORY SUPPORT | anchors reproduce to zero; 50-world severity coverage of V11's S-14/S-15; keep as the standing harness |
| S self and similarity | PROMOTE (S04, S05, S06, S08, S09); EXHAUSTED (S07, S10) | MECHANISM CANDIDATE | confirm S04's selective gain and S06's correction on the untouched lineage; drop hidden continuation and the similarity decomposition |
| Q active reading | STALLED (Q02 closed, Q03 null-sized, Q04 confounded); PROMOTE (Q01 method, Q06 boundary) | INSTRUMENT FAILED (Q02); BOUNDARY ESTABLISHED (Q06) | active reading needs a construction where probes differ by more than 0.05 nats; Q03 and Q04 are promoted mechanically by their criteria and should be read as null and confounded |
| B regimes | PROMOTE (B02, B03, B06); STALLED (B04); BOUNDARY (B05) | MECHANISM CANDIDATE | confirm B02 and B03; B05's readability map is a boundary statement |
| U uptake | PROMOTE (U02, U03, U05, U07) | MECHANISM CANDIDATE | confirm U02; U01's identities hold; U04 and U06 are descriptive |
| R values from opportunities | PROMOTE (R02, R03, R04, R05, R07); STALLED (R06, R08) | MECHANISM CANDIDATE | confirm R02 and R05; the count reader's abstention criterion failed and stays failed |
| T supply ledger | EXHAUSTED for directionality; DESCRIPTIVE for the matrix | DESCRIPTIVE ONLY | no topology claim (equivalence classes reported); the symmetry result closes T05 |
| D many hands | PROMOTE (D02, D03, D05, D06, D07) | MECHANISM CANDIDATE | confirm D03 and D05 |
| F layered intent | BOUNDARY ESTABLISHED (F03); PROMOTE (F04, F05) | BOUNDARY ESTABLISHED | the ruler is validated; the floor is where it is |
| X attacks | PROMOTE the survivors | CONFIRMATORY SUPPORT | eight of twelve survive; the three failures are by construction; X12's far bin is the number to watch on the confirmation lineage |

**STOP READING HERE**

---

## Card ledger

The state of every card, its criterion, its gates and its headline numbers are generated into
[`RESULTS_PACKET.md`](RESULTS_PACKET.md) by `runners/report_v12.py`. Coverage:
`results/v12/COVERAGE.json`; runtime: `results/v12/RUNTIME.json`.

## Intuitive result, then metrics, by trunk

### I. Instruments

- **I01, V11 anchors.** Every committed V11 number reproduces from the V12 primitives at zero
  absolute deviation (S-15: 0.533 → 0.983 identification, L1 0.0093 clean, 0.2493 unbounded,
  0.0096 wrong-expertise, shuffled 0.167; S-14 commissioned and spontaneous separation).
- **I02, randomized severity for S-14/S-15.** Across 50 random world constructions the V11 criteria
  hold in 82% (C1, convergence), 94% (C3, construction B reads a profile from one artifact), 100%
  (C5, commission recovers a missing drive) and 100% (C6, pure compliance collapses to 0.5). Morris
  screening ranks the bimodal profile's mass, signature peak and peaked mass highest; Sobol on those
  three finds total-order indices of 0.75 to 0.90 with first-order indices near zero: what variance
  there is (0.002 on a surface averaging 0.97) is interaction.
- **I03, exact against mean-field.** The two-factor legacy agent's posterior deviates from the
  exact joint by at most 0.012 across the coupling sweep, never confidently wrong.
- **I04, nulls.** No-information and shuffled readers sit at chance (0.176, 0.160 against 0.167).
  Keeping the policy and changing the surface keeps recovery (0.787); keeping the surface and
  changing the policy loses it (0.044). The self prior with two artifacts reaches 0.708, the
  permuted self prior 0.396, the matched generic 0.407.
- **I05, live knobs.** Profile and domain move the emission by JS 0.53 and 0.43, expertise 0.10,
  habit 0.016, regimes 0.008 (surface-matched by construction; bard against concealer 0.018).
- **I06.** Seeds stable, lineages disjoint, outputs unique, no Python-hash seeding, no per-rollout
  file committed under the V12 tree (eleven legacy `points.csv` files predate V12 and are outside it).

### S. Self, similarity, projection, correction

- **S01.** A reader's self-model, measured from its own artifacts, predicts its held-out
  continuations 0.69 [0.61, 0.78] nats better than pooled frequency and 0.018 better than the
  population; it moves by L1 0.10 between surface conventions.
- **S02.** The similarity ruler recovers planted orderings on every axis: Spearman 1.00 (profile),
  1.00 (observation), 0.99 (habit), 0.95 (policy).
- **S03.** Ninety-nine off-ceiling cells exist. There the expert reads goals 0.093 [0.065, 0.128]
  better than the half-corrupted reader and profiles 0.28 nats better. V11's C2 failure was a
  ceiling result, as its record said.
- **S04.** Self-first minus matched generic, log score on the profile after one artifact, by
  quintile of distance from the reader's measured self: +0.600 [0.476, 0.743], −0.088, −0.151,
  −0.202, −0.002. Against the permuted self prior: +0.553 at the nearest fifth, −0.443 at the
  farthest. Hidden next-goal gain: +0.089 near, −0.07 far. Route means at one artifact:
  oracle −0.02, population −1.485, self-first −1.593, permuted −1.602, random-local −1.606,
  generic −1.624; at fifty artifacts every route is within 0.01 of −0.11.
- **S05.** The gain surface over evidence dose: the nearest fifth's gain decays 0.63, 0.53, 0.38,
  0.22, 0.14, 0.07, 0.01 across 1 to 50 artifacts; middle fifths stay negative and shrink; the
  farthest fifth shows a small positive gain at intermediate doses (0.08 to 0.12 at 2 to 8
  artifacts) that the program does not explain. Self-directed error (mass on the reader's own
  profile when it is not the truth) falls 0.247 → 0.011. Calibration error at fifty artifacts is
  0.005 for both routes; asymptotic L1 0.021 against 0.022.
- **S06.** Anchoring after four compatible artifacts 0.048; correction half-life 3.9 artifacts;
  residual self-directed bias 0.064; order effect on the final posterior 0.000.
- **S07.** Hidden continuation: self minus generic −0.007 [−0.014, 0.000] at one artifact,
  +0.006 at four, +0.002 at twelve. Criterion not met.
- **S08.** On the transfer lineage: fresh makers +0.230 near / −0.093 far; dialect convention
  +0.240 / −0.098; anti-similar family −0.105 / −0.094.
- **S09.** Allowed to abstain, at 60% coverage on near makers after one artifact, risk 0.534
  (self-first) against 0.685 (generic); calibration error 0.044 against 0.045.
- **S10.** The measured axes explain 7% of the pair-level gain variance; partial R²: profile
  0.020, observation 0.008, policy 0.004, habit 0.002, regime match 0.000.

### B. Regimes

- **B01.** Entropy and pair-mass gaps across regimes 0.000; population histogram JS 0.013; a
  held-out surface classifier scores 0.307 against a chance of 0.333.
- **B02.** Cooperative assumption on bards +0.79 nats, on concealers −9.9 nats, relative to the
  neutral reader; the matched assumption is best on its own regime at twelve artifacts.
- **B03.** Sticky regime filter recovers a switched regime in 5.1 artifacts on average, 92%
  within twelve; the source-identity reader takes 15.1.
- **B05.** Readability by concealment type (log score on the profile at twelve artifacts, neutral
  reader / concealer-aware reader): deflection −0.40 / −4.25, omission −0.41 / −2.56, mimicry
  −0.55 / −0.02, mixed −0.55 / −1.12, bounded budget −0.47 / −0.64. The concealer-aware reader is
  confidently wrong 37% of the time on deflection and 27% on omission; the neutral reader at most
  4% anywhere. The surface-entropy tell for omission is 0.013 nats.
- **B06.** Own payoff on a next-move task: accurate model 0.727, fooled model 0.638, no model
  0.500. The adoption arm is uninformative as constructed (uniform reader preference).

### U. Uptake

- **U01.** Zero weight bit-identical; the oracle posterior optimal on the softmax objective;
  the uniform posterior equals the population update; a shuffled posterior never beats the oracle.
- **U02.** Wrong-direction movement: wrong-and-confident 0.556, accurate cells at most 0.231.
- **U03.** Uncertainty-aware representations trade regret for direction: MAP regret 0.257 with
  wrong direction 0.304; confidence-gated regret 0.380 with wrong direction 0.237; oracle 0.037 / 0.
- **U04.** Linear effects on movement: reliability +0.62, similarity −0.61, competence +0.04,
  relevance +0.00, accuracy +0.03 (2,304 rows).
- **U05.** Channels after exposure: belief +0.49, preference +0.19, novel-constraint action 0.32;
  process and imitation are surface-overlap statistics on this construction.
- **U06.** A competent value-divergent maker is predicted at 0.89 and raises own-task regret by
  0.63 under unconditional uptake; a concealer with correct technique is predicted at 0.47 and
  costs 0.64; a false context is predicted at 0.00 and costs 0.59.
- **U07.** Reliable counter-evidence reverses harmful movement 85% of the time; posterior on the
  truth 0.62 afterwards, 0.25 left on the false source.
- **U08.** Cumulative movement reaches 0.60 under early reliable context, 0.62 under late, 0.61
  under intermittent conflict; at ten steps 0.47, 0.18, 0.32.

### R. Values from opportunities

- **R01.** Every latent keeps its full conditional entropy (orthogonal design) and gains
  information from records: profile 0.57, habit 0.73, expertise 0.38, goal 0.58 nats.
- **R02.** Held-out choice log score: constrained inversion −0.87 [−0.93, −0.83], partialling
  −1.04, MAP-without-habit −1.30, cost-blind −1.29, count reader −1.30, population −1.29.
- **R03.** Aligned habit: partialling loses 1.49 nats, constrained inversion gains 0.21.
- **R04.** Second domain: profile −1.16, identity-only −2.05, oracle −0.55.
- **R05.** Posterior shift (KL): record reader 0.12 near tie / 0.97 strong; count reader 0.16 / 0.43.
- **R06.** Joint reader profile 0.51, goal 0.71; single-latent readers 0.48, 0.74. Not met.
- **R07.** Gain over frequency: changed cost 0.80, new domain 0.45, commission 0.42, new goal 0.33.
- **R08.** On agreeing menus the count reader abstains 59% (bar 80%), mean posterior 0.47; with
  discriminating menus the record reader reaches 0.98. The abstention half fails.

### T. Supply ledger

- **T01.** Neutral regime: I(profile; goal) 0.324, I(profile; slot) 0, I(goal; slot) 0;
  I(goal; surface) reaches 1.38 of a possible 1.39 by 8 steps; I(profile; surface) saturates at
  0.327, which is I(profile; goal): the surface speaks about the profile only through the one goal
  it carries. Bard regime: slot determined by profile (flagged).
- **T02.** At CURATOR alpha, four steps: goal supplied → profile +0.086; profile supplied →
  goal +0.085; slot supplied → 0.00 either way; wrong goal supplied → profile −0.71; shuffled
  never helps.
- **T03.** Profile → process 0.002; process → goal 0.014 (157 off-ceiling cells).
- **T04.** Correct mechanic against generic: goal +0.23, slot +0.16, profile +0.02; related and
  wrong mechanics ordered between.
- **T05.** Directionality 0.000 in all 57 off-ceiling cells and in all cells: symmetric by identity.
- **T06.** Neutral regime: observe equals intervene (do-gain 0.0023 against observe-gain 0.0023);
  the isolated-slot class in ten worlds, the {chain, flat, isolated} class in two. Bard: observe
  1.57 nats against intervene 0.0003; the {common cause, river, triangle} class in all twelve.

### D. Many hands

- **D01.** Six ecologies matched: quality 0.995 to 0.997, style JS 0.000, surface classifier 0.097
  against a chance of 0.167.
- **D02.** Reach: director goal 1.00, brief 1.00, secondary goal 0.25, local slot 0.25,
  ratification 0.25 (surface reach of the local change 0.18).
- **D03.** Structure reader 0.81; coherence baseline 0.50 (matched by construction).
- **D04.** Director-level attribution 0.92, local 0.51 (chance 0.25), token share 0.27.
- **D05.** Director attribution 0.94 → 0.92 across the rewrite ladder; local 0.49 → 0.26.
- **D06.** Next-intervention log score −1.18 against −1.24 frequency; gain 0.058 [0.014, 0.102].
- **D07.** Abstention 0.97 on identical artifacts; record reader 1.00; later separation 0.87.

### F. Layered intent

- **F01.** Bit difference 0.83; manipulation-off equals flattened exactly.
- **F02.** Upstream and downstream marginals and surface histograms match to JS ≤ 0.004; planted
  MI 0.86 layered against 0.01 flattened.
- **F03.** Dependency ruler AUC 0.51 (4×12), 0.71 (32×12), 0.96 (32×60), 1.00 (128×60); hard
  argmax MI 0.50, 0.71, 0.92, 1.00; sequence bigram at most 0.61; histogram entropy at chance;
  every ruler at chance on null pairs (0.48 to 0.50).
- **F04.** Three-way accuracy 0.95 (1,080 sequences).
- **F05.** Director attribution 1.00 clean, 1.00 under local flattening, 0.92 at half rewrite,
  0.52 at full rewrite, 0.83 under a template change.

### Q. Active reading (re-run under D-V12-1)

- **Q01.** The PyMDP agent's chosen probe is within 0.02 nats of the best probe under its own
  one-step mixture model on 98% of decisions and under the artifact-structured model on 89%
  (criterion 80%). The two models share a best probe on only 68% of decisions; the utility-only
  rival agrees with the structured ranking on 39%. The structured EIG spread across probes is
  0.053 nats, which is the number that governs the rest of the trunk.
- **Q02, closed as an instrument failure.** With the structured likelihood, an
  information-chosen commission and a free look carry the same profile information: 0.190
  against 0.189 nats per probe; information-chosen against random +0.013 [−0.008, 0.036]
  (criterion 0.05). Final log score after four probes: information-chosen −1.03, PyMDP −1.08,
  random −1.03, uncertainty sampling −1.14, free look −1.11. A costly commission is never chosen
  while a free look is available. The first attempt, at ceiling under the mixture likelihood
  (0.42 nats per probe, log scores near −6.5), is kept as `Q02.attempt1.json`.
- **Q03.** Self-first selection of two of six episodes against uncertainty sampling, scored on
  the seventh episode's goal under the same prior: +0.016 [−0.001, 0.035] nats. The oracle pair
  scores −0.87 against −1.27 for a random pair, so 0.40 nats of selection value exist and
  self-first captures 0.02 of them. The criterion (at least zero) is met; the size is a null.
- **Q04.** Per unit cost the most discriminating item is the biography (a label right 70% of the
  time): the exact buyer buys it every time (0.45 nats per purchase), the PyMDP buyer 84% of the
  time with reputation, worth nothing, the other 16%; the polish rival buys reputation every time.
  The criterion is met, and the design confounds the answer: the winning item carries polish 0.9,
  so discrimination and polish are told apart only by the reputation item.
- **Q05.** Exact stopping at zero cost takes all eight probes (regret against hindsight 0.33
  nats; wrong top answer at the stop 30%; 0.96 probes per episode taken after the answer was
  already confident). At cost 0.01 it takes 7.7 probes and PyMDP 0.9 (its cost enters at ten
  times the log-score scale); at cost 0.3 it takes 0.55 probes and stops wrong 69% of the time.
  Regret runs 0.26 to 0.43 nats across the frontier.
- **Q06, boundary.** Against a concealer planting its decoy's commissioned behaviour, the naive
  reader ends 3.8 nats from the truth whatever it probes (information-chosen −3.77, uncertainty
  −4.05, random −3.85); the reader that models the planting ends within 0.4 (−0.32, −0.25,
  −0.38). Choosing probes is worth 0.28 nats to the naive reader and −0.07 to the aware one;
  modelling the planting is worth 3.4. The criterion (some active gain above zero) is met on the
  naive side only.

### B04 (re-run under D-V12-1)

- Regime log score after four challenges: chosen for information about the regime −0.71
  [−0.86, −0.58], random −0.77, uncertainty sampling −0.88. Criterion (aware at least uncertainty)
  met; the margin over random challenges is 0.05 with overlapping intervals. The PyMDP agent
  picks the exact regime-EIG probe 58% of the time. The first attempt, below the uniform prior
  under the mixture likelihood, is kept as `B04.attempt1.json`.

### X. The adversarial matrix, applied to S04's selective gain

Near-fifth and far-fifth gain (self-first minus matched generic, one artifact) under no attack on
the discovery worlds: 0.594 and 0.027. Survival rule: near at least 0.05 and far at most 0.05.

| attack | near | far | reading |
|---|---|---|---|
| X01 source labels shuffled | 0.594 | 0.027 | bit-identical (identity statistic 0) |
| X02 dialect surface convention | 0.594 | 0.027 | bit-identical: the exact reader knows the convention |
| X03 false biography on half the makers | 0.703 | −0.062 | survives; the far bin worsens |
| X04 equifinal twin | 0.153 | 0.168 | the evidence's log-odds between truth and twin is route-independent (identity 2e-14); the selective pattern dissolves into an average over the two labels, as it must |
| X05 permuted self prior | −0.004 | 0.367 | the near gain is gone; what remains lands on far makers by accident |
| X06 evidence order reversed | 0.577 | 0.038 | the two-artifact posterior is order-independent (identity 0); the first-evidence gain moves by 0.02 |
| X07 six fresh random worlds | 0.662 | 0.025 | survives |
| X08 PyMDP mixture reader in place of the exact posterior | 0.207 | −0.076 | a third of the near gain survives the solver; the average over all makers turns negative (−0.12) |
| X09 nearest-centroid histogram reader | 0.380 | −0.049 | two thirds survive a cheap likelihood |
| X10 half the labels wrong | 0.153 | 0.168 | diluted to the same average as X04 |
| X11 makers shaping their goals toward the reader | 0.592 | −0.371 | the near gain is untouched; far makers who imitate the reader cost it 0.37 nats |
| X12 six fresh transfer worlds | 0.671 | 0.080 | survives on the near bin; the far bin sits 0.03 above the rule |

Eight of twelve survive. The three that fail (X04, X05, X10) fail by construction: they remove
the correspondence the gain is made of, or the labels it is scored against. X12's far bin is the
number to watch on the confirmation lineage.

## Controls, repairs, and failures

- **The one-world smoke pass** (every card once on world 0, verdicts redirected to scratch) found
  nine harness defects and five wrong known-answers before the queue started: import depth,
  the sensitivity wrapper's dict interface, a leave-one-out centroid classifier that sits below
  chance by construction (B01, D01), reach measured on feature bits rather than decisions (D02),
  the X attacks seeding their population by attack name (placebo not bit-identical), a fixed-regime
  reader that can in fact flip (B03), a softmax policy scored on regret rather than on the objective
  it maximizes (U01), and the S08 anti-similar family built as a relabelled population rather than
  a per-reader decoy family.
- **T-trunk ceiling.** At the CREATOR tier with eight steps the goal is at ceiling and every
  goal-related supply cell is zero by construction; the battery runs at CURATOR alpha with four
  steps, T03 and T05 sweep tiers, and only off-ceiling cells are read.
- **F03 ladder.** The ruler is validated where blocks are readable before it is read at the floor.
- **Deviation D-V12-1.** The first discovery run scored the Q trunk's and B04's "exact" path with a
  per-feature mixture likelihood, which compounds twenty-four near-identical draws into
  overconfidence: Q02 sat at ceiling (0.005 nats of commission effect against a 0.01 floor, final
  log scores near −6.5) and B04's regime posterior fell below the uniform prior. Both were re-run
  on the artifact-structured likelihood; the first verdicts are kept as `<ID>.attempt1.json`.
- **X survival rule.** The summary flag was tightened from "far gain ≤ 0" to "far gain ≤ 0.05",
  because the far bin sits at 0.03 ± 0.03 under no attack; the X cards were re-run for the flag
  only (numbers unchanged by construction).
- **Gate failures that stand.** Q02's live gate (commissions change information) failed on both
  attempts (0.005 and 0.001 nats against a 0.01 floor); the card is closed, not landed. None
  among landed cards.

## Multiplicity and power

Twelve discovery worlds per card, three seeds where seeds are a factor, sixty makers per world;
hierarchical bootstrap over worlds for every headline interval. Seventy-four cards were run with
no correction across them: the protection is pre-registration of each card's criterion, the
separation of the discovery and confirmation lineages, and the practice of reporting every
criterion that failed (S07, R06, R08, and whichever of the re-run cards fail) beside those that
passed. Many criteria are direction-only and would not survive a size-based reading.

## Runtime and coverage

Discovery lane: 74 cards in 51.7 minutes wall on one thread at below-normal priority, plus the
nineteen-card re-run under D-V12-1 in 18.6 minutes. Coverage: 62/62 mandatory cards resolved
(`results/v12/COVERAGE.json`).

## File paths and hashes

Program lock: `results/v12/prereg_v12_lock.json` (hashes the manifest's cards, the criteria and
the lock module). Every verdict carries the SHA-256 of the module that produced it, the git commit,
and the dirty flag under `produced_by`. Manifest: `results/v12/QUEUE_MANIFEST.json`.

## Frozen confirmation cards

Promotion is mechanical: a card is promoted when its discovery verdict landed, every criterion it
carries passed, and its ceiling is method or constructed mechanism. Twenty-two cards were promoted
and run on the untouched lineage (worlds 100–111, 20.9 minutes wall; verdicts under
`results/validation/soundingline/v12/confirmation/`, ledger `results/v12/CONFIRMATION.json`).
Twenty-one held their criteria; one did not.

- **Held.** S04 (nearest-fifth gain 0.54 against 0.60 in discovery; middle fifths −0.07 to
  −0.12; farthest −0.03; the population prior again beats the self-first average, −1.54 against
  −1.64), S06 (anchoring 0.03, half-life 3.5 artifacts, residual bias 0.08, order effect 0.000),
  S09 (risk 0.53 against 0.68), Q01, Q03 (+0.008 [−0.009, 0.025]: the null holds), Q04, B02
  (+0.84 on bards, −6.1 on concealers), B03 (6.6 artifacts, 88% within twelve, identity-keyed
  reader 15.8), B06 (0.70 against 0.62), U02 (0.54 against 0.24), U03, U05, U07, R02 (−0.84
  against −1.00 for partialling and −1.32 for the floor readers), R03, R04, R05 (0.11 near tie
  against 0.99 strong), R07 (0.82 / 0.44 / 0.44 / 0.33), D02, D06 (0.083 [0.027, 0.142]), F03
  (floor 0.53, readable cells 0.99).
- **Did not hold.** B04. On the untouched lineage the regime-aware challenge policy scores −0.93
  against −0.84 for uncertainty sampling and −0.88 for random challenges; the discovery-lane
  margin of 0.16 nats was noise at eighteen episodes per world. B04 moves to STALLED, and the
  active-reading conclusion stands without its exception.

Cards that were not promoted (S07, R06, R08, Q02, the descriptive T cards) were not run on the
confirmation lineage; their discovery-lane readings are the record.
