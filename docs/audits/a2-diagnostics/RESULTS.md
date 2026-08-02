# Diagnostics on the apparatus

Generated from the verdict files in [results/diagnostics/](../../../results/diagnostics/). Regenerate with `python scripts/write_diagnostics_md.py`.

---

## What this is, and what it is not

**Nothing here asks a question about the world and nothing here fixes anything.** Every check is a diagnostic on the instruments: what are they actually measuring, and over what range can they measure it at all. They exist to decide what the repair work should be, and starting the repair before they report is the mistake the pass was set up to avoid.

Two of them, P-1 and P-2, come from the diagnostics specification. Five more came out of reading the code alongside the validation pass's output, and they run first because two of them change what the expensive sweeps should sweep.

The distinction that matters most for reading what follows: **the validation pass asked whether the recorded answers can be trusted. This pass asks whether the instruments can answer at all.** A finding here does not usually mean a result was wrong. It usually means a result was stated more strongly than the instrument supports.

**The criteria were fixed before any check ran** and hash-locked at `c07837a57450e334`, in [results/diagnostics/criteria.json](../../../results/diagnostics/criteria.json), separate from the validation lock, which has reported and is sealed. Two decisions are recorded in that lock and were made before the sweeps: the estimators P-1 uses for the three parameters that are not hidden states, and P-2's fourth knob.

---

## The eight checks at a glance

| check | what it asks | how it came back |
|---|---|---|
| **P-1** | If the model generates data at a known parameter value, can the value be recovered? | at least one load bearing parameter is unidentifiable |
| **P-2** | Is there a regime where reading the goal is genuinely uncertain? | regime found |
| **D-1** | What is the reader's belief about provenance made of, and which channel wins? | label wins at the default with margin |
| **D-2** | Does uptake rise and fall with how well the reader read the work? | uptake is non monotone in recovery |
| **D-3** | Is 'no two readers agree' measuring disagreement, or shared uncertainty? | disagreement number is not identified on its own |
| **D-4** | How much of the work can the exact solver reach, and how wrong is the shortcut? | coverage gap and the shortcut drifts |
| **D-5** | How many independent things does each pre-registered criterion see? | some criteria cannot separate their own outcomes |
| **D-6** | Are the simulated readers actually distinct? | collisions are cross cell only and benign in direction |

---

## P-1: parameter recovery

Before trusting a model you make it produce data with a setting you chose, then try to read the setting back out of the data. If you cannot, the setting is not measurable, and any finding that depends on it is describing an arbitrary choice rather than anything about the world. This is a routine first check and this project had never run it.

**The specification's estimator only names an estimator for one of the four parameters, and that is worth stating before any number.** Depth is a hidden state, so the reader carries a posterior over it and the posterior mean is an estimate. Trust, readability and the value gate are not hidden states: no agent in the model carries a belief about any of them. So they are fitted instead, by maximising the likelihood of a fixed dataset over a grid, which required writing a capability the project did not have ([ghostscale/fitting.py](../../../ghostscale/fitting.py)).

That forces a distinction which is not a technicality. For depth, recovery asks whether **the reader inside the model** can identify the value. For the other three it asks whether **an ideal analyst holding the correct model** could, which is a strictly easier question. For readability the gap between those two is the entire version 4 reframe.

| parameter | estimator | rank correlation | slope | usable range | classification |
|---|---|---|---|---|---|
| kappa (honest labels throughout) | maximum exact sequence log-evidence over a grid, on a fixed observation tape | not defined | -3.99e-16 | 0% | **UNIDENTIFIABLE** |
| kappa (machine work passed off as human) | maximum exact sequence log-evidence over a grid, on a fixed observation tape | not defined | -3.99e-16 | 0% | **UNIDENTIFIABLE** |
| omega | maximum likelihood under an ANALYST'S model containing omega, which the observer's model does not | 1.00 | 1.00 | 100% | **RECOVERED** |
| mu | posterior mean, read off the reader's own belief | 1.00 | 0.32 | 50% | **COMPRESSED** |
| theta (lambda) | matched to the observed prior drift across a sequence of encounters, because theta enters no observation likelihood at all | 1.00 | 0.95 | 88% | **RECOVERED** |

Four parameters, three estimators, and four different answers.

**kappa, honest labels throughout: UNIDENTIFIABLE.** Rank correlation not defined, slope -0.00, usable range 0% of the swept span. Reading: the estimate is CONSTANT across the whole grid, so there is no ordering to correlate. The likelihood is monotone in the parameter rather than peaked, and the estimate runs to whichever end of the grid the data pushes it. No dataset this model can generate locates this parameter.

**kappa, machine work passed off as human: UNIDENTIFIABLE.** Rank correlation not defined, slope -0.00, usable range 0% of the swept span. Reading: the estimate is CONSTANT across the whole grid, so there is no ordering to correlate. The likelihood is monotone in the parameter rather than peaked, and the estimate runs to whichever end of the grid the data pushes it. No dataset this model can generate locates this parameter.

**omega: RECOVERED.** Rank correlation 1.00, slope 1.00, usable range 100% of the swept span. Reading: ordering, magnitude and range all hold.

**mu: COMPRESSED.** Rank correlation 1.00, slope 0.32, usable range 50% of the swept span. Reading: the ordering survives perfectly and the magnitude does not: a slope of 0.32 means a true change is reported as roughly a third of itself. Directions transfer, magnitudes do not, which is the same conclusion the independent rebuild reached about the label effect by a different route.

**theta (lambda): RECOVERED.** Rank correlation 1.00, slope 0.95, usable range 88% of the swept span. Reading: ordering, magnitude and range all hold.

A parameter classified unidentifiable cannot support a claim. That applies to kappa (honest labels throughout), kappa (machine work passed off as human), and the claims resting on it have to be restated.

The cheap pre-check the spec asks for is necessary and NOT sufficient, and this pass establishes that. Trust passes it comfortably, its likelihood is fully visible in the goal marginal, and it is still not recoverable. The reason is invisible to the pre-check: the likelihood is MONOTONE in trust rather than peaked, because a sharper label channel makes whatever signal arrived more probable under some provenance and the reader is free to move its provenance belief to accommodate it. So the estimate runs to whichever end of the grid the data pushes it, and it runs to opposite ends on the two datasets, which is the signature.

One of the two pairs the spec names cannot be run at all: (mu, effort), because version 5 replaced it as a hidden state with depth. So the joint recovery of the pair cannot be computed at all. What CAN be said about their entanglement is what the construction audit already says: with the effort axis pinned at its maximum, depth still separates by 0.91, so the depth estimate is not the effort knob renamed.

### The cheap pre-check, and what it does not predict

| parameter | quantity | change across the swept range | invariant |
|---|---|---|---|
| kappa | mean over goals of P(signal | provenance, goal) | 0.640 | no |
| omega | mean over foreign goals of P(feature | foreign goal, omega) | 0.144 | no |
| mu | two marginals, because they answer different questions: mean over SUB-GOALS (a design invariant, meant to be zero) and mean over GOALS (the measurement) | 0.111 | no |
| theta | any likelihood | 0 | yes |

**kappa.** A[1] is constant in the goal, so marginalising over goals changes nothing and the parameter is fully visible in the marginal.

**omega.** omega interpolates the whole family toward the human one, so the goal-marginal moves with it and the parameter is visible in the marginal.

**mu.** The sub-goal marginal is invariant to 5.6e-17, which is the design working rather than failing: the mode family averages exactly to the version 4 goal signature and the chains are doubly stochastic, so a deep artifact and a shallow one have identical time-averaged feature histograms and depth lives in the ORDER. The GOAL marginal is NOT invariant, at 0.111, and that is the difference between depth and the rationality parameter it replaced: version 4.5's beta was zero here by construction, which left finite-sample luck in the coupling as its only evidence channel and produced compression at both ends. Depth has a real channel through the sub-goal, which persists in time, so evidence accumulates across a block.

**theta.** theta appears in no observation likelihood whatsoever. It gates what the observer carries forward between encounters, so it is invisible within a single artifact by construction and can only be seen in the trajectory of the prior across a sequence.

### The joint check

| pair | runnable | error correlation | trading off |
|---|---|---|---|
| (kappa, omega) | yes | 0.11 | no |
| (mu, effort) | no | not defined | — |

**(kappa, omega).** a strong correlation between the two recovery errors would mean neither is separately identifiable whatever the single-parameter sweeps say

**(mu, effort).** NOT RUNNABLE, and the reason is the finding. Effort has no estimator anywhere in this model: no agent carries a posterior over it and it enters no likelihood an analyst could profile, because version 5 replaced it as a hidden state with depth. So the joint recovery of the pair cannot be computed at all. What CAN be said about their entanglement is what the construction audit already says: with the effort axis pinned at its maximum, depth still separates by 0.91, so the depth estimate is not the effort knob renamed.

---

## P-2: the goal-difficulty probe

In almost every experiment the simulated reader works out the maker's purpose perfectly. That sounds good and it is actually a problem: if the answer is always right there is no room for anything downstream to vary, so two experiments came back uninterpretable. This looks for a setting where the reader is genuinely unsure, and then checks whether the thing those experiments were trying to measure can move there.

*the spec names three knobs; two were measured dead before the criteria were locked and are run here as single confirmatory cells, and a fourth, reader inexpertise, was added and made primary. Both decisions are in the lock.*

| setting | reads the right purpose | uncertainty about the goal | uptake spread | against the default | disagreement |
|---|---|---|---|---|---|
| d=0.85, T=6 | 0.633 | 0.482 | 0.842 | 1.89x | 1.041 |
| d=0.8, T=6 | 0.761 | 0.418 | 0.820 | 1.84x | 0.787 |
| d=0.7, T=3 | 0.839 | 0.585 | 0.795 | 1.78x | 0.603 |
| d=0.87 | 0.611 | 0.316 | 0.792 | 1.78x | 1.073 |
| d=0.85, T=12 | 0.661 | 0.259 | 0.787 | 1.77x | 0.990 |
| d=0.85 | 0.664 | 0.295 | 0.779 | 1.75x | 0.983 |
| d=0.8, T=3 | 0.683 | 0.718 | 0.775 | 1.74x | 0.957 |
| d=0.82 | 0.719 | 0.261 | 0.764 | 1.71x | 0.877 |
| d=0.8 | 0.767 | 0.236 | 0.758 | 1.70x | 0.778 |
| d=0.85, T=3 | 0.575 | 0.760 | 0.755 | 1.69x | 1.137 |
| d=0.8, T=12 | 0.769 | 0.200 | 0.751 | 1.68x | 0.781 |

**The two knobs the specification names and this pass ran only as confirmatory cells:**

- signature separation at its extreme leaves accuracy at 1.000 and goal uncertainty at 0.003
- goal count doubled leaves accuracy at 1.000 and goal uncertainty at 1.47e-06

A regime exists. **d=0.85, T=6** puts goal accuracy at 0.633, inside the 0.55 to 0.85 band committed before the run, and uptake varies 1.89 times as much across readers as it does at the default. 11 of the 11 cells in the band clear the variance requirement.

The knob that gets there is reader inexpertise, which the spec does not list. The two knobs it does list and this pass ran as confirmatory cells are dead as predicted: signature separation at its extreme leaves accuracy at 1.000; goal count doubled leaves accuracy at 1.000. Separation and goal count change how fast the reader arrives at certainty, not whether it arrives, because every deep look is an independent draw and any non-zero evidence accumulates.

What makes inexpertise different is that it is a SYSTEMATIC error in the reader's own templates rather than a sampling one, so it does not average out however long the run. That is also the model's own account of why a mis-aimed template fails silently, which means the difficulty axis and the two-dimensions result are the same mechanism seen from two directions.

---

## D-1: channel accounting

The reader is told who made the thing, and it also looks at the thing. Both of those are evidence about who made it, both arrive at every glance, and on a lie they disagree. This works out how strong each one is per glance, and therefore which one wins. It needs no simulation at all, which is why it runs first: the answer predicts how several later sweeps will come out.

*expected per-observation log-likelihood ratio for each channel, computed on the likelihood arrays. No simulation. See the module docstring for the one approximation: these are drifts, not trajectories.*

| cell | transparency of the truth | content, per glance | label, per glance | net | crossover trust |
|---|---|---|---|---|---|
| machine work passed off as human | 0.05 | -1.920 | 3.829 | 1.909 | 0.538 |
| machine work labelled honestly | 0.05 | 0 | 0 | 0 | not defined |
| human work falsely labelled machine | 1.00 | -1.491 | 3.829 | 2.337 | 0.408 |
| curated work passed off as fully human | 0.60 | -0.490 | 3.829 | 3.338 | 0.112 |

The reader's belief about who made something is fed by two streams that both arrive every timestep and, on machine work passed off as human, point in opposite directions. The content argues for the truth at -1.920 nats per observation. The label argues for the lie at +3.829 nats per observation at the default trust of 0.9. The label wins, with a margin of +1.909 nats per step, and it wins at any run length.

**The crossover sits at trust = 0.538.** Below it the content overtakes the label as the run lengthens and the reader ends up believing the truth, so the trust exploit is not a fact about labels in general but a fact about labels trusted more than 0.54. That is a narrower claim than the one currently made and it is the accurate one. It also explains, without any new simulation, why the validation pass's robustness matrix found the label effect weakening five-fold at trust 0.30: 0.30 is below the crossover.

**And the arithmetic is checked against measured runs.** Each row is one reader looking at the same mislabelled artifact, at a different level of trust, and the column is how much it believes the label after that many glances.

| trust | 1 glance | 2 | 4 | 10 | all | ends up believing the label |
|---|---|---|---|---|---|---|
| 0.10 | 0.229 | 0.250 | 0.075 | 0.001 | 1.56e-25 | no |
| 0.30 | 0.376 | 0.577 | 0.574 | 0.573 | 2.57e-13 | no |
| 0.50 | 0.535 | 0.832 | 0.947 | 0.999 | 0.042 | no |
| 0.70 | 0.708 | 0.957 | 0.997 | 1.000 | 1.000 | yes |
| 0.90 | 0.898 | 0.997 | 1.000 | 1.000 | 1.000 | yes |

- Every claim phrased as 'a provenance label does X' is really 'a provenance label trusted above 0.54 does X at this content transparency'. The threshold is a property of the two alphas and the signal cardinality, not of the theory.
- The run length is load-bearing wherever the margin is small. It was chosen as an engineering parameter and it should be reported alongside any claim from a near-crossover cell.
- The robustness matrix's kappa row is explained: the weakening at low trust is the crossover, and it could have been predicted from the construction.

---

## D-2: the shape of the uptake measure

Several experiments measure how much a reader 'takes on' from a work, and treat it as going up when the reader understands more. The measure is actually a distance between the reader's belief before and after, and a reader who becomes confidently WRONG has also moved a long way. So the relationship may be U-shaped rather than a slope, and if it is, an experiment whose two conditions sit either side of the bottom of the U finds nothing for reasons unconnected to what it was testing.

| reader inexpertise | reads the right purpose | uptake | spread | share confidently wrong | uptake if wrong | uptake if right |
|---|---|---|---|---|---|---|
| 0 | 1.000 | 3.248 | 0.428 | 0% | not defined | 3.248 |
| 0.10 | 1.000 | 3.248 | 0.428 | 0% | not defined | 3.248 |
| 0.20 | 1.000 | 3.248 | 0.428 | 0% | not defined | 3.248 |
| 0.30 | 1.000 | 3.247 | 0.427 | 0% | not defined | 3.247 |
| 0.40 | 1.000 | 3.243 | 0.429 | 0% | not defined | 3.243 |
| 0.50 | 0.998 | 3.225 | 0.440 | 0% | not defined | 3.228 |
| 0.60 | 0.990 | 3.144 | 0.488 | 0% | 2.459 | 3.156 |
| 0.70 | 0.915 | 2.916 | 0.658 | 2% | 2.614 | 3.001 |
| 0.80 | 0.758 | 2.658 | 0.742 | 14% | 2.730 | 2.785 |
| 0.85 | 0.627 | 2.565 | 0.742 | 24% | 2.774 | 2.687 |
| 0.90 | 0.485 | 2.500 | 0.771 | 34% | 2.876 | 2.557 |
| 0.95 | 0.331 | 2.529 | 0.775 | 47% | 2.895 | 2.573 |
| 1.00 | 0.233 | 2.621 | 0.755 | 58% | 2.971 | 2.504 |

**Uptake is not monotone in how well the reader recovers the goal.** It falls from 3.248 at full expertise to a minimum of 2.500 at inexpertise 0.90, where accuracy is 0.485, and then rises again to 2.621 at zero expertise where accuracy is 0.233. The dip is 16% of the measure's whole range and its minimum is in the interior.

The mechanism is visible in the definition rather than mysterious. Uptake is the distance between what the reader ended up believing and what it started believing. A reader who gets it right ends far from its prior. A reader who gets nothing ends AT its prior. A reader who becomes confidently WRONG also ends far from its prior. Measured here, the confidently wrong readers score 2.654 against 3.054 for the correct ones, which is 87% of the correct readers' uptake for a belief that is false.

**This is a better explanation of the flat depth result than the one on record.** That experiment regressed uptake on depth level, found it flat, and reported its own construction at fault for leaving no headroom. A non-monotone response whose arms straddle the minimum produces the same flat regression, and the two have different repairs: one needs a construction change and the other needs the arms moved to the same side of the trough. The rank correlation between uptake and accuracy across this sweep is 0.93, which is what a flat regression on a U looks like.

The practical consequence for the repair P-2 sets up: a rerun must not sit at the trough. That is inexpertise 0.90 here, and it is close to the middle of the band P-2 identifies as the difficulty regime, so the two constraints pull against each other and the rerun needs both on the table at once.

### The trust factor inside the measure

uptake is defined as this factor times a belief distance, so a sweep that varies trust and reports uptake is reporting the product. Over the range E4 sweeps, the factor alone changes by more than a factor of forty, which is larger than most effects in the project. Across the swept range the factor alone changes by 43.7x.

*uptake is gated on engagement, so a disengaged reader contributes exactly zero and the response has a hard discontinuity in it as well as a trough. The ungated series is reported beside the gated one; the gap between them is the gate.*

---

## D-3: the disagreement statistic

The project's disagreement number counts each reader's single best guess and measures how spread out those guesses are. That works when readers are sure of themselves and differ. It breaks when readers are all equally unsure, because then their best guess is close to a coin toss and the coin tosses scatter on their own. This check builds the coin-toss version of each population, using each reader's actual uncertainty, and asks whether the real number is any bigger.

| cell | uncertainty per reader | modal-goal entropy (shipped) | pairwise divergence (alternative) | vote resample | null degenerate |
|---|---|---|---|---|---|
| E2 machine work passed off as human | 0.090 | 1.379 | 0.498 | 1.379 | yes |
| E2 machine work labelled honestly | 1.293 | 1.378 | 0.044 | 1.379 | no |
| E2 human work read correctly | 7.06e-08 | 0 | 3.63e-09 | 0 | yes |
| E20/E32 fully foreign content (omega = 0) | 1.234 | 1.381 | 0.074 | 1.378 | no |
| E20 partial overlap (omega = 0.10) | 0.486 | 0.662 | 0.104 | 0.746 | no |
| E32 unskilled reader on human work (d = 0.945) | 0.853 | 1.329 | 0.234 | 1.350 | no |

The disagreement statistic cannot be read on its own, and this is a fact about the statistic rather than about any result.

It counts each reader's single best guess and measures how spread out the guesses are. Two completely different populations produce the same spread: readers who are each certain of a DIFFERENT answer, and readers who are all equally unsure and therefore guessing. Both give the same count vector and so the same entropy. The vote-resampling null confirms it from both directions at once, reproducing the observed value in 2 cells because peaked posteriors resample to themselves and in 3 because flat ones resample to noise.

Measured directly: 41% of the variation in the between-reader number across the checked cells is explained by a straight line on the mean WITHIN-reader number. It is close to a restatement of how unsure the readers are.

**This does not invalidate the headline**, because the headline is always the conjunction, and the fabrication index multiplies confidence by disagreement rather than reporting either alone. What it does invalidate is quoting the disagreement figure by itself as a second independent piece of evidence, which is how it reads in the results files and in the README.

**The fix is a better statistic and it is one line.** Mean pairwise Jensen-Shannon divergence between readers' full posteriors is near zero when everyone is equally unsure, because their beliefs are then nearly identical, and large only when readers are confident about different things. It separates 3 pair(s) of cells the shipped statistic reports as indistinguishable: the cells "E2 machine work passed off as human" and "E2 machine work labelled honestly" both score about 1.38 on modal-goal entropy while scoring 0.498 against 0.044 on pairwise divergence; the cells "E2 machine work passed off as human" and "E20/E32 fully foreign content (omega = 0)" both score about 1.38 on modal-goal entropy while scoring 0.498 against 0.074 on pairwise divergence; the cells "E2 machine work labelled honestly" and "E32 unskilled reader on human work (d = 0.945)" both score about 1.38 on modal-goal entropy while scoring 0.044 against 0.234 on pairwise divergence.

Separately, the estimator is downward biased by about (K-1)/2N nats, so numbers reported at validation V-2b per random draw are not on the same scale as those reported at 4000 readers. The bias is analytic, confirmed empirically here, and correctable in one line; the correction is offered as a separate function rather than substituted, because swapping the estimator would change what every committed number means without changing the files.

### How much of the disagreement number is the uncertainty number

| relationship | rank correlation | variance explained by a straight line |
|---|---|---|
| shipped statistic against within-reader uncertainty | 0.54 | 41% |
| alternative statistic against within-reader uncertainty | -0.09 | 13% |

The alternative is the less redundant of the two, which is what makes it worth reporting: it carries information the uncertainty number does not already contain.

**Recommendation.** mean pairwise Jensen-Shannon divergence over the full posteriors, reported beside the modal-goal entropy rather than instead of it, so the existing numbers stay readable.

### The reader-count bias

| reported at | readers | bias, nats | scale sensitive |
|---|---|---|---|
| E2 (version 1 headline) | 4000 | -3.75e-04 | no |
| E17 | 4000 | -3.75e-04 | no |
| E19 | 200 | -0.007 | no |
| E20 | 200 | -0.007 | no |
| E21 | 200 | -0.007 | no |
| E32 | 200 | -0.007 | no |
| E31 | 60 | -0.025 | no |
| validation pass reduced scale | 60 | -0.025 | no |
| validation V-2b per random draw | 15 | -0.100 | yes |

---

## D-4: solver coverage

The validation pass rechecked five experiments with the arithmetic done exactly rather than approximately, and three conclusions changed. This asks the obvious follow-up: how many of the others could even be rechecked, and for the ones that could not, is the approximation drifting a little or a lot?

| | count | which |
|---|---|---|
| checked under exact inference by the validation pass | 5 | E2, E19, E20, E31, E32 |
| made reachable by this pass | 2 | E1, E5 |
| still structurally unreachable | 9 | E10, E11, E13, E16, E18, E6b, E7, E8, E9 |
| reachable but never checked | 14 | E12, E14, E15, E17, E21, E28, E29, E3, E30, E33, E34, E4, E6, N21 |

The validation pass checked 5 of 30 experiments under exact inference, and three of its verdicts moved. 11 experiments were structurally unreachable; this pass makes 2 of those reachable by giving the exact agent its own expected-free-energy accessors, which leaves 9 that still cannot run: E10, E11, E13, E16, E18, E6b, E7, E8, E9. Six of the remaining ones are blocked by Dirichlet learning and three by distributional observations.

Two public claims sit on the learning path and cannot currently be validated at all: the unlabelled learner losing a third of its ability to read human work, and the 31% versus 74% labelling-coverage figures. Those are named here rather than left to be discovered.

On the cells that can be reached, the shortcut's error in the expected free energy is TRANSIENT AND PEAKS EARLY. It is 9.00e-06 nats at the first timestep, which is essentially zero because the belief is a product of its marginals before any evidence arrives. It rises to 0.139 nats at timestep 2, where the joint is furthest from that product (0.058 nats of divergence), and decays to 9.00e-06 nats by the end of the run as the belief concentrates and becomes nearly a point mass, which is nearly a product again.

**That shape is the thing to worry about, because it lands in the decision window.** The expected free energy is what decides whether the reader keeps looking, and the error is largest in the first handful of timesteps. Experiments whose free decision begins after a long forced window see the error only once it has decayed: the overlap sweep forces ten steps and is therefore safe. The one that is not is the selectivity measure, which the version 1 deviations record as being taken over THE FIRST THREE FREE STEPS precisely to catch the decision that matters, and which is also one of the experiments the shortcut could not be swapped out of until this pass. Re-running it under exact inference is now possible and is the cheapest outstanding check in the project.

**dirichlet learning** — the observer updates its own likelihood with Dirichlet counts. ExactAgent refuses this rather than approximating it, so these cannot run under exact inference without an exact learning path being written Affects: E7, E9, E13, E16, E18, E8.

  - an unlabelled learner loses about a third of its ability to read intent from genuine human work, and honest labels cut that error about a hundredfold (E7)
  - 31% of machine content must be labelled for a convention-aware reader and 74% for one who does not know the convention (E16)

**recorded efe terms** — records the pragmatic and epistemic decomposition through pymdp's control helpers, which build predictive states from the factorised marginals. FIXED alongside this pass: ExactAgent.efe_terms and .epistemic_value_about are exact counterparts, so these are now reachable Affects: E1, E5.

  - readers stop looking closely at work with no intent behind it, and the Curator tier costs the most deep looks (E1)
  - trust in the label is a different knob from general decisiveness (E5)

**distributional observations** — the behavioural-regret path passes a distribution rather than an observation index, which the exact filter does not accept Affects: E10, E11, E6b.

  - the reader's own skill caps what can be recovered (E10)
  - distance between belief distributions explains 66% of the variance in actual harm (E11)

**The one code change this pass made.** ExactAgent.efe_terms and ExactAgent.epistemic_value_about. Before this pass metrics.policy_efe_terms raised AttributeError on an exact agent while metrics.epistemic_value succeeded and silently returned the FACTORISED answer, which is the more dangerous of the two failures and is the class of defect this project has been bitten by seven times. A test now fails if the two disagree beyond tolerance on a construction where they must agree.

---

## D-5: criterion power

A measurement rule can be written down in advance, applied exactly, and still be incapable of telling its two answers apart, because it is averaging over too few things. This counts, for every headline criterion in the project, how many independent units it is computed over and how many were available.

| experiment | criterion | statistic | computed over | units | available | under-powered |
|---|---|---|---|---|---|---|
| E31 | update magnitude tracks recovered depth | Spearman rank correlation | the open-gate cells: 2 content levels x 3 labels | 6 | 7200 | yes |
| E20 | confident fabrication peaks in the interior | argmax over a grid, plus an interiority test | the 8 points of the overlap grid | 8 | 8 | yes |
| E20 | the engagement crossing point | linear interpolation of the 0.50 crossing, with an across-seed SE | 60 seed replicates per grid point | 60 | 60 | no |
| E15 | the competence transition is a knee, not a cliff | AIC comparison of three fits, plus a width-versus-evidence test | 15 grid points on the inexpertise axis, at 4 evidence levels | 15 | 15 | no |
| E17 | invention is graded by opacity | tie-aware weak monotonicity across tiers | 4 provenance tiers | 4 | 4 | yes |
| N21 | depth recovery is not effort recovery | ratio of two simple effects | the 4 cells of a 2 x 2 | 4 | 1920 | yes |
| E21 | a counting classifier reproduces the dissociation | comparison of two cell means against the full model's | 3 arms x 20 seeds | 60 | 12000 | no |
| E32 | foreign content and an unskilled reader differ | 5 measures compared at matched overlap, each against a range fraction | 6 matched overlap levels x 2 arms | 12 | 72000 | no |
| E30 | depth changes how much the reader takes on | regression of update magnitude on depth level | 3 depth levels | 3 | 3600 | yes |

5 of 9 primary criteria are computed over fewer than 12 independent units, which is too few for the statistic each one uses to separate its two outcomes: E31 update magnitude tracks recovered depth (6 units); E20 confident fabrication peaks in the interior (8 units); E17 invention is graded by opacity (4 units); N21 depth recovery is not effort recovery (4 units); E30 depth changes how much the reader takes on (3 units).

3 of those had more data available and did not use it. The clearest case is the two-gates result, whose rank correlation runs over six cells while 2,400 per-reader pairs sit in the same run. That is the criterion the project's public headline rests on, and it is also the one the validation pass reported as flipping under exact inference. Both the 0.886 and the 0.600 are inside what six points produce by chance, so the honest statement is not that the verdict flipped but that the criterion was never able to tell.

The remaining cases are ceilings rather than choices: four provenance tiers and three depth levels are what the framework has. Those claims should be stated as orderings rather than as trends, which is a smaller claim and an accurate one.

**E31 — update magnitude tracks recovered depth.** the project's public headline. The validation pass measured 0.886 approximate against 0.600 exact and called it a flip; both sit inside what six points give by chance. Per-reader pairs were available and not used.

**E20 — confident fabrication peaks in the interior.** an argmax over 8 grid points is the design, not a shortfall: the claim IS about which grid point is highest. What it cannot do is put an interval on the location, and the location is what every downstream claim is anchored to.

**E17 — invention is graded by opacity.** four tiers give three steps, which is the minimum a monotonicity claim can be made on. The framework has four tiers, so this is a ceiling rather than a choice, and the claim should be stated as an ordering rather than a trend.

**N21 — depth recovery is not effort recovery.** a ratio of differences between four cell means, with no interval. The validation pass found this verdict reverses under exact inference, and with four cells there is no way to say whether the reversal is real.

**E30 — depth changes how much the reader takes on.** three levels, two of which the experiment itself reports as indistinguishable from each other. That leaves two, and a null on two points is not a null.

---

## D-6: seed independence

Every simulated reader gets its own random seed, derived from which experiment cell it is in, which repeat, and which reader it is. If two readers get the same seed they are the same reader, and some statistics assume they are not. This checks.

| envelope | reader slots | distinct seeds | duplicates | inside a cell and seed | across cells | replacement |
|---|---|---|---|---|---|---|
| E20 overlap sweep | 96000 | 49450 | 46550 | 0 | 46550 | 0 |
| E2 label cells | 320000 | 202010 | 117990 | 0 | 117990 | 0 |
| E31 two gates | 14400 | 14400 | 0 | 0 | 0 | 0 |
| E32 matched arms | 72000 | 42740 | 29260 | 0 | 29260 | 0 |
| validation reduced scale | 7680 | 7680 | 0 | 0 | 0 | 0 |

The per-reader seed function is documented as collision-resistant and is not. On the overlap sweep's own envelope it produces 46550 duplicate seeds across 96000 slots, a duplicate fraction of 48%, and the structure is solvable in closed form: cell c, seed s, reader i receives the same seed as cell c+1, seed s-10, reader i+67, because 100003 - 10*10007 = -67.

**The direction is benign and that is measured rather than assumed.** Every collision is ACROSS cells; there are none inside a single (cell, seed) group, which is the unit the between-reader statistic is computed over. So no disagreement number and no across-seed standard error is affected. What the collisions do is make cross-cell comparisons partly paired, which correlates the errors between cells and makes differences MORE precise than the unpaired standard error implies. Conservative, not anti-conservative.

Two things are still wrong with it. The docstring asserts something false, and the collision structure moves inside a cell under a different choice of cardinalities: it takes only a run with at least 10007 readers, or a change to the multipliers, for the same arithmetic to start duplicating readers within a group. A hash-based replacement removes the class and is provided here, verified collision-free on every envelope above, unwired, because adopting it would change every committed number.

---

## What this pass changed about what may be claimed

1. **kappa (honest labels throughout) cannot be measured.** No dataset this model generates locates it, so it is a free choice of the modeller rather than a quantity. Sweeping it is still legitimate, because a sweep asks what a reader with that disposition would do. Claiming that any real reader has a particular value of it is not, within this framework, a falsifiable claim.
2. **kappa (machine work passed off as human) cannot be measured.** No dataset this model generates locates it, so it is a free choice of the modeller rather than a quantity. Sweeping it is still legitimate, because a sweep asks what a reader with that disposition would do. Claiming that any real reader has a particular value of it is not, within this framework, a falsifiable claim.
3. **mu recovers in order but not in magnitude** (slope 0.32). Directions transfer, sizes do not.
4. A difficulty regime exists, so the generous-fallback experiment and the depth experiment can both be rerun on a fair footing. The knob that gets there is not one of the three the specification named.
5. **Uptake is U-shaped in recovery quality, so it cannot be regressed on a difficulty manipulation without knowing where the arms sit.** Any rerun has to keep both arms on one side of the trough, and the trough is close to the regime P-2 identifies.
6. The disagreement figure may not be quoted on its own. The conjunction it appears in is sound; the figure by itself is close to a restatement of how unsure the readers are. A one-line replacement that separates the cases is recommended.
7. 9 experiments still cannot run under exact inference, including two carrying public claims. An exact learning path is the next thing worth building.
8. These criteria are computed over too few units and had more data available: E31: update magnitude tracks recovered depth; N21: depth recovery is not effort recovery; E30: depth changes how much the reader takes on. The two-gates correlation is the one that matters, because it is the public headline and the validation pass reported it as flipping; at six points neither value could have told.
9. The per-reader seed function is not collision-resistant as documented. The direction is benign and no reported statistic is affected, and the docstring should stop asserting otherwise.

---

## What comes next, and in what order

Recorded so today's work has somewhere to go. None of it is done here.

1. **Restate the claims resting on unidentifiable parameters.** That is the first thing because it is a writing job rather than a compute job, and everything downstream reads better once it is done.
2. **Rerun the two inconclusive experiments in the regime P-2 found**, with both arms on the same side of D-2's trough. Doing it without that constraint would waste a third attempt on the depth experiment.
3. **Build an exact learning path**, which unblocks six experiments including the two with public claims on them.
4. **Re-run the selectivity measure under exact inference.** D-4 shows the approximation's error peaks in exactly the window that measure is taken over, and it is now possible to run and cheap.
5. **Add the pairwise-divergence statistic beside the modal-goal entropy** everywhere disagreement is reported, and correct the reader-count bias at reduced scale.
6. **Then the minimal-model programme**: for each surviving result, find the smallest model that still produces it. D-1 is a down payment on that, since knowing provenance inference is a two-channel race with an analytic crossover tells you which commitment to try removing first.

*Generated from results/diagnostics/ on 2026-07-30 by `scripts/write_diagnostics_md.py`. Every number above is read out of a verdict file; none is typed in.*
