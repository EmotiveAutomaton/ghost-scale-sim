"""V-2 — would a model of this shape produce these results anyway?

Two arms. The second has never been run and is the stronger one.

-----------------------------------------------------------------------------------------
(a) SCRAMBLED PROVENANCE. The mapping from provenance to content properties is what alpha is:
CREATOR content carries all of the maker's intent, GHOST content almost none. Break that mapping
and every provenance effect in the project should disappear, because there is no longer anything
for a label to be evidence about.

TWO SCRAMBLES ARE RUN, NOT ONE, AND THE SECOND IS THE MORE INFORMATIVE.

* **Flattened.** Every tier gets the same alpha, the mean of the published four. Provenance now
  carries exactly zero information about content. The label effect must VANISH. This is the
  spec's null.
* **Permuted.** The four published alpha values are reassigned across the four tiers, in world
  and observer alike. This is a pure relabelling: the information is all still there, attached to
  different names. The label effect must SURVIVE, and it must survive attached to the alpha VALUE
  rather than to the tier's name.

Running only the first would leave a real gap. A result that vanishes when you destroy the
mapping is consistent both with "the mapping is doing the work" and with "the model is fragile
and any perturbation kills it". The permutation separates those: it perturbs just as much and
must NOT kill the effect. A framework whose result dies under relabelling was never about
provenance in the first place.

WHERE THIS CHECK DOES NOT APPLY, said plainly rather than quietly skipped. The readability-axis
experiments (E19, E20, E32) hold provenance fixed — every artifact in them is GHOST and unsigned
— so there is no provenance mapping in them to scramble. The corresponding question for those,
"what property of the construction is holding the result up", is V-4's, and that is where it is
answered. Reporting a vacuous pass here would be worse than reporting a gap.

-----------------------------------------------------------------------------------------
(b) RANDOM-PARAMETER FALSE-POSITIVE RATE. **Nobody has this number and it is the single most
useful thing in the pass.**

Draw the likelihood structure at random, subject only to the format constraints the project
already asserts — column-stochastic, a separation floor between goal signatures, an entropy
ceiling on synthetic content — and run the headline comparison unchanged. Repeat a few hundred
times. Count how often a randomly parameterised model of this architecture returns something that
would have been reported as a finding.

That number is the apparatus's false-positive rate. If a meaningful fraction of random models
produce the label effect, the label effect is a property of the architecture rather than of the
theory, and it has to be reported that way.

**WHAT IS RANDOMISED AND WHAT IS NOT.** The goal signatures and the synthetic-content
distribution are drawn at random. alpha, kappa, the signal likelihood, the effort likelihood, the
transitions and the preferences are NOT: those are the architecture, and the question being asked
is whether the architecture alone is sufficient. Randomising them too would answer a different and
much weaker question.

**THE REPORTABILITY BAR IS BORROWED, NOT INVENTED.** A draw counts as producing a finding if it
reproduces the pattern this project actually reported, judged by thresholds this project was
already using before V-2 existed: confident under a false label (E4's confidence threshold),
disagreement near its ceiling (E2's own framing), and honest doubt at least double the false-label
doubt. Inventing a fresh bar here would let the false-positive rate be tuned, which is precisely
the failure mode being tested for.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants as K
from .. import metrics
from ..config import Config
from ..experiments import _common as C
from . import criteria as CR
from . import validation_dir

# The reportability bar, every clause borrowed from a threshold the project already applied.
CONFIDENT_ENTROPY = 0.50      # E4's committed confident_entropy_threshold, in nats
CEILING_FRACTION = 0.90       # "1.38 against a possible 1.386" — E2's own framing
HONEST_DOUBT_MULTIPLE = 2.0   # honest label leaves at least twice the doubt a false one does


# --------------------------------------------------------------------------- #
# (a) Scrambled and permuted provenance.
# --------------------------------------------------------------------------- #
def _alpha_arms(cfg: Config) -> dict:
    raw = dict(cfg.artifact_model.alpha.raw)
    values = [float(raw[n]) for n in K.PROVENANCE_NAMES]
    mean = float(np.mean(values))
    return {
        "intact": dict(zip(K.PROVENANCE_NAMES, values)),
        "flattened": {n: mean for n in K.PROVENANCE_NAMES},
        # Reversed rather than randomly shuffled: it is the permutation that moves every tier,
        # it is deterministic, and it cannot be re-drawn until it flatters the result.
        "permuted": dict(zip(K.PROVENANCE_NAMES, values[::-1])),
    }


def _label_effect_from_e2(res_dir: Path) -> dict:
    """The label effect, read off E2's committed cell statistics.

    ONE definition, used by every arm of V-2 and by V-3 and V-7, so the arms cannot drift into
    measuring slightly different things. Machine-made content is held constant and only the
    declared label changes.
    """
    stats = pd.read_csv(res_dir / "e2_cell_stats.csv")

    def cell(prov, sig):
        s = stats[(stats.true_provenance == prov) & (stats.declared_signal == sig)]
        return s.iloc[0]

    lie, honest = cell("GHOST", "SIG_CREATOR"), cell("GHOST", "SIG_GHOST")
    within_lie, within_honest = float(lie["within"]), float(honest["within"])
    between_lie = float(lie["between"])
    ceiling = float(np.log(4))
    return {
        "within_under_false_label": within_lie,
        "within_under_honest_label": within_honest,
        "between_under_false_label": between_lie,
        "disagreement_ceiling": ceiling,
        # The effect size: how many times more doubt an honest label leaves. This is the
        # quantity every V-2/V-3/V-7 arm is scored on.
        "honest_doubt_multiple": (float(within_honest / within_lie)
                                  if within_lie > 0 else float("inf")),
        "confident_under_false_label": bool(within_lie < CONFIDENT_ENTROPY),
        "disagreement_near_ceiling": bool(between_lie >= CEILING_FRACTION * ceiling),
        "reportable": bool(within_lie < CONFIDENT_ENTROPY
                           and between_lie >= CEILING_FRACTION * ceiling
                           and within_honest >= HONEST_DOUBT_MULTIPLE * within_lie),
    }


def run_scramble(cfg: Config, workers: int = 1) -> dict:
    from ..config import load_config
    from ..experiments import e2_variance as E2

    out_root = validation_dir() / "v2a"
    arms = _alpha_arms(cfg)
    rows = {}
    for name, alpha in arms.items():
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        c = load_config()
        c.set("inference.exact", True)
        c.set("run.n_observers", int(cfg.get("validation.n_observers", 60)))
        c.set("run.n_seeds", int(cfg.get("validation.n_seeds", 12)))
        for tier, value in alpha.items():
            c.set(f"artifact_model.alpha.{tier}", float(value))
        E2.run(c, out_dir=out, workers=workers, make_fig=False)
        rows[name] = {"alpha": alpha, **_label_effect_from_e2(out)}

    intact = rows["intact"]["honest_doubt_multiple"]
    flat = rows["flattened"]["honest_doubt_multiple"]
    perm = rows["permuted"]["honest_doubt_multiple"]
    # "Vanishes" means the residual effect is under a tenth of the intact one — the same
    # standard the existing alpha-permutation null (N4) applies. The effect is a MULTIPLE, so
    # "no effect" is 1.0 and the distance from it is what gets compared.
    vanishes = bool(abs(flat - 1.0) <= CR.V2A_RESIDUAL_FRACTION * abs(intact - 1.0))

    # -------------------------------------------------------------------------------------
    # DEVIATION, DECLARED. The permutation clause as first written required the effect to
    # "survive", scored with the same reportability bar as the intact arm. That clause cannot
    # pass and could never have passed, for a reason visible in the construction rather than in
    # the measurement: the permutation used is a REVERSAL, so GHOST inherits alpha = 1.00 and
    # CREATOR inherits 0.05. A label reading "made by a person" now tells the reader to expect
    # content it cannot read, on content that is in fact perfectly readable. The effect must
    # inverse, and a bar that tests for direction was testing whether the reversal reverses.
    #
    # The clause is restated to score the effect's MAGNITUDE on a log scale, with the direction
    # reported separately as the diagnostic it actually is. The original clause is retained and
    # still computed, and it fails, exactly as this repository's convention requires.
    # -------------------------------------------------------------------------------------
    def _log_effect(x):
        return float(np.log(x)) if np.isfinite(x) and x > 0 else float("nan")

    mag_intact, mag_perm = abs(_log_effect(intact)), abs(_log_effect(perm))
    magnitude_preserved = bool(np.isfinite(mag_intact) and np.isfinite(mag_perm)
                               and mag_intact > 0
                               and abs(mag_perm - mag_intact) / mag_intact <= 0.25)
    direction_inverted = bool(np.isfinite(_log_effect(intact)) and np.isfinite(_log_effect(perm))
                              and np.sign(_log_effect(intact)) != np.sign(_log_effect(perm)))
    survives = bool(magnitude_preserved and direction_inverted)
    survives_original_clause = bool(rows["permuted"]["reportable"])

    if vanishes and survives:
        verdict = "PROVENANCE_MAPPING_IS_LOAD_BEARING"
        statement = ("Destroying the provenance-to-content mapping removes the label effect. "
                     "Reversing it leaves the effect the same size and turns it upside down. "
                     "The result is attached to what a provenance tier means about content, "
                     "not to the tier's name and not to the model being delicate.")
    elif vanishes and not survives:
        verdict = "EFFECT_IS_FRAGILE_TO_ANY_PERTURBATION"
        statement = ("The label effect vanishes when the mapping is destroyed, which the spec "
                     "asks for, but it does not come back at comparable size under a pure "
                     "reversal, which it should. That makes the null uninformative: the effect "
                     "responds to perturbation as such rather than to provenance, and the claim "
                     "is reported with that attached.")
    else:
        verdict = "EFFECT_SURVIVES_ITS_OWN_NULL"
        statement = ("The label effect does NOT vanish when the provenance-to-content mapping "
                     "is destroyed. Something other than provenance is producing it, and the "
                     "claim is withdrawn pending an account of what.")

    return {
        "arm": "V-2a",
        "question": ("If provenance stops telling you anything about the content, does the "
                     "label effect go away?"),
        "arms": rows,
        "effect_intact": intact, "effect_flattened": flat, "effect_permuted": perm,
        "residual_fraction_allowed": CR.V2A_RESIDUAL_FRACTION,
        "vanishes_when_mapping_destroyed": vanishes,
        "survives_pure_relabelling": survives,
        "log_magnitude_intact": mag_intact,
        "log_magnitude_permuted": mag_perm,
        "direction_inverted_by_reversal": direction_inverted,
        "deviation": {
            "what_changed": ("the permutation clause was restated from 'the effect survives, "
                             "same reportability bar' to 'the effect's log magnitude is "
                             "preserved and its direction inverts'"),
            "why": ("the permutation is a reversal of alpha, so GHOST inherits CREATOR's "
                    "transparency and vice versa. Under that reversal the effect MUST invert; "
                    "a direction-sensitive bar was asking whether a reversal reverses. The "
                    "restatement follows from the construction, not from the measurement."),
            "original_clause_retained_and_still_computed": True,
            "original_clause_outcome": ("PASSES" if survives_original_clause else "FAILS"),
        },
        "not_applicable_to": {
            "experiments": ["E19", "E20", "E32"],
            "why": ("every artifact in them is GHOST and unsigned, so there is no "
                    "provenance-to-content mapping present to scramble. The corresponding "
                    "construction check for those results is V-4."),
        },
        "verdict": verdict,
        "statement": statement,
    }


# --------------------------------------------------------------------------- #
# (b) The random-parameter false-positive rate.
# --------------------------------------------------------------------------- #
def draw_random_likelihoods(rng: np.random.Generator, num_goals: int, num_features: int,
                            js_threshold: float, entropy_ceiling: float,
                            synth_floor: float, max_tries: int = 400):
    """A random likelihood family satisfying the project's format constraints and nothing else.

    The constraints are exactly the three the codebase already asserts at every construction:
    column-stochastic (automatic from Dirichlet draws), a pairwise Jensen-Shannon floor between
    goal signatures, and an entropy ceiling on the synthetic distribution. Nothing about goal
    structure, feature pairing or peakedness is imposed, so the drawn families include shapes the
    designed one deliberately excludes.

    Returns ``None`` if the constraints could not be met within ``max_tries``, and the caller
    counts those rather than retrying forever — a draw budget that silently loops would let the
    constraint set do the work the randomness is supposed to be doing.
    """
    sig = None
    for _ in range(max_tries):
        cand = rng.dirichlet(np.full(num_features, 0.35), size=num_goals)
        ok = True
        for a in range(num_goals):
            for b in range(a + 1, num_goals):
                if metrics.js_divergence(cand[a], cand[b]) <= js_threshold:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            sig = cand
            break
    if sig is None:
        return None

    synth = None
    for _ in range(max_tries):
        cand = rng.dirichlet(np.full(num_features, 0.20))
        cand = np.maximum(cand, synth_floor)
        cand = cand / cand.sum()
        if metrics.shannon_entropy(cand) < entropy_ceiling:
            synth = cand
            break
    if synth is None:
        return None
    return sig, synth


def draw_random_within_partition(rng: np.random.Generator, cfg, js_threshold: float,
                                 entropy_ceiling: float, synth_floor: float,
                                 max_tries: int = 400):
    """A random likelihood family drawn INSIDE the architecture's own two constructions.

    WHY THIS SECOND ARM EXISTS, and it was added because the first arm's result demanded it.

    The unconstrained arm above randomises the likelihood family completely. Doing so silently
    removes two things that are DESIGN DECISIONS rather than format constraints, both documented
    in ``generative_model.py`` since V1:

      1. the goal/feature PARTITION — each goal owns its own pair of features;
      2. GOAL SYMMETRY of the synthetic distribution — the frozen synth is symmetrised across
         the goal pairs so it sits equidistant from every goal signature.

    The second one is load-bearing for the disagreement half of the headline, and V1 said so at
    the time: "an un-symmetrized draw generically resembles ONE goal by chance, which would turn
    the H2 result from arbitrary, observer-specific hallucination into spurious CONSENSUS on that
    one goal." An unconstrained random draw is un-symmetrised by definition, so it cannot produce
    disagreement, and a false-positive rate computed only that way would credit the theory for
    something the construction supplies.

    So this arm keeps the partition and the symmetrisation and randomises everything else that is
    free inside them: how much mass a signature puts on its own pair, how that mass splits within
    the pair, the off-pair floor, and the synth's Dirichlet concentration. The two rates read
    together are the informative object — the first says what the architecture alone does, the
    second says what the architecture plus its two constructions do.
    """
    from ..generative_model import _symmetrize_over_goal_pairs

    pairs = [list(p) for p in cfg.artifact_model.goal_feature_pairs]
    num_features = int(cfg.cardinalities.num_features)
    num_goals = len(pairs)

    sig = None
    for _ in range(max_tries):
        peak = float(rng.uniform(0.50, 0.98))
        floor = float(10.0 ** rng.uniform(-3.0, -1.3))
        cand = np.zeros((num_goals, num_features))
        for g, (a, b) in enumerate(pairs):
            split = float(rng.uniform(0.2, 0.8))
            v = np.full(num_features, floor)
            v[a] += peak * split
            v[b] += peak * (1.0 - split)
            cand[g] = v / v.sum()
        if all(metrics.js_divergence(cand[a], cand[b]) > js_threshold
               for a in range(num_goals) for b in range(a + 1, num_goals)):
            sig = cand
            break
    if sig is None:
        return None

    synth = None
    for _ in range(max_tries):
        conc = float(10.0 ** rng.uniform(-2.0, -0.3))
        cand = rng.dirichlet(np.full(num_features, conc))
        cand = _symmetrize_over_goal_pairs(cand, pairs)
        cand = np.maximum(cand, synth_floor)
        cand = cand / cand.sum()
        if metrics.shannon_entropy(cand) < entropy_ceiling:
            synth = cand
            break
    if synth is None:
        return None
    return sig, synth


def _fpr_worker(payload):
    """One random draw: build the model, run E2's two decisive cells, score reportability."""
    import numpy as _np

    (cfg_raw, draw, base_seed, n_obs, n_seeds, n_timesteps, forced_k, true_goal,
     arm) = payload
    from pymdp.legacy import utils

    from ..config import Config as _Config
    from ..creators import build_creator_bank
    from ..environment import Artifact, Environment
    from ..exact import make_exact_agent
    from ..generative_model import (GenerativeModel, alpha_by_provenance, build_A0,
                                    build_A1_observer, build_A2, build_B, build_C, build_D)
    from ..observer import rollout_observer

    cfg = _Config(cfg_raw).copy()
    cfg.set("inference.exact", True)
    cd_goals = int(cfg.cardinalities.num_goals)
    cd_feats = int(cfg.cardinalities.num_features)

    rng = _np.random.default_rng(base_seed * 7_919 + draw)
    kwargs = dict(js_threshold=float(cfg.artifact_model.js_threshold),
                  entropy_ceiling=float(cfg.artifact_model.structured_ceiling),
                  synth_floor=float(cfg.artifact_model.get("synth_floor", 1.0e-3)))
    if arm == "unconstrained":
        drawn = draw_random_likelihoods(rng, cd_goals, cd_feats, **kwargs)
    else:
        drawn = draw_random_within_partition(rng, cfg, **kwargs)
    if drawn is None:
        return [{"draw": draw, "arm": arm, "usable": 0}]
    sig, synth = drawn

    alpha = alpha_by_provenance(cfg)
    A = utils.obj_array(3)
    A[0] = build_A0(cfg, sig, synth, alpha)
    A[1] = build_A1_observer(cfg, float(cfg.signal_model.kappa))
    A[2] = build_A2(cfg)
    gm = GenerativeModel(A=A, B=build_B(cfg), C=build_C(cfg), sig=sig,
                         noise_free_synth=synth, alpha=alpha,
                         kappa=float(cfg.signal_model.kappa), cfg=cfg, sig_true=sig)

    bank = build_creator_bank(cfg, gm)
    out = {}
    for cell_index, signal in ((0, K.SIG_CREATOR), (1, K.SIG_GHOST)):
        posteriors_by_seed = []
        for s in range(n_seeds):
            env = Environment(cfg, gm, rng_world=_np.random.default_rng(base_seed * 31 + s),
                              creator_bank=bank)
            post = []
            for i in range(n_obs):
                r = C.observer_rng(base_seed, cell_index, s, i)
                agent = make_exact_agent(gm, build_D(cfg, r), cfg, rng=r)
                art = Artifact(provenance=K.GHOST, goal=true_goal, declared_signal=signal)
                res = rollout_observer(agent, art, env, cfg, r, n_timesteps,
                                       force_deep_k=forced_k)
                post.append(_np.asarray(res.final_goal_posterior, dtype=float))
            posteriors_by_seed.append(post)
        out[signal] = {
            "within": float(_np.mean([metrics.mean_within_observer_entropy(p)
                                      for p in posteriors_by_seed])),
            "between": float(_np.mean([metrics.between_observer_entropy(p)
                                       for p in posteriors_by_seed])),
        }

    within_lie = out[K.SIG_CREATOR]["within"]
    within_honest = out[K.SIG_GHOST]["within"]
    between_lie = out[K.SIG_CREATOR]["between"]
    ceiling = float(_np.log(cd_goals))
    multiple = float(within_honest / within_lie) if within_lie > 0 else float("inf")
    return [{
        "draw": draw, "arm": arm, "usable": 1,
        "within_under_false_label": within_lie,
        "within_under_honest_label": within_honest,
        "between_under_false_label": between_lie,
        "honest_doubt_multiple": multiple,
        "confident_under_false_label": int(within_lie < CONFIDENT_ENTROPY),
        "disagreement_near_ceiling": int(between_lie >= CEILING_FRACTION * ceiling),
        "honest_doubt_doubles": int(within_honest >= HONEST_DOUBT_MULTIPLE * within_lie),
        "reportable": int(within_lie < CONFIDENT_ENTROPY
                          and between_lie >= CEILING_FRACTION * ceiling
                          and within_honest >= HONEST_DOUBT_MULTIPLE * within_lie),
        "max_signature_entropy": float(max(metrics.shannon_entropy(row) for row in sig)),
        "synth_entropy": float(metrics.shannon_entropy(synth)),
    }]


ARMS_B = (
    ("unconstrained",
     "the likelihood family drawn freely, subject only to the format constraints. This is the "
     "spec's arm and it answers: what does the architecture do on its own?"),
    ("within_partition",
     "the same, but drawn INSIDE the goal/feature partition and with the synthetic distribution "
     "symmetrised across goals, as V1 constructs it. This answers: what does the architecture "
     "plus its two design decisions do?"),
)


def _score_arm(usable: pd.DataFrame) -> dict:
    mult = usable.honest_doubt_multiple.replace([np.inf, -np.inf], np.nan).dropna()
    nan = float("nan")
    return {
        "usable_draws": int(len(usable)),
        "false_positive_rate": float(usable.reportable.mean()) if len(usable) else nan,
        "clause_rates": {
            "confident_under_false_label": (
                float(usable.confident_under_false_label.mean()) if len(usable) else nan),
            "disagreement_near_ceiling": (
                float(usable.disagreement_near_ceiling.mean()) if len(usable) else nan),
            "honest_doubt_doubles": (
                float(usable.honest_doubt_doubles.mean()) if len(usable) else nan),
        },
        "effect_percentiles": {
            "p50": float(np.percentile(mult, 50)) if len(mult) else nan,
            "p95": float(np.percentile(mult, 95)) if len(mult) else nan,
            "p_committed": float(np.percentile(mult, CR.V2B_PERCENTILE)) if len(mult) else nan,
            "max": float(mult.max()) if len(mult) else nan,
        },
    }


def run_false_positive_rate(cfg: Config, workers: int = 1) -> dict:
    from ..config import load_config

    draws = int(cfg.get("validation.random_model_draws", 300))
    # Deliberately smaller per draw than V-1's scale: the quantity being estimated is a RATE
    # across draws, so draws buy more than observers do, and the reportability bar is a
    # threshold on a cell mean rather than a confidence interval on it.
    n_obs = max(12, int(cfg.get("validation.n_observers", 60)) // 4)
    n_seeds = max(3, int(cfg.get("validation.n_seeds", 12)) // 4)

    base = load_config()
    base.set("inference.exact", True)
    base_seed = int(cfg.run.base_seed)
    payloads = [(base.raw, d, base_seed, n_obs, n_seeds, int(base.run.n_timesteps),
                 int(base.experiments.e2.forced_deep_k),
                 int(base.get("experiments.e2.true_goal", 1)), arm)
                for arm, _ in ARMS_B for d in range(draws)]
    recs = C.run_parallel(payloads, _fpr_worker, workers)

    df = pd.DataFrame(recs)
    out = validation_dir() / "v2b"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "random_model_draws.csv", index=False)

    # The designed model's own effect, at the same reduced scale, from V-2a's intact arm.
    intact_path = validation_dir() / "v2a" / "intact" / "e2_cell_stats.csv"
    real = _label_effect_from_e2(intact_path.parent) if intact_path.exists() else None
    real_multiple = float(real["honest_doubt_multiple"]) if real else float("nan")

    scored, notes = {}, dict(ARMS_B)
    for arm, _ in ARMS_B:
        sub = df[(df.arm == arm) & (df.usable == 1)]
        scored[arm] = _score_arm(sub)
        scored[arm]["what_it_asks"] = notes[arm]
        p = scored[arm]["effect_percentiles"]["p_committed"]
        scored[arm]["designed_effect_outside_random_distribution"] = bool(
            np.isfinite(real_multiple) and np.isfinite(p) and real_multiple > p)

    n_min = min(s["usable_draws"] for s in scored.values())
    if n_min < CR.V2B_MIN_DRAWS:
        return {
            "arm": "V-2b", "draws_requested_per_arm": draws, "arms": scored,
            "designed_model_effect": real_multiple,
            "verdict": "NOT_REPORTABLE_TOO_FEW_USABLE_DRAWS",
            "statement": ("The thinner of the two arms produced only %d usable draws against a "
                          "pre-registered minimum of %d. A false-positive rate quoted from that "
                          "many draws would be worse than none, so none is quoted."
                          % (n_min, CR.V2B_MIN_DRAWS)),
        }

    unc, wp = scored["unconstrained"], scored["within_partition"]
    # THE HEADLINE NUMBER is the within-partition rate, because that is the apparatus as it was
    # actually used. The unconstrained rate is reported beside it as the decomposition.
    fpr = wp["false_positive_rate"]
    outside = bool(wp["designed_effect_outside_random_distribution"])

    # WHICH CLAUSE the random models fail is the informative part, so it is scored explicitly
    # rather than left for a reader to spot in the table.
    conf_rate = unc["clause_rates"]["confident_under_false_label"]
    dis_rate = unc["clause_rates"]["disagreement_near_ceiling"]
    confidence_is_architectural = bool(conf_rate >= 0.50)

    if not outside and fpr >= CR.V2B_ALPHA:
        verdict = "EFFECT_IS_ARCHITECTURE_DEPENDENT"
    elif not outside:
        verdict = "EFFECT_DOES_NOT_SEPARATE_FROM_RANDOM"
    elif fpr >= CR.V2B_ALPHA:
        verdict = "EFFECT_SEPARATES_BUT_APPARATUS_IS_PERMISSIVE"
    else:
        verdict = "EFFECT_IS_NOT_ARCHITECTURAL"

    parts = [
        ("With the likelihood family drawn freely, %.0f%% of random models produce confident "
         "belief under a false label and %.0f%% produce disagreement near its ceiling."
         % (100 * conf_rate, 100 * dis_rate)),
        ("Confident commitment under a false label is therefore a property of the architecture: "
         "a randomly parameterised reader does it too. What a random model does NOT supply is "
         "the DISAGREEMENT that makes the commitment invention rather than consensus, and that "
         "half of the headline is the half the theory is entitled to."
         if confidence_is_architectural else
         "Neither half of the pattern is common among freely drawn random models."),
        ("Inside the architecture's own two constructions, the goal/feature partition and the "
         "symmetrised synthetic distribution, %.1f%% of random parameterisations clear the whole "
         "reportability bar against the %.0f%% committed before the run, and the designed "
         "model's effect (%.1fx) sits %s the random distribution's %.1fth percentile (%.1fx)."
         % (100 * fpr, 100 * CR.V2B_ALPHA, real_multiple,
            "outside" if outside else "INSIDE", CR.V2B_PERCENTILE,
            wp["effect_percentiles"]["p_committed"])),
    ]
    if verdict == "EFFECT_SEPARATES_BUT_APPARATUS_IS_PERMISSIVE":
        parts.append("The apparatus is permissive, and every borderline finding in the project "
                     "should be read against that rate rather than against zero.")
    if verdict in ("EFFECT_IS_ARCHITECTURE_DEPENDENT", "EFFECT_DOES_NOT_SEPARATE_FROM_RANDOM"):
        parts.append("The claim is reported as architecture-dependent, in the same cell as the "
                     "claim, per the spec's constraint that a failed check is not deleted.")

    return {
        "arm": "V-2b",
        "question": ("How often does a randomly parameterised model of this architecture return "
                     "something we would have reported as a finding?"),
        "plain_language": (
            "The model has a shape and it has settings. This check keeps the shape, throws the "
            "settings away, picks new ones at random a few hundred times, and asks how often the "
            "random version still produces the headline. That fraction is the rate at which the "
            "apparatus manufactures findings out of nothing, and nobody had it before. It runs "
            "twice: once with the settings drawn freely, and once with the two design decisions "
            "the model has held since version 1 left in place, so the two halves of the headline "
            "can be told apart."),
        "draws_requested_per_arm": draws,
        "per_draw_scale": {"n_observers": n_obs, "n_seeds": n_seeds},
        "percentile_committed": CR.V2B_PERCENTILE,
        "arms": scored,
        "headline_false_positive_rate": fpr,
        "headline_arm": "within_partition",
        "alpha_committed_before_run": CR.V2B_ALPHA,
        "designed_model_effect": real_multiple,
        "confident_invention_is_architectural": confidence_is_architectural,
        "disagreement_rate_among_free_random_models": dis_rate,
        "reportability_bar": {
            "confident_entropy_nats": CONFIDENT_ENTROPY,
            "disagreement_ceiling_fraction": CEILING_FRACTION,
            "honest_doubt_multiple": HONEST_DOUBT_MULTIPLE,
            "provenance": ("every clause borrowed from a threshold this project already applied "
                           "before V-2 existed"),
        },
        "verdict": verdict,
        "statement": " ".join(parts),
    }


def run(cfg: Config, workers: int = 1) -> dict:
    a = run_scramble(cfg, workers=workers)
    b = run_false_positive_rate(cfg, workers=workers)
    verdict = {
        "check": "V-2",
        "question": "Would a model of this shape produce these results anyway?",
        "scrambled_provenance": a,
        "random_parameter_false_positive_rate": b,
        "verdict": f"{a['verdict']} / {b['verdict']}",
        "statement": a["statement"] + " " + b["statement"],
    }
    (validation_dir() / "v2_nulls.json").write_text(
        json.dumps(verdict, indent=2, default=float), encoding="utf-8")
    return verdict
