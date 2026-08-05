"""T-2 — does goal diversity rise with automaticity, at a fixed decision count?

THE CURATOR'S CLAIM. "Soul" is a variety of motivations, and it travels with expertise: practice
bakes processes in, conscious access to them is lost, and what makes the decision instead is a
drive. So an expert's artifact carries more motivational variety WITHOUT the expert choosing it.

WHAT THIS CAN AND CANNOT TEST, said before the numbers because the framing in the request is
stronger than the model can support.

Nothing in this simulation DERIVES goal-mixture diversity from automaticity. To vary automaticity
I have to set the mixture as a function of a dial, and then "diversity rises with automaticity"
is the wiring rather than a finding. The mechanism half of T-2 is therefore not testable here and
is not claimed. What IS testable, and is worth the run, is the READER-SIDE half:

    when motivational variety goes up at a fixed decision count, does a reader see it as
    posterior breadth WITHOUT losing the goal?

That is S-2's question with the manipulation pointed the other way, and S-2's machinery answers
it: an active terminal goal is drawn per position from a weight vector, so decision density is
identical across arms by construction.

TWO AUTOMATICITY AXES ARE RUN, because they are different claims and only one of them is
circular.

  MIXTURE AUTOMATICITY  the dial sets how many drives feed the emission. Circular as a mechanism
                        test, informative as a reader test. Swept from a point mass on one goal
                        to a flat mixture over all of them.
  DEPTH AUTOMATICITY    mu, the model's own practice variable, held at a FIXED mixture. This is
                        NOT circular -- nothing about mu was built to move breadth -- and it is
                        the honest test of whether practice alone moves what a reader reads about
                        purpose. E43 already ties mu to a maker losing access to its own reasons.

THE TRAP, which is the point of running it here: if goal recovery FALLS as diversity rises, then
"diversity" has quietly become "noise" and the construct is confounded. Process recovery falling
too would mean it has become "shallower". Both are checked and both are reported whichever way
they come out.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ... import constants as K
from ... import metrics
from ...baselines import ObservationTape, TapedEnvironment
from ...config import Config
from ...environment import Artifact
from ...v5_model import make_v5_observer
from ...v6 import SEED_OFFSET, harness as H
from ...methods import gates as G
from ...methods import provenance as PROV
from . import sl_dir
from .common import build
from . import t_common as T

#: Automaticity as concentration of the drive mixture. 0 = one drive (fully deliberate, a single
#: chosen purpose); 1 = flat over all drives (fully automatic, every drive feeding in).
AUTOMATICITY = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Read at the Curator tier with a short look, for S-2's reason: on fully transparent work with a
#: long look this reader's goal posterior collapses to ~1e-10 and breadth is pinned at the floor,
#: so the comparison has no headroom and measures nothing.
READ_TIER = K.CURATOR
READ_GLANCES = 3
N_TIMESTEPS = 24


def _weights(a: float, ng: int) -> np.ndarray:
    """Interpolate from a point mass on goal 0 to a flat mixture. ``a`` is automaticity."""
    point = np.zeros(ng)
    point[0] = 1.0
    flat = np.full(ng, 1.0 / ng)
    w = (1.0 - float(a)) * point + float(a) * flat
    return w / w.sum()


def mixed_deep_features(world, env, creator, actives, mu: int, n_sub: int, ng: int,
                        n_timesteps: int, rng, alpha_override: float | None = None
                        ) -> np.ndarray:
    """DEEP features from ONE practised process serving a MIXTURE of drives.

    THIS REPLACES S-2's EMITTER, WHICH DOES NOT WORK. S-2 varies the active goal by passing a
    fresh ``Artifact`` per position into ``V5Environment.sample_feature``, and that method never
    reads ``artifact.goal`` once a creator is bound -- it returns ``self.creator.next_feature``,
    and the creator holds one fixed goal for the whole artifact. The mixture is drawn and
    discarded. ``scripts/audit_s2_mixture.py`` shows the feature streams are bit-identical with
    the mixture switched off entirely.

    Here the emission is built directly out of the world's own ``subsig[mu, goal, mode]``, which
    is what the creator emits from anyway, with the ACTIVE GOAL substituted per position while
    the MODE TRAJECTORY stays the creator's own. That is the curator's claim made literal: the
    baked-in routine is one routine, and what varies is which drive each decision serves. The
    alpha gate to synthetic noise is kept, so the reading tier still means what it means
    everywhere else in this repository.
    """
    mu_i = list(world.mu_levels).index(int(mu))
    subsig = np.asarray(world.subsig, dtype=float)          # (n_mu, ng, n_sub, n_features)
    synth = np.asarray(world.gm.noise_free_synth, dtype=float)
    alpha = float(env.alpha[READ_TIER]) if alpha_override is None else float(alpha_override)
    modes = H.creator_positions(creator, int(n_timesteps), initial_glance=True)
    out = np.zeros(int(n_timesteps), dtype=int)
    nf = subsig.shape[-1]
    for t in range(int(n_timesteps)):
        if rng.random() < alpha:
            p = subsig[mu_i, int(actives[t]), int(modes[t])]
            out[t] = int(rng.choice(nf, p=p / p.sum()))
        else:
            out[t] = int(rng.choice(nf, p=synth / synth.sum()))
    return out


def _one(world, cfg_r, n_mu, n_sub, ng, mu, beta, w, seed_base, i, n_timesteps=N_TIMESTEPS,
         alpha_override: float | None = None):
    art_rng = np.random.default_rng(seed_base * 31 + i)
    modal = int(np.argmax(w))
    creator, artifact, env = H.make_artifact_and_env(
        world, cfg_r, modal, int(mu), float(beta), n_timesteps, art_rng, provenance=READ_TIER)
    tape = ObservationTape(env, artifact, np.random.default_rng(seed_base * 104729 + i),
                           n_timesteps)
    actives = art_rng.choice(ng, size=n_timesteps, p=w)
    tape.deep = mixed_deep_features(world, env, creator, actives, mu, n_sub, ng, n_timesteps,
                                    art_rng, alpha_override=alpha_override)
    agent = make_v5_observer(world, np.random.default_rng(seed_base * 7907 + i))
    enc = H.run_encounter(world, cfg_r, artifact, TapedEnvironment(tape), agent, creator,
                          np.random.default_rng(seed_base * 7907 + i), n_timesteps,
                          READ_GLANCES, n_sub, n_mu, ng,
                          float(world.cfg.signal_model.kappa), true_goal=modal)
    hmax = float(np.log(max(ng, 2)))
    # The TRUE diversity of what fed the artifact, so the reader's breadth can be regressed on
    # the thing it is supposed to be reading rather than only on the dial.
    counts = np.bincount(actives, minlength=ng).astype(float)
    true_ent = float(metrics.within_observer_entropy(counts / counts.sum())) / hmax
    return {
        "purpose_breadth": float(metrics.within_observer_entropy(enc.goal_posterior)) / hmax,
        "goal_correct": int(enc.correct),
        "goal_error_reduction": float(enc.error_reduction),
        "process": float(enc.process["process_error_reduction"]),
        "process_acc": float(enc.process["process_accuracy"]),
        "distinct_goals_used": int(len(set(actives.tolist()))),
        "true_mixture_breadth": true_ent,
        "recovered_mu": float(enc.recovered_mu),
        "engaged_fraction": float(enc.engaged_fraction),
    }


def run(cfg: Config, n_obs: int = 200) -> dict:
    world, _b, cfg_r, n_mu, n_sub, ng = build(cfg)
    rng = np.random.default_rng(SEED_OFFSET + 91_200)
    rows = []

    # ---- axis 1: mixture automaticity, at fixed depth --------------------------------------
    for mu in (2, 3):
        for a in AUTOMATICITY:
            w = _weights(a, ng)
            for i in range(int(n_obs)):
                r = _one(world, cfg_r, n_mu, n_sub, ng, mu, 1.0, w, 63_000 + mu * 101, i)
                r.update({"axis": "mixture", "automaticity": float(a), "mu": mu,
                          "n_timesteps": N_TIMESTEPS, "i": i})
                rows.append(r)

    # ---- axis 2: depth automaticity, at a FIXED mixture -------------------------------------
    for a in (0.0, 0.5, 1.0):
        w = _weights(a, ng)
        for mu in (1, 2, 3):
            for i in range(int(n_obs)):
                r = _one(world, cfg_r, n_mu, n_sub, ng, mu, 1.0, w, 64_000 + int(a * 10) * 7, i)
                r.update({"axis": "depth", "automaticity": float(a), "mu": mu,
                          "n_timesteps": N_TIMESTEPS, "i": i})
                rows.append(r)

    # ---- axis 3: decision COUNT, the constancy the design rests on --------------------------
    # T-2's whole construction is "at fixed decision count". If breadth moves with the number of
    # positions at a fixed mixture, then the fixed-count constraint is doing work and any result
    # on axis 1 is partly a length effect.
    for n_t in (12, 24, 48):
        for a in (0.0, 1.0):
            w = _weights(a, ng)
            for i in range(int(n_obs)):
                r = _one(world, cfg_r, n_mu, n_sub, ng, 3, 1.0, w, 65_000 + n_t, i,
                         n_timesteps=n_t)
                r.update({"axis": "length", "automaticity": float(a), "mu": 3,
                          "n_timesteps": n_t, "i": i})
                rows.append(r)

    # ---- axis 4: THE DIFFICULTY CONTROL, and the one that decides whether breadth is usable --
    # Breadth rising with diversity is worthless as a signal if breadth also rises whenever the
    # work simply gets harder to read. So: hold the mixture at a POINT MASS -- one drive, no
    # diversity at all -- and make the artifact harder by lowering the channel's alpha until the
    # reader's goal accuracy matches each mixture arm. If breadth still separates the two at
    # matched accuracy, it is reading diversity. If it does not, 'purpose breadth' is a
    # difficulty meter with a suggestive name.
    for alpha in (0.85, 0.70, 0.55, 0.45, 0.35, 0.25, 0.15):
        w = _weights(0.0, ng)
        for i in range(int(n_obs)):
            r = _one(world, cfg_r, n_mu, n_sub, ng, 3, 1.0, w, 66_000 + int(alpha * 100), i,
                     alpha_override=alpha)
            r.update({"axis": "noise", "automaticity": 0.0, "mu": 3,
                      "n_timesteps": N_TIMESTEPS, "alpha": float(alpha), "i": i})
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(sl_dir() / "t2_automaticity_points.csv", index=False)
    (df.groupby(["axis", "automaticity", "mu", "n_timesteps"])
       [["purpose_breadth", "goal_correct", "process", "true_mixture_breadth"]]
       .agg(["mean", "std", "count"])
       .to_csv(sl_dir() / "t2_automaticity_summary.csv"))

    def sel(axis, **kw):
        d = df[df.axis == axis]
        for k, v in kw.items():
            d = d[d[k] == v]
        return d

    # ---- axis 1 verdict ---------------------------------------------------------------------
    mixture = {}
    for mu in (2, 3):
        lo = sel("mixture", mu=mu, automaticity=0.0)
        hi = sel("mixture", mu=mu, automaticity=1.0)
        series = {str(a): {
            "purpose_breadth": float(sel("mixture", mu=mu, automaticity=a).purpose_breadth.mean()),
            "goal_accuracy": float(sel("mixture", mu=mu, automaticity=a).goal_correct.mean()),
            "process": float(sel("mixture", mu=mu, automaticity=a).process.mean()),
            "true_mixture_breadth": float(
                sel("mixture", mu=mu, automaticity=a).true_mixture_breadth.mean()),
        } for a in AUTOMATICITY}
        bs = [series[str(a)]["purpose_breadth"] for a in AUTOMATICITY]
        mixture[f"mu{mu}"] = {
            "series_by_automaticity": series,
            "breadth_rises": T.boot_paired(hi.purpose_breadth.to_numpy(),
                                           lo.purpose_breadth.to_numpy(), rng),
            "goal_accuracy_should_not_fall": T.boot_paired(hi.goal_correct.to_numpy(),
                                                           lo.goal_correct.to_numpy(), rng),
            "process_should_not_fall": T.boot_paired(hi.process.to_numpy(),
                                                     lo.process.to_numpy(), rng),
            "breadth_monotone": bool(np.all(np.diff(bs) >= -1e-9)),
            "breadth_vs_true_mixture_r": float(np.corrcoef(
                sel("mixture", mu=mu).true_mixture_breadth.to_numpy(),
                sel("mixture", mu=mu).purpose_breadth.to_numpy())[0, 1]),
        }

    # ---- axis 2 verdict: does PRACTICE alone move breadth? ----------------------------------
    depth = {}
    for a in (0.0, 0.5, 1.0):
        d1 = sel("depth", automaticity=a, mu=1)
        d3 = sel("depth", automaticity=a, mu=3)
        depth[f"automaticity{a}"] = {
            "breadth_by_mu": {str(m): float(sel("depth", automaticity=a, mu=m)
                                            .purpose_breadth.mean()) for m in (1, 2, 3)},
            "goal_accuracy_by_mu": {str(m): float(sel("depth", automaticity=a, mu=m)
                                                  .goal_correct.mean()) for m in (1, 2, 3)},
            "process_by_mu": {str(m): float(sel("depth", automaticity=a, mu=m)
                                            .process.mean()) for m in (1, 2, 3)},
            "mu3_minus_mu1_breadth": T.boot_paired(d3.purpose_breadth.to_numpy(),
                                                   d1.purpose_breadth.to_numpy(), rng),
            "mu3_minus_mu1_goal_accuracy": T.boot_paired(d3.goal_correct.to_numpy(),
                                                         d1.goal_correct.to_numpy(), rng),
        }

    # ---- axis 3: is the fixed-count constraint load-bearing? --------------------------------
    length = {}
    for a in (0.0, 1.0):
        length[f"automaticity{a}"] = {
            "breadth_by_length": {str(n): float(sel("length", automaticity=a, n_timesteps=n)
                                                .purpose_breadth.mean()) for n in (12, 24, 48)},
            "long_minus_short": T.boot_paired(
                sel("length", automaticity=a, n_timesteps=48).purpose_breadth.to_numpy(),
                sel("length", automaticity=a, n_timesteps=12).purpose_breadth.to_numpy(), rng),
        }

    # ---- axis 4 verdict: breadth at matched goal accuracy -----------------------------------
    noise_pts = []
    for alpha in sorted(df[df.axis == "noise"].alpha.dropna().unique()):
        d = df[(df.axis == "noise") & (df.alpha == alpha)]
        noise_pts.append({"alpha": float(alpha), "goal_accuracy": float(d.goal_correct.mean()),
                          "purpose_breadth": float(d.purpose_breadth.mean())})
    noise_pts.sort(key=lambda r: r["goal_accuracy"])
    # Collapse duplicate accuracies before interpolating. Two alphas can land on the same goal
    # accuracy and np.interp on a non-strictly-increasing x is silently wrong rather than loud.
    _acc, _br = {}, {}
    for p_ in noise_pts:
        _acc.setdefault(p_["goal_accuracy"], []).append(p_["purpose_breadth"])
    acc_axis = sorted(_acc)
    br_axis = [float(np.mean(_acc[a_])) for a_ in acc_axis]
    difficulty = {"noise_curve": noise_pts, "matched": {}}
    for a in AUTOMATICITY:
        d = sel("mixture", mu=3, automaticity=a)
        acc = float(d.goal_correct.mean())
        br = float(d.purpose_breadth.mean())
        matched_br = float(np.interp(acc, acc_axis, br_axis))
        difficulty["matched"][str(a)] = {
            "goal_accuracy": acc, "breadth_mixture": br,
            "breadth_of_equally_hard_single_drive_work": matched_br,
            "excess_breadth_attributable_to_diversity": float(br - matched_br),
            "in_range_of_noise_curve": bool(min(acc_axis) <= acc <= max(acc_axis)),
        }
    # THE BASELINE OFFSET HAS TO BE DIFFERENCED OUT. The automaticity = 0 arm is single-drive
    # work, so it IS a point on the noise curve and its excess should be zero. It is not, because
    # the mixture arms run at the tier's own alpha rather than at a swept one and the two
    # constructions differ slightly. Reporting the raw excess would credit that offset to
    # diversity. What is attributable to diversity is the excess ABOVE the single-drive arm's own
    # excess, and that is the number the verdict reads.
    base_excess = difficulty["matched"]["0.0"]["excess_breadth_attributable_to_diversity"]
    for k2, v2 in difficulty["matched"].items():
        v2["baseline_offset_of_the_single_drive_arm"] = float(base_excess)
        v2["excess_above_single_drive_baseline"] = float(
            v2["excess_breadth_attributable_to_diversity"] - base_excess)
    excesses = [v2["excess_above_single_drive_baseline"]
                for k2, v2 in difficulty["matched"].items()
                if v2["in_range_of_noise_curve"] and float(k2) > 0]
    difficulty["verdict"] = (
        "BREADTH_SEPARATES_DIVERSITY_FROM_DIFFICULTY" if excesses and min(excesses) > 0.02
        else "BREADTH_IS_LARGELY_A_DIFFICULTY_METER")
    difficulty["excess_above_baseline_by_automaticity"] = {
        k2: v2["excess_above_single_drive_baseline"]
        for k2, v2 in difficulty["matched"].items()}
    difficulty["how_to_read"] = (
        "the noise curve is single-drive work made progressively harder to read. If a mixture "
        "arm sits ABOVE that curve at the same goal accuracy, the extra breadth is diversity "
        "rather than difficulty, and purpose_breadth is a usable signal. If it sits on the "
        "curve, breadth cannot tell 'many drives' from 'hard to read' and Sounding Line cannot "
        "use it as a motivational measure.")

    m3 = mixture["mu3"]
    confounded = bool(m3["goal_accuracy_should_not_fall"]["excludes_zero"]
                      and m3["goal_accuracy_should_not_fall"]["difference"] < 0)
    shallower = bool(m3["process_should_not_fall"]["excludes_zero"]
                     and m3["process_should_not_fall"]["difference"] < 0)

    # ---- standing gates ---------------------------------------------------------------------
    # THE LIVE GATE HERE IS THE ONE S-2 NEEDED AND DID NOT HAVE. S-2's mixture was drawn and
    # discarded because V5Environment.sample_feature ignores artifact.goal; the feature streams
    # were bit-identical with the manipulation off. This asserts, every run, that turning the
    # mixture from a point mass to flat actually moves what the reader sees.
    gr = G.GateReport()
    _lo = sel("mixture", mu=3, automaticity=0.0)
    _hi = sel("mixture", mu=3, automaticity=1.0)
    gr.live("mixture_reaches_the_reader",
            float(abs(_hi.purpose_breadth.mean() - _lo.purpose_breadth.mean())), 0.02,
            detail="turning the drive mixture from a point mass to flat must change the reader's "
                   "posterior breadth. S-2 shipped without this check and its manipulation never "
                   "reached the reader at all.")
    gr.live("true_mixture_actually_varies",
            float(_hi.true_mixture_breadth.mean() - _lo.true_mixture_breadth.mean()), 0.5,
            detail="the generative side of the same check: the drawn mixture itself must differ "
                   "between the arms, independently of what the reader made of it.")
    gr.positive("point_mass_mixture_has_zero_true_breadth",
                float(_lo.true_mixture_breadth.mean()), 0.0, 1e-9,
                detail="a mixture concentrated on one drive has zero entropy by construction.")

    verdict = {
        "test": "T-2 — does goal diversity rise with automaticity, at fixed decision count?",
        "for": "Sounding Line, purpose_breadth and the soul-as-variety claim",
        "SCOPE": (
            "the MECHANISM half of T-2 is not testable in this model and is not claimed. Nothing "
            "here derives drive-multiplicity from automaticity; on the mixture axis the diversity "
            "is set by hand, so 'diversity rises with automaticity' is the wiring. What is "
            "measured is the reader-side half: whether a reader sees that variety as posterior "
            "breadth without losing the goal. The DEPTH axis is the non-circular one -- nothing "
            "about mu was built to move breadth."),
        "EMITTER_NOTE": (
            "S-2's emitter does not work and this module does not use it. V5Environment."
            "sample_feature ignores artifact.goal once a creator is bound, so S-2's per-position "
            "mixture is drawn and discarded and its feature streams are bit-identical with the "
            "mixture switched off. See scripts/audit_s2_mixture.py. Here the emission is built "
            "from world.subsig[mu, active_goal, mode] directly, so the drive mixture actually "
            "reaches the reader."),
        "construction": {"read_tier": "CURATOR", "read_glances": READ_GLANCES,
                         "beta": 1.0, "n_goals": int(ng),
                         "automaticity_levels": list(AUTOMATICITY),
                         "note": "decision density identical across mixture arms by construction"},
        "axis_1_mixture_automaticity": mixture,
        "axis_2_depth_automaticity_NON_CIRCULAR": depth,
        "axis_3_decision_count_control": length,
        "axis_4_difficulty_control_DECISIVE": difficulty,
        "construct_is_confounded_with_difficulty": confounded,
        "construct_is_confounded_with_shallowness": shallower,
        "what_would_have_falsified_it": (
            "breadth flat across the automaticity sweep, which would make posterior entropy blind "
            "to motivational variety; or goal accuracy falling with it, which would make "
            "'diversity' a synonym for 'noise' and the construct unusable."),
        "what_this_cannot_show": (
            "that practice CAUSES drive-multiplicity. The depth axis tests whether practice alone "
            "moves breadth at a fixed mixture, which is the nearest non-circular question, and it "
            "is a question about this model's depth construction rather than about people."),
    }
    PROV.stamp(verdict, __file__, gr)
    (sl_dir() / "t2_automaticity.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
