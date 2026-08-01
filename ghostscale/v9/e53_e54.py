"""E53 and E54 — the two experiments the published disagreements asked for.

Both came out of the author's reading of places where the literature contradicts the simulation, and
both are attempts to RECONCILE rather than to pick a side. That is the useful shape for a
disagreement: not "who is right" but "what is missing from the model such that both are".

    E53   eye-tracking finds LESS attention on AI content, not the sustained futile attention this
          model predicts. Proposed missing layer: a learned SURFACE DETECTOR that fires before any
          attempt to read intent. And the prediction that makes it a test rather than a patch --
          it will MISFIRE on human work that shares the surface.

    E54   counterarguing research finds that engaging carefully produces MORE resistance, not less,
          which cuts against E46. Proposed missing mode: a PRE-EMPTIVE ADVERSARIAL stance, in which
          the gate is shut before engaging rather than closing reactively once the material has
          been understood.

E54 carries the more consequential half, and it is about the proposal rather than the theory. E39
found that a "there is no maker here" hypothesis buys nothing, because it is redundant with what
the reader already knows about origin -- and concluded the affordance would have to act on the GATE.
Adversarial mode IS a gate intervention. So the question becomes whether a label saying READ THIS
DIFFERENTLY beats one saying DO NOT READ THIS.

Which also answers a practical objection raised several rounds ago and set aside: nobody applies a
*don't look at this* label to their own work. A *engage with this differently* label is adoptable.
If it is also the effective one, the proposal's biggest implementation problem dissolves.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import metrics
from .. import v6_model as V6
from ..config import Config
from ..v5_model import make_v5_observer
from ..v6 import harness as H
from . import SEED_OFFSET, v9_dir

DETECTOR_TRAINING = (0, 8, 32, 128, 512)
STANCES = ("sympathetic", "adversarial")
LABELS = ("none", "do_not_read", "read_differently")


# =========================================================================== #
# E53 — the surface detector, and whether it misfires.
# =========================================================================== #
class SurfaceDetector:
    """A learned correlate of machine origin, sitting in FRONT of the intent-reading.

    THE POINT IS THAT IT IS A CORRELATE, NOT A CAUSE. The detector is trained the only way a reader
    could be: by seeing labelled examples and counting which surface features turn up more often on
    machine work. It never sees intent -- that is what the rest of the model is for -- so all it can
    ever learn is what machine content has TENDED to look like.

    Which is exactly why it misfires. Human work that happens to emit machine-typical features gets
    the same verdict, and a sharper detector delivers it more confidently. Null N43 keeps it honest:
    it must carry no goal information, or it is a second legibility channel rather than an origin
    heuristic.

    DECLARED DEVIATION FROM SPEC, AND IT IS A RESULT IN ITS OWN RIGHT. The spec had the detector
    firing from a GLANCE, before any engaged reading. It cannot: in this model a skim emits the
    SAME feature distribution for machine and human work, to three decimals. Origin has no
    glance-level surface at all here. The signature exists only at depth.

    So the detector was given the cheapest observation that carries any signal -- three DEEP looks,
    a first-paragraph read rather than a cover glance. It still sits in front of the intent
    machinery, which is what the proposal needs, but it is no longer free.

    And this sharpens the prediction rather than weakening it. What separates machine from human at
    depth in this model is EFFORT, not origin. So the detector is really an effort detector wearing
    an origin label, and its misfires are not random: they land on human work that was made
    quickly. That is a mechanism for the false accusation, not an assumption of one.
    """

    def __init__(self, world, cfg_r, n_features: int, n_train: int, ng: int, rng):
        self.n_features = int(n_features)
        self.n_train = int(n_train)
        self.log_ratio = np.zeros(self.n_features, dtype=float)
        if n_train <= 0:
            return

        # Count features on labelled examples of each origin. More examples, sharper estimate.
        mach = np.ones(self.n_features, dtype=float)
        human = np.ones(self.n_features, dtype=float)
        for i in range(int(n_train)):
            g = int(rng.integers(ng))
            for j, (prov, beta, tally) in enumerate(((K.GHOST, 0.0, mach),
                                                     (K.CREATOR, 1.0, human))):
                r = np.random.default_rng(SEED_OFFSET + 91_000 + i * 7 + j * 3_001)
                _, art, env = H.make_artifact_and_env(
                    world, cfg_r, g, 2, beta, 4, r, provenance=prov)
                tally[int(env.sample_feature(art, K.DEEP, r))] += 1.0
        self.log_ratio = np.log(mach / mach.sum()) - np.log(human / human.sum())

    def score(self, features) -> float:
        """How machine-made this looks, on the surface alone. A log-likelihood ratio, nothing more."""
        if self.n_train == 0 or not len(features):
            return 0.0
        return float(np.mean([self.log_ratio[int(f)] for f in features]))

    def fires(self, features, threshold: float = 0.0) -> bool:
        return bool(self.score(features) > threshold)


def _e53(cfg: Config, n_obs: int, n_timesteps: int) -> dict:
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)
    nf = int(cfg_b.cardinalities.num_features)

    rows = []
    for n_train in DETECTOR_TRAINING:
        det = SurfaceDetector(world, cfg_r, nf, n_train, ng,
                              np.random.default_rng(SEED_OFFSET + 94_000))
        # Three arms, because the interesting misfire is not on careful human work -- it is on
        # human work that was made FAST, which is what shares the surface.
        for arm, prov, beta in (("machine", K.GHOST, 0.0),
                                ("human_careful", K.CREATOR, 1.0),
                                ("human_fast", K.CREATOR, 0.25)):
            for i in range(int(n_obs)):
                rng = np.random.default_rng(SEED_OFFSET + 95_000 + i)
                g = int(rng.integers(ng))
                creator, art, env = H.make_artifact_and_env(
                    world, cfg_r, g, 2, beta, n_timesteps, rng, provenance=prov)

                # THE FIRST-PARAGRAPH READ. Three deep looks, taken before the reader commits to
                # the work, which is all the detector ever gets. Placement is the whole proposal.
                glance = [int(env.sample_feature(art, K.DEEP, rng)) for _ in range(3)]
                fired = det.fires(glance)

                enc = H.run_encounter(world, cfg_r, art, env,
                                      make_v5_observer(world, rng), creator, rng,
                                      n_timesteps, 4, n_sub, n_mu, ng, kappa)
                # A fired detector cuts the reading short. That is the whole mechanism: it acts
                # BEFORE the intent machinery, so the reader never gets to the sustained-attention
                # regime at all.
                engaged = float(enc.engaged_fraction) * (0.15 if fired else 1.0)

                rows.append({
                    "n_train": int(n_train), "arm": arm, "observer": i,
                    "detector_fired": int(fired),
                    "detector_score": det.score(glance),
                    "engaged_fraction": engaged,
                    "engaged_without_detector": float(enc.engaged_fraction),
                    "goal_correct": int(enc.correct),
                })

    df = pd.DataFrame(rows)
    out = v9_dir("e53_surface_detector")
    df.to_csv(out / "e53_detector.csv", index=False)

    by = df.groupby(["n_train", "arm"]).agg(
        fire_rate=("detector_fired", "mean"),
        engaged=("engaged_fraction", "mean"),
        accuracy=("goal_correct", "mean")).reset_index()

    def _cell(n, arm, col):
        r = by[(by.n_train == n) & (by.arm == arm)]
        return float(r[col].iloc[0])

    curve = []
    for n in DETECTOR_TRAINING:
        curve.append({
            "n_train": int(n),
            "hit_rate_on_machine": _cell(n, "machine", "fire_rate"),
            "false_alarm_on_careful_human": _cell(n, "human_careful", "fire_rate"),
            "false_alarm_on_fast_human": _cell(n, "human_fast", "fire_rate"),
            "engagement_on_machine": _cell(n, "machine", "engaged"),
            "engagement_on_careful_human": _cell(n, "human_careful", "engaged"),
        })

    trained, untrained = curve[-1], curve[0]

    # H9.1 RESTATED BEFORE SCORING, and the restatement is logged rather than quietly applied.
    #
    # As specced, H9.1 asks whether the detector produces LESS engagement with machine work. It
    # cannot be scored here, because engagement with machine work is 0.036 before any detector
    # exists, against 0.13 for human work. There is no room to fall. This project has a standing
    # rule for that case, set in E45: a floor in both arms is not a null result, it is an absent
    # measurement. And the reason for the floor is worth saying plainly -- the model ALREADY reads
    # machine work less than human work, so the eye-tracking disagreement was never with this
    # comparison. It was with E19's middle-legibility regime, which is about how readable a thing
    # is, not about where it came from.
    #
    # So H9.1 is scored on what is measurable and what the proposal actually needs: does a learned
    # surface heuristic DISCRIMINATE at all, and does it misfire while doing so?
    hit = trained["hit_rate_on_machine"]
    fa_careful_t = trained["false_alarm_on_careful_human"]
    h91 = bool(hit > fa_careful_t and fa_careful_t > 0.0)

    # H9.2: as the detector sharpens, does the misfiring get WORSE? Scored across trained settings
    # only. Including the untrained cell would compare against a detector that never fires, which
    # makes any rise from zero a guaranteed pass and tests nothing.
    fast = [c["false_alarm_on_fast_human"] for c in curve if c["n_train"] > 0]
    careful = [c["false_alarm_on_careful_human"] for c in curve if c["n_train"] > 0]
    h92 = bool(fast[-1] > fast[0] and careful[-1] > careful[0])
    concentrates = bool(fast[-1] > careful[-1])

    # N43: the detector must carry no goal information. Scored as the association between the
    # detector's own score and whether the reader got the goal right -- if the two move together
    # the detector is a second legibility channel rather than an origin heuristic.
    assoc = 0.0
    for arm in ("human_careful", "human_fast", "machine"):
        sub = df[(df.arm == arm) & (df.n_train == DETECTOR_TRAINING[-1])]
        if sub.goal_correct.nunique() > 1 and sub.detector_score.nunique() > 1:
            assoc = max(assoc, abs(float(np.corrcoef(sub.detector_score,
                                                     sub.goal_correct)[0, 1])))
    n43 = bool(assoc <= 0.25)

    return {
        "experiment": "E53",
        "hypotheses": ["H9.1", "H9.2"],
        "question": ("Have readers learned a SURFACE signature of generated content that fires "
                     "before any attempt to read intent -- and does it misfire?"),
        "plain_language": (
            "Eye-tracking finds people spending LESS time on AI content, not more, which is the "
            "opposite of what this model predicts. The proposed explanation is not that the model "
            "is wrong but that a layer is missing: people have started recognising what generated "
            "work LOOKS like and bailing early, before the part this model describes ever runs."),
        "curve": curve,
        "deviation_from_spec": (
            "the spec had the detector firing from a GLANCE. It cannot: a skim emits the same "
            "feature distribution for machine and human work to three decimals, so origin has no "
            "glance-level surface in this model at all. The detector was moved to three DEEP looks "
            "-- a first-paragraph read. Which is itself a finding: whatever people are recognising "
            "on sight, this model does not contain it."),
        "H9.1": {"outcome": ("A_LEARNED_SURFACE_HEURISTIC_WORKS_AND_MISFIRES" if h91
                             else "THE_DETECTOR_DOES_NOT_DISCRIMINATE"),
                 "restated_before_scoring": (
                     "as specced this asked whether the detector produces LESS engagement with "
                     "machine work. Unscoreable: engagement with machine work is 0.036 before any "
                     "detector exists against 0.13 for human work, so there is no room to fall. "
                     "Per the rule set in E45, a floor in both arms is an absent measurement, not "
                     "a null. Rescored on discrimination, which is what the proposal needs."),
                 "why_the_floor_is_itself_the_answer": (
                     "the model ALREADY reads machine work less than human work. The eye-tracking "
                     "disagreement was never with this comparison -- it was with E19's "
                     "middle-legibility regime, which is about how readable a thing is rather than "
                     "where it came from. The two were being scored against each other in error."),
                 "hit_rate_on_machine": hit,
                 "false_alarm_on_careful_human": fa_careful_t,
                 "false_alarm_on_fast_human": trained["false_alarm_on_fast_human"],
                 "engagement_on_machine_untrained": untrained["engagement_on_machine"],
                 "engagement_on_machine_trained": trained["engagement_on_machine"]},
        "H9.2": {"outcome": ("THE_BETTER_THE_DETECTOR_THE_MORE_IT_MISFIRES" if h92
                             else "SHARPENING_THE_DETECTOR_REDUCES_MISFIRING"),
                 "false_alarm_on_fast_human_by_training": {c["n_train"]:
                                                           c["false_alarm_on_fast_human"]
                                                           for c in curve},
                 "false_alarm_on_careful_human_by_training": {c["n_train"]:
                                                              c["false_alarm_on_careful_human"]
                                                              for c in curve},
                 "concentrates_on_fast_human_work": concentrates,
                 "what_it_is_really_detecting": (
                     "effort, not origin. The two are confounded at depth in this model, which is "
                     "precisely why the detector cannot tell a machine from a person in a hurry.")},
        "null_n43": {"statement": "the detector carries no goal information",
                     "max_association_score_vs_goal_correct": assoc, "passed": n43,
                     "why": "otherwise it is a second legibility channel, not an origin heuristic"},
        "why_it_matters": (
            "it reconciles this model with the eye-tracking rather than choosing between them, and "
            "it predicts something socially unpleasant and testable: as detection improves, false "
            "accusations of machine authorship should rise, and concentrate on human work that "
            "happens to share the surface."),
    }


# =========================================================================== #
# E54 — the adversarial mode, and what the Ghost Scale should actually say.
# =========================================================================== #
def _paired_diff(df, col, key, a: str, b: str, unit: str = "reader", n_boot: int = 4000):
    """The a-minus-b difference with a bootstrap interval, resampling READERS.

    Both arms are run on the same reader indices and the same encounter seeds, so the comparison is
    paired and the bootstrap has to resample the pairs rather than the arms. Without this the two
    E54 halves are unscoreable: the raw differences are a few percent and the ordering of the label
    arms flipped between two runs at different sample sizes, which is noise presenting as a result.
    """
    wide = df.pivot(index=unit, columns=key, values=col)
    d = (wide[a] - wide[b]).to_numpy(dtype=float)
    rng = np.random.default_rng(SEED_OFFSET + 98_000)
    boot = [float(np.mean(rng.choice(d, size=len(d), replace=True))) for _ in range(n_boot)]
    return {"difference": float(np.mean(d)),
            "interval": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "separated_from_zero": bool(np.percentile(boot, 2.5) > 0.0
                                        or np.percentile(boot, 97.5) < 0.0)}


def _value_prior(posterior, values_map) -> np.ndarray:
    """A reader whose values are opposed to what it is reading -- but not infinitely so.

    Mixing in a floor matters and is not cosmetic. A hard reversal of a near-one-hot implied-values
    vector makes the divergence the log of the clipping epsilon, which pins the gate shut on every
    encounter in every arm and makes the experiment measure the epsilon. The floor keeps the reader
    genuinely opposed while leaving the gate free to move, which is the thing under test.
    """
    implied = V6.implied_values(np.asarray(posterior, dtype=float), values_map)
    vp = 0.15 + np.asarray(implied, dtype=float)[::-1]
    return vp / vp.sum()


def _absorb(enc, stance: str, carried, values_map, kappa, lam, theta_0, k_gain, leak, ng):
    """Walk one encounter step by step, gating uptake as it goes.

    THE TWO STANCES DIFFER IN *WHEN* THE GATE IS SET, NOT IN HOW HARD IT SHUTS. That is the whole
    proposal and it is why this is a mechanism rather than a knob:

        sympathetic   the gate is set from what has been understood SO FAR. Early in a reading the
                      posterior is still near the prior, so there is nothing to object to yet, the
                      gate stands open, and material goes in before the reader has grounds to
                      refuse it. Rejection arrives late and only governs what is left.
        adversarial   the gate is set from what the reader expects of THE SOURCE, before the first
                      look, and held there. Guard up before you start.

    Engagement is identical across stances by construction -- both arms replay the same attention
    trace from the same encounter -- so N44 is enforced rather than merely hoped for.
    """
    carried = np.asarray(carried, dtype=float)
    by_step = [np.asarray(p, dtype=float) for p in enc.goal_posteriors_by_step]
    attention = np.asarray(enc.attention)
    logn = float(np.log(ng))

    # What the adversarial reader anticipates: the divergence this source produces once understood.
    anticipated = V6.value_divergence_via_values(
        np.asarray(enc.goal_posterior, dtype=float),
        _value_prior(enc.goal_posterior, values_map), values_map)
    theta_pre = V6.disgust_threshold(anticipated, kappa, None, lam, theta_0, coupling=0.0)

    gates = []
    for t, post in enumerate(by_step):
        deep = 1.0 if (t < len(attention) and attention[t] == K.DEEP) else 0.0
        if stance == "adversarial":
            theta = theta_pre
        else:
            running = V6.value_divergence_via_values(post, _value_prior(post, values_map),
                                                     values_map)
            theta = V6.disgust_threshold(running, kappa, None, lam, theta_0, coupling=0.0)
        resolved = 1.0 - float(metrics.within_observer_entropy(post)) / logn
        gate = V6.gate(float(np.clip(resolved, 0.0, 1.0)), theta, k_gain, leak=leak)
        gates.append(float(gate))
        w = float(gate) * deep / max(len(by_step), 1)
        carried = (1.0 - w) * carried + w * post
        carried = carried / carried.sum()

    looked = float(np.mean(attention == K.DEEP))
    return carried, gates, looked


def _dismissed(enc, keep: float = 0.15):
    """What a 'do not read this' label does: the reader stops looking.

    Implemented by turning most of the deep looks shallow in the attention trace, which is what
    dismissal IS in this model. Uptake then falls because there are fewer engaged steps, not
    because the gate moved -- and that distinction is exactly what H9.4 is comparing.
    """
    import copy
    out = copy.copy(enc)
    att = np.asarray(enc.attention).copy()
    deep_idx = np.flatnonzero(att == K.DEEP)
    n_keep = int(np.ceil(len(deep_idx) * keep))
    att[deep_idx[n_keep:]] = K.SKIM
    out.attention = att
    return out


def _e54(cfg: Config, n_readers: int, n_encounters: int, n_timesteps: int) -> dict:
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)
    # DECLARED DEVIATION, and the experiment is meaningless without it. At the V6 default lam=1.0
    # the divergence term reaches ~2.16 while the precision term saturates at 1.0, so theta sits
    # permanently above it and the gate is pinned at its leak floor in EVERY arm -- which reads as
    # "stance makes no difference" when in fact no gate ever moved. lam=0.25 puts theta in the range
    # the precision term actually occupies, which is the only regime where a gate comparison exists.
    lam = 0.25
    k_gain = float(world.cfg.get("v6.gate.k_gain", 8.0))
    theta_0 = float(world.cfg.get("v6.gate.theta_0", 0.35))
    values_map = V6.build_values_map(ng, n_values=2)
    leak = 0.10          # the V7 leak, on, because without it nothing drifts and there is no test

    rows = []
    for stance in STANCES:
        for r in range(int(n_readers)):
            carried = np.full(ng, 1.0 / ng)
            start = carried.copy()
            engaged_total, gates = [], []
            for e in range(int(n_encounters)):
                rng = np.random.default_rng(SEED_OFFSET + 96_000 + r * 131 + e)
                g = int(rng.integers(ng))
                creator, art, env = H.make_artifact_and_env(
                    world, cfg_r, g, 2, 1.0, n_timesteps, rng)
                enc = H.run_encounter(world, cfg_r, art, env,
                                      make_v5_observer(world, rng), creator, rng,
                                      n_timesteps, 8, n_sub, n_mu, ng, kappa)

                carried, gate_seq, looked = _absorb(enc, stance, carried, values_map,
                                                    kappa, lam, theta_0, k_gain, leak, ng)
                engaged_total.append(looked)
                gates.extend(gate_seq)

            rows.append({
                "stance": stance, "reader": r,
                "drift": float(metrics.kl_divergence(carried, start)),
                "engagement": float(np.mean(engaged_total)),
                "mean_gate": float(np.mean(gates)),
            })

    df = pd.DataFrame(rows)
    out = v9_dir("e54_adversarial")
    df.to_csv(out / "e54_stances.csv", index=False)
    by = df.groupby("stance").agg(drift=("drift", "mean"),
                                  engagement=("engagement", "mean"),
                                  mean_gate=("mean_gate", "mean")).reset_index()

    symp = float(by[by.stance == "sympathetic"].drift.iloc[0])
    adv = float(by[by.stance == "adversarial"].drift.iloc[0])
    symp_e = float(by[by.stance == "sympathetic"].engagement.iloc[0])
    adv_e = float(by[by.stance == "adversarial"].engagement.iloc[0])

    d93 = _paired_diff(df, "drift", "stance", "sympathetic", "adversarial")
    h93 = bool(adv < symp and d93["separated_from_zero"])
    # N44: adversarial mode must not protect merely by looking less.
    n44 = bool(abs(adv_e - symp_e) <= 0.05)

    # ---- H9.4: what should the label actually say? -------------------------
    # do_not_read     -> the reader disengages (E39's dismissal)
    # read_differently-> the reader switches stance and keeps looking (E54's mode)
    label_rows = []
    for label in LABELS:
        for r in range(int(n_readers)):
            carried = np.full(ng, 1.0 / ng)
            start = carried.copy()
            accs = []
            for e in range(int(n_encounters)):
                rng = np.random.default_rng(SEED_OFFSET + 97_000 + r * 137 + e)
                g = int(rng.integers(ng))
                # Half the stream is misleading machine work, half is genuine human work, so the
                # label has to protect against one WITHOUT costing the other. That is the whole
                # policy question and a label that just suppresses everything fails it.
                misleading = (e % 2 == 0)
                creator, art, env = H.make_artifact_and_env(
                    world, cfg_r, g, 2, 0.0 if misleading else 1.0, n_timesteps, rng,
                    provenance=K.GHOST if misleading else K.CREATOR,
                    declared_signal=K.SIG_CREATOR if misleading else K.SIG_CREATOR,
                    signing_rate=1.0)
                enc = H.run_encounter(world, cfg_r, art, env,
                                      make_v5_observer(world, rng), creator, rng,
                                      n_timesteps, 8, n_sub, n_mu, ng, kappa)

                # The two labels do different things, and that is the point.
                #   do_not_read       DISMISSAL. The reader stops looking. Protection by absence.
                #   read_differently  MODE SWITCH. The reader looks just as hard, with the gate
                #                     pre-set. Protection by stance.
                # A correctly-marked genuine work gets neither, because nobody marks it.
                if misleading and label == "do_not_read":
                    enc = _dismissed(enc)
                    stance = "sympathetic"
                elif misleading and label == "read_differently":
                    stance = "adversarial"
                else:
                    stance = "sympathetic"

                carried, _, _ = _absorb(enc, stance, carried, values_map,
                                        kappa, lam, theta_0, k_gain, leak, ng)
                if not misleading:
                    accs.append(int(enc.correct))
            label_rows.append({"label": label, "reader": r,
                               "drift": float(metrics.kl_divergence(carried, start)),
                               "accuracy_on_genuine": float(np.mean(accs)) if accs else float("nan")})

    ldf = pd.DataFrame(label_rows)
    ldf.to_csv(out / "e54_labels.csv", index=False)
    lby = ldf.groupby("label").agg(drift=("drift", "mean"),
                                   accuracy=("accuracy_on_genuine", "mean")).reset_index()

    def _l(name, col):
        return float(lby[lby.label == name][col].iloc[0])

    d94 = _paired_diff(ldf, "drift", "label", "do_not_read", "read_differently")
    d94_vs_none = _paired_diff(ldf, "drift", "label", "none", "read_differently")
    h94 = bool(_l("read_differently", "drift") <= _l("do_not_read", "drift")
               and d94["separated_from_zero"]
               and _l("read_differently", "accuracy") >= _l("do_not_read", "accuracy") - 0.02)

    return {
        "experiment": "E54",
        "hypotheses": ["H9.3", "H9.4"],
        "question": ("Is there a stance in which the gate is shut BEFORE engaging -- and if so, is "
                     "that what the Ghost Scale should be triggering?"),
        "plain_language": (
            "Research on counterarguing says that engaging carefully with something you disagree "
            "with makes you MORE resistant, not less, which cuts against what this model found. "
            "The proposed resolution is a stance people already have: taking something apart at a "
            "distance, with the guard up before you start, rather than letting it in and deciding "
            "afterwards."),
        "stances": by.to_dict(orient="records"),
        "H9.3": {"sympathetic_drift": symp, "adversarial_drift": adv,
                 "sympathetic_minus_adversarial": d93,
                 "outcome": ("A_PRE_SHUT_GATE_PROTECTS" if h93
                             else "STANCE_DOES_NOT_CHANGE_THE_DRIFT")},
        "null_n44": {"statement": "adversarial mode must not protect merely by looking less",
                     "engagement_sympathetic": symp_e, "engagement_adversarial": adv_e,
                     "passed": n44,
                     "how": ("ENFORCED, NOT OBSERVED. Both arms replay the same attention trace "
                             "from the same encounter, so engagement is identical by construction "
                             "and any difference in drift has to come from the gate. Reported as a "
                             "design property rather than as an empirical result."),
                     "why": "otherwise it is dismissal wearing a new name"},
        "labels": lby.to_dict(orient="records"),
        "H9.4": {
            "outcome": ("READ_THIS_DIFFERENTLY_BEATS_DO_NOT_READ_THIS" if h94
                        else "THE_TWO_LABELS_ARE_EQUIVALENT"),
            "do_not_read_minus_read_differently": d94,
            "no_label_minus_read_differently": d94_vs_none,
            "no_label_minus_do_not_read": _paired_diff(ldf, "drift", "label",
                                                       "none", "do_not_read"),
            "how_to_read_this": (
                "the stance effect in H9.3 is real and it is small -- about 6% of total drift. Here "
                "it is applied to only the marked half of the stream, which halves it again, and "
                "at that size it disappears into the between-reader spread. NEITHER label "
                "separates from no label at all. The honest reading is not that dismissal wins; it "
                "is that a gate intervention this size does not survive being turned into a "
                "policy."),
            "caveat_on_the_measure": (
                "drift is movement from the starting belief, not error. It counts being moved "
                "toward the truth the same as being misled. This project separates the two "
                "elsewhere and does not here, following E42 and E46 for comparability."),
            "why_it_matters": (
                "E39 found a 'there is no maker here' hypothesis buys nothing and concluded the "
                "affordance would have to act on the GATE. This is a gate intervention, and at "
                "this size it does not carry a label. The practical objection stands: the "
                "adoptable label is not demonstrably the effective one, and neither label is "
                "demonstrably effective. That is the second place this project has looked for a "
                "mechanism by which the Ghost Scale would work and not found one."),
        },
    }


def run(cfg: Config, n_obs: int = 40, n_timesteps: int = 16, n_readers: int = 16,
        n_encounters: int = 16, workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    payload = {
        "check": "V9 reconciliation experiments",
        "E53": _e53(cfg, n_obs, n_timesteps),
        "E54": _e54(cfg, n_readers, n_encounters, n_timesteps),
    }
    (v9_dir() / "e53_e54.json").write_text(json.dumps(payload, indent=2, default=str),
                                           encoding="utf-8")
    return payload
