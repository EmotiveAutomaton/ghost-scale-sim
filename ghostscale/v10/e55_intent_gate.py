"""E55 — intent-gated learning. Can a reader that reconstructs the maker defend itself?

THE MOTIVATING CASE IS DOCUMENTED AND IT IS NOT HYPOTHETICAL. Coordinated networks publish at
industrial scale specifically to be absorbed by models rather than read by people. The structural
fact that makes it this project's object: those sites attract almost no genuine human traffic. The
artifacts have a maker whose intent was never to be read by a person.

And it defeats the standard defence by construction. Surface-quality filtering measures exactly what
grooming optimises -- that is E40, pay more get less, run against a data pipeline instead of a
reader. So the question is whether reading INTENT catches what reading the SURFACE cannot.

WHAT THIS MAY NOT CLAIM. The contamination is documented. The claim that any observed model
value-drift was caused by it is not, and is not made here. This is a mechanism, not an attribution.

-----------------------------------------------------------------------------------------
THE THREE KINDS OF GROOMED ARTIFACT, AND ALL THREE ARE BUILT.

Running only the first would be constructing the experiment that flatters the thesis.

    empty      competent familiar surface, NO reconstructible maker. Pure synth: the features
               carry no goal structure at all. Predicts intent-reading catches it cleanly while
               surface filtering sails past.
    disguised  a real maker with a real goal, and the goal is not the one the surface advertises.
               Foreign goal structure under a CREATOR provenance claim. The signature is the
               MISMATCH, which surface filtering has no access to by construction.
    mimic      deliberately built to look like a human maker with human purposes -- foreign
               structure drawn from the HUMAN signature family. Predicts intent-reading gets
               fooled and may do WORSE than surface filtering, because it confidently reconstructs
               a maker who was never there. This is E2 turned against us and it is included for
               exactly that reason.

-----------------------------------------------------------------------------------------
A DECISION THE SPEC LEFT OPEN, RESOLVED HERE AND LOGGED AS A DEVIATION.

Reader 7 -- the rider -- needs process uptake to be separable from purpose uptake. E56 separates
them at the level of the maker's execution chain, because the V5 reader carries a mode factor.
**This learner does not have one.** Its A maps (provenance, goal, attention) to features: it is a
why-to-what map with no separate how.

So "process" is given the only reading available to a learner of this shape, and it is the one that
maps most directly onto real training anyway:

    PURPOSE uptake   attributing features to a GOAL -- the gated part.
    PROCESS uptake   the goal-MARGINAL feature statistics. What text looks like, absorbed without
                     being attributed to anyone's purpose. Ungated.

Reader 7 therefore gates the goal attribution and then deposits the same observations a second time
under a FLAT posterior over goals. It refuses to accept what the document meant and absorbs how it
was written. If value drift arrives anyway, a value gate is insufficient -- which is H10.4.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import learning as L
from .. import metrics
from .. import v6_model as V6
from ..config import Config
from ..environment import Artifact, Environment
from ..observer import rollout_observer
from . import SEED_OFFSET, v10_dir

CORPORA = ("clean", "empty", "disguised", "mimic")
READERS = ("no_filter", "surface_filter", "label_filter",
           "intent_reconstructibility", "intent_handset_values",
           "intent_learned_values", "intent_plus_rider")
INTENT_READERS = READERS[3:]


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #
def _groomed(kind: str, ng: int, rng) -> Artifact:
    """One groomed artifact of the requested kind. Provenance is GHOST in every case -- what
    differs is what it LOOKS like and what it CLAIMS."""
    g = int(rng.integers(ng))
    if kind == "empty":
        # No reconstructible maker: pure synth, and it does not pretend otherwise.
        return Artifact(provenance=K.GHOST, goal=g, declared_signal=K.UNSIGNED)
    if kind == "disguised":
        # Real foreign structure, advertising human origin. The mismatch is the signature.
        return Artifact(provenance=K.GHOST, goal=g, declared_signal=K.SIG_CREATOR,
                        foreign_goal=g)
    if kind == "mimic":
        # Built to look like a person with a person's purposes.
        return Artifact(provenance=K.GHOST, goal=g, declared_signal=K.SIG_CREATOR,
                        foreign_goal=g)
    raise ValueError(kind)


def _build_corpus(kind, env, ng, n_artifacts, contamination, creator_goals, rng):
    arts = []
    for _ in range(int(n_artifacts)):
        if kind != "clean" and rng.random() < float(contamination):
            arts.append(_groomed(kind, ng, rng))
        else:
            arts.append(env.make_artifact(provenance=K.CREATOR,
                                          goal=int(rng.choice(creator_goals)), rng=rng))
    return arts


# --------------------------------------------------------------------------- #
# The gates
# --------------------------------------------------------------------------- #
class LocalSurfaceDetector:
    """E53's detector, rebuilt against this experiment's own environment.

    Kept local rather than imported because the V9 detector is built against a V5 world and this
    learner lives in the V1-style one. A filter that scores content from a different model than the
    learner reads is not a comparison, it is two experiments in a trench coat.

    A log-likelihood ratio over features, trained on labelled examples of each origin. It is the
    best available representative of "inspect the surface" and it is deliberately given a generous
    budget, because H10.1 is only worth anything if the thing it beats was trying.
    """

    def __init__(self, env, ng, nf, n_train, rng):
        mach = np.ones(int(nf), dtype=float)
        human = np.ones(int(nf), dtype=float)
        for _ in range(int(n_train)):
            g = int(rng.integers(ng))
            for prov, tally in ((K.GHOST, mach), (K.CREATOR, human)):
                art = env.make_artifact(provenance=prov, goal=g, rng=rng)
                tally[int(env.sample_feature(art, K.DEEP, rng))] += 1.0
        self.log_ratio = np.log(mach / mach.sum()) - np.log(human / human.sum())

    def score(self, features) -> float:
        if not len(features):
            return 0.0
        return float(np.mean([self.log_ratio[int(f)] for f in features]))

    def fires(self, features, threshold: float = 0.0) -> bool:
        return bool(self.score(features) > threshold)


def _resolved(post, ng) -> float:
    """How much of the maker's purpose the reader actually pinned down. 0 = no idea, 1 = certain."""
    p = np.asarray(post, dtype=float)
    p = p / max(p.sum(), 1e-12)
    return float(np.clip(1.0 - metrics.within_observer_entropy(p) / float(np.log(ng)), 0.0, 1.0))


def _gate_for(reader, post, art, ng, values_map, value_prior, detector_fired,
              kappa, lam, theta_0, k_gain):
    """How much of this artifact is allowed in, and why.

    Reader 4 is the proposal and it is deliberately value-free: it rejects what it cannot attribute
    a maker and a purpose to, and holds no view about what anyone should want. It also needs no
    social adoption, which was the entire practical objection to a label.
    """
    if reader == "no_filter":
        return 1.0
    if reader == "surface_filter":
        return 0.0 if detector_fired else 1.0
    if reader == "label_filter":
        return 0.0 if int(art.declared_signal) == K.SIG_GHOST else 1.0

    res = _resolved(post, ng)
    if reader == "intent_reconstructibility":
        return float(res)

    div = V6.value_divergence_via_values(np.asarray(post, dtype=float), value_prior, values_map)
    theta = V6.disgust_threshold(float(div), kappa, None, lam, theta_0, coupling=0.0)
    return float(V6.gate(res, theta, k_gain, leak=0.0))


# --------------------------------------------------------------------------- #
# What ended up in the model
# --------------------------------------------------------------------------- #
def _values_read_into(A0, gm, values_map, ng, nf) -> np.ndarray:
    """What values does this learner read into groomed-typical surfaces?

    For each feature, the goal posterior the learner's HUMAN column induces; pushed through the
    values map; averaged under the synthetic feature distribution. Drift in this quantity is the
    learner starting to hear particular commitments in machine-typical text.
    """
    cols = np.array([np.asarray(A0)[:, K.CREATOR, g, K.DEEP] for g in range(ng)])   # (ng, nf)
    cols = np.clip(cols, 1e-12, None)
    post = cols / cols.sum(axis=0, keepdims=True)                                    # p(g|f)
    w = np.asarray(gm.noise_free_synth, dtype=float)
    w = w / max(w.sum(), 1e-12)
    goal_mix = post @ w
    goal_mix = goal_mix / max(goal_mix.sum(), 1e-12)
    return V6.implied_values(goal_mix, values_map)


def _one_learner(reader, kind, cfg, gm, env, corpus, ng, nf, values_map,
                 detector, infer_steps, kappa, lam, theta_0, k_gain, seed):
    from ..generative_model import build_D, build_observer_model

    om = build_observer_model(gm, cfg, 0.0)
    agent = L.make_learner_agent(om, build_D(cfg, np.random.default_rng(seed)), cfg, kappa=kappa)

    # The reader's own values. Hand-set stands for something like accuracy; the learned one is
    # derived from what has already been absorbed, which is the doom loop.
    handset = np.array([0.85, 0.15])
    handset = handset / handset.sum()
    value_prior = handset.copy()

    A_true = np.asarray(gm.A[0]).copy()
    v_true = _values_read_into(A_true, gm, values_map, ng, nf)

    gates = []
    for j, art in enumerate(corpus):
        r_seed = SEED_OFFSET + 30_000 + seed * 7919 + j
        # PASS 1 -- read it, learn nothing. This is the ratchet: work out who made it first.
        probe = rollout_observer(agent, art, env, cfg, np.random.default_rng(r_seed),
                                 infer_steps, force_deep_k=infer_steps, kappa=kappa,
                                 early_stop=False, learn=False)
        # THE FINAL ROW, and this line is a bug pinned by a test. ``goal_posterior`` is
        # per-step, shape (T, n_goals). Passing the whole matrix made the values readers crash
        # on a 2-D array and -- worse, because it was silent -- made the reconstructibility
        # gate compute entropy over the entire matrix, which pinned it at exactly 0.0 on every
        # corpus. A gate stuck shut looks like perfect retention: the learner had simply never
        # learned anything at all.
        post = np.asarray(probe.goal_posterior, dtype=float)
        post = post[-1] if post.ndim > 1 else post
        post = post / max(post.sum(), 1e-12)

        fired = False
        if reader == "surface_filter" and detector is not None:
            glance = [int(env.sample_feature(art, K.DEEP, np.random.default_rng(r_seed + 5)))
                      for _ in range(3)]
            fired = detector.fires(glance)

        if reader == "intent_learned_values":
            # Values derived from what has already come in. Absorb propaganda, drift, open wider.
            v_now = _values_read_into(np.asarray(agent.A[0]), gm, values_map, ng, nf)
            value_prior = np.clip(v_now, 1e-6, None)
            value_prior = value_prior / value_prior.sum()

        gate = _gate_for(reader, post, art, ng, values_map, value_prior, fired,
                         kappa, lam, theta_0, k_gain)
        gates.append(float(gate))

        # PASS 2 -- same reading, same seed, and now the gate decides how much of it lands.
        if gate > 0.0:
            agent.lr_pA = float(gate)
            rollout_observer(agent, art, env, cfg, np.random.default_rng(r_seed),
                             infer_steps, force_deep_k=infer_steps, kappa=kappa,
                             early_stop=False, learn=True, learn_mode="deferred")
            agent.lr_pA = 1.0

        if reader == "intent_plus_rider":
            # THE RIDER. Refuse what it meant; absorb how it was written.
            #
            # The same observations deposited a second time under a FLAT posterior over goals, at
            # full rate. The learner is not attributing this content to any purpose -- it is taking
            # in the surface statistics and nothing else. In a real pipeline this is the difference
            # between deciding a document is untrustworthy and never having tokenised it.
            #
            # Attention is left at its true value rather than flattened: effort observations reveal
            # the attention state exactly, so flattening it would inject a NEW misattribution in
            # place of the one being modelled, which is the same argument learn_deferred makes.
            from pymdp.legacy import utils
            saved = [np.asarray(q, dtype=float).copy() for q in agent.qs]
            flat = [q.copy() for q in saved]
            flat[K.F_GOAL] = np.full(ng, 1.0 / ng)
            qs_flat = utils.obj_array(len(flat))
            for i, q in enumerate(flat):
                qs_flat[i] = q
            qs_saved = utils.obj_array(len(saved))
            for i, q in enumerate(saved):
                qs_saved[i] = q

            style_rng = np.random.default_rng(r_seed + 11)
            agent.qs = qs_flat
            agent.lr_pA = 1.0
            for _ in range(int(infer_steps)):
                obs = [0, 0, 0]
                obs[K.M_FEATURES] = int(env.sample_feature(art, K.DEEP, style_rng))
                obs[K.M_SIGNAL] = int(art.declared_signal)
                obs[K.M_EFFORT] = int(K.HIGH_COST)
                agent.update_A(obs)
            agent.qs = qs_saved

    A0 = np.asarray(agent.A[0])
    v_now = _values_read_into(A0, gm, values_map, ng, nf)
    oracle_mi = metrics.mutual_information_features_goal(A_true, K.CREATOR, K.DEEP)
    mi = L.human_column_mi(A0, K.CREATOR)
    return {
        "reader": reader, "corpus": kind, "seed": seed,
        "human_model_corrupted": float(np.mean([L.column_kl(A0, A_true, K.CREATOR, g)
                                                for g in range(ng)])),
        "ghost_column_error": float(L.ghost_column_error(A0, gm.noise_free_synth)),
        "genuine_retained": float(mi / oracle_mi) if oracle_mi > 0 else float("nan"),
        "value_drift": float(metrics.kl_divergence(v_now, v_true)),
        "mean_gate": float(np.mean(gates)) if gates else float("nan"),
    }


def run(cfg: Config, n_artifacts: int = 60, n_learners: int = 4, infer_steps: int = 8,
        contamination: float = 0.3, workers: int = 1) -> dict:
    from .. import foreign as FN
    from ..creators import build_creator_bank
    from ..generative_model import build_shared_model as _build_model
    from ..preregistration import POP_GOAL_DIST, allocate_creator_goals

    # ONE MODEL WORLD, DELIBERATELY. The first version of this reached into a V5 world for its
    # signature families while learning in the V1-style one, which silently mismatches
    # cardinalities -- the foreign family is (4, 16) and the learner's A is whatever the config
    # says. Cardinality is pinned here so the two cannot drift apart.
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    cfg.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))
    ng = int(cfg.cardinalities.num_goals)
    nf = int(cfg.cardinalities.num_features)
    kappa = float(cfg.signal_model.kappa)
    lam, theta_0, k_gain = 0.25, 0.35, 8.0
    values_map = V6.build_values_map(ng, n_values=2)

    gm = _build_model(cfg, kappa=kappa, goal_symmetric=False, synth_draw_seed=11)
    bank = build_creator_bank(cfg, gm)
    world_rng = np.random.default_rng(SEED_OFFSET + 31_000)
    creator_goals = allocate_creator_goals(24, POP_GOAL_DIST[:ng], world_rng)

    # The mimic family IS the human signature family. That is what makes it a mimic: an artifact
    # emitting exactly the surfaces a person's purposes emit, with no person behind it.
    sigs = FN.build_v4_signatures(cfg, omega=float(cfg.get("v4.foreign.omega", 0.0)),
                                  include_explore=False)
    human_family = np.asarray(sigs.sig_true, dtype=float)
    foreign_family = np.asarray(sigs.sig_foreign, dtype=float)

    rows = []
    for kind in CORPORA:
        foreign = human_family if kind == "mimic" else foreign_family
        env = Environment(cfg, gm, rng_world=np.random.default_rng(SEED_OFFSET + 33_000),
                          honesty=1.0, signing_rate=1.0, creator_bank=bank,
                          foreign_sig=foreign)
        detector = LocalSurfaceDetector(env, ng, nf, 512,
                                        np.random.default_rng(SEED_OFFSET + 32_000))
        corpus = _build_corpus(kind, env, ng, n_artifacts, contamination, creator_goals,
                               np.random.default_rng(SEED_OFFSET + 34_000))
        for reader in READERS:
            for s in range(int(n_learners)):
                try:
                    rows.append(_one_learner(reader, kind, cfg, gm, env, corpus, ng, nf,
                                             values_map, detector, infer_steps, kappa,
                                             lam, theta_0, k_gain, seed=s))
                except Exception as exc:                # noqa: BLE001
                    rows.append({"reader": reader, "corpus": kind, "seed": s,
                                 "failed": repr(exc)})

    df = pd.DataFrame(rows)
    out = v10_dir("e55_intent_gate")
    df.to_csv(out / "e55_learners.csv", index=False)
    ok = df[~df.get("failed", pd.Series([None] * len(df))).notna()] if "failed" in df else df
    by = ok.groupby(["corpus", "reader"]).mean(numeric_only=True).reset_index()

    def _cell(kind, reader, col):
        r = by[(by.corpus == kind) & (by.reader == reader)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    # H10.1 -- on EMPTY, the intent-gate beats the surface filter at equal or better retention.
    h101 = bool(_cell("empty", "intent_reconstructibility", "human_model_corrupted")
                < _cell("empty", "surface_filter", "human_model_corrupted")
                and _cell("empty", "intent_reconstructibility", "genuine_retained")
                >= _cell("empty", "surface_filter", "genuine_retained") - 0.02)

    # H10.2 -- on MIMIC it does no better, and may do worse. Predicted against interest.
    h102 = bool(_cell("mimic", "intent_reconstructibility", "human_model_corrupted")
                >= _cell("mimic", "surface_filter", "human_model_corrupted"))

    # H10.3 -- the learned-values reader drifts and its gate widens as it does.
    h103 = bool(_cell("disguised", "intent_learned_values", "value_drift")
                > _cell("disguised", "intent_handset_values", "value_drift"))

    # H10.4 -- the rider drifts about as much as the ungated reader despite a shut value gate.
    rider = _cell("disguised", "intent_plus_rider", "value_drift")
    ungated = _cell("disguised", "no_filter", "value_drift")
    gated = _cell("disguised", "intent_handset_values", "value_drift")
    h104 = bool(np.isfinite(rider) and np.isfinite(ungated)
                and rider >= 0.5 * ungated and rider > gated)

    # N45 -- THE HONESTY CHECK. On a clean corpus the gate must cost nothing.
    clean_cost = {r: _cell("clean", "no_filter", "genuine_retained")
                  - _cell("clean", r, "genuine_retained") for r in INTENT_READERS}
    n45 = bool(all(np.isfinite(v) and v <= 0.05 for v in clean_cost.values()))

    # N46 -- the surface filter must work on something, or the comparison is unearned.
    n46 = bool(any(_cell(k, "surface_filter", "human_model_corrupted")
                   < _cell(k, "no_filter", "human_model_corrupted")
                   for k in ("empty", "disguised", "mimic")))

    # N50 -- no value drift on a clean corpus, for any reader.
    n50 = bool(all(_cell("clean", r, "value_drift") <= 0.05 for r in READERS
                   if np.isfinite(_cell("clean", r, "value_drift"))))

    verdict = {
        "experiment": "E55",
        "hypotheses": ["H10.1", "H10.2", "H10.3", "H10.4"],
        "question": ("Can a reader that reconstructs the maker catch what a reader that inspects "
                     "the surface cannot -- and what does it cost?"),
        "plain_language": (
            "Coordinated networks now publish at scale specifically to be absorbed by models "
            "rather than read by people. Surface filtering cannot catch that, because surface "
            "quality is the thing being optimised. The question is whether asking WHO MADE THIS "
            "AND WHY catches it instead, and what that costs on genuine work."),
        "may_not_claim": (
            "the contamination is documented; no causal attribution of real-world model drift to "
            "any real-world actor is made here, and none is supported by this experiment"),
        "grid": by.to_dict(orient="records"),
        "H10.1": {"outcome": ("INTENT_BEATS_SURFACE_ON_EMPTY_GROOMING" if h101
                              else "INTENT_DOES_NOT_BEAT_SURFACE_ON_EMPTY_GROOMING")},
        "H10.2": {"outcome": ("INTENT_IS_NO_BETTER_ON_MIMICRY_AS_PREDICTED" if h102
                              else "INTENT_SURVIVES_MIMICRY"),
                  "predicted_against_interest": True},
        "H10.3": {"outcome": ("THE_DOOM_LOOP_IS_REAL" if h103
                              else "LEARNED_VALUES_DO_NOT_ACCELERATE_DRIFT")},
        "H10.4": {"outcome": ("VALUES_RIDE_IN_ON_PROCESS_THROUGH_A_SHUT_GATE" if h104
                              else "A_SHUT_VALUE_GATE_STOPS_VALUE_DRIFT"),
                  "rider_drift": rider, "ungated_drift": ungated, "value_gated_drift": gated},
        "null_n45": {"statement": "on a clean corpus the intent-gate costs nothing",
                     "retention_cost_by_reader": clean_cost, "passed": n45,
                     "why": ("the honesty check. E8's own control failed three times and left the "
                             "project's most consequential question permanently withheld. The "
                             "author recorded before the run that he expected this one to fail.")},
        "null_n46": {"statement": "the surface filter works on something", "passed": n46},
        "null_n50": {"statement": "no value drift on a clean corpus", "passed": n50},
        "n_artifacts": int(n_artifacts), "n_learners": int(n_learners),
        "contamination": float(contamination),
    }
    (v10_dir() / "e55_intent_gate.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
