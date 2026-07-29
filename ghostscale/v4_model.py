"""V4 model assembly — a world and an observer built over the 16-feature partition.

This module deliberately does NOT modify ``generative_model.py``'s builders. V1-V3 keep their
own construction, their own cardinality, and their own seed streams, so nothing here can move
a previously reported number. V4 reaches into the shared pieces it can reuse (B, C, D, A[1],
A[2], the agent constructor) and builds its own A[0] and its own signature family.

WHAT IS DIFFERENT FROM V1-V3, AND WHY EACH DIFFERENCE EXISTS.

  * 16 features rather than 8, so ``foreign_basis`` has a disjoint partition to live in
    (decision D1; see ``foreign.py`` for the full argument).
  * The observer's goal space is 4 or 5 states depending on whether EXPLORE is enabled. The
    world's CONTENT is always produced over the four real human goals plus, in the
    exploratory-human condition, sig_EXPLORE. Observer and world are allowed to differ here
    on purpose: that asymmetry is the whole of E19.
  * Foreign content is emitted from ``sig_foreign[k]``, which appears NOWHERE in the
    observer's A[0]. That absence is the misspecification. The observer has to explain
    out-of-family structure with in-family hypotheses, which is the mechanism V4 claims
    produces confident wrong answers.

C stays exactly zero over features and signals (null N7), unchanged, and it is asserted here
as well as in the shared builder. An observer that wants artifacts to be human would make
every V4 result worthless in the same way it would a V1 one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymdp.legacy import utils

from . import constants as K
from . import foreign as FN
from .config import Config, load_config
from .generative_model import (GenerativeModel, build_A1_observer, build_A2, build_B,
                               build_C, build_D, build_noise_free_synth,
                               alpha_by_provenance, assert_preferences_zero,
                               assert_A_column_stochastic, make_agent)


def load_v4_config(quick: bool = False, include_explore: bool = True,
                   overrides: dict | None = None) -> Config:
    """A config at V4 cardinality. The goal count follows ``include_explore``, because the
    two E19 arms differ in exactly that one number and everything downstream (B[1], D[1],
    A[0]'s third axis) has to agree with it."""
    base = load_config(quick=quick)
    n_features = int(base.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4))
    n_real = int(base.get("v4.cardinalities.num_real_goals", FN.NUM_REAL_GOALS))
    ov = {
        "cardinalities.num_features": n_features,
        "cardinalities.num_goals": n_real + (1 if include_explore else 0),
    }
    if overrides:
        ov.update(overrides)
    return load_config(quick=quick, overrides=ov)


def build_v4_A0(cfg: Config, sig_observer: np.ndarray, noise_free_synth: np.ndarray,
                alpha: np.ndarray) -> np.ndarray:
    """A[0], shape (16, provenance, observer_goals, attention).

    Same mixture rule as V1-V3, so that at omega = 1 with EXPLORE disabled the structure is
    the V3 structure widened rather than replaced. That is what N16 will check at stage 3.
    """
    nf = int(cfg.cardinalities.num_features)
    ng = int(cfg.cardinalities.num_goals)
    npv = int(cfg.cardinalities.num_provenance)
    na = int(cfg.cardinalities.num_attention)
    assert sig_observer.shape == (ng, nf), (
        f"sig_observer {sig_observer.shape} does not match ({ng}, {nf}); the observer's goal "
        f"space and the config's num_goals have drifted apart")
    A0 = np.zeros((nf, npv, ng, na))
    for p in range(npv):
        a = float(alpha[p])
        for g in range(ng):
            A0[:, p, g, K.DEEP] = a * sig_observer[g] + (1.0 - a) * noise_free_synth
            A0[:, p, g, K.SKIM] = noise_free_synth
    return A0


def build_v4_synth(cfg: Config) -> np.ndarray:
    """``noise_free_synth`` over 16 features.

    V4 keeps it because A[0] still needs a SKIM column and a background for the alpha
    mixture, but its theoretical role has shrunk: in V4 the thing that breaks the observer is
    foreign STRUCTURE, not the absence of structure. The draw is symmetrised over the HUMAN
    pairs only, for the V1 reason (an unsymmetrised draw resembles one goal by chance and
    turns confident disagreement into spurious consensus), and floored for the D3b reason.
    """
    nf = int(cfg.cardinalities.num_features)
    seed = int(cfg.get("v4.foreign.draw_seed", 41)) + 1000
    conc = float(cfg.artifact_model.noise_free_synth_concentration)
    rng = np.random.default_rng(seed)
    synth = rng.dirichlet(np.full(nf, conc))
    # Symmetrise across the human goal pairs so the synth background is equidistant from
    # every real goal signature.
    for pair in FN.HUMAN_PAIRS:
        synth[pair] = synth[pair].mean()
    grouped = np.array([synth[p].sum() for p in FN.HUMAN_PAIRS])
    even = grouped.mean()
    for pair in FN.HUMAN_PAIRS:
        synth[pair] *= even / max(synth[pair].sum(), 1e-300)
    floor = float(cfg.get("artifact_model.synth_floor", 1.0e-3))
    synth = np.maximum(synth, floor)
    return synth / synth.sum()


@dataclass
class V4World:
    """The world model plus the pieces the environment needs that V1-V3 kept elsewhere."""
    gm: GenerativeModel
    sigs: FN.V4Signatures
    cfg: Config
    include_explore: bool

    @property
    def num_observer_goals(self) -> int:
        return int(self.cfg.cardinalities.num_goals)


def build_v4_world(cfg: Config, omega: float, include_explore: bool,
                   kappa: float | None = None, foreign_seed: int | None = None,
                   run_assertions: bool = True) -> V4World:
    """Build the V4 world model. Runs every C1/C2 assertion before returning anything."""
    kappa = float(cfg.signal_model.kappa) if kappa is None else float(kappa)
    sigs = FN.build_v4_signatures(cfg, omega=omega, include_explore=include_explore,
                                  foreign_seed=foreign_seed)
    synth = build_v4_synth(cfg)
    alpha = alpha_by_provenance(cfg)

    A = utils.obj_array(3)
    A[0] = build_v4_A0(cfg, sigs.sig_observer, synth, alpha)
    A[1] = build_A1_observer(cfg, kappa)
    A[2] = build_A2(cfg)

    gm = GenerativeModel(A=A, B=build_B(cfg), C=build_C(cfg), sig=sigs.sig_observer,
                         noise_free_synth=synth, alpha=alpha, kappa=kappa, cfg=cfg,
                         d=0.0, sig_true=sigs.sig_observer, synth_k=-1, synth_lean=0.0,
                         goal_symmetric=True)
    if run_assertions:
        assert_preferences_zero(gm.C)
        assert_A_column_stochastic(gm.A, cfg, where="v4 world")
    return V4World(gm=gm, sigs=sigs, cfg=cfg, include_explore=include_explore)


def make_v4_observer(world: V4World, rng: np.random.Generator, kappa: float | None = None,
                     d_i: float = 0.0):
    """One observer over the V4 world. Heterogeneous D, as V1 requires (null N3).

    ``d_i`` is V2 C1's INEXPERTISE, carried into V4 for E32. V4 never swept it — the reframe
    was about the content, so every V4 observer was a perfect reader of its own family — and
    V5 C3 is precisely the question of whether moving the content and blunting the reader are
    the same operation, so the axis has to exist on both sides of one design.

    At ``d_i = 0`` this returns the V4 observer unchanged and DRAWS NO VARIATE from the
    signature stream, which is V2's N8 hazard and its fix: ``observer_sig_rng`` spawns an
    independent child stream so the D-prior sequence is bit-identical whatever d is. Without
    that, turning expertise on would shift every downstream draw and E32's two arms would
    differ for a reason having nothing to do with C3.
    """
    from .generative_model import build_observer_model
    from .observer import observer_sig_rng

    D = build_D(world.cfg, rng)
    if float(d_i) <= 0.0:
        return make_agent(world.gm, D, world.cfg, kappa=kappa)
    gm_i = build_observer_model(world.gm, world.cfg, float(d_i),
                                rng_sig=observer_sig_rng(rng), kappa=kappa)
    return make_agent(gm_i, D, world.cfg, kappa=kappa)


# --------------------------------------------------------------------------- #
# Reading the posterior.
# --------------------------------------------------------------------------- #
def explore_mass(goal_posterior: np.ndarray, include_explore: bool) -> float:
    """Posterior mass the observer put on EXPLORE. Zero by definition when EXPLORE is not in
    the hypothesis space, which keeps the two arms' CSVs the same shape."""
    if not include_explore:
        return 0.0
    return float(np.asarray(goal_posterior, dtype=float)[FN.EXPLORE])


def real_goal_posterior(goal_posterior: np.ndarray) -> np.ndarray:
    """The posterior restricted to the four REAL goals and renormalised (decision D4).

    Every V1-V3 metric that consumes a goal posterior assumes four entries. Marginalising
    EXPLORE out and renormalising keeps those metrics comparable across V3 and V4 and across
    E19's two arms; the EXPLORE mass itself is reported separately rather than discarded,
    because in E19 it IS the measurement.

    The alternative rule, treating EXPLORE as a fifth thing people can want, would change
    every downstream number and break the boundary regression N16 is meant to check. This
    choice is fixed here, before E19 runs, rather than after seeing which one is kinder.
    """
    q = np.asarray(goal_posterior, dtype=float)[:FN.NUM_REAL_GOALS]
    s = q.sum()
    return q / s if s > 0 else np.full(FN.NUM_REAL_GOALS, 1.0 / FN.NUM_REAL_GOALS)
