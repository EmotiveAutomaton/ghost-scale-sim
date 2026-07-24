"""prior_strength calibration for the Learner observer (DECISION D2). Gates E7/E8/E9.

WHY THIS EXISTS. pymdp adds ``calc_pA_info_gain`` to the expected free energy RAW, with no
coefficient. At the natural prior strength that term is ~130x the scale of everything else
in this model:

    prior_strength   G(DEEP) - G(SKIM)          (oracle observer, for reference: +1.03)
        0.5              +265.9
        1.0              +133.8
        4.0               +34.8
       16.0               +10.0
       64.0                +3.8

Left uncalibrated the learner engages DEEP on everything forever. That deletes E9's
STARVATION arm outright — disengagement can never occur, so there is nothing to measure —
and makes E8's engagement panel meaningless.

D2 ALSO REQUIRES DECOUPLING. ``prior_strength`` would otherwise do two jobs at once: it sets
the EFE scale of the parameter info-gain term AND it is the inverse learning rate (a strong
prior is a slow learner). Since E7's headline measurement is a convergence RATE, letting the
engagement calibration silently set the learning speed would confound it. So learning speed
is set independently by ``lr_pA`` and is not touched by this procedure.

THE ONE-PASS LIMIT (as directed). This calibration gets exactly one pass. If no candidate
satisfies both criteria, we do NOT iterate, retune, or widen the grid: we switch
``use_param_info_gain`` off for every learner experiment and record it as a deviation. Under
D1's seeding the learner already has a reason to engage GHOST content — it believes GHOST is
CREATOR-like until it learns otherwise — so the term is not load-bearing for that motivation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Config
from . import constants as K
from .generative_model import build_shared_model, build_D
from .creators import build_creator_bank
from .environment import Environment
from .observer import make_observer, rollout_observer, find_named_policies
from . import learning as L


def _efe_gap(agent) -> float:
    """G(all-DEEP) - G(all-SKIM) from the agent's current beliefs. Positive => engages."""
    q_pi, G = agent.infer_policies()
    deep, skim = find_named_policies(agent)
    idx = {tuple(map(tuple, np.asarray(p))): i for i, p in enumerate(agent.policies)}
    return float(G[idx[tuple(map(tuple, np.asarray(deep)))]]
                 - G[idx[tuple(map(tuple, np.asarray(skim)))]])


def write_criterion(cfg: Config, path: Path) -> dict:
    """Write the acceptance criterion BEFORE the sweep runs (D2).

    The point of writing it first is the same as E6b's pre-registration: a threshold chosen
    after seeing the sweep is not a threshold.
    """
    crit = {
        "decision": "D2",
        "written_before_sweep": True,
        "quantity": "learning.prior_strength",
        "rule": "the SMALLEST swept prior_strength satisfying BOTH criteria below",
        "criterion_a": {
            "what": "learner's G(DEEP)-G(SKIM) on CREATOR content, relative to the oracle's",
            "test": "gap_learner <= 2.0 * gap_oracle",
            "why": "the parameter info-gain term must not dominate the EFE; the learner "
                   "should engage for roughly the reasons the oracle does",
        },
        "criterion_b": {
            "what": "engagement on GHOST content for a learner whose GHOST column has converged",
            "test": "cum_deep < 1.0 over 20 free steps",
            "why": "E9's STARVATION channel requires that disengagement remains possible; "
                   "if a converged learner still engages GHOST, starvation cannot occur and "
                   "the arm does not exist",
        },
        "decoupling": "learning speed is set by lr_pA and is NOT adjusted by this procedure",
        "one_pass_limit": {
            "rule": "exactly one pass; no iteration, retuning, or grid widening",
            "on_failure": "set learning.use_param_info_gain=false for ALL learner "
                          "experiments and record it as a deviation in RESULTS_V2.md",
        },
        "sweep": list(cfg.get("learning.calibration_sweep", [1.0, 4.0, 16.0, 64.0, 256.0])),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(crit, indent=2), encoding="utf-8")
    return crit


def _oracle_gap(cfg: Config, gm) -> float:
    agent = make_observer(gm, cfg, np.random.default_rng(0))
    bank = build_creator_bank(cfg, gm)
    env = Environment(cfg, gm, rng_world=np.random.default_rng(1), creator_bank=bank)
    art = env.make_artifact(provenance=K.CREATOR, goal=1, rng=np.random.default_rng(2))
    agent.reset()
    agent.infer_states(env.observation(art, K.SKIM, np.random.default_rng(3)))
    return _efe_gap(agent)


def _learner_gap(cfg: Config, gm, strength: float) -> float:
    D = build_D(cfg, np.random.default_rng(0))
    agent = L.make_learner_agent(gm, D, cfg, prior_strength=strength)
    bank = build_creator_bank(cfg, gm)
    env = Environment(cfg, gm, rng_world=np.random.default_rng(1), creator_bank=bank)
    art = env.make_artifact(provenance=K.CREATOR, goal=1, rng=np.random.default_rng(2))
    agent.reset()
    agent.infer_states(env.observation(art, K.SKIM, np.random.default_rng(3)))
    return _efe_gap(agent)


def _converged_ghost_engagement(cfg: Config, gm, strength: float,
                                n_expose: int = 120, T_free: int = 20) -> float:
    """Expose a learner to GHOST content until its GHOST column has converged, then measure
    engagement when it is FREE to disengage. Criterion (b)."""
    D = build_D(cfg, np.random.default_rng(4))
    agent = L.make_learner_agent(gm, D, cfg, prior_strength=strength)
    bank = build_creator_bank(cfg, gm)
    world = np.random.default_rng(5)
    env = Environment(cfg, gm, rng_world=world, creator_bank=bank)

    # Forced exposure so the column can actually converge regardless of engagement policy.
    for _ in range(n_expose):
        art = env.make_artifact(provenance=K.GHOST, goal=int(world.integers(4)), rng=world)
        agent.reset()
        for _ in range(6):
            action = np.zeros(len(agent.num_controls)); action[K.F_ATTENTION] = K.DEEP
            agent.action = action
            obs = env.observation(art, K.DEEP, world)
            agent.infer_states(obs)
            L.learn_step(agent, obs, cfg)

    # Now free: does it disengage?
    deeps = []
    for i in range(10):
        art = env.make_artifact(provenance=K.GHOST, goal=int(world.integers(4)), rng=world)
        res = rollout_observer(agent, art, env, cfg, np.random.default_rng(900 + i), T_free,
                               learn=False, early_stop=False)
        deeps.append(res.cum_deep)
    return float(np.mean(deeps))


def calibrate(cfg: Config, results_dir: Path) -> dict:
    """Run the one calibration pass and write the result."""
    crit = write_criterion(cfg, results_dir / "v2_calibration_criterion.json")
    gm = build_shared_model(cfg, goal_symmetric=False,
                            synth_draw_seed=int(cfg.get("experiments.e6b.seed_scan_start", 1)))

    gap_oracle = _oracle_gap(cfg, gm)
    rows = []
    for s in crit["sweep"]:
        gap = _learner_gap(cfg, gm, float(s))
        ghost_deep = _converged_ghost_engagement(cfg, gm, float(s))
        rows.append({
            "prior_strength": float(s),
            "gap_learner": gap,
            "gap_oracle": gap_oracle,
            "ratio": gap / gap_oracle if gap_oracle != 0 else float("inf"),
            "criterion_a_pass": bool(gap <= 2.0 * gap_oracle),
            "ghost_cum_deep": ghost_deep,
            "criterion_b_pass": bool(ghost_deep < 1.0),
        })

    passing = [r for r in rows if r["criterion_a_pass"] and r["criterion_b_pass"]]
    chosen = min(passing, key=lambda r: r["prior_strength"]) if passing else None
    out = {
        "criterion": crit,
        "gap_oracle": gap_oracle,
        "sweep_results": rows,
        "chosen_prior_strength": chosen["prior_strength"] if chosen else None,
        "PASSED": chosen is not None,
        "fallback_applied": chosen is None,
        "action": ("set learning.prior_strength = %g" % chosen["prior_strength"] if chosen
                   else "ONE-PASS LIMIT REACHED (D2): set learning.use_param_info_gain=false "
                        "for all learner experiments and record as a deviation"),
    }
    (results_dir / "v2_calibration.json").write_text(json.dumps(out, indent=2),
                                                     encoding="utf-8")
    return out


def main():
    from .experiments import _common as C
    ap = C.standard_argparser("D2 — prior_strength calibration (gates E7/E8/E9)")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    res_dir, _ = C.ensure_dirs(args.out)
    out = calibrate(cfg, res_dir)

    print("D2 calibration — criterion written BEFORE the sweep, one pass only.\n")
    print(f"  oracle G(DEEP)-G(SKIM) on CREATOR = {out['gap_oracle']:+.4f}\n")
    print(f"  {'strength':>9} {'gap':>10} {'ratio':>8} {'(a)':>5} {'ghost_deep':>11} {'(b)':>5}")
    for r in out["sweep_results"]:
        print(f"  {r['prior_strength']:>9.1f} {r['gap_learner']:>10.3f} {r['ratio']:>8.2f} "
              f"{str(r['criterion_a_pass']):>5} {r['ghost_cum_deep']:>11.2f} "
              f"{str(r['criterion_b_pass']):>5}")
    print(f"\n  PASSED: {out['PASSED']}")
    print(f"  ACTION: {out['action']}")


if __name__ == "__main__":
    main()
