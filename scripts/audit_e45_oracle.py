"""Poking E45's "the simulator needs no examples" claim until it breaks.

THE OBJECTION, WHICH IS CORRECT: you cannot simulate a maker from nothing. If a reader clears an
80% bar with zero worked examples, it was not learning. It was told.

    e45 line 121:  agent = make_agent(gm, d, cfg)          <- gm is the WORLD model
    e45 line 107:  env   = Environment(cfg, gm, rng, ...)  <- the SAME gm emits the features

``build_shared_model`` calls itself ground truth in its own docstring. So the simulator's
likelihood is not an estimate of the environment's likelihood. It IS the environment's likelihood,
the same object. The counter meanwhile estimates that likelihood by counting samples.

That is not sample efficiency. It is an oracle against a learner, and an oracle needs no data by
definition, which is why "zero" looked like a finding when it is a tautology.

Three checks:

  P1  Go below the sweep floor, as asked. n_train = 1, 2, 3. Does the counter do anything
      interesting down there, and is the simulator invariant (it must be, structurally)?
  P2  TAKE THE ORACLE AWAY. ``build_observer_model(world, cfg, d_i)`` is the codebase's own way of
      giving a reader its OWN likelihood rather than the world's: sig_i is sig_true perturbed by
      d_i. d=0 is the oracle E45 ships. Sweep d and watch what survives.
  P3  The same, on the held-out goal, which is the sharper hypothesis. If the zero-shot advantage
      is also an oracle effect it will die at the same rate.
"""
import sys

import numpy as np

sys.path.insert(0, ".")

from ghostscale import constants as K
from ghostscale.baselines import ObservationTape, TapedEnvironment, run_no_tom_classifier
from ghostscale.config import load_config
from ghostscale.environment import Artifact, Environment
from ghostscale.generative_model import (build_D, build_observer_model, build_shared_model,
                                         make_agent)
from ghostscale.observer import rollout_observer
from ghostscale.v7 import SEED_OFFSET
from ghostscale.v7.e45_tom_efficiency import N_HELD_OUT, TEST_TIER, _train_on

N_OBS, N_TIMESTEPS, FORCED_K = 150, 3, 3


def paired(cfg, gm, n_train, held, base_seed, d_i=0.0):
    """E45's loop, with the simulator's likelihood allowed to differ from the world's."""
    ng = int(cfg.cardinalities.num_goals)
    held = set(np.atleast_1d(held).tolist()) if held is not None else set()
    train_goals = [g for g in range(ng) if g not in held]
    test_goals = [g for g in range(ng) if not held or g in held]

    sim, cnt = [], []
    for i in range(N_OBS):
        rng = np.random.default_rng(SEED_OFFSET + base_seed + i)
        env = Environment(cfg, gm, rng, honesty=1.0, signing_rate=0.0)
        clf = _train_on(cfg, env, ng, train_goals, n_train, rng)

        g = int(rng.choice(test_goals))
        art = Artifact(provenance=TEST_TIER, goal=g, declared_signal=K.UNSIGNED)
        tape = ObservationTape(env, art, rng, N_TIMESTEPS)
        d = build_D(cfg, rng)

        # d_i = 0 returns the world model itself, which is exactly what E45 ships.
        obs_gm = build_observer_model(gm, cfg, d_i,
                                      np.random.default_rng(7_000_000 + base_seed + i))
        res = rollout_observer(make_agent(obs_gm, d, cfg), art, TapedEnvironment(tape), cfg, rng,
                               n_timesteps=N_TIMESTEPS, force_deep_k=FORCED_K, early_stop=False)
        sim.append(int(int(np.argmax(np.asarray(res.final_goal_posterior, float))) == g))

        out = run_no_tom_classifier(cfg, clf, tape, art, d, N_TIMESTEPS, FORCED_K,
                                    stop_confidence=0.99)
        cnt.append(int(int(np.argmax(np.asarray(out.final_goal_posterior, float))) == g))
    return float(np.mean(sim)), float(np.mean(cnt))


def main():
    cfg = load_config()
    cfg.set("inference.exact", True)
    gm = build_shared_model(cfg)
    ng = int(cfg.cardinalities.num_goals)
    held = list(range(ng - N_HELD_OUT, ng))
    bar = 0.80

    print("=" * 76)
    print("P0  is the simulator's likelihood the SAME OBJECT the world emits from?")
    obs0 = build_observer_model(gm, cfg, 0.0, np.random.default_rng(0))
    print(f"     build_observer_model(world, d=0) is world: {obs0 is gm}")
    print(f"     A[0] identical to the environment's:       "
          f"{np.array_equal(np.asarray(obs0.A[0]), np.asarray(gm.A[0]))}")
    print("     -> the reader is not estimating the emission map. It has it.")

    print()
    print("=" * 76)
    print("P1  below the sweep floor, all four goals in play")
    for n_train in (1, 2, 3, 4, 8):
        s, c = paired(cfg, gm, n_train, None, 46_000)
        flag = "  <- counter clears the bar" if c >= bar else ""
        print(f"     n_train {n_train:3d}   simulator {s:.4f}   counter {c:.4f}{flag}")

    print()
    print("=" * 76)
    print("P2  TAKE THE ORACLE AWAY: the simulator gets its own likelihood, not the world's")
    print("     d is INEXPERTISE. d=0 is what E45 ships. Counter held at 512 examples.")
    for d_i in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        s, c = paired(cfg, gm, 512, None, 46_000, d_i=d_i)
        verdict = "simulator wins" if s > c else "COUNTER WINS"
        print(f"     d={d_i:<5}  simulator {s:.4f}   counter {c:.4f}   {verdict}")

    print()
    print("=" * 76)
    print("P3  the same sweep on the HELD-OUT goal, which is the sharper hypothesis")
    for d_i in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        s, c = paired(cfg, gm, 512, held, 48_000, d_i=d_i)
        print(f"     d={d_i:<5}  simulator {s:.4f}   counter {c:.4f}   (chance 0.5000)")


if __name__ == "__main__":
    main()
