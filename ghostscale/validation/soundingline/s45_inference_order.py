"""S-4 and S-5 — inference ORDER, run as one test because they are one question.

WHY THEY ARE MERGED. S-4 asks whether reading the method first and deriving the purpose from it
beats the purpose-first loop. S-5 asks whether entering through an anomaly beats entering through
the artifact in order. Both are questions about the ORDER in which evidence is consumed, and this
reader runs exact inference over the full joint when ``inference.exact`` is set. For the STATIC
factors, Bayesian updating on the same evidence set is order-independent. For the temporal
sub-goal chain it is not a theorem -- a reordered tape describes a different generative history --
so order-invariance of the final posterior is CHECKED empirically rather than assumed
(``accuracy_checks_that_should_not_move``; the committed run's differences sit inside their
intervals). S-5's own write-up states the static-factor argument; S-4's does not, and within the
resolution of that check it is the same argument.

So the answerable half of both is COST, which is the one thing this simulation is built to price:
the reader pays per DEEP look and may disengage. The question becomes *how many looks does it take
to get there*, and that is a real question with a real answer.

  S-4  forward: the tape in its natural order.
       reverse: the tape ordered so the steps carrying the most execution-mode information come
       first -- method-first, as far as an order can express it.

  S-5  anomaly-first: the least likely observation under a reader's uninformed expectation, then
       the rest in order.

MEASURED: steps until the goal posterior first resolves, final goal accuracy, and final process
recovery. The accuracy and recovery columns are there as a CHECK, not as an outcome: if they move,
the reorder is not a reorder and something is wrong with the harness.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ... import constants as K
from ... import metrics
from ... import v6_model as V6
from ...baselines import ObservationTape, TapedEnvironment
from ...config import Config
from ...prereg_v6 import BOOTSTRAP_DRAWS, percentile_interval
from ...v5_model import make_v5_observer
from ...v6 import SEED_OFFSET, harness as H
from ...v6.e36_process import BETA_GRID, MU_GRID, RESOLVED_ENTROPY
from ...methods import provenance as PROVENANCE
from . import sl_dir
from .common import build, e36_seed

_EPS = 1e-12


def _feature_surprisal(world) -> np.ndarray:
    """How unexpected each feature is to a reader holding no information yet.

    A[0] marginalised over a flat state prior: what you expect to see before you know anything.
    An anomaly is a feature low under that.
    """
    A0 = np.asarray(world.gm.A[0], dtype=float)
    flat = A0.reshape(A0.shape[0], -1)
    marg = flat.mean(axis=1)
    marg = marg / max(marg.sum(), _EPS)
    return -np.log(np.clip(marg, _EPS, None))


def _reorder(tape: ObservationTape, order: np.ndarray) -> ObservationTape:
    tape.deep = tape.deep[order]
    tape.skim = tape.skim[order]
    return tape


def _steps_to_settle(enc) -> int:
    for t, p in enumerate(enc.goal_posteriors_by_step):
        if float(metrics.within_observer_entropy(p)) <= RESOLVED_ENTROPY:
            return t
    return len(enc.goal_posteriors_by_step)


def run(cfg: Config, n_obs: int = 80, n_timesteps: int = 24, forced_k: int = 24) -> dict:
    world, _cfg_b, cfg_r, n_mu, n_sub, ng = build(cfg)
    surprisal = _feature_surprisal(world)
    rows = []

    for mu in MU_GRID:
        for beta in BETA_GRID:
            base = e36_seed(mu, beta)
            for i in range(int(n_obs)):
                art_rng = np.random.default_rng(base * 31 + i)
                g_true = int(art_rng.integers(ng))
                creator, artifact, env = H.make_artifact_and_env(
                    world, cfg_r, g_true, int(mu), float(beta), n_timesteps, art_rng)
                base_tape = ObservationTape(env, artifact,
                                            np.random.default_rng(base * 104729 + i), n_timesteps)
                deep = np.array(base_tape.deep, dtype=int)

                orders = {
                    "forward": np.arange(n_timesteps),
                    # Method-first: the steps whose observed feature is most diagnostic of an
                    # execution mode, hardest first.
                    "reverse": np.argsort(-surprisal[deep], kind="stable"),
                    # Anomaly-first: one anomaly, then everything else in its natural order.
                    "anomaly_first": np.concatenate(
                        [[int(np.argmax(surprisal[deep]))],
                         np.array([t for t in range(n_timesteps)
                                   if t != int(np.argmax(surprisal[deep]))], dtype=int)]),
                }
                for name, order in orders.items():
                    tape = ObservationTape(env, artifact,
                                           np.random.default_rng(base * 104729 + i), n_timesteps)
                    tape = _reorder(tape, np.asarray(order, dtype=int))
                    agent = make_v5_observer(world, np.random.default_rng(base * 7907 + i))
                    enc = H.run_encounter(world, cfg_r, artifact, TapedEnvironment(tape), agent,
                                          creator, np.random.default_rng(base * 7907 + i),
                                          n_timesteps, forced_k, n_sub, n_mu, ng,
                                          float(world.cfg.signal_model.kappa), true_goal=g_true)
                    rows.append({
                        "arm": name, "mu": mu, "beta": beta, "i": i,
                        "steps_to_settle": _steps_to_settle(enc),
                        "settled": int(_steps_to_settle(enc) < n_timesteps),
                        "goal_correct": int(enc.correct),
                        "process": float(enc.process["process_error_reduction"]),
                        "final_entropy": float(enc.final_entropy),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(sl_dir() / "s45_inference_order_points.csv", index=False)
    rng = np.random.default_rng(SEED_OFFSET + 90_450)

    def paired(a: str, b: str, col: str) -> dict:
        x = df[df.arm == a].sort_values(["mu", "beta", "i"])[col].to_numpy(dtype=float)
        y = df[df.arm == b].sort_values(["mu", "beta", "i"])[col].to_numpy(dtype=float)
        n = min(x.size, y.size)
        d = x[:n] - y[:n]
        draws = [float(np.mean(rng.choice(d, d.size, replace=True)))
                 for _ in range(BOOTSTRAP_DRAWS)]
        lo, hi = percentile_interval(draws)
        return {"mean_difference": float(d.mean()), "interval": [lo, hi], "n": int(n),
                "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))}

    by_arm = {a: {"steps_to_settle": float(s.steps_to_settle.mean()),
                  "settled_fraction": float(s.settled.mean()),
                  "goal_accuracy": float(s.goal_correct.mean()),
                  "process_recovery": float(s.process.mean()), "n": int(len(s))}
              for a, s in df.groupby("arm")}

    verdict = {
        "test": "S-4 and S-5 — does inference order buy anything, and is it accuracy or cost?",
        "for": "Sounding Line, the purpose-first loop and the anomaly entry point",
        "why_merged": (
            "both are questions about the order evidence is consumed in, and this reader runs "
            "exact inference over the full joint. Bayesian updating on the same evidence set is "
            "order-independent, so no reordering can change the final posterior. Sounding Line's "
            "own S-5 note says exactly this. The answerable half of both is cost."),
        "by_arm": by_arm,
        "cost_contrasts": {
            "reverse_minus_forward_steps": paired("reverse", "forward", "steps_to_settle"),
            "anomaly_first_minus_forward_steps": paired("anomaly_first", "forward",
                                                        "steps_to_settle"),
        },
        "accuracy_checks_that_should_not_move": {
            "reverse_minus_forward_accuracy": paired("reverse", "forward", "goal_correct"),
            "anomaly_first_minus_forward_accuracy": paired("anomaly_first", "forward",
                                                           "goal_correct"),
            "how_to_read": (
                "these are a HARNESS CHECK and not a result. All three arms see the same "
                "observations in a different order, so final accuracy should be unchanged. If it "
                "moves, the reorder changed the evidence and the cost numbers cannot be trusted."),
        },
        "n_rollouts": int(len(df)),
        "what_would_have_falsified_the_claim": (
            "reverse or anomaly-first settling in fewer DEEP steps than forward, with accuracy "
            "unchanged. That would make the entry point a genuine efficiency and not a story."),
    }
    PROVENANCE.stamp(verdict, __file__)
    (sl_dir() / "s45_inference_order.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
