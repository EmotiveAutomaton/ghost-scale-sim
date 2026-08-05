"""T-10 — the decision's IDENTITY is unrecoverable. Is its LOCATION?

T-3 ANSWERED A QUESTION AND CLOSED A DOOR. The sub-goal posterior floors at about 2.33 effective
modes in the most favourable regime the model admits, so "a decision was recovered" never becomes
a well-defined event and every instrument built on counting decisions should be abandoned rather
than repaired.

That is a finding about WHICH mode. It is not a finding about WHEN the mode changed, and those are
different questions with different answers and very different consequences for an instrument.

  which  needs the posterior to concentrate on one of four modes. It does not.
  when   needs the posterior to MOVE when the maker switches. It might, while staying diffuse
         the entire time -- a belief can travel from one uncertain state to another uncertain
         state, and the travel is visible even when neither endpoint is.

**If boundaries are recoverable and identities are not, "how many decisions" is dead but "where
the seams are" is alive, and those are different instruments.** T-3's negative retires a family of
designs; this asks whether a neighbouring family survives it. That is worth one module.

THE MEASUREMENT. The creator's execution chain switches mode at known steps. The reader never sees
them. Score three detectors, all computed from the reader's own beliefs with no access to truth:

  travel      per-step L1 movement of the sub-goal posterior -- T-5's best single feature, and the
              obvious candidate for "the belief moved because the maker moved".
  entropy     per-step change in sub-goal entropy: does uncertainty spike at a seam?
  catch22     whether any canonical feature of the trajectory tracks the SWITCH RATE, which is a
              weaker claim than locating individual seams and a more transportable one.

Scored as AUC over steps -- is a switch step ranked above a non-switch step -- with a
circular-shift null, which preserves the trajectory's own autocorrelation and destroys only its
alignment to the truth. A plain permutation null would be too easy to beat: a smooth series beats
shuffled noise for reasons that have nothing to do with seams.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ... import constants as K
from ... import metrics
from ...config import Config
from ...methods import gates as G
from ...methods import provenance as PROVENANCE
from ...methods import trajectory as TRAJ
from ...v5_model import make_v5_observer, marginal_subgoal
from ...v6 import SEED_OFFSET, harness as H
from . import sl_dir
from . import t_common as T
from .t3_countability import _world
from .t5_detection import auc

N_TIMESTEPS = 48
#: Dwell is the axis T-3 found binding, so it is the axis a boundary detector has to be swept on:
#: a maker who switches every other step and one who never switches are different instruments'
#: problems.
DWELLS = (4.0, 9.2, 20.0)
MUS = (2, 3)
BETAS = (1.0, 0.25)


def _encounter(world, cfg_r, mu, beta, g, n_mu, n_sub, ng, rng_a, rng_o):
    from ...observer import rollout_observer
    creator, artifact, env = H.make_artifact_and_env(
        world, cfg_r, int(g), int(mu), float(beta), N_TIMESTEPS, rng_a, provenance=K.CREATOR)
    agent = make_v5_observer(world, rng_o)
    res = rollout_observer(agent, artifact, env, cfg_r, rng_o, n_timesteps=N_TIMESTEPS,
                           force_deep_k=N_TIMESTEPS, initial_glance=True, early_stop=False)
    rows = np.asarray(res.goal_posterior, dtype=float)
    S = np.asarray([marginal_subgoal(r, n_mu, ng, n_sub) for r in rows], dtype=float)
    ent = np.array([metrics.within_observer_entropy(x) for x in S])
    travel = np.concatenate([[0.0], np.abs(np.diff(S, axis=0)).sum(axis=1)])
    d_ent = np.concatenate([[0.0], np.abs(np.diff(ent))])
    modes = np.asarray(H.creator_positions(creator, N_TIMESTEPS, initial_glance=True), dtype=int)
    switch = np.concatenate([[0], (np.diff(modes) != 0).astype(int)])
    return travel, d_ent, ent, switch


def _shift_null(signal: np.ndarray, switch: np.ndarray, rng, draws: int = 200) -> float:
    """AUC under circular shifts: keeps the signal's autocorrelation, destroys its alignment."""
    n = signal.size
    out = []
    for _ in range(int(draws)):
        k = int(rng.integers(1, n))
        s = np.roll(signal, k)
        pos, neg = s[switch == 1], s[switch == 0]
        if pos.size and neg.size:
            a = auc(pos, neg)
            if np.isfinite(a):
                out.append(a)
    return float(np.mean(out)) if out else float("nan")


def run(cfg: Config, n_obs: int = 120) -> dict:
    rng = np.random.default_rng(SEED_OFFSET + 92_000)
    rows, rate_rows = [], []

    for dwell in DWELLS:
        world, cfg_r, n_mu, n_sub, ng = _world(cfg, 1.0, dwell, 4)
        for mu in MUS:
            for beta in BETAS:
                seed = 76_000 + int(dwell * 10) + mu * 7 + int(beta * 100)
                for i in range(int(n_obs)):
                    travel, d_ent, ent, switch = _encounter(
                        world, cfg_r, mu, beta, int(i % ng), n_mu, n_sub, ng,
                        np.random.default_rng(seed * 31 + i),
                        np.random.default_rng(seed * 7907 + i))
                    if switch.sum() == 0 or switch.sum() == switch.size:
                        continue
                    row = {"dwell": dwell, "mu": mu, "beta": beta, "i": i,
                           "n_switches": int(switch.sum()),
                           "auc_travel": auc(travel[switch == 1], travel[switch == 0]),
                           "auc_entropy_change": auc(d_ent[switch == 1], d_ent[switch == 0])}
                    row["null_travel"] = _shift_null(travel, switch, rng, draws=60)
                    row["null_entropy_change"] = _shift_null(d_ent, switch, rng, draws=60)
                    rows.append(row)
                    # Switch RATE, a weaker and more transportable target than seam location.
                    f = TRAJ.features(ent)
                    if "skipped" not in f:
                        rate_rows.append({"dwell": dwell, "mu": mu, "beta": beta,
                                          "switch_rate": float(switch.mean()), **f})

    df = pd.DataFrame(rows)
    df.to_csv(sl_dir() / "t10_boundary_recovery_points.csv", index=False)
    gr = G.GateReport()

    def cell(d):
        return {
            "n": int(len(d)),
            "auc_travel": float(d.auc_travel.mean()),
            "null_travel": float(d.null_travel.mean()),
            "lift_travel": float(d.auc_travel.mean() - d.null_travel.mean()),
            "auc_entropy_change": float(d.auc_entropy_change.mean()),
            "null_entropy_change": float(d.null_entropy_change.mean()),
            "lift_entropy_change": float(d.auc_entropy_change.mean()
                                         - d.null_entropy_change.mean()),
            "mean_switches_per_artifact": float(d.n_switches.mean()),
        }

    by_cell = {f"dwell{dw}_mu{mu}_beta{b}": cell(df[(df.dwell == dw) & (df.mu == mu)
                                                    & (df.beta == b)])
               for dw in DWELLS for mu in MUS for b in BETAS
               if len(df[(df.dwell == dw) & (df.mu == mu) & (df.beta == b)])}
    overall = cell(df) if len(df) else {}
    lift = T.boot_paired(df.auc_travel.to_numpy(), df.null_travel.to_numpy(), rng) \
        if len(df) else {}

    # ---- switch-rate readability, if catch22 is present ---------------------------------------
    rate = {"skipped": TRAJ.available()[1]}
    if rate_rows:
        rd = pd.DataFrame(rate_rows)
        feats = [c for c in rd.columns
                 if c not in ("dwell", "mu", "beta", "switch_rate")]
        cors = {}
        for f in feats:
            x, y = rd[f].to_numpy(dtype=float), rd.switch_rate.to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 20 and np.std(x[ok]) > 1e-12:
                cors[f] = float(np.corrcoef(x[ok], y[ok])[0, 1])
        if cors:
            best = max(cors, key=lambda k: abs(cors[k]))
            rate = {"n": int(len(rd)), "n_features": len(cors),
                    "best_feature": best, "best_abs_correlation": float(abs(cors[best])),
                    "top5": dict(sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:5]),
                    "how_to_read": (
                        "correlation between a canonical trajectory feature and the maker's true "
                        "switch rate, across artifacts. Locating individual seams is the strong "
                        "claim; reading the RATE off the trajectory is the weak one, and the weak "
                        "one is what would transport to text.")}

    # ---- gates ---------------------------------------------------------------------------------
    gr.identity("circular_null_sits_at_chance", abs(float(df.null_travel.mean()) - 0.5), 0.0, 0.05,
                detail=("a circular shift destroys alignment to the truth while preserving the "
                        "trajectory's own autocorrelation, so its AUC must sit at chance. If it "
                        "does not, the switch mask itself is structured and the lift is an "
                        "artifact of where switches fall rather than of the signal."))
    gr.positive("switches_actually_occur",
                float(df.n_switches.mean()), float(df.n_switches.mean()), 1e-9,
                detail="artifacts with no switch or all switches are dropped; this records how "
                       "many seams a scored artifact actually has.")
    gr.live("travel_responds_to_dwell",
            float(abs(by_cell.get(f"dwell4.0_mu3_beta{BETAS[0]}", {}).get("lift_travel", 0.0)
                      - by_cell.get(f"dwell20.0_mu3_beta{BETAS[0]}", {}).get("lift_travel", 0.0))),
            0.0,
            detail="how much the lift moves between the shortest and longest dwell. Recorded "
                   "rather than thresholded: a detector that is flat in dwell is still a "
                   "detector, it is just one whose difficulty does not depend on the maker.")

    verdict = {
        "test": "T-10 — is the LOCATION of a decision recoverable when its IDENTITY is not?",
        "for": "Sounding Line; whether T-3's negative retires seam-finding as well as counting",
        "WHY_THIS_EXISTS": (
            "T-3 established that the sub-goal posterior floors at ~2.33 effective modes and never "
            "resolves onto one, so 'which decision' is undefined. 'When did the maker switch' is a "
            "different question: a belief can travel from one uncertain state to another, and the "
            "travel is visible even when neither endpoint is."),
        "method": {
            "signals": ["per-step L1 travel of the sub-goal posterior",
                        "per-step absolute change in sub-goal entropy"],
            "target": "the creator's true mode-switch steps, which the reader never sees",
            "null": ("circular shift of the signal, which preserves its autocorrelation and "
                     "destroys only its alignment. A permutation null would be too easy: a smooth "
                     "series beats shuffled noise for reasons unrelated to seams."),
            "n_timesteps": N_TIMESTEPS, "dwells": list(DWELLS), "n_per_cell": int(n_obs),
        },
        "overall": overall,
        "by_cell": by_cell,
        "travel_lift_over_null": lift,
        "switch_rate_readability": rate,
        "what_would_have_falsified_it": (
            "travel AUC at the circular-shift null. That would mean the reader's belief does not "
            "move when the maker switches, and T-3's negative would retire seam-finding too."),
        "what_this_cannot_show": (
            "that seams are findable in text. The switch times here are ground truth from a "
            "generative chain; a real corpus has no such labels, which is why the switch-RATE "
            "result matters more than the seam-location one -- a rate can be validated against a "
            "held-out property of a corpus, an individual seam cannot."),
    }
    PROVENANCE.stamp(verdict, __file__, gr)
    (sl_dir() / "t10_boundary_recovery.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
