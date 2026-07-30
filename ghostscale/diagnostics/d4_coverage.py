"""D-4 — how much of the body of work can the exact solver reach, and how far does the shortcut drift?

The validation pass checked five experiment families under exact inference and three of its verdicts
moved. That raises an obvious question it did not answer: **how much of the rest is even reachable?**

Three things block the exact solver, and they are different in kind:

* **Dirichlet learning.** `ExactAgent` refuses `update_A` rather than approximating it. Every
  experiment whose observer LEARNS is therefore unreachable, and that is six of them, including two
  that carry public claims.
* **The recorded expected-free-energy terms.** E1 records the pragmatic and epistemic decomposition
  through `metrics.policy_efe_terms`, which builds its predictive states from the factorised
  marginals. Fixed alongside this pass: `ExactAgent` now exposes exact counterparts, so E1 and E5
  become reachable, and this module measures how far the two disagree.
* **Distributional observations.** The behavioural-regret path feeds a distribution rather than an
  index, which the exact filter does not accept.

WHY THE DRIFT MEASUREMENT MATTERS MORE THAN THE COUNT. A coverage count says what was not checked. It
does not say whether checking would have mattered. So this also measures, on the cells it can reach,
how far the mean-field expected free energy drifts from the exact one AS EVIDENCE ACCUMULATES. At the
first timestep the belief is a product of marginals by construction, so the two agree exactly and any
comparison there is vacuous. The drift is what tells you whether the unreachable experiments are
sitting on a small error or a large one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants as K
from .. import metrics
from ..config import Config
from ..experiments._common import REPO_ROOT
from . import diagnostics_dir

# The blockers, assigned by reading the call sites rather than by guessing.
BLOCKERS = {
    "dirichlet_learning": {
        "why": ("the observer updates its own likelihood with Dirichlet counts. ExactAgent refuses "
                "this rather than approximating it, so these cannot run under exact inference "
                "without an exact learning path being written"),
        "experiments": ["E7", "E9", "E13", "E16", "E18", "E8"],
        "public_claims_at_risk": [
            "an unlabelled learner loses about a third of its ability to read intent from genuine "
            "human work, and honest labels cut that error about a hundredfold (E7)",
            "31% of machine content must be labelled for a convention-aware reader and 74% for one "
            "who does not know the convention (E16)",
        ],
    },
    "recorded_efe_terms": {
        "why": ("records the pragmatic and epistemic decomposition through pymdp's control helpers, "
                "which build predictive states from the factorised marginals. FIXED alongside this "
                "pass: ExactAgent.efe_terms and .epistemic_value_about are exact counterparts, so "
                "these are now reachable"),
        "experiments": ["E1", "E5"],
        "public_claims_at_risk": [
            "readers stop looking closely at work with no intent behind it, and the Curator tier "
            "costs the most deep looks (E1)",
            "trust in the label is a different knob from general decisiveness (E5)",
        ],
    },
    "distributional_observations": {
        "why": ("the behavioural-regret path passes a distribution rather than an observation "
                "index, which the exact filter does not accept"),
        "experiments": ["E10", "E11", "E6b"],
        "public_claims_at_risk": [
            "the reader's own skill caps what can be recovered (E10)",
            "distance between belief distributions explains 66% of the variance in actual harm "
            "(E11)",
        ],
    },
}

CHECKED_BY_VALIDATION = ["E2", "E19", "E20", "E31", "E32"]
ALL_EXPERIMENTS = ["E1", "E2", "E3", "E4", "E5", "E6", "E6b", "E7", "E8", "E9", "E10", "E11",
                   "E12", "E13", "E14", "E15", "E16", "E17", "E18", "E19", "E20", "E21", "E28",
                   "E29", "E30", "E31", "E32", "E33", "E34", "N21"]


def _measure_efe_drift(cfg: Config) -> list:
    """How far mean-field EFE drifts from exact EFE as evidence accumulates.

    Run on the three E2 cells, forced DEEP so the tape is identical, with both quantities read off
    the SAME exact agent at every step. Reading them off one agent is the point: it isolates the
    estimator from the trajectory, so the difference is the factorisation and nothing else.
    """
    from ..environment import Artifact, Environment
    from ..exact import make_exact_agent
    from ..generative_model import build_D, build_shared_model
    from ..observer import find_named_policies

    rows = []
    gm = build_shared_model(cfg)
    for name, prov, sig in (("machine work passed off as human", K.GHOST, K.SIG_CREATOR),
                            ("machine work labelled honestly", K.GHOST, K.SIG_GHOST),
                            ("human work read correctly", K.CREATOR, K.SIG_CREATOR)):
        env = Environment(cfg, gm, np.random.default_rng(3), honesty=1.0, signing_rate=1.0)
        art = Artifact(provenance=prov, goal=int(cfg.get("experiments.e2.true_goal", 1)),
                       declared_signal=sig)
        agent = make_exact_agent(gm, build_D(cfg, np.random.default_rng(5)), cfg)
        agent.reset()
        deep, skim = find_named_policies(agent)
        obs_rng = np.random.default_rng(9)
        for t in range(int(cfg.run.n_timesteps)):
            # Both estimators, same belief, same policy.
            mf_prag, mf_epi = metrics.policy_efe_terms(agent, deep)
            ex_prag, ex_epi = agent.efe_terms(deep)
            mf_goal = metrics.epistemic_value(agent, deep)
            ex_goal = agent.epistemic_value_about(deep)
            rows.append({
                "cell": name, "t": t + 1,
                "meanfield_epistemic_total": mf_epi, "exact_epistemic_total": ex_epi,
                "epistemic_total_abs_error": abs(mf_epi - ex_epi),
                "meanfield_epistemic_goal": mf_goal, "exact_epistemic_goal": ex_goal,
                "epistemic_goal_abs_error": abs(mf_goal - ex_goal),
                "meanfield_pragmatic": mf_prag, "exact_pragmatic": ex_prag,
                "pragmatic_abs_error": abs(mf_prag - ex_prag),
                # How far the belief is from being a product of its own marginals. This is the
                # quantity mean field assumes is zero, so it is the right x-axis for the error.
                "joint_vs_product_kl": _joint_product_kl(agent),
            })
            agent.infer_states(env.observation(art, K.DEEP, obs_rng))
            agent.action = np.zeros(len(agent.num_controls))
            agent.action[K.F_ATTENTION] = K.DEEP
    return rows


def _joint_product_kl(agent) -> float:
    """KL( joint belief || product of its own marginals ), in nats. Zero iff mean field is exact."""
    b = agent.qs_joint
    prod = np.asarray(agent.qs[0], dtype=float)
    for f in range(1, agent.num_factors):
        prod = np.multiply.outer(prod, np.asarray(agent.qs[f], dtype=float))
    prod = prod.ravel()
    m = b > 0
    return float(np.sum(b[m] * np.log(b[m] / np.clip(prod[m], 1e-300, None))))


def run(cfg: Config, workers: int = 1) -> dict:
    out = diagnostics_dir("d4_coverage")
    blocked = sorted({e for v in BLOCKERS.values() for e in v["experiments"]})
    now_reachable = BLOCKERS["recorded_efe_terms"]["experiments"]
    still_blocked = sorted(set(blocked) - set(now_reachable))
    unchecked_but_reachable = sorted(
        set(ALL_EXPERIMENTS) - set(blocked) - set(CHECKED_BY_VALIDATION))

    drift = _measure_efe_drift(cfg)
    pd.DataFrame(drift).to_csv(out / "efe_drift.csv", index=False)
    df = pd.DataFrame(drift)
    worst = df.loc[df.epistemic_total_abs_error.idxmax()]
    late = df[df.t >= df.t.max() - 4]

    early = df[df.t <= 5]
    summary = {
        "epistemic_total_max_abs_error": float(df.epistemic_total_abs_error.max()),
        "epistemic_total_mean_abs_error_early": float(early.epistemic_total_abs_error.mean()),
        "epistemic_total_mean_abs_error_late": float(late.epistemic_total_abs_error.mean()),
        "epistemic_goal_max_abs_error": float(df.epistemic_goal_abs_error.max()),
        "pragmatic_max_abs_error": float(df.pragmatic_abs_error.max()),
        "max_joint_vs_product_kl": float(df.joint_vs_product_kl.max()),
        "worst_cell": str(worst["cell"]), "worst_t": int(worst["t"]),
        "error_at_t1": float(df[df.t == 1].epistemic_total_abs_error.max()),
    }

    n_total = len(ALL_EXPERIMENTS)
    coverage = {
        "experiments_total": n_total,
        "checked_under_exact_by_the_validation_pass": CHECKED_BY_VALIDATION,
        "structurally_unreachable_before_this_pass": blocked,
        "made_reachable_by_this_pass": now_reachable,
        "still_structurally_unreachable": still_blocked,
        "reachable_but_never_checked": unchecked_but_reachable,
        "fraction_checked": len(CHECKED_BY_VALIDATION) / n_total,
        "fraction_still_unreachable": len(still_blocked) / n_total,
    }

    # THE ERROR IS TRANSIENT, and which window it lands in is what decides whether it matters. It
    # peaks where the joint is furthest from a product of its marginals, which is a few steps in:
    # before any evidence the belief IS a product, and once the belief has concentrated it is nearly
    # a point mass and therefore nearly a product again. The interesting number is the peak and the
    # timestep it falls on, not a late average.
    drift_is_small = summary["epistemic_total_max_abs_error"] < 0.01
    if still_blocked and not drift_is_small:
        verdict = "COVERAGE_GAP_AND_THE_SHORTCUT_DRIFTS"
    elif still_blocked:
        verdict = "COVERAGE_GAP_BUT_THE_SHORTCUT_STAYS_CLOSE"
    elif not drift_is_small:
        verdict = "FULL_COVERAGE_BUT_THE_SHORTCUT_DRIFTS"
    else:
        verdict = "COVERAGE_ADEQUATE"

    statement = (
        "The validation pass checked %d of %d experiments under exact inference, and three of its "
        "verdicts moved. %d experiments were structurally unreachable; this pass makes %d of those "
        "reachable by giving the exact agent its own expected-free-energy accessors, which leaves "
        "%d that still cannot run: %s. Six of the remaining ones are blocked by Dirichlet learning "
        "and three by distributional observations.\n\n"
        "Two public claims sit on the learning path and cannot currently be validated at all: the "
        "unlabelled learner losing a third of its ability to read human work, and the 31%% versus "
        "74%% labelling-coverage figures. Those are named here rather than left to be discovered.\n\n"
        % (len(CHECKED_BY_VALIDATION), n_total, len(blocked), len(now_reachable),
           len(still_blocked), ", ".join(still_blocked)))

    statement += (
        "On the cells that can be reached, the shortcut's error in the expected free energy is "
        "TRANSIENT AND PEAKS EARLY. It is %.2e nats at the first timestep, which is essentially zero "
        "because the belief is a product of its marginals before any evidence arrives. It rises to "
        "%.3f nats at timestep %d, where the joint is furthest from that product (%.3f nats of "
        "divergence), and decays to %.2e nats by the end of the run as the belief concentrates and "
        "becomes nearly a point mass, which is nearly a product again.\n\n"
        "%s"
        % (summary["error_at_t1"], summary["epistemic_total_max_abs_error"], summary["worst_t"],
           summary["max_joint_vs_product_kl"], summary["epistemic_total_mean_abs_error_late"],
           ("The peak error is under a hundredth of a nat, so the shortcut is close everywhere and "
            "the unreachable experiments are probably sitting on a small error."
            if drift_is_small else
            "**That shape is the thing to worry about, because it lands in the decision window.** "
            "The expected free energy is what decides whether the reader keeps looking, and the "
            "error is largest in the first handful of timesteps. Experiments whose free decision "
            "begins after a long forced window see the error only once it has decayed: the overlap "
            "sweep forces ten steps and is therefore safe. The one that is not is the selectivity "
            "measure, which the version 1 deviations record as being taken over THE FIRST THREE "
            "FREE STEPS precisely to catch the decision that matters, and which is also one of the "
            "experiments the shortcut could not be swapped out of until this pass. Re-running it "
            "under exact inference is now possible and is the cheapest outstanding check in the "
            "project.")))

    payload = {
        "check": "D-4",
        "question": ("How much of the work can the exact solver reach, and how wrong is the "
                     "shortcut where it can be measured?"),
        "plain_language": (
            "The validation pass rechecked five experiments with the arithmetic done exactly rather "
            "than approximately, and three conclusions changed. This asks the obvious follow-up: "
            "how many of the others could even be rechecked, and for the ones that could not, is "
            "the approximation drifting a little or a lot?"),
        "coverage": coverage,
        "blockers": BLOCKERS,
        "efe_drift_summary": summary,
        "efe_drift_note": (
            "Both estimators are read off the SAME exact agent at every step, so the difference is "
            "the factorisation and not a different trajectory. At t = 1 the belief IS a product of "
            "its marginals, so agreement there is a construction fact and not evidence."),
        "the_fix_made_here": (
            "ExactAgent.efe_terms and ExactAgent.epistemic_value_about. Before this pass "
            "metrics.policy_efe_terms raised AttributeError on an exact agent while "
            "metrics.epistemic_value succeeded and silently returned the FACTORISED answer, which "
            "is the more dangerous of the two failures and is the class of defect this project has "
            "been bitten by seven times. A test now fails if the two disagree beyond tolerance on a "
            "construction where they must agree."),
        "verdict": verdict,
        "statement": statement,
    }
    (diagnostics_dir() / "d4_coverage.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload
