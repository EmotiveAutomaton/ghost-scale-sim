"""E56 — is the gate selective? The tennis players.

Two opponents who come to understand each other better by playing each other. Everyone recognises
the trope and nobody has a mechanism for it. This model has one, and it was sitting in three
results that were never put together:

    E36     recovering someone's intent roughly DOUBLES how much of their method you pick up.
    E31/E30 depth moves how much of the PROCESS transfers, and provably cannot move how much of the
            PURPOSE does, because the construction holds purpose equally readable at every depth.
    E54     a gate shut BEFORE engaging protects where one shut after does not -- by about 6%.

Nobody asked whether that gate is SELECTIVE.

-----------------------------------------------------------------------------------------
THE MECHANISM, AND IT IS NOT STIPULATED -- IT FALLS OUT OF THE TIMING.

E54 established that the two stances differ in WHEN the gate is set, not in how hard it shuts. Put
that next to what resolves when:

    PROCESS accrues CONTINUOUSLY. Every step of the reading carries evidence about the maker's
            execution chain, and the reader is picking it up from the first look.
    PURPOSE resolves LATE. The posterior over the goal is near the prior early and only sharpens
            once enough has been seen.

A gate that is shut from step zero therefore blocks the thing that arrives late and cannot block the
thing that has already been arriving. **You do not adopt your opponent's aims. You adopt their
technique.** And the horror version is the same sentence with the gate slightly leakier: you take
the method first, and the commitments come in riding on it, because practised method is precisely
where a maker's unreportable commitments live (E43).

-----------------------------------------------------------------------------------------
H10.5  Adversarial engagement suppresses GOAL and VALUE uptake substantially more than it
       suppresses PROCESS uptake.

H10.6  Process uptake PREDICTS subsequent value uptake at a fixed level of goal uptake.

N48    The design must vary process uptake while holding goal uptake fixed, or it cannot answer its
       own question. Guaranteed by E31's construction -- goal readability is equal at every depth --
       and ASSERTED here rather than assumed, because the failure of exactly this precondition is
       what produced three inconsistent passes in V6 and was only caught in V7.

THE AUTHOR'S PRIOR IS RECORDED IN THE SPEC AND IT IS EXPENSIVE. He predicts values ride in on
process, citing mere exposure -- absorption with no evaluative step anywhere -- and has stated that
a null here would be read as evidence the MODEL is wrong rather than as an uninteresting negative.
That is why this experiment runs before E55 rather than after.
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
from ..v9.e53_e54 import _absorb, _value_prior
from . import SEED_OFFSET, v10_dir

STANCES = ("sympathetic", "adversarial")


def _uptake(enc, gates, values_map, ng, n_sub) -> dict:
    """Three uptakes off one reading, each accrued PER STEP and weighted by the gate as it stood.

    THE BUG THIS REPLACES, CAUGHT ON THE FIRST RUN AND WORTH RECORDING. The first version scored
    process per step but goal and values against the FINAL gate, on the reasoning that purpose
    resolves late. That makes the comparison degenerate: by the end of a reading the sympathetic
    reader's running divergence has reached the same value the adversarial reader anticipated from
    the start, so the two arms have the SAME final gate by construction. Goal and value ratios came
    back at exactly 1.000 -- which is not a null result, it is an erased measurement.

    All three channels now accrue the same way, per step, weighted by that step's gate. The
    difference between them is therefore purely about WHEN their information arrives, which is the
    mechanism actually under test rather than an artifact of three different weighting rules:

        process   evidence about the maker's execution chain, arriving at every step
        goal      the INCREMENT in goal knowledge from one step to the next
        values    the increment in implied-values knowledge over the same step

    A uniform gate suppresses all three by the same ratio. Only a schedule can separate them.
    """
    gates = np.asarray(gates, dtype=float)
    chance = float(np.log(1.0 / max(int(n_sub), 1)))
    steps = [np.asarray(p, dtype=float) for p in enc.goal_posteriors_by_step]
    prior = np.asarray(enc.goal_prior, dtype=float)
    truth = np.zeros(ng)
    truth[int(enc.true_goal)] = 1.0
    v_true = V6.implied_values(truth, values_map)

    def _g(t):
        return float(gates[t]) if t < len(gates) else float(gates[-1])

    # PROCESS -- per-step evidence about the execution chain.
    lps = []
    for t, (q, s_true) in enumerate(zip(enc.subgoal_posteriors, enc.true_modes)):
        p = np.asarray(q, dtype=float)
        p = p / max(p.sum(), 1e-12)
        lps.append(_g(t) * (float(np.log(max(p[int(s_true)], 1e-12))) - chance))
    process_uptake = float(np.mean(lps)) if lps else float("nan")

    # GOAL and VALUES -- the increment each step made, gated at that step.
    g_inc, v_inc = [], []
    prev = prior
    prev_v = V6.implied_values(prev, values_map)
    for t, post in enumerate(steps):
        g_inc.append(_g(t) * float(metrics.error_reduction(post, prev, int(enc.true_goal))))
        cur_v = V6.implied_values(post, values_map)
        v_inc.append(_g(t) * (float(metrics.kl_divergence(prev_v, v_true))
                              - float(metrics.kl_divergence(cur_v, v_true))))
        prev, prev_v = post, cur_v

    final = np.asarray(enc.goal_posterior, dtype=float)
    return {"process_uptake": process_uptake,
            "goal_uptake": float(np.sum(g_inc)) if g_inc else float("nan"),
            "value_uptake": float(np.sum(v_inc)) if v_inc else float("nan"),
            "final_gate": float(gates[-1]) if len(gates) else 0.0,
            "mean_gate": float(np.mean(gates)) if len(gates) else 0.0,
            "process_ungated": float(enc.process["process_error_reduction"]),
            "goal_ungated": float(metrics.error_reduction(final, prior, int(enc.true_goal)))}


def run(cfg: Config, n_obs: int = 40, n_timesteps: int = 16, depths=(1, 2, 3),
        workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)
    k_gain = float(world.cfg.get("v6.gate.k_gain", 8.0))
    theta_0 = float(world.cfg.get("v6.gate.theta_0", 0.35))
    lam = 0.25          # the V9 value, for the reason recorded in E54: at 1.0 no gate ever moves
    leak = 0.10
    values_map = V6.build_values_map(ng, n_values=2)

    depths = tuple(int(d) for d in depths if int(d) <= int(n_mu))

    rows = []
    for stance in STANCES:
        for depth in depths:
            for i in range(int(n_obs)):
                rng = np.random.default_rng(SEED_OFFSET + 10_000 + i * 13 + depth)
                g = int(rng.integers(ng))
                creator, art, env = H.make_artifact_and_env(
                    world, cfg_r, g, depth, 1.0, n_timesteps, rng)
                enc = H.run_encounter(world, cfg_r, art, env,
                                      make_v5_observer(world, rng), creator, rng,
                                      n_timesteps, 8, n_sub, n_mu, ng, kappa)
                carried = np.full(ng, 1.0 / ng)
                _, gates, looked = _absorb(enc, stance, carried, values_map,
                                           kappa, lam, theta_0, k_gain, leak, ng)
                r = _uptake(enc, gates, values_map, ng, n_sub)
                r.update({"stance": stance, "depth": depth, "observer": i,
                          "engaged": looked})
                rows.append(r)

    df = pd.DataFrame(rows)
    out = v10_dir("e56_selective_gate")
    df.to_csv(out / "e56_uptake.csv", index=False)

    by = df.groupby(["stance", "depth"]).mean(numeric_only=True).reset_index()
    st = df.groupby("stance").mean(numeric_only=True).reset_index()

    def _s(stance, col):
        return float(st[st.stance == stance][col].iloc[0])

    # ---- H10.5: is the suppression selective? -----------------------------
    # Scored as the RATIO of adversarial to sympathetic uptake on each channel. A uniform gate
    # suppresses both channels by the same ratio; a selective one does not.
    def _ratio(col):
        s = _s("sympathetic", col)
        a = _s("adversarial", col)
        return float(a / s) if abs(s) > 1e-12 else float("nan")

    r_proc, r_goal, r_val = _ratio("process_uptake"), _ratio("goal_uptake"), _ratio("value_uptake")
    h105 = bool(np.isfinite(r_proc) and np.isfinite(r_goal)
                and r_proc > r_goal and r_proc > r_val)

    # ---- N48: goal readability must be flat across depth -------------------
    # E31's construction guarantees it. Asserted rather than assumed, because the failure of this
    # precondition is what produced three inconsistent passes in V6.
    goal_by_depth = df.groupby("depth")["goal_ungated"].mean()
    proc_by_depth = df.groupby("depth")["process_ungated"].mean()
    goal_spread = float(goal_by_depth.max() - goal_by_depth.min())
    proc_spread = float(proc_by_depth.max() - proc_by_depth.min())
    n48 = bool(proc_spread > goal_spread)

    # ---- H10.6: does process predict value uptake at fixed goal uptake? ----
    # Partial correlation of process with value, controlling for goal. Plain least squares:
    # residualise both against goal uptake, then correlate the residuals.
    def _resid(y, x):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        Xd = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        return y - Xd @ beta

    def _partial(frame):
        s = frame[np.isfinite(frame.process_uptake) & np.isfinite(frame.value_uptake)]
        if len(s) <= 8 or s.goal_uptake.nunique() <= 1:
            return float("nan"), [float("nan"), float("nan")]
        rp = _resid(s.process_uptake, s.goal_uptake)
        rv = _resid(s.value_uptake, s.goal_uptake)
        if np.std(rp) <= 1e-12 or np.std(rv) <= 1e-12:
            return float("nan"), [float("nan"), float("nan")]
        r = float(np.corrcoef(rp, rv)[0, 1])
        rng = np.random.default_rng(SEED_OFFSET + 11_000)
        boot = []
        for _ in range(2000):
            idx = rng.integers(0, len(rp), len(rp))
            a, b = rp[idx], rv[idx]
            if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                boot.append(float(np.corrcoef(a, b)[0, 1]))
        ci = ([float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
              if boot else [float("nan"), float("nan")])
        return r, ci

    partial, partial_ci = _partial(df)
    h106 = bool(np.isfinite(partial) and partial > 0.2)

    # DECLARED POST-HOC SPLIT, and the pooled number above is the one that decides.
    # The rider mechanism matters specifically for the reader whose gate is SHUT -- the claim is
    # that values arrive despite the guard, not that they arrive generally. Reported beside the
    # pre-registered statistic, never in place of it.
    by_stance_partial = {}
    for s in STANCES:
        r, ci = _partial(df[df.stance == s])
        by_stance_partial[s] = {"partial_r": r, "interval": ci}

    verdict = {
        "experiment": "E56",
        "hypotheses": ["H10.5", "H10.6"],
        "question": ("Is the gate selective -- does adversarial reading block purpose and values "
                     "while letting method through?"),
        "plain_language": (
            "Two opponents come to understand each other better by playing. The proposed mechanism "
            "is not that the guard fails, but that it is pointed at the wrong thing: method arrives "
            "continuously from the first look, purpose only resolves late, and a guard raised "
            "before you start can only block what has not arrived yet."),
        "by_stance_and_depth": by.to_dict(orient="records"),
        "H10.5": {
            "outcome": ("THE_GATE_BLOCKS_PURPOSE_AND_PASSES_METHOD" if h105
                        else "THE_GATE_SUPPRESSES_BOTH_CHANNELS_ALIKE"),
            "adversarial_over_sympathetic": {"process": r_proc, "goal": r_goal, "value": r_val},
            "how_to_read": ("a ratio near 1.0 means that channel was not suppressed. A uniform "
                            "gate gives all three the same ratio; only a schedule can separate "
                            "them."),
        },
        "H10.6": {
            "outcome": ("METHOD_UPTAKE_CARRIES_VALUE_UPTAKE" if h106
                        else "METHOD_AND_VALUE_UPTAKE_ARE_INDEPENDENT"),
            "partial_correlation_controlling_for_goal_uptake": partial,
            "interval": partial_ci,
            "pre_registered_bar": 0.2,
            "by_stance_declared_post_hoc": by_stance_partial,
            "why_it_matters": (
                "this is the rider mechanism. If it holds, a reader can refuse someone's purpose "
                "and still acquire their commitments by taking up their method -- which is what "
                "E55's reader 7 tests in a learner, and what makes a value gate insufficient."),
        },
        "null_n48": {
            "statement": "the design varies process uptake while holding goal readability fixed",
            "goal_spread_across_depth": goal_spread,
            "process_spread_across_depth": proc_spread,
            "passed": n48,
            "why": ("E31's construction holds the goal equally readable at every depth. If process "
                    "did not vary more than goal, this design cannot answer its own question -- "
                    "the failure that produced three inconsistent passes in V6."),
        },
        "authors_recorded_prior": (
            "values ride in on process, citing mere exposure. The author stated before the run "
            "that a null here would be read as evidence the MODEL is wrong rather than as an "
            "uninteresting negative."),
        "n_obs": int(n_obs), "depths": list(depths),
    }
    (v10_dir() / "e56_selective_gate.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
