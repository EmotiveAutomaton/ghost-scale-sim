"""V-11c — what V-11a's control was throwing away, and how big the effect actually is.

V-11a compared the real settling split against a sham split drawn from the distribution of settling
times, and reported that the sham reproduced two thirds of the gain. That is a real control and it
is too conservative in a way that is visible in its own output:

    settled_at:  t=2 -> 104 readings, t=3 -> 37, t=4 -> 28,  out of 320
    |real - sham|:  1 -> 58,  2 -> 45

Settling times bunch hard at the front, so a sham drawn from that pool lands within a step or two of
the truth about a third of the time. **Those shams are also post-settling.** The control was
subtracting part of the signal along with the clock, which is why what survived looked small.

Three things this runs instead.

  1  PERI-SETTLING ALIGNMENT. Forget windows. Take the reader's log-probability on the maker's true
     execution mode at each step, align every reading on its own settling step, and average. If
     resolving the goal unlocks the method there is a STEP CHANGE at zero. If the plate was
     measuring the clock there is a smooth ramp through it. A two-window split cannot tell those
     apart and this can.

  2  A FAR SHAM. The same split test with the sham forced at least four steps away from the truth,
     which is what the control was supposed to be.

  3  THE DENOMINATOR. +0.020 nats sounds like nothing until you ask nothing compared to what. The
     whole process signal a reader ever extracts in E36 is about 0.09 nats. The right unit is the
     share of recoverable process information, not the raw nat.

And one prediction the theory makes that nobody has scored: BETA is goal legibility. At beta = 1.0
the goal is fully readable; at 0.10 the craft is visible and the goal is not. If understanding the
purpose is what unlocks the method, the unlock must SHRINK as the goal becomes unreadable. A
dose-response over a knob the hypothesis names is worth more than any single contrast.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import metrics
from .. import v6_model as V6
from ..config import Config
from ..prereg_v6 import BOOTSTRAP_DRAWS, percentile_interval
from ..v5_model import make_v5_observer
from ..v6 import SEED_OFFSET, harness as H
from ..v6.e36_process import BETA_GRID, MU_GRID, RESOLVED_ENTROPY

_EPS = 1e-12
LAGS = tuple(range(-6, 7))       # steps either side of the settling event
FAR = 4                          # a sham must be at least this far from the truth to count


REPO = __import__("pathlib").Path(__file__).resolve().parents[2]


def _out():
    """V-11 is a validation pass, so it writes under results/validation/ like
    every other one. It wrote into results/v6/ for a week, which put a check ON
    version 6 inside version 6's own results, and that was simply untidy."""
    d = REPO / "results" / "validation" / "v11"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _per_step_logp(enc, n_sub: int) -> np.ndarray:
    """Information the reader carries about the true execution mode, step by step, in nats."""
    out = []
    for q, s in zip(enc.subgoal_posteriors, enc.true_modes):
        p = np.asarray(q, dtype=float)
        p = p / max(p.sum(), _EPS)
        out.append(float(np.log(max(p[int(s)], _EPS)) - np.log(1.0 / n_sub)))
    return np.asarray(out, dtype=float)


def _gain_at(enc, split: int, n_sub: int) -> float:
    before = V6.process_recovery(enc.subgoal_posteriors[:split],
                                 enc.true_modes[:split], n_sub)["process_error_reduction"]
    after = V6.process_recovery(enc.subgoal_posteriors[split:],
                                enc.true_modes[split:], n_sub)["process_error_reduction"]
    return float(after) - float(before)


def run(cfg: Config, n_obs: int = 120, n_timesteps: int = 24, forced_k: int = 24) -> dict:
    world, _cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    rows, traces = [], []

    for mu in MU_GRID:
        if mu == 1:
            continue
        for beta in BETA_GRID:
            base = 30_000 + mu * 100 + int(beta * 100)      # E36's own seeds
            for i in range(int(n_obs)):
                art_rng = np.random.default_rng(base * 31 + i)
                g_true = int(art_rng.integers(ng))
                creator, artifact, env = H.make_artifact_and_env(
                    world, cfg_r, g_true, int(mu), float(beta), n_timesteps, art_rng)
                agent = make_v5_observer(world, np.random.default_rng(base * 7907 + i))
                enc = H.run_encounter(world, cfg_r, artifact, env, agent, creator,
                                      np.random.default_rng(base * 7907 + i),
                                      n_timesteps, forced_k, n_sub, n_mu, ng,
                                      float(world.cfg.signal_model.kappa))
                ents = [float(metrics.within_observer_entropy(p))
                        for p in enc.goal_posteriors_by_step]
                settled = next((t for t, h in enumerate(ents) if h <= RESOLVED_ENTROPY), None)
                if settled is None or not (2 <= settled <= len(ents) - 3):
                    continue
                lp = _per_step_logp(enc, n_sub)
                rows.append({"mu": mu, "beta": beta, "settled_at": settled,
                             "n_steps": len(ents), "goal_correct": int(enc.correct),
                             "whole_process": float(enc.process["process_error_reduction"]),
                             "real_gain": _gain_at(enc, settled, n_sub)})
                traces.append((settled, lp, mu, beta, int(enc.correct), enc))

    if not rows:
        return {"check": "V-11c", "outcome": "NOT_MEASURABLE"}
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED_OFFSET + 11_200)

    # ---- 1. peri-settling alignment -------------------------------------- #
    peri = {int(k): [] for k in LAGS}
    for settled, lp, *_ in traces:
        for k in LAGS:
            t = settled + k
            if 0 <= t < len(lp):
                peri[int(k)].append(float(lp[t]))
    curve = {int(k): (float(np.mean(v)) if v else float("nan")) for k, v in peri.items()}
    counts = {int(k): int(len(v)) for k, v in peri.items()}

    # THE EVENT IS AT LAG -1, NOT LAG 0, AND THAT IS NOT A FUDGE. Settling is DETECTED by an
    # entropy threshold, and the observation that drops the entropy below the threshold is the
    # observation carrying the information. The detector therefore fires one step AFTER the
    # arrival. A pre-window that includes lag -1 is a pre-window containing the event, which is
    # what V-11a's split did. Both conventions are reported.
    EVENT = -1
    pre = [curve[k] for k in LAGS if k < EVENT and np.isfinite(curve[k])]
    post = [curve[k] for k in LAGS if k >= EVENT and np.isfinite(curve[k])]
    step = float(np.mean(post) - np.mean(pre))
    naive_step = float(np.mean([curve[k] for k in LAGS if k >= 0 and np.isfinite(curve[k])])
                       - np.mean([curve[k] for k in LAGS if k < 0 and np.isfinite(curve[k])]))

    # Is that a STEP or a RAMP? A ramp would keep climbing at the same rate through the event, so
    # fit the pre-event slope and ask what the post-event level would have been if it had just
    # carried on. Whatever is above that line arrived AT the event.
    ks = np.array([k for k in LAGS if k < EVENT and np.isfinite(curve[k])], dtype=float)
    vs = np.array(pre, dtype=float)
    slope, intercept = np.polyfit(ks, vs, 1) if ks.size >= 2 else (0.0, float(np.mean(vs)))
    post_ks = np.array([k for k in LAGS if k >= EVENT and np.isfinite(curve[k])], dtype=float)
    extrapolated = float(np.mean(slope * post_ks + intercept))
    above_the_ramp = float(np.mean(post) - extrapolated)

    # ---- 2. a sham that is actually far away ----------------------------- #
    far_real, far_sham = [], []
    for settled, lp, mu, beta, ok, enc in traces:
        lo, hi = 2, len(lp) - 3
        cands = [c for c in range(lo, hi + 1) if abs(c - settled) >= FAR]
        if not cands:
            continue
        far_real.append(_gain_at(enc, settled, n_sub))
        far_sham.append(_gain_at(enc, int(rng.choice(cands)), n_sub))
    fr, fs = np.asarray(far_real), np.asarray(far_sham)
    fd = fr - fs
    fdraws = [float(np.mean(rng.choice(fd, fd.size, replace=True))) for _ in range(BOOTSTRAP_DRAWS)]
    flo, fhi = percentile_interval(fdraws)

    # ---- 3. dose-response on goal legibility ----------------------------- #
    by_beta = {}
    for beta in BETA_GRID:
        sub = df[df.beta == beta]
        if not len(sub):
            continue
        g = sub.real_gain.to_numpy()
        draws = [float(np.mean(rng.choice(g, g.size, replace=True))) for _ in range(BOOTSTRAP_DRAWS)]
        blo, bhi = percentile_interval(draws)
        by_beta[float(beta)] = {"gain": float(g.mean()), "interval": [blo, bhi], "n": int(g.size)}
    legible = by_beta.get(1.0, {}).get("gain", float("nan"))
    illegible = by_beta.get(0.10, {}).get("gain", float("nan"))

    # ---- 4. does it matter whether you settled on the RIGHT goal? -------- #
    right = df[df.goal_correct == 1].real_gain.to_numpy()
    wrong = df[df.goal_correct == 0].real_gain.to_numpy()
    cdraws = ([float(np.mean(rng.choice(right, right.size, replace=True))
                     - np.mean(rng.choice(wrong, wrong.size, replace=True)))
               for _ in range(BOOTSTRAP_DRAWS)] if right.size and wrong.size else [])
    clo, chi = percentile_interval(cdraws) if cdraws else (float("nan"), float("nan"))
    correctness = {"settled_on_the_right_goal": float(right.mean()) if right.size else None,
                   "settled_on_the_wrong_goal": float(wrong.mean()) if wrong.size else None,
                   "difference": (float(right.mean() - wrong.mean())
                                  if right.size and wrong.size else None),
                   "interval": [clo, chi],
                   "excludes_zero_positive": bool(np.isfinite(clo) and clo > 0.0),
                   "n_right": int(right.size), "n_wrong": int(wrong.size),
                   "how_to_read": (
                       "settling is COMMITMENT; settling on the RIGHT goal is UNDERSTANDING. If "
                       "both unlock equally then what matters is that the reader stopped being "
                       "uncertain rather than what it became certain of, and the hypothesis is "
                       "about confidence rather than about purpose.")}

    # ---- 5. the denominator ---------------------------------------------- #
    whole = float(df.whole_process.mean())
    verdict = {
        "check": "V-11c — what V-11a's control threw away, and the effect in interpretable units",
        "why": ("V-11a drew its sham split from the distribution of settling times, and settling "
                "times bunch at the front of the reading, so a third of shams landed within two "
                "steps of the truth and were themselves post-settling. The control was "
                "subtracting signal along with the clock."),
        "peri_settling": {
            "curve_by_lag": curve,
            "n_by_lag": counts,
            "arrival_lag": EVENT,
            "mean_before": float(np.mean(pre)),
            "mean_after": float(np.mean(post)),
            "step_at_arrival": step,
            "step_if_you_put_the_event_at_lag_zero": naive_step,
            "pre_event_slope_per_step": float(slope),
            "what_a_pure_ramp_predicts_after": extrapolated,
            "above_the_ramp": above_the_ramp,
            "how_to_read": (
                "if the reader were simply accumulating, the pre-event slope carried forward would "
                "predict the post-event level. Whatever sits above that line arrived at the "
                "settling step and not before it."),
        },
        "far_sham": {
            "real_gain": float(fr.mean()), "sham_gain": float(fs.mean()),
            "difference": float(fd.mean()), "interval": [flo, fhi],
            "excludes_zero_positive": bool(np.isfinite(flo) and flo > 0.0),
            "n": int(fd.size), "min_distance": FAR,
        },
        "dose_response_on_goal_legibility": {
            "by_beta": by_beta,
            "legible_minus_illegible": float(legible - illegible),
            "how_to_read": (
                "beta is how readable the goal is. The hypothesis says the unlock is caused by "
                "recovering the purpose, so it must shrink as the purpose becomes unrecoverable. "
                "Nothing about elapsed time cares about beta, so a gradient here is evidence no "
                "clock can produce."),
        },
        "settling_right_versus_wrong": correctness,
        "effect_in_context": {
            "whole_reading_process_information": whole,
            "share_of_all_process_information": (float(fd.mean()) / whole) if whole else None,
            "how_to_read": (
                "the raw number is in nats and nats are small here. The total process information "
                "a reader extracts across an entire reading is the denominator that makes it "
                "legible."),
        },
        "n_rollouts": int(len(df)),
    }
    df.to_csv(_out() / "v11c_peri_settling.csv", index=False)
    (_out() / "v11c_peri_settling.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
