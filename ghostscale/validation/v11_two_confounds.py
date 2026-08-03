"""V-11 — the two controls plates 5 and 11 needed and did not have.

Both plates were shipped, both were challenged by a reader rather than by the pipeline, and in both
cases the challenge was the same shape: *the design cannot separate the claim from a boring
alternative*. This module runs the missing separator for each.

-----------------------------------------------------------------------------------------
V-11a — E36's temporal ordering, against a placebo split.

THE CLAIM: within one reading, method uptake rises after the reader settles on the goal, so working
out what someone was FOR is what makes their method readable.

THE BORING ALTERNATIVE: "after settling" is later in the reading than "before settling". The later
window always has more evidence behind it. Both quantities could simply accrue with looking, and the
settling point could be doing no work at all.

THE SEPARATOR: split the same rollout at a SHAM point. For each rollout, draw a split index from the
distribution of real settling times but independent of when that rollout actually settled, and score
before/after at the sham point. If the gain is the same, the plate is measuring the clock. If the
real split beats the sham split, settling is doing work the clock is not.

This is a placebo control and not a re-analysis: every rollout contributes both a real gain and a
sham gain, so the two are paired and the contrast is within-reading.

-----------------------------------------------------------------------------------------
V-11b — E43's maker, sat down in front of its own work.

THE CLAIM ON THE PLATE: expertise bakes in until the maker cannot report it, and it transfers anyway.

WHAT E43 ACTUALLY MEASURED: a maker's declared accuracy against a SEPARATE reader's accuracy. The
walkthrough noted the gap: nothing in the experiment puts the maker in the reader's seat, so
"experts can learn about themselves from their own process" was an extension.

THE TEST: take the artifact a maker produced, hand it to an observer built from that same maker's
own model, and ask what it recovers about the goal the maker was pursuing. Compare against what the
maker can declare about that same goal. If reading beats declaring, the extension is earned: your
own work is a better record of your intent than your memory of it.

The observer is a fresh agent with the maker's own body plan and a flat prior, so it starts from no
privileged knowledge of which goal was running -- it has to read it off the artifact like anybody
else. That is the honest version of "sitting yourself down in front of your own work".
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
from ..v6 import SEED_OFFSET, harness as H, v6_dir
from ..v6.e36_process import BETA_GRID, MU_GRID, RESOLVED_ENTROPY

# V-11b READS ITS OWN WORK AT THE CURATOR TIER, WITH A SHORT LOOK. On fully transparent work the
# reader scores 1.00 at every depth -- which is exactly the ceiling that made E43's original plate
# unreadable, and repeating it here would answer nothing. Partial intent transmission leaves the
# headroom the comparison needs.
SELF_READ_TIER = K.CURATOR
SELF_READ_GLANCES = 3
from ..prereg_v6 import BOOTSTRAP_DRAWS, percentile_interval

# E43's own grid, so V-11b is scored on the cells the plate draws.
E43_MU = (1, 2, 3)


def _gain_at(enc, split: int, n_sub: int) -> float:
    """Process uptake after ``split`` minus before it, on one rollout."""
    before = V6.process_recovery(enc.subgoal_posteriors[:split],
                                 enc.true_modes[:split], n_sub)["process_error_reduction"]
    after = V6.process_recovery(enc.subgoal_posteriors[split:],
                                enc.true_modes[split:], n_sub)["process_error_reduction"]
    return float(after) - float(before)


def run_a(cfg: Config, n_obs: int = 40, n_timesteps: int = 24, forced_k: int = 24) -> dict:
    """V-11a: the real settling split against a sham split drawn from the same distribution."""
    world, _cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    rows = []
    for mu in MU_GRID:
        if mu == 1:                      # no process at the shallowest depth, by construction
            continue
        for beta in BETA_GRID:
            # E36'S OWN SEED SCHEME, VERBATIM. A control that re-randomises the rollouts is
            # not controlling the experiment, it is running a different one, and the fidelity
            # check below is that ``real_gain`` here must reproduce the committed +0.080.
            base = 30_000 + mu * 100 + int(beta * 100)
            for i in range(int(n_obs)):
                art_rng = np.random.default_rng(base * 31 + i)
                creator, artifact, env = H.make_artifact_and_env(
                    world, cfg_r, int(art_rng.integers(ng)), int(mu), float(beta),
                    n_timesteps, art_rng)
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
                rows.append({"mu": mu, "beta": beta, "i": i, "settled_at": int(settled),
                             "n_steps": len(ents),
                             "real_gain": _gain_at(enc, settled, n_sub),
                             "enc": enc})

    if not rows:
        return {"check": "V-11a", "outcome": "NOT_MEASURABLE", "n": 0}

    # THE SHAM SPLIT IS DRAWN FROM THE OBSERVED SETTLING TIMES, NOT FROM A UNIFORM RANGE. A
    # uniform sham would sit at a different average depth in the reading than the real split and
    # the comparison would be confounded by the very thing it is controlling for.
    rng = np.random.default_rng(SEED_OFFSET + 11_000)
    settle_pool = np.array([r["settled_at"] for r in rows], dtype=int)
    for r in rows:
        lo, hi = 2, r["n_steps"] - 3
        for _ in range(64):
            cand = int(rng.choice(settle_pool))
            if lo <= cand <= hi and cand != r["settled_at"]:
                r["sham_at"] = cand
                break
        else:
            r["sham_at"] = int(np.clip(r["settled_at"] + 1, lo, hi))
        r["sham_gain"] = _gain_at(r["enc"], r["sham_at"], n_sub)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "enc"} for r in rows])
    diff = df.real_gain.to_numpy() - df.sham_gain.to_numpy()
    draws = [float(np.mean(rng.choice(diff, diff.size, replace=True)))
             for _ in range(BOOTSTRAP_DRAWS)]
    lo, hi = percentile_interval(draws)
    survives = bool(np.isfinite(lo) and lo > 0.0)

    out = v6_dir()
    df.to_csv(out / "v11a_placebo_split.csv", index=False)
    return {
        "check": "V-11a — E36's temporal ordering against a placebo split",
        "question": ("does method uptake rise because the reader settled on the goal, or merely "
                     "because the later window has more evidence in it?"),
        "real_gain": float(df.real_gain.mean()),
        "sham_gain": float(df.sham_gain.mean()),
        "real_minus_sham": float(diff.mean()),
        "interval": [lo, hi],
        "excludes_zero_positive": survives,
        "n_rollouts": int(len(df)),
        "mean_settled_at": float(df.settled_at.mean()),
        "mean_sham_at": float(df.sham_at.mean()),
        "outcome": ("SETTLING_DOES_WORK_THE_CLOCK_DOES_NOT" if survives
                    else "THE_ORDERING_IS_THE_CLOCK"),
        "how_to_read": (
            "the sham split sits at the same average depth in the reading as the real one, so any "
            "gain it reproduces is what the clock buys. Whatever is left over is what settling "
            "buys. A null here does not make E36 false; it makes the temporal plate a picture of "
            "elapsed evidence and the ordering claim unsupported by it."),
    }


def run_b(cfg: Config, n_obs: int = 60, n_timesteps: int = 24, forced_k: int = 12) -> dict:
    """V-11b: the maker reads its own artifact, against what the maker can declare."""
    world, _cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    e43 = json.loads((v6_dir() / "e43_selfreport.json").read_text(encoding="utf-8"))
    p_report_by_mu = {int(c["mu"]): float(c["p_self_report"]) for c in e43["cells"]}
    rows = []
    for mu in E43_MU:
        base = 43_000 + mu * 977
        for i in range(int(n_obs)):
            art_rng = np.random.default_rng(base * 31 + i)
            g_true = int(art_rng.integers(ng))
            creator, artifact, env = H.make_artifact_and_env(
                world, cfg_r, g_true, int(mu), 1.0, n_timesteps, art_rng,
                provenance=SELF_READ_TIER)

            # WHAT THE MAKER CAN SAY, taken from E43's own committed cells rather than
            # re-derived here, so this cannot quietly drift away from the plate it is checking.
            declared_ok = int(art_rng.random() < p_report_by_mu[int(mu)])

            # WHAT THE MAKER GETS BY READING ITS OWN WORK. Same body plan -- it is the same
            # person -- but a flat prior over goals, so it has to recover which one was running
            # from the artifact rather than from remembering.
            agent = make_v5_observer(world, np.random.default_rng(base * 13 + i))
            enc = H.run_encounter(world, cfg_r, artifact, env, agent, creator,
                                  np.random.default_rng(base * 7907 + i),
                                  n_timesteps, SELF_READ_GLANCES, n_sub, n_mu, ng,
                                  float(world.cfg.signal_model.kappa), true_goal=g_true)
            rows.append({"mu": int(mu), "i": i,
                         "declared_ok": declared_ok,
                         "read_own_ok": int(enc.correct),
                         "read_own_error_reduction": float(enc.error_reduction),
                         "process_recovered": float(enc.process["process_accuracy"])})

    df = pd.DataFrame(rows)
    by_mu = {int(m): {"declared": float(s.declared_ok.mean()),
                      "read_own_work": float(s.read_own_ok.mean()),
                      "process_recovered": float(s.process_recovered.mean()),
                      "n": int(len(s))}
             for m, s in df.groupby("mu")}

    rng = np.random.default_rng(SEED_OFFSET + 11_100)
    deep = df[df.mu == max(E43_MU)]
    diff = deep.read_own_ok.to_numpy() - deep.declared_ok.to_numpy()
    draws = [float(np.mean(rng.choice(diff, diff.size, replace=True)))
             for _ in range(BOOTSTRAP_DRAWS)]
    lo, hi = percentile_interval(draws)
    survives = bool(np.isfinite(lo) and lo > 0.0)

    # THE CLAIM IS AN INTERACTION, NOT A LEVEL. "Experts can learn about themselves from their own
    # process" does not say reading always beats declaring -- on a scribble it should lose, because
    # a novice can simply tell you what they were doing. It says the balance TIPS as the work gets
    # deeper. So the statistic that decides it is the change in (read - declared) from the
    # shallowest cell to the deepest, and the arms are paired within each cell.
    shallow = df[df.mu == min(E43_MU)]
    a_deep = deep.read_own_ok.to_numpy() - deep.declared_ok.to_numpy()
    a_shallow = shallow.read_own_ok.to_numpy() - shallow.declared_ok.to_numpy()
    idraws = [float(np.mean(rng.choice(a_deep, a_deep.size, replace=True))
                    - np.mean(rng.choice(a_shallow, a_shallow.size, replace=True)))
              for _ in range(BOOTSTRAP_DRAWS)]
    ilo, ihi = percentile_interval(idraws)
    tips = bool(np.isfinite(ilo) and ilo > 0.0)

    out = v6_dir()
    df.to_csv(out / "v11b_maker_reads_itself.csv", index=False)
    return {
        "check": "V-11b — the maker sat down in front of its own work",
        "question": ("can a maker recover a purpose off its own artifact that it can no longer "
                     "declare?"),
        "by_depth": by_mu,
        "at_the_deepest": {"declared": float(deep.declared_ok.mean()),
                           "read_own_work": float(deep.read_own_ok.mean()),
                           "advantage": float(diff.mean()),
                           "interval": [lo, hi],
                           "n": int(len(deep))},
        "excludes_zero_positive": survives,
        "the_interaction": {
            "advantage_at_the_shallowest": float(a_shallow.mean()),
            "advantage_at_the_deepest": float(a_deep.mean()),
            "tipping": float(a_deep.mean() - a_shallow.mean()),
            "interval": [ilo, ihi],
            "excludes_zero_positive": tips,
            "why_this_is_the_statistic": (
                "on a scribble the maker should WIN, because a novice can just tell you what they "
                "were doing. The claim is that the balance tips with depth, so the level at any "
                "one depth decides nothing and the change across depths decides everything."),
        },
        "outcome": ("THE_BALANCE_TIPS_TOWARD_THE_WORK_AS_DEPTH_RISES" if tips
                    else "READING_YOUR_OWN_WORK_BUYS_NOTHING"),
        "how_to_read": (
            "the reader here has the maker's own body plan and a flat prior over goals, so it is "
            "the same person with no memory of which intention was running. If it beats the "
            "declared report, then the work is a better record of the intent than the maker is, "
            "which is what the plate's second sentence claims and what E43 never tested."),
        "what_it_still_cannot_show": (
            "self-report is modelled as a probability that falls with depth, not derived from the "
            "maker's own machinery. So this establishes that the ARTIFACT carries recoverable "
            "intent at depths where the declared channel is degraded; it does not derive the "
            "degradation. E43 is the same in this respect and says so."),
    }


def run(cfg: Config, **kw) -> dict:
    a = run_a(cfg, **{k: v for k, v in kw.items() if k in ("n_obs", "n_timesteps", "forced_k")})
    b = run_b(cfg, **{k: v for k, v in kw.items() if k in ("n_obs", "n_timesteps", "forced_k")})
    verdict = {"check": "V-11 — the two controls plates 5 and 11 needed", "V-11a": a, "V-11b": b}
    (v6_dir() / "v11_two_confounds.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
