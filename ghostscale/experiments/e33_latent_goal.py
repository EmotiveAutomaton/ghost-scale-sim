"""E33 — the latent goal (V5 C4).

Can an observer recover a goal the creator does not know it has? Under what conditions does it
beat the creator's own self-model? Does latent/declared divergence raise extractable depth?
And can a generative system produce the thing at all?

The construction, the three arms, and the restated categorical claim are documented in
``ghostscale/latent_goal.py``. This module runs them.

-----------------------------------------------------------------------------------------
WHY THIS HAS ITS OWN ROLLOUT LOOP. ``rollout_observer`` builds a three-modality observation
and hardcodes the free initial glance as ``[feature, signal, effort]``. E33 has a fourth
modality — the creator's self-report — so reusing that loop would mean widening a function
every V1-V4.5 experiment depends on, for one experiment's benefit. The loop here is short
because E33 forces DEEP throughout: depth lives in the ORDER of execution modes, and a reader
free to stop after ten steps has seen about one and a half blocks of three and a half, so
engagement is not a variable this experiment can afford to also be measuring. E30 measures it.

-----------------------------------------------------------------------------------------
WHAT IS MEASURED, AND AGAINST WHAT.

* ``latent_correct`` — did the observer recover the goal that actually drove the creator.
* ``declared_correct`` — did it recover the goal the creator would report.
* ``self_report_correct`` — did the CREATOR's own report name its latent goal. This is the
  baseline for "beats the self-model", and it MOVES with the divergence rate rather than
  sitting at a constant, which is what makes the comparison a comparison with the creator.
* ``recovered_mu`` — how much structure the reader thinks is behind the work. This is where
  C4's third question is answered, and it is why the trace was built into the plan.
* arm separation on ``recovered_mu`` — the categorical claim. The human arm must separate from
  both others while the deceptive and generative arms do not separate from each other.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants as K
from .. import foreign as FN
from .. import latent_goal as LG
from .. import metrics
from .. import prereg_v5_c234 as P5C
from ..config import Config
from ..v5_model import (MU_LEVELS, build_v5_world, load_v5_config, make_v5_observer,
                        marginal_goal, marginal_mu, marginal_subgoal, mgs_index, recovered_mu)
from . import _common as C

ARMS = P5C.E33_ARMS


def _state_goal_map(n_mu: int, ng: int, n_sub: int):
    lut = np.zeros(n_mu * ng * n_sub, dtype=int)
    for mi in range(n_mu):
        for g in range(ng):
            for s in range(n_sub):
                lut[mgs_index(mi, g, s, ng, n_sub)] = g
    return lut


def _e33_worker(payload):
    (cfg_raw, cell_index, arm, p_div, seed_rep, base_seed, n_obs, n_timesteps) = payload
    from ..config import Config as _Config

    cfg = _Config(cfg_raw).copy()
    cfg.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))

    world = build_v5_world(cfg)
    n_mu, n_sub, ng = len(MU_LEVELS), world.n_subgoals, FN.NUM_REAL_GOALS
    n_states = n_mu * ng * n_sub
    rho = float(cfg.get("experiments.e33.rho", 0.8))
    suppression = float(cfg.get("experiments.e33.suppression", 0.9))
    mu_true = 3
    mi_true = list(MU_LEVELS).index(mu_true)

    lut = _state_goal_map(n_mu, ng, n_sub)
    A3 = LG.build_declaration_likelihood(ng, n_states, int(cfg.cardinalities.num_attention),
                                         rho, lambda s: lut[s])

    art_rng = np.random.default_rng(C.observer_seed(base_seed, cell_index, seed_rep, 9_999))
    latent = int(art_rng.integers(ng))

    # The declaration. The generative arm confabulates from the artifact's own emission
    # marginal, so it needs that marginal first — which is the goal-averaged mode family at
    # this depth, i.e. what the output actually looks like.
    emission_marginal = world.subsig[mi_true, latent].mean(axis=0)
    decl = LG.draw_declaration(arm, latent, ng, p_div, art_rng,
                               emission_marginal=emission_marginal,
                               sig_true=world.sigs.sig_true)

    # THE TRACE. Only the self-unaware creator's conflict marks the work; the deceptive and
    # generative arms emit exactly what an ordinary creator of that latent goal would.
    chain = world.chains[latent]
    if arm == "human_unaware" and decl.diverged:
        chain = LG.conflict_chain(chain, decl.declared, n_sub, suppression)

    # Draw the artifact's mode trajectory once. An artifact is a fixed object; every observer
    # reads the same one (E19/E21's lesson).
    traj = np.empty(n_timesteps + 1, dtype=int)
    traj[0] = int(art_rng.integers(n_sub))
    for t in range(1, traj.size):
        traj[t] = int(art_rng.choice(n_sub, p=chain[:, traj[t - 1]]))
    emis = world.subsig[mi_true, latent]

    # IS THE TRACE REAL IN THE ARTIFACT, independently of whether any observer reads it?
    # A world-side measurement: how far this artifact's realised mode histogram sits from the
    # uniform one every un-conflicted plan produces. This is what separates the pre-registered
    # TRACE_IS_REAL_BUT_UNREAD branch from NO_CATEGORICAL_DIFFERENCE, and without it a null in
    # the observer's posterior could not be told apart from a null in the construction.
    hist = np.bincount(traj[:n_timesteps], minlength=n_sub).astype(float)
    hist /= hist.sum()
    uni = np.full(n_sub, 1.0 / n_sub)
    trace_kl = float(metrics.kl_divergence(hist, uni))

    recs = []
    for i in range(n_obs):
        rng = C.observer_rng(base_seed, cell_index, seed_rep, i)
        # The self-report channel is added AT CONSTRUCTION. pymdp caches the modality count
        # and the Markov-blanket map when the Agent is built, so an A array grown afterwards
        # raises inside the inference loop instead of being used.
        agent = make_v5_observer(world, rng, extra_modalities=[A3])
        agent.reset()

        # THE SELF-REPORT IS DELIVERED ONCE, AT THE FIRST STEP, AND THAT IS NOT A DETAIL.
        # A creator says what it was doing one time; it does not repeat itself every time the
        # reader looks at another part of the work. Delivered every timestep, one declaration
        # compounds into 24 independent likelihood factors — at rho = 0.8 that is a ratio of
        # 12 to 1 raised to the 24th — and the observer simply believes the label whatever the
        # content says. Measured that way, latent recovery was 0.00 at every divergence rate
        # in all three arms, which is a measurement of the delivery schedule and nothing else.
        said = decl.obs if decl.obs >= 0 else LG.no_declaration_index(ng)
        none_obs = LG.no_declaration_index(ng)
        for t in range(n_timesteps):
            f = int(rng.choice(emis.shape[1], p=emis[traj[t]]))
            qs = agent.infer_states([f, K.UNSIGNED, K.HIGH_COST,
                                     said if t == 0 else none_obs])

        q = np.asarray(qs[1], dtype=float)
        g_post = marginal_goal(q, n_mu, ng, n_sub)
        modal = int(np.argmax(g_post))
        recs.append({
            "arm": arm, "p_divergence": float(p_div), "seed_rep": seed_rep, "observer": i,
            "latent": latent, "declared": decl.declared, "diverged": int(decl.diverged),
            "latent_correct": int(modal == latent),
            "declared_correct": int(modal == decl.declared),
            "self_report_correct": int(decl.declared == latent),
            "recovered_mu": recovered_mu(q, n_mu, ng, n_sub),
            "mu_entropy": float(metrics.shannon_entropy(marginal_mu(q, n_mu, ng, n_sub))),
            "subgoal_entropy": float(metrics.shannon_entropy(
                marginal_subgoal(q, n_mu, ng, n_sub))),
            "final_entropy": float(metrics.within_observer_entropy(g_post)),
            # World-side: is the mark on the work, whether or not it is read?
            "artifact_trace_kl": trace_kl,
            "posterior": g_post.tolist(),
        })
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    res_dir, fig_dir = C.ensure_dirs(out_dir)
    prereg = res_dir / "v5_c234_preregistration.json"
    P5C.ensure_preregistration_c234(cfg, prereg)

    base_seed = int(cfg.run.base_seed if seed is None else seed)
    n_obs = int(cfg.get("experiments.e33.n_observers", 100))
    n_seeds = int(cfg.get("experiments.e33.n_seeds", 20))
    n_timesteps = int(cfg.get("experiments.e33.n_timesteps", 24))

    payloads, ci = [], 0
    for arm in ARMS:
        for p in P5C.E33_DIVERGENCE_GRID:
            for s in range(n_seeds):
                payloads.append((cfg.raw, ci, arm, float(p), s, base_seed, n_obs,
                                 n_timesteps))
            ci += 1
    recs = C.run_parallel(payloads, _e33_worker, workers)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "posterior"} for r in recs])
    df.to_csv(res_dir / "e33_points.csv", index=False)
    stats = cell_stats(recs)
    stats.to_csv(res_dir / "e33_cell_stats.csv", index=False)

    verdict = build_verdict(stats)
    (res_dir / "e33_verdict.json").write_text(json.dumps(verdict, indent=2, default=float),
                                              encoding="utf-8")
    if make_fig:
        make_figure(stats, verdict, fig_dir / "e33_latent_goal.png")
    return stats


def cell_stats(recs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(recs)
    rows = []
    for (arm, p), sub in df.groupby(["arm", "p_divergence"]):
        div = sub[sub.diverged == 1]
        rows.append({
            "arm": arm, "p_divergence": float(p), "n": int(len(sub)),
            "diverged_fraction": float(sub.diverged.mean()),
            "latent_recovery": float(sub.latent_correct.mean()),
            "declared_recovery": float(sub.declared_correct.mean()),
            "self_report_accuracy": float(sub.self_report_correct.mean()),
            "latent_recovery_when_diverged": (float(div.latent_correct.mean())
                                              if len(div) else float("nan")),
            "recovered_mu": float(sub.recovered_mu.mean()),
            "recovered_mu_when_diverged": (float(div.recovered_mu.mean())
                                           if len(div) else float("nan")),
            "subgoal_entropy": float(sub.subgoal_entropy.mean()),
            "final_entropy": float(sub.final_entropy.mean()),
            "artifact_trace_kl": float(sub.artifact_trace_kl.mean()),
            "artifact_trace_kl_when_diverged": (float(div.artifact_trace_kl.mean())
                                                if len(div) else float("nan")),
        })
    return pd.DataFrame(rows).sort_values(["arm", "p_divergence"]).reset_index(drop=True)


def build_verdict(stats: pd.DataFrame) -> dict:
    hu = stats[stats.arm == "human_unaware"].sort_values("p_divergence")

    # THE TRACE, measured on DIVERGED artifacts only and on recovered depth. Comparing all
    # artifacts would dilute it with the agreeing ones, which carry no trace in any arm and
    # are identical across arms by construction.
    def diverged_mu(arm):
        s = stats[(stats.arm == arm) & (stats.p_divergence > 0)]
        v = s.recovered_mu_when_diverged.dropna()
        return float(v.mean()) if len(v) else float("nan")

    mu_h, mu_d, mu_g = (diverged_mu(a) for a in ("human_unaware", "deceptive", "generative"))
    separation = {
        "human_vs_deceptive": abs(mu_h - mu_d),
        "human_vs_generative": abs(mu_h - mu_g),
        "deceptive_vs_generative": abs(mu_d - mu_g),
        "recovered_mu_when_diverged": {"human_unaware": mu_h, "deceptive": mu_d,
                                       "generative": mu_g},
    }

    verdict = P5C.e33_verdict(
        latent_recovery=hu.latent_recovery.tolist(),
        declared_recovery=hu.declared_recovery.tolist(),
        self_model_accuracy=hu.self_report_accuracy.tolist(),
        depth_by_divergence=hu.recovered_mu.tolist(),
        arm_separation=separation)
    def diverged_trace(arm):
        s = stats[(stats.arm == arm) & (stats.p_divergence > 0)]
        v = s.artifact_trace_kl_when_diverged.dropna()
        return float(v.mean()) if len(v) else float("nan")

    verdict["artifact_side_trace"] = {
        "mode_histogram_kl_from_uniform_when_diverged": {
            a: diverged_trace(a) for a in ARMS},
        "note": ("a WORLD-SIDE measurement: is the mark on the work at all, independently of "
                 "whether an observer reads it. If the human arm's artifacts differ here while "
                 "the observer's posterior does not, the trace is real and unread — which is "
                 "a different result from there being no trace, and the two would otherwise "
                 "be indistinguishable."),
    }
    verdict["experiment"] = "E33 — the latent goal"
    verdict["divergence_grid"] = hu.p_divergence.tolist()
    verdict["beats_self_model_by_rate"] = [
        {"p_divergence": float(r.p_divergence),
         "observer_recovers_latent": float(r.latent_recovery),
         "creator_self_report_names_latent": float(r.self_report_accuracy),
         "margin": float(r.latent_recovery - r.self_report_accuracy)}
        for r in hu.itertuples()]
    verdict["cell_table"] = stats.to_dict(orient="records")
    return verdict


def make_figure(stats: pd.DataFrame, verdict: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..figures import set_style
    set_style()

    hu = stats[stats.arm == "human_unaware"].sort_values("p_divergence")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    ax.plot(hu.p_divergence, hu.latent_recovery, marker="o",
            label="reader finds the real reason")
    ax.plot(hu.p_divergence, hu.self_report_accuracy, marker="s", ls="--", color="C7",
            label="maker's own account is right")
    ax.set(xlabel="how often the maker is wrong about themselves",
           ylabel="share correct", ylim=(0, 1.05),
           title="Can a reader know you\nbetter than you know yourself?")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(hu.p_divergence, hu.recovered_mu, marker="o", color="C4")
    ax.set(xlabel="how often the maker is wrong about themselves",
           ylabel="how much thought the reader thinks went in",
           title="Does self-blindness make\nthe work read as deeper?")

    ax = axes[2]
    sep = verdict["arm_separation"]["recovered_mu_when_diverged"]
    names = ["doesn't know\nitself", "lying", "no self to\nknow"]
    ax.bar(range(3), [sep["human_unaware"], sep["deceptive"], sep["generative"]], 0.6,
           color=["C2", "C3", "C0"])
    ax.set_xticks(range(3), names, fontsize=8)
    ax.set(ylabel="how much thought the reader thinks went in",
           title="Only one of these\nmarks the work")

    fig.suptitle(f"E33 — {verdict['verdict'].replace('_', ' ').lower()}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E33 — the latent goal")
    args = ap.parse_args()
    cfg = load_v5_config(quick=args.quick)
    if args.seed is not None:
        cfg.set("run.base_seed", int(args.seed))
    stats = run(cfg, out_dir=Path(args.out) if args.out else None,
                workers=args.workers or C.default_workers())
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
