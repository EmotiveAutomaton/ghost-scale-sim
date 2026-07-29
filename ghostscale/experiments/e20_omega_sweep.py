"""E20 — the overlap sweep, with V4.5 §6's required addition (V4 §E20).

THE VARIABLE. omega is how much of the creator's generative structure the observer's
hypothesis space can express. omega = 1 is a shared likelihood family (human); omega = 0 is
fully foreign; in between is where curation, hybrid production, and any partially-shared
architecture live. V4 §0.3 says this parameter subsumes the morphological-distance condition,
the curation condition, and the human/synthetic confound, and that no separate animal
experiment should be built.

TWO QUESTIONS, and V4.5 §6 promotes the second from a column to a headline.

1. **Where does confident fabrication peak?** V4 §E20 predicts LOW BUT NONZERO omega: enough
   in-family structure to make an explanation seem available, not enough to make it correct.
   The unidentifiability model V1-V3 ran on could not generate that prediction — it predicts
   failure to be monotone in how much goal signal is present — so an interior peak is the
   strongest single argument for the reframe, and a peak at zero means the reframe buys
   nothing here.

2. **Where does the metabolic prediction flip?** E19's unpredicted finding is that observers
   do NOT disengage from goal-foreign content: 0.746 engagement across free steps, no
   resolution, ``crash_signature`` false in every cell. That inverts V1-V3's prediction that
   people disengage from synthetic content and save the effort.

   This matters beyond the model. **H1 and H3 in the preprint both assume goal-empty content,
   and both invert under goal-foreign.** Goal-empty predicts an autonomic drop within 2-4
   seconds. Goal-foreign predicts sustained arousal with no resolution. A single pupillometry
   measurement discriminates them, which makes this the sharpest and cheapest empirical
   prediction the framework has produced — and E20's crossing point is what tells a human
   study where to look.

So engagement and ``crash_signature`` are primary outcomes here, reported first, not secondary
columns.

DESIGN. Foreign content at each omega, unsigned, one artifact per (omega, seed) seen by every
observer in the cell — E2's design, load-bearing for the between-observer measure exactly as
it was in E19. The observer holds the four real goals; EXPLORE is disabled, because E19 showed
it inert on foreign content (0.2036 mass against a 0.2000 flat baseline, chosen slightly LESS
often than chance), so an EXPLORE arm would double the sweep to re-measure a null.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants as K
from .. import foreign as FN
from .. import metrics
from .. import prereg_v4 as P4
from .. import prereg_v4_5 as P45
from ..config import Config
from ..creators import HumanCreator
from ..environment import Artifact, Environment
from ..observer import rollout_observer
from ..v4_model import build_v4_world, load_v4_config, make_v4_observer
from . import _common as C

OMEGA_GRID = P45.E20_OMEGA_GRID


def _e20_worker(payload):
    (cfg_raw, cell_index, omega, seed_rep, base_seed, n_obs, n_timesteps, forced_k) = payload
    from ..config import Config as _Config

    cfg = _Config(cfg_raw).copy()
    cfg.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))

    # The OBSERVER's model is built at omega = 0 in the sense that matters: its likelihood
    # family is its own four goal signatures, always. omega governs the WORLD's foreign
    # family, which is what ``build_v4_world`` mixes. The observer never sees omega and has no
    # parameter for it — that asymmetry is the misspecification the whole experiment is about.
    world = build_v4_world(cfg, omega=omega, include_explore=False)
    creators = {g: HumanCreator(cfg, world.sigs.sig_true, g)
                for g in range(FN.NUM_REAL_GOALS)}
    env = Environment(cfg, world.gm, np.random.default_rng(base_seed + cell_index),
                      honesty=1.0, signing_rate=0.0, creator_bank=creators,
                      foreign_sig=world.sigs.sig_foreign)

    art_rng = np.random.default_rng(C.observer_seed(base_seed, cell_index, seed_rep, 9_999))
    k = int(art_rng.integers(FN.NUM_REAL_GOALS))
    artifact = Artifact(provenance=K.GHOST, goal=k, declared_signal=K.UNSIGNED,
                        foreign_goal=k)

    recs = []
    for i in range(n_obs):
        rng = C.observer_rng(base_seed, cell_index, seed_rep, i)
        agent = make_v4_observer(world, rng)
        res = rollout_observer(agent, artifact, env, cfg, rng, n_timesteps=n_timesteps,
                               force_deep_k=forced_k, initial_glance=True)
        post = np.asarray(res.final_goal_posterior, dtype=float)
        free = np.asarray(res.attention)[forced_k:]
        recs.append({
            "omega": float(omega), "seed_rep": seed_rep, "observer": i,
            "foreign_goal": k,
            # At omega = 1 the foreign family IS the human family, so the foreign goal index
            # is also the correct in-family goal and accuracy is meaningful. Below 1 it is the
            # goal the observer WOULD be right about if the overlap were complete, which is
            # the right reference for asking whether partial overlap misleads it.
            "matches_foreign_goal": int(int(np.argmax(post)) == k),
            "modal_goal": int(np.argmax(post)),
            "final_entropy": float(metrics.within_observer_entropy(post)),
            "engaged_fraction": float(np.mean(free == K.DEEP)) if free.size else float("nan"),
            "cum_deep": int(res.cum_deep),
            "posterior": post.tolist(),
        })
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    res_dir, fig_dir = C.ensure_dirs(out_dir)
    prereg_path = res_dir / "v4_5_e20_preregistration.json"
    P45.write_preregistration_e20(cfg, prereg_path)

    base_seed = int(cfg.run.base_seed if seed is None else seed)
    n_obs = int(cfg.get("experiments.e20.n_observers", 200))
    n_seeds = int(cfg.get("experiments.e20.n_seeds", 20))
    forced_k = int(cfg.get("experiments.e20.forced_deep_k", 10))
    n_timesteps = int(cfg.get("experiments.e20.n_timesteps", 24))

    payloads, ci = [], 0
    for om in OMEGA_GRID:
        for s in range(n_seeds):
            payloads.append((cfg.raw, ci, float(om), s, base_seed, n_obs, n_timesteps,
                             forced_k))
        ci += 1
    recs = C.run_parallel(payloads, _e20_worker, workers)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "posterior"} for r in recs])
    df.to_csv(res_dir / "e20_points.csv", index=False)

    stats = cell_stats(recs)
    stats.to_csv(res_dir / "e20_omega_sweep.csv", index=False)

    verdict = build_verdict(stats)
    (res_dir / "e20_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    if make_fig:
        make_figure(stats, verdict, fig_dir / "e20_omega_sweep.png")
    return stats


def cell_stats(recs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(recs)
    rows = []
    for om, sub in df.groupby("omega"):
        betweens, per_seed_engaged = [], []
        for _, g in sub.groupby("seed_rep"):
            betweens.append(metrics.between_observer_entropy(
                [np.asarray(p, dtype=float) for p in g["posterior"]]))
            per_seed_engaged.append(float(g.engaged_fraction.mean()))
        within = float(sub.final_entropy.mean())
        between = float(np.mean(betweens))
        engaged = float(sub.engaged_fraction.mean())
        crash = P4.crash_signature(within, engaged)
        rows.append({
            "omega": float(om), "n": int(len(sub)),
            # PRIMARY, per V4.5 §6.
            "engaged_fraction": engaged,
            "engaged_fraction_sd": float(sub.engaged_fraction.std()),
            "crashed": bool(crash["crashed"]),
            "unresolved": bool(crash["unresolved"]),
            "disengaged": bool(crash["disengaged"]),
            "fabrication_index": P45.fabrication_index(within, between,
                                                       FN.NUM_REAL_GOALS),
            "fabrication_index_strict": P45.fabrication_index_strict(
                within, float(sub.matches_foreign_goal.mean()), FN.NUM_REAL_GOALS),
            # The standard error that matters for the crossing point is ACROSS SEEDS, not
            # across observers: every observer in a seed reads the same artifact, so they are
            # not independent draws and an observer-level SE would understate the uncertainty
            # by roughly sqrt(n_observers).
            "engaged_sem_across_seeds": float(
                np.std(per_seed_engaged, ddof=1) / np.sqrt(len(per_seed_engaged))
                if len(per_seed_engaged) > 1 else float("nan")),
            # The components, always reported beside the index.
            "within_observer": within,
            "within_observer_sd": float(sub.final_entropy.std()),
            "between_observer": between,
            "between_observer_sd": float(np.std(betweens)),
            "matches_foreign_goal": float(sub.matches_foreign_goal.mean()),
        })
    return pd.DataFrame(rows).sort_values("omega").reset_index(drop=True)


def build_verdict(stats: pd.DataFrame) -> dict:
    v = P45.e20_verdict(stats.omega.tolist(), stats.engaged_fraction.tolist(),
                        stats.crashed.tolist(), stats.fabrication_index.tolist(),
                        stats.matches_foreign_goal.tolist(),
                        fabrication_strict=stats.fabrication_index_strict.tolist(),
                        engaged_sem=stats.engaged_sem_across_seeds.tolist())
    v["experiment"] = "E20 — the overlap sweep"
    v["cell_table"] = stats.to_dict(orient="records")
    return v


def make_figure(stats: pd.DataFrame, verdict: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..figures import set_style
    set_style()

    om = stats.omega.values
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    ax.errorbar(om, stats.engaged_fraction, yerr=stats.engaged_fraction_sd, marker="o",
                capsize=3, color="C0")
    ax.axhline(0.5, ls=":", color="k", lw=1)
    cross = verdict.get("engagement_crossing_omega")
    if cross is not None:
        ax.axvline(cross, ls="--", color="firebrick", lw=1.2,
                   label=f"attention gives way at\noverlap = {cross:.2f}")
        ax.legend(fontsize=7)
    ax.set(xlabel="how much the reader's world overlaps the maker's",
           ylabel="share of free time spent looking closely", ylim=(-0.05, 1.05),
           title="THE METABOLIC QUESTION\nWhere does it stop holding attention?")

    ax = axes[1]
    ax.plot(om, stats.fabrication_index, marker="o", color="C3")
    pk = verdict["fabrication_peak_omega"]
    ax.axvline(pk, ls="--", color="C3", lw=1.2, label=f"peak at overlap = {pk:.2f}")
    ax.set(xlabel="how much the reader's world overlaps the maker's",
           ylabel="sure of itself AND no two readers agree",
           title="Confident invention")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(om, stats.within_observer, marker="o", label="each reader's own uncertainty")
    ax.plot(om, stats.between_observer, marker="s", label="disagreement between readers")
    ax.plot(om, stats.matches_foreign_goal, marker="^", label="reads the real purpose")
    ax.axhline(float(np.log(4)), ls=":", color="k", lw=1, label="total disagreement")
    ax.set(xlabel="how much the reader's world overlaps the maker's",
           ylabel="nats / share", title="The components")
    ax.legend(fontsize=7)

    fig.suptitle(f"E20 — {verdict['outcome'].replace('_', ' ').lower()}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E20 — the overlap sweep")
    args = ap.parse_args()
    cfg = load_v4_config(quick=args.quick, include_explore=False)
    if args.seed is not None:
        cfg.set("run.base_seed", int(args.seed))
    out = Path(args.out) if args.out else None
    stats = run(cfg, out_dir=out, workers=args.workers or C.default_workers())
    print(stats.to_string(index=False))
    v = json.loads((C.ensure_dirs(out)[0] / "e20_verdict.json").read_text(encoding="utf-8"))
    print("\n" + v["statement"])
    print("engagement crossing:", v["engagement_crossing_omega"])


if __name__ == "__main__":
    main()
