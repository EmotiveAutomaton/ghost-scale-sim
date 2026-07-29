"""E19 — Does the crash survive a generous hypothesis space? (V4 spec §2, stage 1.)

THE QUESTION. V1-V3's crash could be an artifact of an artificially small hypothesis space.
The observer held four specific goals and no fallback, so of course it failed to explain
content produced outside them. EXPLORE is the most generous fallback the theory permits: under
the FEP, epistemic foraging is a real policy, so "they were reducing uncertainty" is always
available to an observer modelling a creator as a fellow agent, and it covers an enormous
fraction of low-stakes human production.

If the crash survives EXPLORE, the result is stronger than anything in V1-V3, because the
obvious deflationary explanation was given its best shot. If EXPLORE absorbs foreign content,
the crash was hypothesis-space poverty and V4 spec §2 requires that be reported plainly and
prominently.

-----------------------------------------------------------------------------------------
DESIGN. Two arms crossed with three content types.

  arm            observer goal space
  explore_off    the four real goals. The V1-V3 observer, widened to 16 features.
  explore_on     the four real goals PLUS EXPLORE.

  content              provenance   features drawn from            what it tests
  human_directed       CREATOR      HumanCreator(sig_true[g])      the ordinary case
  human_exploratory    CREATOR      HumanCreator(sig_EXPLORE)      POSITIVE CONTROL
  foreign              GHOST        sig_foreign[k] at omega = 0    the decisive cell

Content is generated identically in both arms. Only the observer's hypothesis space differs,
which is the whole point: any difference between arms is attributable to EXPLORE and to
nothing else. The creator bank is therefore built once from the full five-signature set and
shared, rather than derived from the arm's own (four- or five-goal) model.

EVERYTHING IS UNSIGNED. ``signing_rate = 0``, so no artifact carries a provenance tag and the
content has to carry the whole load. With a tag present the observer could infer GHOST from
the label and disengage on that basis, which would answer a question about the Ghost Scale
rather than the question E19 is asking about hypothesis spaces.

THE POSITIVE CONTROL IS NOT DECORATION. If EXPLORE fails to absorb exploratory HUMAN content
then it is not functioning as a fallback at all, and its failure to absorb foreign content
carries no information: a hypothesis that explains nothing cannot be said to have failed to
explain foreign content in particular. ``prereg_v4.e19_verdict`` returns INCONCLUSIVE in that
case rather than crediting it as a win.
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
from ..config import Config
from ..creators import HumanCreator
from ..environment import Artifact, Environment
from ..observer import rollout_observer
from ..v4_model import (build_v4_world, explore_mass, load_v4_config, make_v4_observer,
                        real_goal_posterior)
from . import _common as C

ARMS = ("explore_off", "explore_on")
CONTENT = ("human_directed", "human_exploratory", "foreign")


def build_content_creators(cfg: Config, sigs: FN.V4Signatures) -> dict[int, HumanCreator]:
    """The world's creators: one per real goal, plus one that emits sig_EXPLORE.

    Built from the FULL signature set regardless of the observer's arm, so that the content
    an observer sees is identical in explore_off and explore_on. Wiring the bank to the arm's
    own model would confound the manipulation with the stimulus.
    """
    full = np.vstack([sigs.sig_true, sigs.sig_explore[None, :]])
    return {g: HumanCreator(cfg, full, g) for g in range(full.shape[0])}


def _e19_worker(payload):
    (cfg_raw, cell_index, arm, content, seed_rep, base_seed, n_obs, n_timesteps,
     forced_k, omega) = payload
    from ..config import Config as _Config

    # The parent's config already carries every V4 setting (and --quick, if it was set). The
    # ONE thing that differs per arm is the size of the observer's goal space, so it is set
    # here rather than reloading from disk in the subprocess. Reloading would silently drop
    # any override the parent applied.
    include_explore = (arm == "explore_on")
    cfg = _Config(cfg_raw).copy()
    n_real = int(cfg.get("v4.cardinalities.num_real_goals", FN.NUM_REAL_GOALS))
    cfg.set("cardinalities.num_goals", n_real + (1 if include_explore else 0))
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))
    world = build_v4_world(cfg, omega=omega, include_explore=include_explore)
    creators = build_content_creators(cfg, world.sigs)
    env = Environment(cfg, world.gm, np.random.default_rng(base_seed + cell_index),
                      honesty=1.0, signing_rate=0.0, creator_bank=creators,
                      foreign_sig=world.sigs.sig_foreign)

    # ONE artifact per (cell, seed_rep), seen by every observer in it. This is E2's design and
    # it is load-bearing for the between-observer measure: disagreement only means anything
    # when the thing being disagreed about is held fixed.
    #
    # An earlier draft drew a fresh goal PER OBSERVER, which silently turned that column into
    # a measurement of the sampler. It showed up as human_directed scoring 1.379 nats of
    # "disagreement" against a 1.386 ceiling while its accuracy was 1.000 — observers who all
    # get the right answer cannot be disagreeing, so the statistic was not measuring what its
    # name said. Goal variety now comes from the 20 seed_reps instead.
    art_rng = np.random.default_rng(C.observer_seed(base_seed, cell_index, seed_rep, 9_999))
    if content == "foreign":
        k = int(art_rng.integers(FN.NUM_REAL_GOALS))
        artifact = Artifact(provenance=K.GHOST, goal=k, declared_signal=K.UNSIGNED,
                            foreign_goal=k)
        true_goal = -1                          # no in-family goal is correct, by construction
    elif content == "human_exploratory":
        artifact = Artifact(provenance=K.CREATOR, goal=FN.EXPLORE, declared_signal=K.UNSIGNED)
        true_goal = FN.EXPLORE
    else:
        g = int(art_rng.integers(FN.NUM_REAL_GOALS))
        artifact = Artifact(provenance=K.CREATOR, goal=g, declared_signal=K.UNSIGNED)
        true_goal = g

    recs = []
    for i in range(n_obs):
        rng = C.observer_rng(base_seed, cell_index, seed_rep, i)
        art = artifact
        agent = make_v4_observer(world, rng)
        res = rollout_observer(agent, art, env, cfg, rng, n_timesteps=n_timesteps,
                               force_deep_k=forced_k, initial_glance=True)

        post = np.asarray(res.final_goal_posterior, dtype=float)
        real = real_goal_posterior(post)
        # Engagement is measured over the FREE steps only. The forced phase is the same in
        # every cell by construction, so including it would dilute the contrast with a
        # constant and make every arm look more engaged than it chose to be.
        free = np.asarray(res.attention)[forced_k:]
        engaged = float(np.mean(free == K.DEEP)) if free.size else float("nan")

        recs.append({
            "arm": arm, "content": content, "omega": omega, "seed_rep": seed_rep,
            "observer": i, "true_goal": true_goal,
            "explore_mass": explore_mass(post, include_explore),
            "final_entropy": float(metrics.within_observer_entropy(post)),
            "real_goal_entropy": float(metrics.within_observer_entropy(real)),
            "modal_real_goal": int(np.argmax(real)),
            "modal_goal_full": int(np.argmax(post)),
            "correct": int(np.argmax(real) == true_goal) if 0 <= true_goal < FN.NUM_REAL_GOALS
                       else -1,
            "engaged_fraction": engaged,
            "cum_deep": int(res.cum_deep),
            # Whether this observer's single best explanation was "they were exploring".
            # Needed because between-observer disagreement is measured on the REAL-goal
            # posterior (decision D4), which for exploratory content reports the arbitrary
            # residual left AFTER EXPLORE is marginalised out. Without this column the
            # positive control looks like total disagreement when the observers in fact
            # agree, almost unanimously, on the explanation that matters.
            "modal_is_explore": int(include_explore and int(np.argmax(post)) == FN.EXPLORE),
            "real_posterior": real.tolist(),
            "full_posterior": post.tolist(),
        })
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    # PRE-REGISTRATION FIRST. No inference happens before the criteria, and sig_EXPLORE
    # itself, are on disk and hash-locked (V4 spec §6, N17).
    prereg_path = res_dir / "v4_preregistration.json"
    P4.write_preregistration_v4(cfg, prereg_path)
    P4.assert_prereg_locked_v4(prereg_path)
    P4.assert_explore_matches_prereg(cfg, prereg_path)

    base_seed = int(cfg.run.base_seed if seed is None else seed)
    e = cfg.get("experiments.e19", {})
    n_obs = int(cfg.get("experiments.e19.n_observers", 200))
    n_seeds = int(cfg.get("experiments.e19.n_seeds", 20))
    forced_k = int(cfg.get("experiments.e19.forced_deep_k", 10))
    n_timesteps = int(cfg.get("experiments.e19.n_timesteps", 24))

    payloads, ci = [], 0
    for arm in ARMS:
        for content in CONTENT:
            for s in range(n_seeds):
                payloads.append((cfg.raw, ci, arm, content, s, base_seed, n_obs,
                                 n_timesteps, forced_k, 0.0))
            ci += 1
    recs = C.run_parallel(payloads, _e19_worker, workers)

    # Both posteriors KEPT (V4.5 A2) — see the note in e2_variance.run. ``real_posterior`` is
    # the four-real-goal object decision D4 fixes as the comparable one, so it is what the
    # calibration analysis scores; ``full_posterior`` carries the EXPLORE mass alongside it.
    df = pd.DataFrame(recs)
    df.to_csv(res_dir / "e19_explore.csv", index=False)

    stats = cell_stats(recs)
    stats.to_csv(res_dir / "e19_cell_stats.csv", index=False)

    verdict = build_verdict(stats, cfg)
    (res_dir / "e19_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    if make_fig:
        make_figure(stats, verdict, fig_dir / "e19_explore.png")
    return stats


def cell_stats(recs: list[dict]) -> pd.DataFrame:
    """Per-cell means, plus the between-observer disagreement computed per seed and averaged.

    Disagreement is measured on the REAL-goal posterior (decision D4), so it is directly
    comparable across the two arms and against V1-V3's E2 and E17, whose ceiling is the same
    ln(4) = 1.386.
    """
    df = pd.DataFrame(recs)
    rows = []
    for (arm, content), sub in df.groupby(["arm", "content"]):
        betweens, betweens_full = [], []
        for _, g in sub.groupby("seed_rep"):
            betweens.append(metrics.between_observer_entropy(
                [np.asarray(p, dtype=float) for p in g["real_posterior"]]))
            betweens_full.append(metrics.between_observer_entropy(
                [np.asarray(p, dtype=float) for p in g["full_posterior"]]))
        rows.append({
            "arm": arm, "content": content, "n": int(len(sub)),
            "explore_mass": float(sub.explore_mass.mean()),
            "explore_mass_sd": float(sub.explore_mass.std()),
            "final_entropy": float(sub.final_entropy.mean()),
            "final_entropy_sd": float(sub.final_entropy.std()),
            "real_goal_entropy": float(sub.real_goal_entropy.mean()),
            "engaged_fraction": float(sub.engaged_fraction.mean()),
            "engaged_fraction_sd": float(sub.engaged_fraction.std()),
            "between_observer": float(np.mean(betweens)),
            "between_observer_sd": float(np.std(betweens)),
            "between_observer_full": float(np.mean(betweens_full)),
            "modal_is_explore": float(sub.modal_is_explore.mean()),
            "accuracy": float(sub[sub.correct >= 0].correct.mean())
                        if (sub.correct >= 0).any() else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["content", "arm"]).reset_index(drop=True)


def build_verdict(stats: pd.DataFrame, cfg: Config) -> dict:
    cells, crashes = {}, {}
    for r in stats.itertuples():
        key = f"{r.content}/{r.arm}"
        cells[key] = P4.explore_absorbed(r.explore_mass, r.final_entropy, r.engaged_fraction)
        crashes[key] = P4.crash_signature(r.final_entropy, r.engaged_fraction)
    verdict = P4.e19_verdict(cells)
    verdict["crash_signature"] = crashes
    verdict["experiment"] = "E19 — does the crash survive a generous hypothesis space?"
    verdict["stage"] = 1
    verdict["cell_table"] = stats.to_dict(orient="records")
    return verdict


def make_figure(stats, verdict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..figures import set_style
    set_style()

    content_label = {"human_directed": "human work,\nmade on purpose",
                     "human_exploratory": "human work,\nmade while exploring",
                     "foreign": "machine work,\nforeign purpose"}
    arm_label = {"explore_off": "reader without the\n'they were exploring' option",
                 "explore_on": "reader WITH the\n'they were exploring' option"}
    colours = {"explore_off": "grey", "explore_on": "C0"}
    order = list(CONTENT)
    x = np.arange(len(order))
    w = 0.38

    def val(content, arm, col):
        return float(stats[(stats.content == content) & (stats.arm == arm)][col].iloc[0])

    fig, axes = plt.subplots(1, 4, figsize=(20.0, 4.8))

    ax = axes[0]
    for j, arm in enumerate(ARMS):
        vals = [val(c, arm, "explore_mass") for c in order]
        errs = [val(c, arm, "explore_mass_sd") for c in order]
        ax.bar(x + (j - 0.5) * w, vals, w, yerr=errs, capsize=3,
               color=colours[arm], label=arm_label[arm])
    ax.axhline(P4.E19_ABSORB_MASS, color="k", ls=":", lw=1,
               label=f"counts as 'explained away' ({P4.E19_ABSORB_MASS})")
    ax.axhline(1.0 / (FN.NUM_REAL_GOALS + 1), color="firebrick", ls="--", lw=1,
               label="what a reader with no idea would give it")
    ax.set_xticks(x, [content_label[c] for c in order], fontsize=8)
    ax.set(ylabel="belief that the maker was just exploring",
           title="Does 'they were just exploring'\nexplain the machine work away?")
    ax.legend(fontsize=7)

    ax = axes[1]
    for j, arm in enumerate(ARMS):
        vals = [val(c, arm, "modal_is_explore") for c in order]
        ax.bar(x + (j - 0.5) * w, vals, w, color=colours[arm], label=arm_label[arm])
    ax.axhline(1.0 / (FN.NUM_REAL_GOALS + 1), color="firebrick", ls="--", lw=1,
               label="what picking at random would give")
    ax.set_xticks(x, [content_label[c] for c in order], fontsize=8)
    ax.set(ylim=(0, 1.05),
           ylabel="share of readers who settled on 'they were exploring'",
           title="Do readers AGREE that exploring\nis the best explanation?")
    ax.legend(fontsize=7)

    ax = axes[2]
    for j, arm in enumerate(ARMS):
        vals = [val(c, arm, "engaged_fraction") for c in order]
        errs = [val(c, arm, "engaged_fraction_sd") for c in order]
        ax.bar(x + (j - 0.5) * w, vals, w, yerr=errs, capsize=3,
               color=colours[arm], label=arm_label[arm])
    ax.set_xticks(x, [content_label[c] for c in order], fontsize=8)
    ax.set(ylim=(0, 1.15), ylabel="share of free time spent looking closely",
           title="Readers do NOT give up on machine work.\n"
                 "They keep paying and never get there.")
    # Zero here means two opposite things, so the chart has to say which is which.
    ax.annotate("zero = solved it\nand stopped", xy=(0, 0.02), xytext=(0, 0.30),
                ha="center", fontsize=7, color="darkgreen",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1))
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[3]
    for j, arm in enumerate(ARMS):
        vals = [val(c, arm, "between_observer") for c in order]
        errs = [val(c, arm, "between_observer_sd") for c in order]
        ax.bar(x + (j - 0.5) * w, vals, w, yerr=errs, capsize=3,
               color=colours[arm], label=arm_label[arm])
    ax.axhline(float(np.log(FN.NUM_REAL_GOALS)), color="k", ls=":", lw=1,
               label="no two readers agree")
    ax.set_xticks(x, [content_label[c] for c in order], fontsize=8)
    ax.set(ylabel="disagreement about the everyday purpose (nats)",
           title="Disagreement about WHICH everyday purpose\n"
                 "(exploring is set aside here, see panel 2)")
    ax.legend(fontsize=7, loc="center left")

    fig.suptitle("E19 — The most generous excuse available does not explain machine work away. "
                 "It does explain human dabbling.", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E19 — does the crash survive a generous hypothesis space?")
    args = ap.parse_args()
    cfg = load_v4_config(quick=args.quick, include_explore=True)
    if args.seed is not None:
        cfg.set("run.base_seed", int(args.seed))
    out = Path(args.out) if args.out else None
    stats = run(cfg, out_dir=out, workers=args.workers or C.default_workers())
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
