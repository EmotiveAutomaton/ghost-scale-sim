"""E21 — Is the machinery necessary? (V4 spec §2 stage 2; V4.5 §0 priority 2.)

THE QUESTION A REFEREE WILL ASK AND NOBODY HAS ANSWERED. The framework's results are produced
by an active-inference observer with a generative model of another agent. Does that apparatus
do any work, or would a heuristic reproduce the same phenomena for a fraction of the
machinery?

V4.5 §2 and §7 both require that this experiment be ABLE to return "the machinery is
unnecessary", and that the answer go in the first line of the section. That is why the arms
are given every advantage the design can afford them: a common stimulus, the same
heterogeneity, and a free sweep over any parameter they have that has no principled value.

-----------------------------------------------------------------------------------------
THE TWO SIGNATURES, AND WHY THEY ARE THE RIGHT ONES.

Several baselines WILL reproduce disengagement. Disengagement is cost-benefit arithmetic and
needs no theory of mind, so a result showing that a heuristic disengages from cheap content
is not evidence against anything. The design therefore does not test for it.

  1. E2's SIMULTANEOUS confidence/disagreement dissociation. A heuristic can be confidently
     wrong. It should not be confidently wrong in a way that differs per observer while every
     observer is individually certain, on content that contains no goal.
  2. E19's sustained-but-unresolved attention on foreign content. That is a prediction about
     an agent that keeps EXPECTING to learn something specific. An effort heuristic has no
     expectation to be wrong about, so arm D should fail it — and if arm D reproduces it
     anyway, that is informative about how little machinery the phenomenon needs.

-----------------------------------------------------------------------------------------
DESIGN. content x declared signal x arm.

  content            what the observer is looking at
  human_directed     CREATOR artifact from a real creator policy. The CONTROL, and it is
                     load-bearing twice: it is the positive control for goal recovery, and
                     it is the specificity clause for the engagement signature.
  goal_empty         GHOST artifact, pure noise_free_synth. THIS IS THE STIMULUS E2 ACTUALLY
                     RAN ON — V1-V3's model of synthetic content, lifted to 16 features.
  goal_foreign       GHOST artifact from sig_foreign at omega = 0. V4's model of it.

  declared signal    SIG_CREATOR (the lie / the fabrication condition), SIG_GHOST (truthful,
                     the control that shows what honest labelling buys), UNSIGNED (no label:
                     content carries the whole load, which is the condition E19 measured its
                     engagement result under).

Both models of synthetic content are run because the dissociation is a V1 result about
goal-empty content while V4.5 is a delta on a model that REPLACED that stimulus. Asking the
necessity question about only one of them would answer it for only one model of synthetic
content, and the honest answer may differ between them.

The observer's goal space is the four real goals. EXPLORE is E19's manipulation and adding it
here would confound the necessity question with the hypothesis-space question.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import baselines as BL
from .. import constants as K
from .. import foreign as FN
from .. import metrics
from .. import prereg_v4_5 as P45
from ..config import Config
from ..creators import HumanCreator
from ..environment import Artifact, Environment
from ..generative_model import alpha_by_provenance, build_D
from ..observer import rollout_observer
from ..v4_model import build_v4_world, load_v4_config, make_v4_observer
from . import _common as C

ARMS = P45.E21_ARMS
CONTENT = ("human_directed", "goal_empty", "goal_foreign")
SIGNALS = (("SIG_CREATOR", K.SIG_CREATOR), ("SIG_GHOST", K.SIG_GHOST), ("UNSIGNED", K.UNSIGNED))

# Arm D's disengagement threshold and arm E's stopping confidence have no principled value.
# Both are swept and the arm is credited if ANY setting reproduces a signature — see the
# pre-registration's "arm_D_is_given_its_best_shot". The full grid is reported.
#
# The grid's SCALE was set by arithmetic, not by tuning: adding one observation to a histogram
# of n moves it by roughly 1/n in total variation, so over 24 timesteps the movement a
# threshold has to bracket runs from about 0.04 down. A grid that topped out at 0.05 left arm
# D engaged for the whole rollout at four of its five settings, which would have failed it by
# choice of constant rather than on the merits. The grid now spans "gives up almost at once"
# to "never gives up".
D_THRESHOLD_GRID = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
E_CONFIDENCE_GRID = (0.60, 0.80, 0.95, 0.99)


def _make_artifact(content: str, signal: int, rng: np.random.Generator) -> tuple[Artifact, int]:
    """One artifact, plus the goal that counts as correct (-1 where none does)."""
    g = int(rng.integers(FN.NUM_REAL_GOALS))
    if content == "human_directed":
        return Artifact(provenance=K.CREATOR, goal=g, declared_signal=signal), g
    if content == "goal_empty":
        # Pure noise_free_synth: the artifact carries an assigned goal that its features do
        # not transmit. This is V1's GHOST stimulus and it is what E2 measured.
        return Artifact(provenance=K.GHOST, goal=g, declared_signal=signal), -1
    return Artifact(provenance=K.GHOST, goal=g, declared_signal=signal, foreign_goal=g), -1


def _e21_worker(payload):
    (cfg_raw, cell_index, content, sig_name, signal, seed_rep, base_seed, n_obs,
     n_timesteps, forced_k, n_train) = payload
    from ..config import Config as _Config

    cfg = _Config(cfg_raw).copy()
    cfg.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))

    world = build_v4_world(cfg, omega=0.0, include_explore=False)
    creators = {g: HumanCreator(cfg, world.sigs.sig_true, g)
                for g in range(FN.NUM_REAL_GOALS)}
    env = Environment(cfg, world.gm, np.random.default_rng(base_seed + cell_index),
                      honesty=1.0, signing_rate=0.0, creator_bank=creators,
                      foreign_sig=world.sigs.sig_foreign)
    alpha = alpha_by_provenance(cfg)
    n_features = int(cfg.cardinalities.num_features)

    # ONE artifact per (cell, seed_rep), seen by every observer in it — E2's design, and
    # load-bearing for the between-observer measure exactly as it was in E19.
    art_rng = np.random.default_rng(C.observer_seed(base_seed, cell_index, seed_rep, 9_999))
    artifact, true_goal = _make_artifact(content, signal, art_rng)

    recs = []
    for i in range(n_obs):
        rng = C.observer_rng(base_seed, cell_index, seed_rep, i)

        # THE MATCHED HETEROGENEITY. One D draw and one observation tape per observer, both
        # shared by every arm, so the arms differ in their machinery and in nothing else.
        D = build_D(cfg, rng)
        tape_rng = np.random.default_rng(
            C.observer_seed(base_seed, cell_index, seed_rep, 50_000 + i))
        tape = BL.ObservationTape(env, artifact, tape_rng, n_timesteps)
        clf_rng = np.random.default_rng(
            C.observer_seed(base_seed, cell_index, seed_rep, 70_000 + i))

        results: dict[str, BL.BaselineResult | object] = {}

        # ---- arm A: the full active-inference observer ----------------------------------
        agent = make_v4_observer(world, np.random.default_rng(
            C.observer_seed(base_seed, cell_index, seed_rep, 90_000 + i)))
        agent.D = D
        res_a = rollout_observer(agent, artifact, BL.TapedEnvironment(tape), cfg, rng,
                                 n_timesteps=n_timesteps, force_deep_k=forced_k,
                                 initial_glance=True)
        results["A_active_inference"] = BL.BaselineResult(
            final_goal_posterior=np.asarray(res_a.final_goal_posterior, dtype=float),
            attention=np.asarray(res_a.attention),
            engaged_fraction=float(np.mean(np.asarray(res_a.attention)[forced_k:] == K.DEEP)))

        # ---- arm B: same inference, engagement policy REMOVED (always DEEP) --------------
        agent_b = make_v4_observer(world, np.random.default_rng(
            C.observer_seed(base_seed, cell_index, seed_rep, 90_000 + i)))
        agent_b.D = D
        res_b = rollout_observer(agent_b, artifact, BL.TapedEnvironment(tape), cfg, rng,
                                 n_timesteps=n_timesteps, force_deep_k=n_timesteps,
                                 initial_glance=True)
        results["B_bayesian_always_deep"] = BL.BaselineResult(
            final_goal_posterior=np.asarray(res_b.final_goal_posterior, dtype=float),
            attention=np.asarray(res_b.attention), engaged_fraction=1.0)

        # ---- arm C: the label-truster ----------------------------------------------------
        results["C_label_truster"] = BL.run_label_truster(
            cfg, tape, artifact, D, alpha, n_timesteps, forced_k)

        # ---- arm D: the effort heuristic, over its whole threshold grid -------------------
        for thr in D_THRESHOLD_GRID:
            results[f"D_effort_heuristic@{thr}"] = BL.run_effort_heuristic(
                cfg, tape, artifact, D, n_features, n_timesteps, forced_k, threshold=thr)

        # ---- arm E: the no-ToM classifier, over its whole confidence grid -----------------
        clf = BL.NoToMClassifier(cfg, env, FN.NUM_REAL_GOALS, n_features, clf_rng,
                                 n_train=n_train)
        for conf in E_CONFIDENCE_GRID:
            results[f"E_no_tom_classifier@{conf}"] = BL.run_no_tom_classifier(
                cfg, clf, tape, artifact, D, n_timesteps, forced_k, stop_confidence=conf)

        for arm, r in results.items():
            post = np.asarray(r.final_goal_posterior, dtype=float)
            recs.append({
                "arm": arm, "content": content, "declared_signal": sig_name,
                "seed_rep": seed_rep, "observer": i, "true_goal": true_goal,
                "modal_goal": int(np.argmax(post)),
                "correct": int(int(np.argmax(post)) == true_goal) if true_goal >= 0 else -1,
                "within_entropy": float(metrics.within_observer_entropy(post)),
                "engaged_fraction": float(r.engaged_fraction),
                "posterior": post.tolist(),
            })
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    # PRE-REGISTRATION FIRST, and V4's stays binding alongside it.
    prereg_path = res_dir / "v4_5_preregistration.json"
    P45.write_preregistration_v4_5(cfg, prereg_path)
    P45.assert_prereg_locked_v4_5(prereg_path)

    base_seed = int(cfg.run.base_seed if seed is None else seed)
    n_obs = int(cfg.get("experiments.e21.n_observers", 200))
    n_seeds = int(cfg.get("experiments.e21.n_seeds", 20))
    forced_k = int(cfg.get("experiments.e21.forced_deep_k", 10))
    n_timesteps = int(cfg.get("experiments.e21.n_timesteps", 24))
    n_train = int(cfg.get("experiments.e21.classifier_train_n", 200))

    payloads, ci = [], 0
    for content in CONTENT:
        for sig_name, signal in SIGNALS:
            for s in range(n_seeds):
                payloads.append((cfg.raw, ci, content, sig_name, signal, s, base_seed,
                                 n_obs, n_timesteps, forced_k, n_train))
            ci += 1
    recs = C.run_parallel(payloads, _e21_worker, workers)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "posterior"} for r in recs])
    df.to_csv(res_dir / "e21_points.csv", index=False)

    stats = cell_stats(recs)
    stats.to_csv(res_dir / "e21_cell_stats.csv", index=False)

    verdict = build_verdict(stats)
    (res_dir / "e21_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    if make_fig:
        make_figure(stats, verdict, fig_dir / "e21_model_comparison.png")
    return stats


def cell_stats(recs: list[dict]) -> pd.DataFrame:
    """Per-(arm, content, signal) means, with between-observer disagreement computed per seed
    and averaged — the same construction E2 and E19 use, so the numbers are comparable to
    both."""
    df = pd.DataFrame(recs)
    rows = []
    for (arm, content, sig), sub in df.groupby(["arm", "content", "declared_signal"]):
        betweens = []
        for _, g in sub.groupby("seed_rep"):
            betweens.append(metrics.between_observer_entropy(
                [np.asarray(p, dtype=float) for p in g["posterior"]]))
        rows.append({
            "arm": arm, "content": content, "declared_signal": sig, "n": int(len(sub)),
            "within_observer": float(sub.within_entropy.mean()),
            "within_observer_sd": float(sub.within_entropy.std()),
            "between_observer": float(np.mean(betweens)),
            "between_observer_sd": float(np.std(betweens)),
            "engaged_fraction": float(sub.engaged_fraction.mean()),
            "engaged_fraction_sd": float(sub.engaged_fraction.std()),
            "accuracy": (float(sub[sub.correct >= 0].correct.mean())
                         if (sub.correct >= 0).any() else float("nan")),
        })
    return pd.DataFrame(rows).sort_values(
        ["content", "declared_signal", "arm"]).reset_index(drop=True)


def _get(stats: pd.DataFrame, arm: str, content: str, sig: str, col: str) -> float:
    sub = stats[(stats.arm == arm) & (stats.content == content)
                & (stats.declared_signal == sig)]
    return float(sub[col].iloc[0]) if len(sub) else float("nan")


def _base_arm(arm: str) -> str:
    return arm.split("@", 1)[0]


def build_verdict(stats: pd.DataFrame) -> dict:
    """Score every arm on both signatures, then collapse the swept arms to their best case.

    "Best case" is the point of the sweep: arms D and E each have one parameter with no
    principled value, so crediting them only at some arbitrary setting would be a way of
    failing them by choice of constant. An arm reproduces a signature if ANY setting in its
    grid does, and the grid is reported so that a reader can see whether the crediting
    setting was a knife edge or the whole range.
    """
    arms = sorted(stats.arm.unique())

    # The dissociation is measured where E2 measured it: synthetic content under a HUMAN
    # claim, with the truthfully-labelled cell as the secondary label-induction control.
    diss_by_arm_and_content: dict[str, dict[str, dict]] = {}
    for content in ("goal_empty", "goal_foreign"):
        for arm in arms:
            d = P45.reproduces_dissociation(
                within=_get(stats, arm, content, "SIG_CREATOR", "within_observer"),
                between=_get(stats, arm, content, "SIG_CREATOR", "between_observer"),
                control_within=_get(stats, arm, content, "SIG_GHOST", "within_observer"))
            diss_by_arm_and_content.setdefault(content, {})[arm] = d

    # The engagement signature is measured unsigned, as E19 measured it.
    eng_by_arm = {}
    for arm in arms:
        eng_by_arm[arm] = P45.reproduces_foreign_engagement(
            engaged=_get(stats, arm, "goal_foreign", "UNSIGNED", "engaged_fraction"),
            final_entropy=_get(stats, arm, "goal_foreign", "UNSIGNED", "within_observer"),
            control_engaged=_get(stats, arm, "human_directed", "UNSIGNED", "engaged_fraction"),
            control_entropy=_get(stats, arm, "human_directed", "UNSIGNED", "within_observer"))

    def collapse(per_arm: dict) -> tuple[dict, dict]:
        """Reduce swept variants to one entry per base arm, keeping the best case."""
        best: dict[str, dict] = {}
        grid: dict[str, dict] = {}
        for arm, d in per_arm.items():
            base = _base_arm(arm)
            grid.setdefault(base, {})[arm] = d
            cur = best.get(base)
            if cur is None or (d["reproduces"] and not cur["reproduces"]):
                best[base] = dict(d, crediting_variant=arm)
        return best, grid

    diss_best, diss_grid = collapse(diss_by_arm_and_content["goal_empty"])
    diss_best_f, diss_grid_f = collapse(diss_by_arm_and_content["goal_foreign"])
    eng_best, eng_grid = collapse(eng_by_arm)

    # The verdict is driven by the stimulus E2 actually ran on, with the foreign-content
    # result reported beside it rather than folded in.
    verdict = P45.e21_verdict(diss_best, eng_best)
    verdict["experiment"] = "E21 — is the active-inference machinery necessary?"
    verdict["stage"] = "V4 stage 2 / V4.5 priority 2"
    verdict["dissociation_on_goal_empty_content"] = diss_best
    verdict["dissociation_on_goal_foreign_content"] = diss_best_f
    verdict["parameter_grids"] = {
        "dissociation_goal_empty": diss_grid,
        "dissociation_goal_foreign": diss_grid_f,
        "foreign_engagement": eng_grid,
    }
    verdict["label_induction_secondary"] = {
        arm: {"goal_empty": diss_best[arm]["label_induced"],
              "goal_foreign": diss_best_f[arm]["label_induced"]}
        for arm in diss_best}
    verdict["cell_table"] = stats.to_dict(orient="records")
    return verdict


def make_figure(stats: pd.DataFrame, verdict: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..figures import set_style
    set_style()

    label = {"A_active_inference": "full model", "B_bayesian_always_deep": "B: always look",
             "C_label_truster": "C: trust the label", "D_effort_heuristic": "D: effort rule",
             "E_no_tom_classifier": "E: classifier"}
    base_arms = list(P45.E21_ARMS)

    def best_variant(base: str, key: str) -> str:
        grid = verdict["parameter_grids"][key].get(base, {})
        for arm, d in grid.items():
            if d["reproduces"]:
                return arm
        return next(iter(grid), base)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    x = np.arange(len(base_arms))

    ax = axes[0]
    for j, content in enumerate(("goal_empty", "goal_foreign")):
        key = f"dissociation_{content}"
        w = [_get(stats, best_variant(b, key), content, "SIG_CREATOR", "within_observer")
             for b in base_arms]
        ax.bar(x + (j - 0.5) * 0.38, w, 0.38, label=content.replace("_", "-"))
    ax.axhline(P45.DISSOC_CONFIDENT_ENTROPY, color="k", ls=":", lw=1,
               label="counts as 'certain'")
    ax.set_xticks(x, [label[b] for b in base_arms], fontsize=7, rotation=20)
    ax.set(ylabel="how certain each reader is (nats, lower = surer)",
           title="Certainty on machine work\npassed off as human")
    ax.legend(fontsize=7)

    ax = axes[1]
    for j, content in enumerate(("goal_empty", "goal_foreign")):
        key = f"dissociation_{content}"
        b_ = [_get(stats, best_variant(b, key), content, "SIG_CREATOR", "between_observer")
              for b in base_arms]
        ax.bar(x + (j - 0.5) * 0.38, b_, 0.38, label=content.replace("_", "-"))
    ax.axhline(P45.DISSOC_DISAGREE_ENTROPY, color="k", ls=":", lw=1,
               label="counts as 'no two agree'")
    ax.set_xticks(x, [label[b] for b in base_arms], fontsize=7, rotation=20)
    ax.set(ylabel="disagreement between readers (nats)",
           title="Do readers disagree with each other\nabout what it was for?")
    ax.legend(fontsize=7)

    ax = axes[2]
    for j, (content, sig) in enumerate((("human_directed", "UNSIGNED"),
                                        ("goal_foreign", "UNSIGNED"))):
        e = [_get(stats, best_variant(b, "foreign_engagement"), content, sig,
                  "engaged_fraction") for b in base_arms]
        ax.bar(x + (j - 0.5) * 0.38, e, 0.38,
               label="human work" if content == "human_directed" else "machine work")
    ax.axhline(P45.FOREIGN_ENGAGED_FLOOR, color="k", ls=":", lw=1, label="counts as 'kept at it'")
    ax.set_xticks(x, [label[b] for b in base_arms], fontsize=7, rotation=20)
    ax.set(ylim=(0, 1.1), ylabel="share of free time spent looking closely",
           title="Does the reader keep paying attention\nto machine work and get nowhere?")
    ax.legend(fontsize=7)

    fig.suptitle(f"E21 — {verdict['verdict'].replace('_', ' ').lower()}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E21 — is the active-inference machinery necessary?")
    args = ap.parse_args()
    cfg = load_v4_config(quick=args.quick, include_explore=False)
    if args.seed is not None:
        cfg.set("run.base_seed", int(args.seed))
    out = Path(args.out) if args.out else None
    stats = run(cfg, out_dir=out, workers=args.workers or C.default_workers())
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
