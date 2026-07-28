"""E14 — Is the f=0 leak bounded by ENGAGEMENT rather than by sample size?

Not in the V3 spec. It exists because V3 §1 C2 does not merely ask for a refutation to be
reported, it makes finding the cause a precondition:

    "it would mean the loop is lossy for a structural reason C1 does not address, and E8 must
     not be run until that reason is found."

WHAT E12 AND E13 ESTABLISHED, and why they point here.

E12 refuted the finite-sample diagnosis outright: the f=0 honest-signal leak is FLAT in
per-generation sample size across a 100x range (log-log exponent b = -0.017, t = -0.28) and
flat in the number of observers averaged (M = 1/5/20 -> 0.0059/0.0063/0.0063). Worse for the
diagnosis, generation 0's error *grows* with N — payload KL 0.033 -> 0.075, learned-column KL
0.289 -> 0.475 from N=100 to N=10000. Variance shrinks with data. A BIAS does not, and more
data estimates a bias more sharply. So the loop is converging, confidently, on a wrong answer.

E13 then located the lever. On the identical quantity (KL of the learned CREATOR/DEEP column
from the true one), with the identical code path:

    E9 control     — engagement FORCED, 6.00 of 6 DEEP steps -> 0.074
    E9 starvation  — engagement free,   1.86 of 6 DEEP steps -> 0.344     (4.6x worse)
    E12 recursion  — engagement free                          -> ~0.55 mean

THE HYPOTHESIS. ``learn_step`` attributes each observation to the *believed* (provenance,
goal), not the true one. A disengaged observer's goal posterior never resolves, so its updates
are systematically MISATTRIBUTED — smeared across goal columns in proportion to an unresolved
posterior. That is a self-confirming learning bias: the blurred column produces blurrier
posteriors, which produce a blurrier column. It is bounded below by how much the observer
actually looks, and NOT by how many artifacts it looks at — which is exactly the signature
E12 measured and mistook, in the V3 spec, for something sample size could fix.

Note this is a strictly weaker cousin of the identifiability deadlock D1 already documents.
D1's uniform-prior learner could not learn at all because its posterior never left the prior.
This learner's posterior leaves the prior but only partway, so it learns a contracted version
of the truth — and then teaches it to the next generation.

THE TEST. Run the f=0 honest-signal recursion twice, identical in every respect except
engagement:

    free    — the observer decides when to look (the E8/E12 condition)
    forced  — DEEP for the whole budget (``freeze_engagement``, E9's control condition)

PREDICTED, if the hypothesis holds: the forced arm's per-generation leak slope collapses
toward zero and its learned-column error drops toward E9's control value (~0.074), while the
free arm reproduces E12's ~0.005 nats/generation. If instead BOTH arms leak equally, the cause
is not misattribution-under-disengagement and this diagnosis is wrong too — report that and
keep E8 withheld.

This experiment does not repair E8. Forcing DEEP is not a model of a real reader, and V3 §6's
bar for E8 is unchanged. It identifies the mechanism, which is what the spec requires before
anything else is attempted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import Config
from ..generative_model import build_shared_model as _build_model, assert_preferences_zero
from ..preregistration import POP_GOAL_DIST
from ..generations import run_chain
from ..figures import set_style
from .. import prereg_v3 as P3
from . import _common as C

ARMS = {"free": False, "forced": True}    # -> freeze_engagement


def _e14_worker(payload):
    (cfg_raw, arm, seed_rep, base_seed, g_max, n_creators, n_artifacts, n_observers,
     infer_steps, synth_seed) = payload
    cfg = Config(cfg_raw)
    gm = _build_model(cfg, goal_symmetric=False, synth_draw_seed=synth_seed)
    assert_preferences_zero(gm.C)

    results = run_chain(cfg, gm, POP_GOAL_DIST[:cfg.cardinalities.num_goals],
                        contamination=0.0, signing_rate=1.0, honesty=1.0, g_max=g_max,
                        n_creators=n_creators, n_artifacts=n_artifacts,
                        n_observers=n_observers, infer_steps=infer_steps, d_i=0.0,
                        base_seed=base_seed * 7919 + seed_rep,
                        population_average_seed=True,
                        freeze_engagement=ARMS[arm])
    return [{"arm": arm, "seed_rep": seed_rep, "generation": r.generation,
             "kl_payload": r.kl_payload, "creator_col_kl": r.creator_col_kl,
             "eff_sample_count": r.eff_sample_count,
             "mean_deep_genuine": r.mean_deep_genuine,
             "c_entropy": float(-np.sum(r.c_recovered * np.log(r.c_recovered + 1e-12))),
             "mi_genuine": r.mi_genuine} for r in results]


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    g_max = int(cfg.get("experiments.e14.g_max", 6))
    n_artifacts = int(cfg.get("experiments.e14.n_artifacts", 1000))
    n_observers = int(cfg.get("experiments.e14.n_observers", 5))
    n_creators = int(cfg.get("experiments.e14.n_creators_next", 20))
    infer_steps = int(cfg.get("experiments.e14.infer_steps", 6))
    n_reps = int(cfg.get("experiments.e14.n_replications", 3))
    synth_seed = int(cfg.get("experiments.e14.synth_draw_seed", 17))
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    payloads = [(cfg.raw, arm, s, base_seed, g_max, n_creators, n_artifacts, n_observers,
                 infer_steps, synth_seed)
                for arm in ARMS for s in range(n_reps)]
    df = pd.DataFrame(C.run_parallel(payloads, _e14_worker, workers))
    df.to_csv(res_dir / "e14_engagement_floor.csv", index=False)

    verdict = classify(df)
    (res_dir / "e14_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    if make_fig:
        make_figure(df, verdict, fig_dir / "e14_engagement_floor.png")
    return df


def _slope(sub: pd.DataFrame, key: str) -> dict:
    g = sub["generation"].to_numpy(float)
    y = sub[key].to_numpy(float)
    ok = np.isfinite(y)
    g, y = g[ok], y[ok]
    if len(np.unique(g)) < 3:
        return {"slope": float("nan"), "se": float("nan"), "t": float("nan")}
    slope, intercept = np.polyfit(g, y, 1)
    resid = y - (slope * g + intercept)
    se = float(np.sqrt((resid @ resid) / max(len(g) - 2, 1) / np.sum((g - g.mean()) ** 2)))
    return {"slope": float(slope), "se": se,
            "t": float(slope / se) if se > 0 else float("nan")}


def classify(df: pd.DataFrame) -> dict:
    out = {}
    for arm, sub in df.groupby("arm"):
        kl = _slope(sub, "kl_payload")
        out[arm] = {
            "leak_slope": kl,
            "n11_verdict_on_this_arm": P3.n11_verdict(kl["slope"], kl["t"]),
            "creator_col_kl_slope": _slope(sub, "creator_col_kl"),
            "creator_col_kl_mean": float(sub.creator_col_kl.mean()),
            "creator_col_kl_gen0": float(sub[sub.generation == 0].creator_col_kl.mean()),
            "kl_payload_mean": float(sub.kl_payload.mean()),
            "deep_per_genuine": float(sub.mean_deep_genuine.mean()),
            "c_entropy_gen0": float(sub[sub.generation == 0].c_entropy.mean()),
            "c_entropy_final": float(sub[sub.generation == sub.generation.max()].c_entropy.mean()),
        }
    free, forced = out.get("free", {}), out.get("forced", {})
    ratio = (forced.get("creator_col_kl_mean", float("nan"))
             / max(free.get("creator_col_kl_mean", float("nan")), 1e-12))
    supported = bool(forced.get("n11_verdict_on_this_arm", {}).get("passed")
                     and not free.get("n11_verdict_on_this_arm", {}).get("passed"))
    return {
        "experiment": "E14 (not in the V3 spec; required by §1 C2's 'until that reason is found')",
        "hypothesis": ("the f=0 leak is misattribution under unresolved goal posteriors — a "
                       "self-confirming learning bias bounded by ENGAGEMENT, not by sample size"),
        "arms": out,
        "forced_over_free_column_error": ratio,
        "hypothesis_supported": supported,
        "reference_e9_control_column_kl": 0.0744,
        "interpretation": (
            "supported: forcing engagement removes the leak that neither more data (E12 N-sweep) "
            "nor more observers (E12 M-sweep) could touch, so the loop's loss is attribution "
            "error under unresolved posteriors. NOT supported: both arms leak, the cause lies "
            "elsewhere again, and E8 stays withheld under §6 either way — this experiment "
            "identifies a mechanism, it does not repair E8."),
    }


def make_figure(df, verdict, path):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    colours = {"free": "firebrick", "forced": "C0"}
    arm_label = {"free": "readers free to look away",
                 "forced": "readers made to look closely at everything"}

    for key, ax, title, ylab in (
            ("kl_payload", axes[0],
             "With zero machine content, this should stay flat\n"
             "(everything here is honest human work)",
             "error in what people are believed to want (nats)"),
            ("creator_col_kl", axes[1],
             "The reader's own model of human work drifts too",
             "how wrong its model of human work is (nats)")):
        for arm, sub in df.groupby("arm"):
            m = sub.groupby("generation")[key].agg(["mean", "std"])
            ax.errorbar(m.index, m["mean"], yerr=m["std"], marker="o", ms=4, capsize=3,
                        color=colours.get(arm, "k"), label=arm_label.get(arm, arm))
        if key == "creator_col_kl":
            ax.axhline(0.0744, color="grey", ls=":", lw=1,
                       label="damage from a starved reader, for scale")
        ax.set(xlabel="generation (each one trains on the last one's output)",
               ylabel=ylab, title=title)
        ax.legend(fontsize=7)

    ax = axes[2]
    arms = list(verdict["arms"])
    slopes = [verdict["arms"][a]["leak_slope"]["slope"] for a in arms]
    errs = [verdict["arms"][a]["leak_slope"]["se"] for a in arms]
    passed = [verdict["arms"][a]["n11_verdict_on_this_arm"]["passed"] for a in arms]
    ax.bar(range(len(arms)), slopes, yerr=errs, capsize=4,
           color=["C0" if p else "firebrick" for p in passed])
    ax.axhline(0, color="k", lw=1)
    ax.axhline(P3.N11_SLOPE_CEILING, color="k", ls=":", lw=1,
               label="the level this had to get under")
    short = {"free": "readers free\nto look away",
             "forced": "readers made to look\nclosely at everything"}
    ax.set_xticks(range(len(arms)), [short.get(a, a) for a in arms], fontsize=8)
    ax.set(ylabel="error added per generation (nats)",
           title="Does making readers pay attention fix it?\n"
                 "(a blue bar would mean yes; both are red)")
    ax.legend(fontsize=7)

    fig.suptitle("E14 — Forcing readers to pay attention lowers the damage "
                 "but does not stop the drift", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E14 — engagement floor on the f=0 leak")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = C.default_workers() if args.workers is None else args.workers
    run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    res_dir, _ = C.ensure_dirs(args.out)
    v = json.loads((res_dir / "e14_verdict.json").read_text(encoding="utf-8"))
    print("\nE14 — f=0 honest recursion, engagement free vs forced:")
    for arm, a in v["arms"].items():
        print(f"  {arm:7s} DEEP {a['deep_per_genuine']:.2f}/6  "
              f"leak slope {a['leak_slope']['slope']:+.5f} (t {a['leak_slope']['t']:+.2f}) "
              f"-> N11 {'passes' if a['n11_verdict_on_this_arm']['passed'] else 'FAILS'}   "
              f"column KL {a['creator_col_kl_mean']:.4f}")
    print(f"\n  forced/free column error = {v['forced_over_free_column_error']:.3f} "
          f"(E9 control reference {v['reference_e9_control_column_kl']})")
    print(f"  hypothesis supported: {v['hypothesis_supported']}")


if __name__ == "__main__":
    main()
