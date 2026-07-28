"""E18 — Does fixing the ESTIMATOR remove the generational contraction?

The V3 round ended with the contraction located but not repaired: ``learn_step`` commits
Dirichlet counts at every timestep using the posterior held at that moment, so the first DEEP
observation of every artifact is filed before the goal has resolved. A fixed fraction of all
evidence, about ``1/infer_steps``, is misattributed, and the learned column comes out flatter
than the truth by a ratio that does not shrink with corpus size, does not shrink with observer
averaging, and is only weakly reduced by engagement.

-----------------------------------------------------------------------------------------
WHY THIS EXPERIMENT EXISTS RATHER THAN AN ``infer_steps = 24`` RE-RUN.

Raising the inference budget divides the bias; it does not remove it. Measured, one
generation, as the contraction ratio r = KL(learned || uniform) / KL(true || uniform):

    infer_steps        2       4       6      12      24
    online   1 - r  0.147   0.140   0.078   0.035   0.014
    deferred 1 - r  0.056  -0.001  -0.006   0.005   0.002

The online estimator's bias tracks 1/steps and never reaches zero. A result that clears its
gate only at a hand-picked budget invites the obvious question — why 24 and not 12 — whose
honest answer would be "because that is where it dropped below our threshold". That is
choosing the operating point to pass one's own test, which is the failure decision D1 was
written to prevent, and it would make E8 a tuned number rather than a result.

The deferred estimator is flat in the budget from four steps on (below that the E-step
genuinely has not converged, and the residue is real rather than an artifact). Flat is the
property that matters: it means the bias is ABSENT rather than DIVIDED, so nothing about the
result depends on where the budget was set.

THE FIX, and it adds no parameter. Inference over one artifact is an E-step; the Dirichlet
update is an M-step; committing counts mid-E-step is the error. Deferred commitment buffers
the artifact's observations, lets inference finish, and only then deposits each observed
feature under the resolved posterior. See ``learning.learn_deferred``.
-----------------------------------------------------------------------------------------

DESIGN. The f = 0 honest-signal chain — the condition that failed N11 in V2 and again in V3 —
run at the SAME ``infer_steps`` V3 used, with the estimator as the only variable:

    online    the V1/V2/V3 estimator (the control; must reproduce V3's leak)
    deferred  commitment deferred to the end of each artifact's inference

PREDICTED, if early-step misattribution is the whole of the generational contraction: the
deferred arm's leak slope falls below the pre-registered N11 ceiling of 0.001 nats/generation
AND its C_recovered entropy stops climbing toward uniform. The online arm must still leak, or
the comparison is uninformative.

FALSIFIED if the deferred arm still contracts at a similar rate. Then a second contributor
exists, the single-generation measurement does not explain the chain, and E8 stays withheld —
the estimator fix would be necessary but not sufficient.

This experiment is a candidate route to a REPORTABLE E8. It is not itself E8, and passing here
does not license reporting E8: that requires re-running E12's gate and N11 at full scale under
the fixed estimator.
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

MODES = ("online", "deferred")
UNIFORM_H = float(np.log(4))
TRUE_H = 1.2799          # H(C_true) for [0.40, 0.30, 0.20, 0.10]


def _e18_worker(payload):
    (cfg_raw, mode, seed_rep, base_seed, g_max, n_creators, n_artifacts, n_observers,
     infer_steps, synth_seed, freeze) = payload
    cfg = Config(cfg_raw)
    gm = _build_model(cfg, goal_symmetric=False, synth_draw_seed=synth_seed)
    assert_preferences_zero(gm.C)

    results = run_chain(cfg, gm, POP_GOAL_DIST[:cfg.cardinalities.num_goals],
                        contamination=0.0, signing_rate=1.0, honesty=1.0, g_max=g_max,
                        n_creators=n_creators, n_artifacts=n_artifacts,
                        n_observers=n_observers, infer_steps=infer_steps, d_i=0.0,
                        base_seed=base_seed * 7919 + seed_rep,
                        population_average_seed=True, learn_mode=mode,
                        freeze_engagement=bool(freeze))
    return [{"learn_mode": mode, "seed_rep": seed_rep, "generation": r.generation,
             "kl_payload": r.kl_payload, "creator_col_kl": r.creator_col_kl,
             "eff_sample_count": r.eff_sample_count,
             "mean_deep_genuine": r.mean_deep_genuine,
             "c_entropy": float(-np.sum(r.c_recovered * np.log(r.c_recovered + 1e-12)))}
            for r in results]


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    g_max = int(cfg.get("experiments.e18.g_max", 6))
    n_artifacts = int(cfg.get("experiments.e18.n_artifacts", 1000))
    n_observers = int(cfg.get("experiments.e18.n_observers", 5))
    n_creators = int(cfg.get("experiments.e18.n_creators_next", 20))
    infer_steps = int(cfg.get("experiments.e18.infer_steps", 6))
    n_reps = int(cfg.get("experiments.e18.n_replications", 3))
    synth_seed = int(cfg.get("experiments.e18.synth_draw_seed", 17))
    freeze = bool(cfg.get("experiments.e18.freeze_engagement", True))
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    payloads = [(cfg.raw, mode, s, base_seed, g_max, n_creators, n_artifacts, n_observers,
                 infer_steps, synth_seed, freeze)
                for mode in MODES for s in range(n_reps)]
    df = pd.DataFrame(C.run_parallel(payloads, _e18_worker, workers))
    df["infer_steps"] = infer_steps
    df.to_csv(res_dir / "e18_deferred_estimator.csv", index=False)

    verdict = analyse(df, infer_steps)
    (res_dir / "e18_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    if make_fig:
        make_figure(df, verdict, fig_dir / "e18_deferred_estimator.png")
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


def analyse(df: pd.DataFrame, infer_steps: int) -> dict:
    arms = {}
    for mode, sub in df.groupby("learn_mode"):
        kl = _slope(sub, "kl_payload")
        first = sub[sub.generation == sub.generation.min()]
        last = sub[sub.generation == sub.generation.max()]
        h0, h1 = float(first.c_entropy.mean()), float(last.c_entropy.mean())
        arms[mode] = {
            "leak_slope": kl,
            "n11_verdict": P3.n11_verdict(kl["slope"], kl["t"]),
            "creator_col_kl_slope": _slope(sub, "creator_col_kl"),
            "creator_col_kl_gen0": float(first.creator_col_kl.mean()),
            "creator_col_kl_final": float(last.creator_col_kl.mean()),
            "kl_payload_gen0": float(first.kl_payload.mean()),
            "kl_payload_final": float(last.kl_payload.mean()),
            "c_entropy_gen0": h0, "c_entropy_final": h1,
            # Fraction of the distance from H(C_true) to H(uniform) travelled by the last
            # generation. This is the contraction, expressed so the two arms are comparable.
            "fraction_to_uniform_gen0": (h0 - TRUE_H) / (UNIFORM_H - TRUE_H),
            "fraction_to_uniform_final": (h1 - TRUE_H) / (UNIFORM_H - TRUE_H),
        }
    online, deferred = arms.get("online", {}), arms.get("deferred", {})
    fixed = bool(deferred.get("n11_verdict", {}).get("passed"))
    control_ok = bool(not online.get("n11_verdict", {}).get("passed"))
    return {
        "experiment": "E18 — deferred commitment versus the online estimator",
        "infer_steps": infer_steps,
        "note": ("run at V3's own inference budget, with the estimator as the only variable. "
                 "The deferred estimator's single-generation bias is flat in the budget, so "
                 "nothing here depends on where the budget was set."),
        "arms": arms,
        "control_arm_still_leaks": control_ok,
        "deferred_passes_n11": fixed,
        "verdict": (
            "the estimator fix removes the generational contraction at V3's own inference "
            "budget: the deferred arm satisfies the pre-registered N11 criterion while the "
            "online control still leaks. This is a candidate route to a reportable E8 — it "
            "does NOT itself license reporting E8, which requires re-running E12's gate and "
            "N11 at full scale under the fixed estimator."
            if fixed and control_ok else
            "the estimator fix does NOT remove the generational contraction. Early-step "
            "misattribution is therefore not the whole of it: a second contributor exists, the "
            "single-generation measurement does not explain the chain, and E8 stays withheld."
            if control_ok else
            "UNINFORMATIVE: the online control did not leak in this configuration, so the "
            "comparison establishes nothing. Check the configuration before reading the "
            "deferred arm."),
    }


def make_figure(df, verdict, path):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    colours = {"online": "firebrick", "deferred": "C0"}
    mode_name = {"online": "old: files evidence as it goes",
                 "deferred": "fixed: waits until it has decided"}

    for key, ax, ylab, title in (
            ("kl_payload", axes[0], "error in what people are believed to want (nats)",
             "With zero machine content, this should stay flat\n"
             "(everything here is honest human work)"),
            ("creator_col_kl", axes[1], "how wrong its model of human work is (nats)",
             "The reader's own model of human work")):
        for mode, sub in df.groupby("learn_mode"):
            m = sub.groupby("generation")[key].agg(["mean", "std"])
            ax.errorbar(m.index, m["mean"], yerr=m["std"], marker="o", ms=4, capsize=3,
                        color=colours.get(mode, "k"), label=mode_name.get(mode, mode))
        ax.set(xlabel="generation (each one trains on the last one's output)",
               ylabel=ylab, title=title)
        ax.legend(fontsize=8)

    ax = axes[2]
    for mode, sub in df.groupby("learn_mode"):
        m = sub.groupby("generation").c_entropy.mean()
        frac = (m - TRUE_H) / (UNIFORM_H - TRUE_H)
        ax.plot(m.index, frac, "-o", ms=4, color=colours.get(mode, "k"),
                label=mode_name.get(mode, mode))
    ax.axhline(0.0, color="grey", ls=":", lw=1, label="what people actually want")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="no preferences left at all")
    ax.set(xlabel="generation (each one trains on the last one's output)",
           ylabel="how far preferences have flattened out",
           title="Does the picture of what people want\nslide toward 'anything goes'?")
    ax.legend(fontsize=7)

    fig.suptitle("E18 — The drift was a bookkeeping bug in our own code, "
                 "not a fact about the world", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E18 — deferred commitment versus the online estimator")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = C.default_workers() if args.workers is None else args.workers
    run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    res_dir, _ = C.ensure_dirs(args.out)
    v = json.loads((res_dir / "e18_verdict.json").read_text(encoding="utf-8"))
    print(f"\nE18 — f=0 honest chain at infer_steps={v['infer_steps']}, estimator as the only variable:")
    for mode, a in v["arms"].items():
        s = a["leak_slope"]
        print(f"  {mode:9s} leak slope {s['slope']:+.5f} (t {s['t']:+.2f}) -> N11 "
              f"{'PASSES' if a['n11_verdict']['passed'] else 'fails'}   "
              f"column KL {a['creator_col_kl_gen0']:.4f} -> {a['creator_col_kl_final']:.4f}   "
              f"to-uniform {a['fraction_to_uniform_gen0']:.2f} -> {a['fraction_to_uniform_final']:.2f}")
    print(f"\n  {v['verdict']}")


if __name__ == "__main__":
    main()
