"""E13 — The freeze and the leak on one axis (V3 spec §1 C4). STAGE 2.

V2 left one effect unexplained: under starvation the learned model did not blur toward
flatness as predicted, it **ossified** — shape error frozen at ~0.24 and contamination-
independent across f = 0 / 0.3 / 0.6. V3 asks whether that freeze and E8's leak are the same
finite-sample effect.

-----------------------------------------------------------------------------------------
DECISION D7 — ONE QUANTITY, ONE AXIS, AND OUTCOME 2 EXPECTED.

V3 §1 C4 asks whether the two "fall on the same curve". As written that is not well defined:
V2 reports the E9 freeze in FEATURE space (per-column KL of the learned A, averaged over the
human tiers) and the E8 leak in GOAL space (payload KL). There is no mapping between them,
and no stated threshold for what "the same curve" would mean.

So both halves of E13 report the IDENTICAL quantity, through the identical code path:

        y = mean over goals of  KL( learned CREATOR/DEEP column || true CREATOR/DEEP column )
        x = effective Dirichlet sample count  (concentration mass - prior)

and the classification threshold is pre-registered (``prereg_v3.E13_TOLERANCE_FACTOR``)
before the run rather than chosen once the scatter is visible.

AND THE FRAMING IS SOFTENED, DELIBERATELY. The V3 §0 hypothesis — that the freeze and the
leak are *the same effect* — is not what this experiment is built to confirm. It tests
whether they lie on **a shared finite-sample axis, on which they may sit at opposite ends**.
The reason is visible in V2's own numbers and should be said before the run, not after:

    E9 starvation: engagement collapses to 1.87 DEEP steps of a possible 6, so genuine
        content stops producing updates and the column never departs from its (informative,
        D1-seeded) prior.  ->  LOW effective sample count, error is PRIOR-ANCHORING.
    E8 honest arm: the signal concentrates updates on the CREATOR column, which sharpens
        fast around a finite-sample estimate that then compounds.
        ->  HIGH effective sample count, error is SAMPLING NOISE.

On this axis those are opposite signatures, so **outcome 2 (two distinct effects) is the
expected result and is recorded as such in the pre-registration**. That is not a failure of
the V3 redo. It is the redo establishing that the framework has TWO finite-sample failure
modes where it assumed one — which V3 §6 requires this experiment to be capable of returning,
and §5 requires to be reported in those words rather than explained away.

THE THREE OUTCOMES (V3 §1 C4), all reportable, decided by ``prereg_v3.e13_classify``:
  1. shared axis      — E9's point falls on the recursion's fitted finite-sample curve
  2. distinct effects — it does not; an open problem in the framework  [EXPECTED]
  3. freeze vanished  — the starvation arm no longer freezes at all; D4 confirmed outright
-----------------------------------------------------------------------------------------
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
from ..constants import CREATOR, DEEP
from ..generative_model import (build_shared_model as _build_model, assert_preferences_zero,
                                build_observer_model, build_D)
from ..creators import build_creator_bank
from ..environment import Environment
from ..observer import rollout_observer
from ..preregistration import POP_GOAL_DIST, allocate_creator_goals
from ..generations import run_chain
from .. import learning as L
from .. import prereg_v3 as P3
from ..figures import set_style
from . import _common as C

# Only the two arms C4 needs: the freeze itself, and the reference that says whether it froze.
ARMS = {"starvation_only": (1.0, False), "control": (1.0, True)}   # (honesty, freeze_engagement)


def _column_kl(A_learned, A_true, num_goals: int) -> float:
    """THE shared quantity (D7). Identical call in both halves of this experiment."""
    return float(np.mean([L.column_kl(A_learned, A_true, CREATOR, g) for g in range(num_goals)]))


# --------------------------------------------------------------------------- #
# Half 1 — E9's starvation arm, instrumented.
# --------------------------------------------------------------------------- #
def _e13_starve_worker(payload):
    (cfg_raw, cell_index, arm, kappa, f, seed_rep, base_seed, n_creators, n_artifacts,
     n_observers, infer_steps, synth_seed, trace_every) = payload
    cfg = Config(cfg_raw)
    num_goals = int(cfg.cardinalities.num_goals)
    honesty, freeze = ARMS[arm]

    gm = _build_model(cfg, kappa=kappa, goal_symmetric=False, synth_draw_seed=synth_seed)
    assert_preferences_zero(gm.C)                     # N7
    bank = build_creator_bank(cfg, gm)
    A_true = np.asarray(gm.A[0])

    pop_rng = np.random.default_rng(base_seed * 13 + seed_rep)
    creator_goals = allocate_creator_goals(n_creators, POP_GOAL_DIST[:num_goals], pop_rng)
    world = np.random.default_rng(base_seed * 31 + seed_rep * 7)
    env = Environment(cfg, gm, rng_world=world, honesty=honesty, signing_rate=1.0,
                      creator_bank=bank)
    corpus = env.draw_corpus(n_artifacts, contamination=f, creator_goals=creator_goals,
                             rng=world)

    strength = float(cfg.get("learning.prior_strength", 1.0))
    recs = []
    for i in range(n_observers):
        r = C.observer_rng(base_seed, cell_index, seed_rep, i)
        om = build_observer_model(gm, cfg, 0.0)
        agent = L.make_learner_agent(om, build_D(cfg, r), cfg, kappa=kappa)
        prev = np.asarray(agent.A[0])[:, CREATOR, :, DEEP].copy()
        n_deep = n_gen = 0
        for j, art in enumerate(corpus):
            rr = np.random.default_rng(base_seed * 90001 + seed_rep * 337 + i * 7919 + j)
            res = rollout_observer(agent, art, env, cfg, rr, infer_steps,
                                   force_deep_k=(infer_steps if freeze else 0),
                                   kappa=kappa, early_stop=False, learn=True)
            if art.provenance == CREATOR:
                n_gen += 1
                n_deep += res.cum_deep
            if (j + 1) % trace_every == 0:
                col = np.asarray(agent.A[0])[:, CREATOR, :, DEEP]
                mass = np.asarray(agent.pA[0])[:, CREATOR, :, DEEP].sum(axis=0)
                recs.append({
                    "half": "starvation", "arm": arm, "contamination": f,
                    "seed_rep": seed_rep, "observer": i, "artifact_index": j + 1,
                    "eff_sample_count": float(np.mean(mass - strength)),
                    "creator_col_kl": _column_kl(agent.A[0], A_true, num_goals),
                    "col_change_rate": float(np.abs(col - prev).sum() / trace_every),
                    "deep_per_genuine": n_deep / max(n_gen, 1)})
                prev = col.copy()
    return recs


# --------------------------------------------------------------------------- #
# Half 2 — the f=0 honest-signal recursion, instrumented on the same axis.
# --------------------------------------------------------------------------- #
def _e13_recursion_worker(payload):
    (cfg_raw, seed_rep, base_seed, g_max, n_creators, n_artifacts, n_observers,
     infer_steps, synth_seed, trace_every) = payload
    cfg = Config(cfg_raw)
    gm = _build_model(cfg, goal_symmetric=False, synth_draw_seed=synth_seed)
    assert_preferences_zero(gm.C)

    sink: list = []
    run_chain(cfg, gm, POP_GOAL_DIST[:cfg.cardinalities.num_goals],
              contamination=0.0, signing_rate=1.0, honesty=1.0, g_max=g_max,
              n_creators=n_creators, n_artifacts=n_artifacts, n_observers=n_observers,
              infer_steps=infer_steps, d_i=0.0, base_seed=base_seed * 7919 + seed_rep,
              population_average_seed=True, trace_sink=sink, trace_every=trace_every)
    for row in sink:
        row.update({"half": "recursion", "arm": "honest_f0", "contamination": 0.0,
                    "seed_rep": seed_rep})
    return sink


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    e = cfg.experiments.e13
    n_artifacts, n_observers = int(e.n_artifacts), int(e.n_observers)
    n_creators = int(cfg.get("experiments.e13.n_creators", 50))
    infer_steps = int(cfg.get("experiments.e13.infer_steps", 6))
    n_reps = int(cfg.get("experiments.e13.n_replications", 3))
    f_levels = list(cfg.get("experiments.e13.contamination_levels", [0.0, 0.3, 0.6]))
    kappa = float(cfg.get("experiments.e13.kappa", 0.9))
    synth_seed = int(cfg.get("experiments.e13.synth_draw_seed", 17))
    trace_every = int(cfg.get("experiments.e13.trace_every", 25))
    rec_n = int(cfg.get("experiments.e13.recursion_n_artifacts", 1000))
    rec_g = int(cfg.get("experiments.e13.recursion_g_max", 4))
    rec_obs = int(cfg.get("experiments.e13.recursion_n_observers", 5))
    rec_cre = int(cfg.get("experiments.e13.recursion_n_creators", 20))
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    starve_payloads, ci = [], 0
    for arm in ARMS:
        for f in f_levels:
            for s in range(n_reps):
                starve_payloads.append((cfg.raw, ci, arm, kappa, f, s, base_seed, n_creators,
                                        n_artifacts, n_observers, infer_steps, synth_seed,
                                        trace_every))
            ci += 1
    rec_payloads = [(cfg.raw, s, base_seed, rec_g, rec_cre, rec_n, rec_obs, infer_steps,
                     synth_seed, trace_every) for s in range(n_reps)]

    recs = C.run_parallel(starve_payloads, _e13_starve_worker, workers)
    recs += C.run_parallel(rec_payloads, _e13_recursion_worker, workers)
    df = pd.DataFrame(recs)
    df.to_csv(res_dir / "e13_freeze_leak_signature.csv", index=False)

    verdict = classify(df)
    (res_dir / "e13_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    if make_fig:
        make_e13_figure(df, verdict, fig_dir / "e13_shared_signature.png")
    return df


def _endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """The final traced panel per (half, arm, f, rep, observer) — where each run ended up."""
    keys = ["half", "arm", "contamination", "seed_rep", "observer"]
    return df.sort_values("artifact_index").groupby(keys, as_index=False).last()


def classify(df: pd.DataFrame) -> dict:
    """Fit the recursion's finite-sample curve and place the freeze against it (D7)."""
    rec = df[df.half == "recursion"]
    ends = _endpoints(df)
    starve = ends[(ends.half == "starvation") & (ends.arm == "starvation_only")]
    control = ends[(ends.half == "starvation") & (ends.arm == "control")]

    # y = c * x**b, fitted in log-log on the recursion's traced panels.
    x = rec["eff_sample_count"].to_numpy(float)
    y = rec["creator_col_kl"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() >= 3:
        b, log_c = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
        c = float(np.exp(log_c))
    else:
        b, c = float("nan"), float("nan")

    freeze_n_eff = float(starve["eff_sample_count"].mean()) if len(starve) else float("nan")
    freeze_kl = float(starve["creator_col_kl"].mean()) if len(starve) else float("nan")
    control_kl = float(control["creator_col_kl"].mean()) if len(control) else float("nan")

    outcome = P3.e13_classify(c, b, freeze_n_eff, freeze_kl, control_kl)

    starve_rate = float(starve["col_change_rate"].mean()) if len(starve) else float("nan")
    control_rate = float(control["col_change_rate"].mean()) if len(control) else float("nan")
    rec_rate = float(rec.sort_values("artifact_index").groupby(
        ["seed_rep", "generation", "observer"], as_index=False).last()
        ["col_change_rate"].mean()) if len(rec) else float("nan")

    return {
        "experiment": "E13",
        "spec": "V3 §1 C4, §5 (report which outcome obtained)",
        "shared_quantity": "mean over goals of KL(learned CREATOR/DEEP column || true), nats",
        "shared_axis": "effective Dirichlet sample count (concentration mass - prior)",
        "recursion_fit": {"c": c, "b": b, "n_points": int(ok.sum()),
                          "mean_eff_sample_count": float(np.mean(x[ok])) if ok.sum() else None,
                          "mean_creator_col_kl": float(np.mean(y[ok])) if ok.sum() else None},
        "freeze": {"eff_sample_count": freeze_n_eff, "creator_col_kl": freeze_kl,
                   "col_change_rate": starve_rate,
                   "deep_per_genuine": float(starve["deep_per_genuine"].mean()) if len(starve) else None,
                   "contamination_independence": (
                       starve.groupby("contamination").creator_col_kl.mean().round(4).to_dict()
                       if len(starve) else {})},
        "control": {"creator_col_kl": control_kl, "col_change_rate": control_rate,
                    "deep_per_genuine": float(control["deep_per_genuine"].mean()) if len(control) else None},
        "recursion_col_change_rate": rec_rate,
        "classification": outcome,
        "reporting_requirement": (
            "V3 §5: state explicitly which of the three C4 outcomes obtained. If they are "
            "different effects, flag the second one as an open problem in the framework, in "
            "those words."),
    }


def make_e13_figure(df, verdict, path):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    rec = df[df.half == "recursion"]
    ends = _endpoints(df)
    starve = ends[(ends.half == "starvation") & (ends.arm == "starvation_only")]
    control = ends[(ends.half == "starvation") & (ends.arm == "control")]

    ax = axes[0]
    ax.scatter(rec.eff_sample_count, rec.creator_col_kl, s=10, alpha=0.35, color="C0",
               label="E8/E12 honest f=0 recursion")
    fit = verdict["recursion_fit"]
    if np.isfinite(fit["b"]) and len(rec):
        xs = np.linspace(max(rec.eff_sample_count.min(), 1e-6), rec.eff_sample_count.max(), 50)
        ax.plot(xs, fit["c"] * xs ** fit["b"], color="C0", lw=2,
                label=f"fit: {fit['c']:.2f}·n^{fit['b']:.2f}")
    if len(starve):
        ax.scatter(starve.eff_sample_count, starve.creator_col_kl, s=45, marker="D",
                   color="firebrick", label="E9 starvation (the freeze)", zorder=5)
    if len(control):
        ax.scatter(control.eff_sample_count, control.creator_col_kl, s=45, marker="s",
                   color="grey", label="E9 control", zorder=5)
    ax.set(xscale="log", yscale="log", xlabel="effective Dirichlet sample count",
           ylabel="KL(learned CREATOR column || true)  [nats]",
           title=f"The shared axis\noutcome {verdict['classification']['outcome']}: "
                 f"{verdict['classification']['label']}")
    ax.legend(fontsize=7)

    ax = axes[1]
    for (half, arm), sub in df.groupby(["half", "arm"]):
        m = sub.groupby("artifact_index").col_change_rate.mean().sort_index()
        ax.plot(m.index, m.values, marker="o", ms=3, label=f"{half}/{arm}")
    ax.set(yscale="log", xlabel="artifacts observed",
           ylabel="|Δ learned column| per observation",
           title="Does the rate of change collapse?\n(C4's premature-convergence signature)")
    ax.legend(fontsize=7)

    ax = axes[2]
    if len(starve):
        s = starve.groupby("contamination").creator_col_kl.agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="D", capsize=3,
                    color="firebrick", label="starvation")
    if len(control):
        cc = control.groupby("contamination").creator_col_kl.agg(["mean", "std"])
        ax.errorbar(cc.index, cc["mean"], yerr=cc["std"], marker="s", capsize=3,
                    color="grey", label="control")
    ax.set(xlabel="contamination f", ylabel="KL(learned CREATOR column || true)  [nats]",
           title="V2's freeze signature: flat in f\n(is it still there under C1?)")
    ax.legend(fontsize=7)

    fig.suptitle("E13 — The freeze and the leak on one finite-sample axis", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E13 — freeze/leak shared-signature diagnostic")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = C.default_workers() if args.workers is None else args.workers
    run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    res_dir, _ = C.ensure_dirs(args.out)
    v = json.loads((res_dir / "e13_verdict.json").read_text(encoding="utf-8"))

    print("\nE13 — the shared quantity, KL(learned CREATOR column || true):")
    print(f"  recursion (honest f=0): fit  kl = {v['recursion_fit']['c']:.3f} * "
          f"n_eff ^ {v['recursion_fit']['b']:.3f}")
    print(f"  freeze  (E9 starvation): kl = {v['freeze']['creator_col_kl']:.4f} at "
          f"n_eff = {v['freeze']['eff_sample_count']:.1f}, "
          f"DEEP/genuine = {v['freeze']['deep_per_genuine']}")
    print(f"  control (E9, forced DEEP): kl = {v['control']['creator_col_kl']:.4f}")
    print(f"  freeze vs f: {v['freeze']['contamination_independence']}")
    cl = v["classification"]
    print(f"\n  C4 OUTCOME {cl['outcome']} — {cl['label']}")
    print(f"    {cl['statement']}")


if __name__ == "__main__":
    main()
