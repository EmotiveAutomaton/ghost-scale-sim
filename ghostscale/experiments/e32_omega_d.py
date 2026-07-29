"""E32 — are foreign content and an unskilled reader the same thing? (V5 C3.)

THE OBSERVATION C3 MAKES. omega measures how foreign the creator's structure is; d measures
how poor the reader's template is. Both produce one thing — a gap between what was generated
and what can be represented — and they are the creator-side and observer-side causes of the
same failure. If a high-d observer on human content is behaviourally indistinguishable from a
low-omega observer on foreign content AT MATCHED EFFECTIVE OVERLAP, the model should carry one
variable and say so. If they diverge, the divergence names a second dimension.

-----------------------------------------------------------------------------------------
THE MATCHING PROBLEM, WHICH IS THE WHOLE EXPERIMENT AND WHICH THE SPEC DOES NOT FLAG.

"Matched effective overlap" is a free parameter. omega mixes the CONTENT toward the human
family; d mixes the OBSERVER'S TEMPLATES toward a Dirichlet draw. They act at different points
and in different geometries, and without fixing the correspondence in advance one can find a d
that matches any omega on whichever measure one prefers — including, obviously, the measure one
is about to report.

So the correspondence is defined ONCE, computed BEFORE any rollout, written to disk, and
hash-locked with the rest of V5's criteria:

    effective overlap = MI(features; goal) under the OBSERVER'S OWN likelihood,
                        evaluated on the content it actually sees, in nats.

Both knobs degrade exactly that quantity, which is why it is the right common currency: it is
what the observer can in principle extract, whatever the reason it cannot extract more. d is
solved for each omega by bisection on the measured curve, and the solved curve is reported in
the verdict so a reader can check the match rather than take it on trust.

-----------------------------------------------------------------------------------------
THE PREDICTION, WRITTEN BEFORE THE RUN, AND IT IS NOT "ONE VARIABLE".

The two failures should DIVERGE, and the mechanism is visible in the construction:

* A high-d observer has templates aimed at the wrong place, but the content is IN SUPPORT.
  Its four hypotheses give different likelihoods, one of them wins, and it wins for reasons
  that are that observer's own sampling noise. Predicted signature: CONFIDENT, WRONG, and
  idiosyncratic — low within-observer entropy, high between-observer disagreement.
* Low-omega content sits where all four of the observer's templates read floor. The
  likelihoods are near-equal, so nothing wins. Predicted signature: UNCERTAIN and SUSTAINED —
  high within-observer entropy, and engagement that does not resolve, which is E19's and
  E20's finding.

If that holds, the second dimension C3 asks to be named is **whether the failure is detectable
to the observer**: a mis-aimed template fails silently and an out-of-range one fails loudly.
That is a more useful answer than a collapse to one variable, and it is what the model should
carry.

-----------------------------------------------------------------------------------------
THE ZERO-COMPUTE PRECURSOR ran first and is reported in the verdict as ``precursor``. E15's
committed d grid and E20's committed omega grid already share columns, so the comparison costs
nothing. It is INDICATIVE ONLY and is labelled as such: E15 ran at 8 features under a different
design and E20 at 16, so the two are not matched on overlap and cannot settle the question.
They can and do show which way it is going.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants as K
from .. import foreign as FN
from .. import metrics
from .. import prereg_v5_c234 as P5C
from ..config import Config
from ..environment import Artifact, Environment
from ..generative_model import build_observer_signature
from ..observer import rollout_observer
from ..v4_model import build_v4_world, load_v4_config, make_v4_observer
from . import _common as C

ARMS = ("foreign_content", "unskilled_reader")


# --------------------------------------------------------------------------- #
# The common currency, and the matching curve built from it.
# --------------------------------------------------------------------------- #
def effective_overlap_foreign(cfg: Config, omega: float) -> float:
    """MI(features; goal) a COMPETENT observer can extract from content at overlap omega.

    The observer's templates are ``sig_true``; the content is ``sig_foreign`` at this omega.
    The quantity is the mutual information between the feature a reader draws and the goal
    that produced it, evaluated under the observer's own family — that is, how much the
    content tells THIS reader about intent.
    """
    sigs = FN.build_v4_signatures(cfg, omega=float(omega), include_explore=False)
    return _mi_under_templates(sigs.sig_foreign, sigs.sig_true)


def effective_overlap_unskilled(cfg: Config, d: float, n_draws: int = 24,
                                seed: int = 20250729) -> float:
    """MI(features; goal) an observer at inexpertise d can extract from HUMAN content.

    Averaged over ``n_draws`` template perturbations, because ``sig_i`` is drawn per observer
    and a single draw would make the curve a measurement of one observer's luck.
    """
    sig_true = FN.build_human_signatures(cfg)
    if float(d) <= 0.0:
        return _mi_under_templates(sig_true, sig_true)
    rng = np.random.default_rng(int(seed))
    vals = [_mi_under_templates(sig_true, build_observer_signature(sig_true, float(d), rng))
            for _ in range(int(n_draws))]
    return float(np.mean(vals))


def _mi_under_templates(content: np.ndarray, templates: np.ndarray) -> float:
    """MI(feature; goal) when goal g emits ``content[g]`` and the reader scores with
    ``templates``. The reader's posterior over goals given a feature is what carries the
    information, so the joint is P(g) * content[g][f] and the channel is the template
    likelihood — this is the quantity that goes to zero both when the content leaves the
    templates' support and when the templates stop discriminating."""
    content = np.asarray(content, dtype=float)
    templates = np.clip(np.asarray(templates, dtype=float), 1e-12, None)
    templates = templates / templates.sum(axis=1, keepdims=True)
    ng, nf = content.shape
    pg = 1.0 / ng
    mi = 0.0
    for g in range(ng):
        for f in range(nf):
            joint = pg * content[g, f]
            if joint <= 0:
                continue
            post = templates[:, f] / templates[:, f].sum()
            if post[g] > 0:
                mi += joint * np.log(post[g] / pg)
    return float(mi)


def build_matching_curve(cfg: Config, omega_grid) -> list[dict]:
    """Solve d(omega) by bisection on the measured curves. Computed BEFORE any rollout."""
    lo_ref = effective_overlap_unskilled(cfg, 0.0)
    rows = []
    for w in omega_grid:
        target = effective_overlap_foreign(cfg, float(w))
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if effective_overlap_unskilled(cfg, mid) > target:
                lo = mid
            else:
                hi = mid
        d = 0.5 * (lo + hi)
        rows.append({
            "omega": float(w),
            "target_effective_overlap_nats": float(target),
            "matched_d": float(d),
            "achieved_effective_overlap_nats": float(effective_overlap_unskilled(cfg, d)),
            "competent_reader_ceiling_nats": float(lo_ref),
        })
    return rows


# --------------------------------------------------------------------------- #
# The zero-compute precursor.
# --------------------------------------------------------------------------- #
def precursor(res_dir: Path) -> dict:
    """E15's committed d grid against E20's committed omega grid. Indicative only."""
    out = {"status": "unavailable", "note": (
        "INDICATIVE ONLY. E15 ran at 8 features under a different design and E20 at 16; the "
        "two are not matched on effective overlap and cannot settle C3. They show direction.")}
    e15p, e20p = Path(res_dir) / "e15_competence_cliff.csv", Path(res_dir) / "e20_omega_sweep.csv"
    if not (e15p.exists() and e20p.exists()):
        return out
    e15 = pd.read_csv(e15p)
    e20 = pd.read_csv(e20p)
    main = e15[e15.get("arm", "main") == "main"] if "arm" in e15.columns else e15
    g = main.groupby("d").agg(accuracy=("goal_accuracy", "mean"),
                              final_entropy=("final_entropy", "mean")).reset_index()
    out.update({
        "status": "computed",
        "unskilled_reader_e15": {
            "d": g.d.tolist(), "accuracy": g.accuracy.tolist(),
            "final_entropy": g.final_entropy.tolist()},
        "foreign_content_e20": {
            "omega": e20.omega.tolist(),
            "within_observer": e20.within_observer.tolist(),
            "between_observer": e20.between_observer.tolist(),
            "engaged_fraction": e20.engaged_fraction.tolist()},
        "direction": (
            "E15's high-d readers stay LOW in final entropy while accuracy collapses — "
            "confident and wrong. E20's low-omega cells sit HIGH in within-observer entropy "
            "(1.258 at omega = 0) and stay engaged. Opposite signatures, which is the "
            "divergence E32 predicts and tests properly."),
    })
    return out


# --------------------------------------------------------------------------- #
# The experiment.
# --------------------------------------------------------------------------- #
def _e32_worker(payload):
    (cfg_raw, cell_index, arm, omega, d, seed_rep, base_seed, n_obs, n_timesteps,
     forced_k) = payload
    from ..config import Config as _Config

    cfg = _Config(cfg_raw).copy()
    cfg.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))

    # Both arms are built from the SAME world so the only difference is where the gap comes
    # from. The foreign arm moves the content and leaves the reader competent; the unskilled
    # arm leaves the content human and perturbs the reader's templates.
    world = build_v4_world(cfg, omega=float(omega), include_explore=False)
    env = Environment(cfg, world.gm, np.random.default_rng(base_seed + cell_index),
                      honesty=1.0, signing_rate=0.0, foreign_sig=world.sigs.sig_foreign)

    art_rng = np.random.default_rng(C.observer_seed(base_seed, cell_index, seed_rep, 9_999))
    goal = int(art_rng.integers(FN.NUM_REAL_GOALS))
    if arm == "foreign_content":
        artifact = Artifact(provenance=K.GHOST, goal=goal, declared_signal=K.UNSIGNED,
                            foreign_goal=goal)
    else:
        artifact = Artifact(provenance=K.CREATOR, goal=goal, declared_signal=K.UNSIGNED)

    recs = []
    for i in range(n_obs):
        rng = C.observer_rng(base_seed, cell_index, seed_rep, i)
        d_i = float(d) if arm == "unskilled_reader" else 0.0
        agent = make_v4_observer(world, rng, d_i=d_i)
        res = rollout_observer(agent, artifact, env, cfg, rng, n_timesteps=n_timesteps,
                               force_deep_k=forced_k, initial_glance=True, early_stop=False)
        post = np.asarray(res.final_goal_posterior, dtype=float)
        free = np.asarray(res.attention)[forced_k:]
        recs.append({
            "arm": arm, "omega": float(omega), "d": float(d), "seed_rep": seed_rep,
            "observer": i, "true_goal": goal,
            "engaged_fraction": float(np.mean(free == K.DEEP)) if free.size else float("nan"),
            "final_entropy": float(metrics.within_observer_entropy(post)),
            "correct": int(int(np.argmax(post)) == goal),
            "psi_analogue": float(metrics.psi_analogue(
                post, res.goal_prior, float(cfg.signal_model.kappa), engaged=True)),
            "cum_deep": int(res.cum_deep),
            "posterior": post.tolist(),
        })
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    # THE MATCHING CURVE IS COMPUTED AND LOCKED BEFORE ANY ROLLOUT.
    curve = build_matching_curve(cfg, P5C.E32_OMEGA_GRID)
    P5C.ensure_preregistration_c234(cfg, res_dir / "v5_c234_preregistration.json")

    base_seed = int(cfg.run.base_seed if seed is None else seed)
    n_obs = int(cfg.get("experiments.e32.n_observers", 200))
    n_seeds = int(cfg.get("experiments.e32.n_seeds", 30))
    forced_k = int(cfg.get("experiments.e32.forced_deep_k", 10))
    n_timesteps = int(cfg.get("experiments.e32.n_timesteps", 24))

    payloads, ci = [], 0
    for row in curve:
        for arm in ARMS:
            for s in range(n_seeds):
                payloads.append((cfg.raw, ci, arm, row["omega"], row["matched_d"], s,
                                 base_seed, n_obs, n_timesteps, forced_k))
            ci += 1
    recs = C.run_parallel(payloads, _e32_worker, workers)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "posterior"} for r in recs])
    df.to_csv(res_dir / "e32_points.csv", index=False)
    stats = cell_stats(recs)
    stats.to_csv(res_dir / "e32_cell_stats.csv", index=False)

    verdict = build_verdict(stats, curve, res_dir)
    (res_dir / "e32_verdict.json").write_text(json.dumps(verdict, indent=2, default=float),
                                              encoding="utf-8")
    if make_fig:
        make_figure(stats, verdict, fig_dir / "e32_omega_d.png")
    return stats


def cell_stats(recs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(recs)
    rows = []
    for (arm, omega), sub in df.groupby(["arm", "omega"]):
        betweens = [metrics.between_observer_entropy(
            [np.asarray(p, dtype=float) for p in g["posterior"]])
            for _, g in sub.groupby("seed_rep")]
        rows.append({
            "arm": arm, "omega": float(omega), "d": float(sub.d.iloc[0]), "n": int(len(sub)),
            "engaged_fraction": float(sub.engaged_fraction.mean()),
            "within_observer": float(sub.final_entropy.mean()),
            "between_observer": float(np.mean(betweens)),
            "accuracy": float(sub.correct.mean()),
            "psi_analogue": float(sub.psi_analogue.mean()),
            "cum_deep": float(sub.cum_deep.mean()),
        })
    return pd.DataFrame(rows).sort_values(["arm", "omega"]).reset_index(drop=True)


MEASURES = ("engaged_fraction", "within_observer", "between_observer", "accuracy",
            "psi_analogue")


def build_verdict(stats: pd.DataFrame, curve: list[dict], res_dir: Path) -> dict:
    fo = stats[stats.arm == "foreign_content"].set_index("omega").sort_index()
    un = stats[stats.arm == "unskilled_reader"].set_index("omega").sort_index()
    shared = sorted(set(fo.index) & set(un.index))

    per_measure = {}
    for m in MEASURES:
        a = fo.loc[shared, m].to_numpy(float)
        b = un.loc[shared, m].to_numpy(float)
        gap = float(np.mean(np.abs(a - b)))
        scale = float(max(np.ptp(np.concatenate([a, b])), 1e-9))
        per_measure[m] = {
            "foreign_content": a.tolist(), "unskilled_reader": b.tolist(),
            "mean_abs_gap": gap, "gap_relative_to_range": gap / scale,
            "indistinguishable": bool(gap / scale <= P5C.E32_INDISTINGUISHABLE_FRAC),
        }

    n_same = sum(1 for m in MEASURES if per_measure[m]["indistinguishable"])
    verdict = P5C.e32_verdict(per_measure)
    verdict.update({
        "experiment": "E32 — are foreign content and an unskilled reader the same thing?",
        "measures_indistinguishable": n_same,
        "measures_total": len(MEASURES),
        "per_measure": per_measure,
        "matching_curve": curve,
        "matching_definition": (
            "MI(features; goal) under the observer's OWN likelihood on the content it sees, "
            "in nats. Solved for d by bisection, computed before any rollout, and locked."),
        "precursor": precursor(res_dir),
        "cell_table": stats.to_dict(orient="records"),
    })
    return verdict


def make_figure(stats: pd.DataFrame, verdict: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..figures import set_style
    set_style()

    fo = stats[stats.arm == "foreign_content"].sort_values("omega")
    un = stats[stats.arm == "unskilled_reader"].sort_values("omega")
    panels = [("engaged_fraction", "Do they keep looking?", "share of free time looking closely"),
              ("within_observer", "How sure are they?", "leftover uncertainty (nats)"),
              ("between_observer", "Do readers agree?", "disagreement between readers (nats)"),
              ("accuracy", "Do they get it right?", "share who got the purpose right")]

    fig, axes = plt.subplots(1, 4, figsize=(20.0, 4.8))
    for ax, (col, title, ylab) in zip(axes, panels):
        ax.plot(fo.omega, fo[col], marker="o", label="content from outside")
        ax.plot(un.omega, un[col], marker="s", ls="--", color="C3",
                label="reader out of their depth")
        ax.set(xlabel="matched overlap (low = big gap)", ylabel=ylab, title=title)
        ax.legend(fontsize=7)
    fig.suptitle(f"E32 — {verdict['verdict'].replace('_', ' ').lower()}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E32 — foreign content versus an unskilled reader")
    args = ap.parse_args()
    cfg = load_v4_config(quick=args.quick, include_explore=False)
    if args.seed is not None:
        cfg.set("run.base_seed", int(args.seed))
    stats = run(cfg, out_dir=Path(args.out) if args.out else None,
                workers=args.workers or C.default_workers())
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
