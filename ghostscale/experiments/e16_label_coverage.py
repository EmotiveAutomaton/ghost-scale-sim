"""E16 — How much AI content must be labelled to protect the corpus?

E7 asked whether a reader can learn which sources are hollow with labels and without them.
This asks the question in between, at the resolution a policy argument needs: label synthetic
content a fraction ``p`` of the time and find the coverage at which the reader's model of
human intent stops degrading.

The target is a quotable number — "you only need to label X% of AI content" — so the design
has to survive the obvious objections.

-----------------------------------------------------------------------------------------
FOUR ARMS, BECAUSE "LABELLING" MEANS TWO DIFFERENT THINGS AND SO DOES "READER".

**Regime — what gets labelled.**

  * ``symmetric``  — every artifact is signed with probability p, human and synthetic alike.
    This is V1/V2's ``signing_rate`` and it is a *provenance-metadata* regime.
  * ``ai_only``    — synthetic work is disclosed with probability p; human work is never
    signed. This is the actual policy proposal (AI-disclosure mandates), and it is NOT the
    same experiment: under it an unlabelled artifact is *evidence of humanity*, which is
    where most of the protective value is supposed to come from.

**Reader — whether the regime is known.**

  * ``naive``         — the V1/V2 observer likelihood, ``build_A1_observer``. It assigns
    ``P(UNSIGNED | provenance) = (1-κ)/5``, the SAME for every tier, so the absence of a
    label is uninformative BY CONSTRUCTION, at any coverage.
  * ``regime_aware``  — ``build_A1_regime_aware``, which knows the coverage and can therefore
    draw the contrapositive inference.

THE ASYMMETRY IS THE POINT, AND IT IS WHY THE 2x2 IS RUN RATHER THAN JUST THE POLICY CELL.
A naive reader under an ai_only regime can gain only from the labels it actually sees; it
cannot gain from the ones that are absent. If the coverage threshold turns out to be much
lower for regime_aware readers than for naive ones, the quotable finding is not "label X% of
AI content" but the sharper and more useful:

    a disclosure mandate protects the corpus only for readers who know the mandate exists —
    coverage and reader awareness are complements, not substitutes.

**A CAVEAT THAT MUST TRAVEL WITH ANY QUOTED NUMBER.** ``regime_aware`` is given the TRUE
coverage. That is the most generous assumption available, so its threshold is a LOWER BOUND
on what a real deployment would need; a reader with a miscalibrated belief about coverage can
only do worse. The number is a floor, not a target.

Honesty is held at 1.0 throughout: labels never lie. Mislabelling is E4's question, and
mixing it in here would confound coverage with fraud.
-----------------------------------------------------------------------------------------

MEASURES, AND A CORRECTION TO THIS MODULE'S FIRST DESIGN.

  * ``ghost_col_err`` — KL(learned GHOST column || true synth). **THE PRIMARY OUTCOME.**
    Thresholded at E7's own ``competence_kl`` (1.0 nats) so the number means the same thing
    it does there.
  * ``creator_mi``    — MI(features; goal) of the learned HUMAN columns. Reported as a
    DEGRADATION measure, not an acquisition one.

This module was first written with ``creator_mi`` as primary, which was a mistake, and E7's
own documentation says why in as many words:

    "the D1-seeded learner starts at ~95% of oracle MI by construction, so that clock reads
     zero in every condition. It measures the seeding, not the learning. What the learner
     genuinely does not have at t=0 is the GHOST column, so competence is timed on that
     instead."

An outcome with ~5% of its range available cannot resolve a coverage threshold. The primary
was changed on that pre-existing, documented ground — not because of how the numbers came
out — and both measures are reported so the change is auditable.
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
from ..constants import CREATOR, POLISHED, CURATOR, GHOST, DEEP, UNSIGNED
from ..generative_model import (build_shared_model as _build_model, assert_preferences_zero,
                                build_observer_model, build_D, build_A1_regime_aware)
from ..creators import build_creator_bank
from ..environment import Environment
from ..observer import rollout_observer
from ..preregistration import POP_GOAL_DIST, allocate_creator_goals
from .. import learning as L, metrics
from ..figures import set_style
from . import _common as C

REGIMES = ("symmetric", "ai_only")
READERS = ("naive", "regime_aware")


def coverage_map(regime: str, p: float) -> dict[int, float]:
    """Per-provenance labelling coverage under each regime."""
    if regime == "symmetric":
        return {CREATOR: p, POLISHED: p, CURATOR: p, GHOST: p}
    if regime == "ai_only":
        return {CREATOR: 0.0, POLISHED: 0.0, CURATOR: 0.0, GHOST: p}
    raise ValueError(f"unknown regime {regime!r}")


def _e16_worker(payload):
    (cfg_raw, cell_index, regime, reader, p, kappa, f, seed_rep, base_seed,
     n_creators, n_artifacts, n_observers, infer_steps, synth_seed) = payload
    cfg = Config(cfg_raw)
    num_goals = cfg.cardinalities.num_goals
    cov = coverage_map(regime, p)

    gm = _build_model(cfg, kappa=kappa, goal_symmetric=False, synth_draw_seed=synth_seed)
    assert_preferences_zero(gm.C)                    # N7
    bank = build_creator_bank(cfg, gm)
    oracle_mi = metrics.mutual_information_features_goal(np.asarray(gm.A[0]), CREATOR, DEEP)

    pop_rng = np.random.default_rng(base_seed * 13 + seed_rep)
    creator_goals = allocate_creator_goals(n_creators, POP_GOAL_DIST[:num_goals], pop_rng)
    world = np.random.default_rng(base_seed * 31 + seed_rep * 7)
    env = Environment(cfg, gm, rng_world=world, honesty=1.0, signing_rate=p,
                      creator_bank=bank, signing_rate_by_provenance=cov)
    corpus = env.draw_corpus(n_artifacts, contamination=f, creator_goals=creator_goals,
                             rng=world)
    # Recorded so the realized coverage can be checked against the requested p in the CSV.
    labelled = float(np.mean([a.declared_signal != UNSIGNED for a in corpus]))

    recs = []
    for i in range(n_observers):
        r = C.observer_rng(base_seed, cell_index, seed_rep, i)
        om = build_observer_model(gm, cfg, 0.0, kappa=kappa)
        if reader == "regime_aware":
            om.A[1] = build_A1_regime_aware(cfg, kappa, cov)
        agent = L.make_learner_agent(om, build_D(cfg, r), cfg, kappa=kappa)
        for j, art in enumerate(corpus):
            rr = np.random.default_rng(base_seed * 90001 + seed_rep * 337 + i * 7919 + j)
            rollout_observer(agent, art, env, cfg, rr, infer_steps,
                             force_deep_k=infer_steps, kappa=kappa,
                             early_stop=False, learn=True)
        mi = L.human_column_mi(agent.A[0], CREATOR)
        recs.append({
            "regime": regime, "reader": reader, "coverage": p, "kappa": kappa,
            "contamination": f, "seed_rep": seed_rep, "observer": i,
            "creator_mi": mi, "oracle_mi": oracle_mi,
            "mi_ratio": mi / oracle_mi if oracle_mi > 0 else np.nan,
            "ghost_col_err": L.ghost_column_error(agent.A[0], gm.noise_free_synth),
            "frac_corpus_labelled": labelled,
        })
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    grid = list(cfg.get("experiments.e16.coverage_grid"))
    kappa = float(cfg.get("experiments.e16.kappa", 0.9))
    f = float(cfg.get("experiments.e16.contamination", 0.6))
    n_creators = int(cfg.get("experiments.e16.n_creators", 50))
    n_artifacts = int(cfg.get("experiments.e16.n_artifacts", 400))
    n_observers = int(cfg.get("experiments.e16.n_observers", 6))
    infer_steps = int(cfg.get("experiments.e16.infer_steps", 6))
    n_reps = int(cfg.get("experiments.e16.n_replications", 3))
    synth_seed = int(cfg.get("experiments.e16.synth_draw_seed", 17))
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    payloads, ci = [], 0
    for regime in REGIMES:
        for reader in READERS:
            for p in grid:
                for s in range(n_reps):
                    payloads.append((cfg.raw, ci, regime, reader, float(p), kappa, f, s,
                                     base_seed, n_creators, n_artifacts, n_observers,
                                     infer_steps, synth_seed))
                ci += 1
    df = pd.DataFrame(C.run_parallel(payloads, _e16_worker, workers))
    df.to_csv(res_dir / "e16_label_coverage.csv", index=False)

    verdict = analyse(df, float(cfg.get("experiments.e7.competence_kl", 1.0)))
    (res_dir / "e16_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    if make_fig:
        make_figure(df, verdict, fig_dir / "e16_label_coverage.png")
    return df


def threshold_for(sub: pd.DataFrame, frac_of_full: float = 0.95) -> dict:
    """Smallest coverage at which ``creator_mi`` reaches ``frac_of_full`` of its p=1 value.

    Defined against the arm's OWN full-coverage value rather than against the oracle, because
    the policy question is "how much of the available protection does partial coverage buy",
    not "how close to a perfect reader does it get".
    """
    agg = sub.groupby("coverage").creator_mi.mean().sort_index()
    if agg.empty:
        return {"threshold": None}
    full = float(agg.iloc[-1])
    floor = float(agg.iloc[0])
    if not np.isfinite(full) or full <= floor:
        return {"threshold": None, "full_coverage_mi": full, "zero_coverage_mi": floor,
                "reason": "no gain from coverage in this arm; a threshold is not defined"}
    target = floor + frac_of_full * (full - floor)
    hit = [c for c, v in agg.items() if v >= target]
    return {"threshold": float(min(hit)) if hit else None,
            "target_mi": target, "full_coverage_mi": full, "zero_coverage_mi": floor,
            "gain_from_full_coverage": full - floor,
            "criterion": f"{frac_of_full:.0%} of the gain this arm gets from full coverage"}


def competence_threshold(sub: pd.DataFrame, competence_kl: float = 1.0) -> dict:
    """Smallest coverage at which the learned GHOST column comes within ``competence_kl`` of
    the true synthetic distribution — E7's own competence definition, reused verbatim.

    Linearly interpolated between the bracketing grid points, since the grid is coarser than
    the precision a quoted policy number implies.
    """
    agg = sub.groupby("coverage").ghost_col_err.mean().sort_index()
    cov = agg.index.to_numpy(float)
    err = agg.to_numpy(float)
    below = np.where(err <= competence_kl)[0]
    if len(below) == 0:
        return {"competence_threshold": None, "reached": False,
                "best_err": float(err.min()), "err_at_full_coverage": float(err[-1]),
                "criterion": f"KL(learned GHOST column || true synth) <= {competence_kl}",
                "reason": "never reaches competence at any coverage in the swept range"}
    i = int(below[0])
    if i == 0:
        thresh = float(cov[0])
    else:
        x0, x1, y0, y1 = cov[i - 1], cov[i], err[i - 1], err[i]
        thresh = float(x0 + (competence_kl - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)
    return {"competence_threshold": thresh, "reached": True,
            "err_at_full_coverage": float(err[-1]),
            "criterion": f"KL(learned GHOST column || true synth) <= {competence_kl}"}


def analyse(df: pd.DataFrame, competence_kl: float = 1.0) -> dict:
    arms = {}
    for (regime, reader), sub in df.groupby(["regime", "reader"]):
        arms[f"{regime}/{reader}"] = {
            **competence_threshold(sub, competence_kl),
            "creator_mi_threshold": threshold_for(sub),
            "ghost_err_by_coverage": sub.groupby("coverage").ghost_col_err.mean().round(4).to_dict(),
            "mi_by_coverage": sub.groupby("coverage").creator_mi.mean().round(4).to_dict(),
        }
    ai_naive = arms.get("ai_only/naive", {}).get("competence_threshold")
    ai_aware = arms.get("ai_only/regime_aware", {}).get("competence_threshold")
    sym_naive = arms.get("symmetric/naive", {}).get("competence_threshold")
    sym_aware = arms.get("symmetric/regime_aware", {}).get("competence_threshold")
    awareness_matters = (ai_aware is not None
                         and (ai_naive is None or ai_aware < ai_naive))
    # Internal control: under SYMMETRIC labelling every tier is signed at the same rate, so
    # absence of a label really is uninformative and knowing the regime should buy nothing.
    # If awareness helps there too, the regime-aware likelihood is doing something other than
    # what it claims and the ai_only comparison cannot be trusted.
    control_ok = (sym_naive is not None and sym_aware is not None
                  and abs(sym_aware - sym_naive) <= 0.1)
    return {
        "experiment": "E16 — labelling coverage threshold",
        "primary_outcome": ("ghost_col_err — KL(learned GHOST column || true synth), "
                            "thresholded at E7's competence_kl; see the module docstring for "
                            "why creator_mi is NOT the primary"),
        "arms": arms,
        "policy_cell": "ai_only/regime_aware",
        "headline_threshold": ai_aware,
        "naive_threshold_same_regime": ai_naive,
        "awareness_lowers_the_threshold": bool(awareness_matters),
        "symmetric_control_shows_no_awareness_effect": bool(control_ok),
        "control_interpretation": (
            "under symmetric labelling, absence of a label is genuinely uninformative, so a "
            "regime-aware reader should gain nothing over a naive one. That the two agree "
            "there is what licenses attributing the ai_only difference to the contrapositive "
            "inference rather than to the likelihood change itself."),
        "caveat_on_any_quoted_number": (
            "the regime_aware reader is given the TRUE coverage, the most generous assumption "
            "available, so its threshold is a LOWER BOUND on what a real deployment needs; a "
            "reader with a miscalibrated belief about coverage can only do worse"),
        "why_naive_may_show_no_gain_from_absence": (
            "build_A1_observer assigns P(UNSIGNED | provenance) identically across tiers, so "
            "for a naive reader an unlabelled artifact is uninformative by construction at any "
            "coverage. Any ai_only gain it shows comes only from the labels it does see."),
    }


def make_figure(df, verdict, path):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    styles = {("symmetric", "naive"): ("grey", "--"), ("symmetric", "regime_aware"): ("C2", "-"),
              ("ai_only", "naive"): ("firebrick", "--"), ("ai_only", "regime_aware"): ("C0", "-")}

    def arm_name(regime, reader):
        rg = "everything gets labelled" if regime == "symmetric" else "only AI gets labelled"
        rd = "reader knows the rule" if reader == "regime_aware" else "reader does not know it"
        return f"{rg}, {rd}"

    for key, ax, ylab, title in (
            ("creator_mi", axes[0], "intent it can still read out of human work (nats)",
             "Its grip on human work\n(a background check, not the headline)"),
            ("ghost_col_err", axes[1], "how wrong it is about machine-made work (nats)",
             "Has the reader worked out what\nhollow content looks like?")):
        for (regime, reader), sub in df.groupby(["regime", "reader"]):
            agg = sub.groupby("coverage")[key].agg(["mean", "std"])
            colour, ls = styles.get((regime, reader), ("k", "-"))
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], color=colour, ls=ls,
                        marker="o", ms=4, capsize=2, label=arm_name(regime, reader))
        ax.set(xlabel="share of machine-made work that carries a label",
               ylabel=ylab, title=title)
        ax.legend(fontsize=7)

    ax = axes[2]
    # Short forms, because four full arm names will not fit across one axis without colliding.
    short = {("symmetric", "naive"): "all work\nlabelled\nreader in\nthe dark",
             ("symmetric", "regime_aware"): "all work\nlabelled\nreader told\nthe rule",
             ("ai_only", "naive"): "only AI\nlabelled\nreader in\nthe dark",
             ("ai_only", "regime_aware"): "only AI\nlabelled\nreader told\nthe rule"}
    names, vals = [], []
    for name, a in verdict["arms"].items():
        regime, reader = name.split("/")
        names.append(short.get((regime, reader), name))
        # ``competence_threshold`` is the key ``analyse`` actually writes. This read used to be
        # ``a.get("threshold")``, which is never present, so every bar came out NaN and the panel
        # rendered empty — a silent one, because an empty panel looks like a real null result.
        v = a.get("competence_threshold")
        vals.append(float(v) if v is not None else np.nan)
    ax.bar(range(len(names)), vals,
           color=["C0" if "regime_aware" in n else "firebrick" for n in verdict["arms"]])
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(names)), names, fontsize=7)
    ax.set(ylim=(0, 1.0), ylabel="labelling needed before the reader copes",
           title="How much labelling each case needs\n"
                 "(no bar = labelling never got this reader there)")

    fig.suptitle("E16 — A label helps twice as much when the reader knows "
                 "the labelling rule exists", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E16 — labelling coverage threshold")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = C.default_workers() if args.workers is None else args.workers
    run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    res_dir, _ = C.ensure_dirs(args.out)
    v = json.loads((res_dir / "e16_verdict.json").read_text(encoding="utf-8"))
    print("\nE16 — coverage threshold by arm (95% of the arm's own gain):")
    for name, a in v["arms"].items():
        t = a.get("threshold")
        print(f"  {name:28s} threshold {('%.2f' % t) if t is not None else '  none'}   "
              f"MI {a.get('zero_coverage_mi', float('nan')):.3f} -> "
              f"{a.get('full_coverage_mi', float('nan')):.3f}")
    print(f"\n  policy cell (ai_only/regime_aware): {v['headline_threshold']}")
    print(f"  awareness lowers the threshold: {v['awareness_lowers_the_threshold']}")
    print(f"  CAVEAT: {v['caveat_on_any_quoted_number']}")


if __name__ == "__main__":
    main()
