"""E29 — Do the three gates dissociate? (V4.5 §5.)

THE EXPERIMENT THAT DETERMINES WHETHER THE DECOMPOSITION IS REAL OR A RELABELING. Three gates
with different action points is a harder object to identify than one scalar: more parameters,
more freedom, less falsifiability per experiment. V4.5 §5 says E29 is the mitigation, and that
if it fails, to say so plainly rather than keeping the decomposition on theoretical grounds.

-----------------------------------------------------------------------------------------
WHY THIS RUNS OVER MULTIPLE ENCOUNTERS, WHICH IS THE DESIGN DECISION THE WHOLE EXPERIMENT
RESTS ON.

theta gates THE UPDATE — what the observer carries forward — and not the likelihood. That is
V4.5 §3.2's table and it is the only placement that makes theta a gate on integration rather
than on perception. The consequence, worked out before the run rather than discovered in the
output: within a SINGLE artifact, a closed theta changes nothing observable. Inference is
untouched, so engagement and resolution are bit-identical to the open-theta cell, and the only
things that differ are the update scalar and the divergence — both of which are how "closed
theta" is DEFINED. A single-artifact E29 would therefore report that theta dissociates, on the
strength of a tautology.

So each observer sees a SEQUENCE of artifacts from one source, and the gated update is what
its prior becomes for the next one. Now theta has somewhere to show up that is not its own
definition: after n encounters, an observer with a closed gate has a prior that has not moved,
and one with an open gate has a prior that has walked toward what it kept recovering. The
primary update measure is therefore CUMULATIVE PRIOR DRIFT, D_KL(D_final || D_initial), which
is a behavioural quantity rather than a restatement of the manipulation.

-----------------------------------------------------------------------------------------
DESIGN. Three kappa_p levels x two beta levels x two theta levels = 12 cells.

V4.5 §5 says 2 x 2 x 2, and then its own signature table splits the low-kappa_p row by content
type, because that split is E19's finding and is why E19 had to run first. So the kappa_p axis
has three levels, not two:

  kappa_p high          human content, alpha = 1.0. There is plenty to read.
  kappa_p low, empty    human content at alpha = 0.05. The intent does not survive
                        transmission, so the features are essentially noise_free_synth. This
                        is V1-V3's goal-EMPTY case, produced on human content so the axis
                        stays a manipulation of legibility rather than of provenance.
  kappa_p low, foreign  foreign content at omega = 0. Structured, goal-directed, and outside
                        the observer's family. V4's goal-FOREIGN case.

  beta      low = 0.25, high = 1.0. The grid endpoints, EXCLUDING beta = 0: E28 established
            that at beta = 0 the content carries literally zero goal information, so low beta
            and low kappa_p are the same condition there by construction and the cell could
            not discriminate anything. Stated here rather than chosen after seeing which value
            made the predicted table come out.

  theta     open   = lambda 0: the gate ignores value divergence entirely and the update
                     reduces to V1's psi_analogue exactly.
            closed = lambda 4 with a value prior P_c peaked on a goal the artifact is NOT
                     about. High value divergence with an ACTIVE gate, which is what the
                     signature table's "closed theta" row means. The gate is shut BY
                     DIVERGENCE rather than shut unconditionally — a gate that closes whatever
                     was recovered is an off switch, not a values gate.

-----------------------------------------------------------------------------------------
THE DECISIVE CONTRAST is low beta against closed theta. Both produce full extraction with
little integration. They differ in WHY: low beta says the trajectory is weak evidence, closed
theta says the recovered goal is unacceptable. If those two cells are behaviourally
indistinguishable on every measure, the decomposition is not earning its parameters.

One honesty flag is reported alongside the verdict and should be read first:
``distinguishable_on_behaviour_alone``. The divergence spike separates closed theta from low
beta partly BY CONSTRUCTION, since divergence is how closed theta is defined. If divergence is
the only thing that separates them, the decomposition has not been shown to do behavioural
work, whatever the headline verdict says.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants as K
from .. import foreign as FN
from .. import metrics
from .. import prereg_v4_5 as P45
from ..config import Config
from ..creators import HumanCreator
from ..environment import Artifact, Environment
from ..metrics import normalize
from ..observer import rollout_observer
from ..v4_5_model import (F_BETA, build_v45_world, demonstrator_signature, gated_update,
                          load_v4_5_config, make_v45_observer, recovered_beta)
from . import _common as C

KAPPA_P_LEVELS = ("high", "low_empty", "low_foreign")
BETA_LEVELS = {"low": 0.25, "high": 1.0}
THETA_LEVELS = ("open", "closed")

# The four cells V4.5 §5's signature table names, as (kappa_p, beta, theta) coordinates.
NAMED_CELLS = {
    "reference": ("high", "high", "open"),
    "low_kappa_p_goal_empty": ("low_empty", "high", "open"),
    "low_kappa_p_goal_foreign": ("low_foreign", "high", "open"),
    "low_beta": ("high", "low", "open"),
    "closed_theta": ("high", "high", "closed"),
}


def _e29_worker(payload):
    (cfg_raw, cell_index, kappa_p, beta_name, theta_name, seed_rep, base_seed, n_obs,
     n_encounters, n_timesteps, forced_k) = payload
    from ..config import Config as _Config

    cfg = _Config(cfg_raw).copy()
    cfg.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))

    true_beta = BETA_LEVELS[beta_name]
    world = build_v45_world(cfg)

    # The world's creators emit at the true beta, using the observer's own demonstrator
    # definition so the two cannot silently disagree about what beta means.
    world_sig = demonstrator_signature(world.sigs.sig_true, world.sigs.sig_explore, true_beta)
    creators = {g: HumanCreator(cfg, world_sig, g) for g in range(FN.NUM_REAL_GOALS)}

    # kappa_p is manipulated on the CHANNEL (alpha) for the goal-empty level, so the content
    # stays human and the axis stays a manipulation of legibility rather than of provenance.
    alpha_overrides = {"CREATOR": 0.05} if kappa_p == "low_empty" else None
    env = Environment(cfg, world.gm, np.random.default_rng(base_seed + cell_index),
                      honesty=1.0, signing_rate=0.0, creator_bank=creators,
                      alpha_overrides=alpha_overrides,
                      foreign_sig=world.sigs.sig_foreign)

    theta_base = float(cfg.get("v4_5.theta.base_open", 10.0) if theta_name == "open"
                       else cfg.get("v4_5.theta.base_closed", 0.0))
    lam = float(cfg.get("v4_5.theta.lambda_open", 0.0) if theta_name == "open"
                else cfg.get("v4_5.theta.lambda_closed", 4.0))
    eta = float(cfg.get("experiments.e29.integration_rate", 0.5))
    kappa = float(cfg.signal_model.kappa)

    art_rng = np.random.default_rng(C.observer_seed(base_seed, cell_index, seed_rep, 9_999))
    source_goal = int(art_rng.integers(FN.NUM_REAL_GOALS))

    # P_c — what the observer VALUES. For the closed-theta arm it is peaked on a goal the
    # source is not pursuing, which is what makes the recovered goal unacceptable; for the
    # open arm it is peaked on the source's own goal, so divergence stays low and the two
    # arms differ in divergence as well as in lambda. Aligning P_c with the source in the
    # open arm matters: leaving it misaligned there would mean the "open" cells also carry
    # high divergence, and the divergence column would stop tracking the manipulation.
    value_goal = source_goal if theta_name == "open" else (source_goal + 2) % FN.NUM_REAL_GOALS
    value_prior = np.full(FN.NUM_REAL_GOALS, 0.05)
    value_prior[value_goal] = 1.0
    value_prior = value_prior / value_prior.sum()

    if kappa_p == "low_foreign":
        artifact = Artifact(provenance=K.GHOST, goal=source_goal,
                            declared_signal=K.UNSIGNED, foreign_goal=source_goal)
    else:
        artifact = Artifact(provenance=K.CREATOR, goal=source_goal,
                            declared_signal=K.UNSIGNED)

    recs = []
    for i in range(n_obs):
        rng = C.observer_rng(base_seed, cell_index, seed_rep, i)
        agent = make_v45_observer(world, rng)
        prior0 = np.asarray(agent.D[K.F_GOAL], dtype=float).copy()

        engaged_acc, entropy_acc, psi_acc, div_acc, theta_acc, beta_acc = [], [], [], [], [], []
        for _ in range(n_encounters):
            res = rollout_observer(agent, artifact, env, cfg, rng, n_timesteps=n_timesteps,
                                   force_deep_k=forced_k, initial_glance=True,
                                   extra_factors=(F_BETA,))
            post = np.asarray(res.final_goal_posterior, dtype=float)
            free = np.asarray(res.attention)[forced_k:]
            engaged_acc.append(float(np.mean(free == K.DEEP)) if free.size else float("nan"))
            entropy_acc.append(float(metrics.within_observer_entropy(post)))
            beta_acc.append(recovered_beta(res.final_extra_posterior[F_BETA],
                                           world.beta_levels))

            g = gated_update(post, np.asarray(agent.D[K.F_GOAL], dtype=float), kappa,
                             engaged=True, value_prior=value_prior,
                             theta_base=theta_base, lam=lam)
            psi_acc.append(g["psi_gated"])
            div_acc.append(g["value_divergence"])
            theta_acc.append(g["theta"])

            # THE CARRIED-FORWARD UPDATE, gated by theta. This is the only place theta acts,
            # and it is why the experiment runs over a sequence: a gate on integration is
            # invisible in a design with nothing to integrate into.
            step = float(g["theta"]) * eta
            agent.D[K.F_GOAL] = normalize(
                (1.0 - step) * np.asarray(agent.D[K.F_GOAL], dtype=float) + step * post)

        prior_final = np.asarray(agent.D[K.F_GOAL], dtype=float)
        recs.append({
            "kappa_p": kappa_p, "beta": beta_name, "theta": theta_name,
            "seed_rep": seed_rep, "observer": i, "source_goal": source_goal,
            "engaged_fraction": float(np.mean(engaged_acc)),
            "final_entropy": float(entropy_acc[-1]),
            "mean_entropy": float(np.mean(entropy_acc)),
            "psi_per_encounter": float(np.mean(psi_acc)),
            # THE PRIMARY UPDATE MEASURE: how far the observer's carried belief actually
            # moved over the whole sequence. Behavioural, not a restatement of the gate.
            "prior_drift": float(metrics.kl_divergence(prior_final, prior0)),
            "prior_mass_on_source_goal": float(prior_final[source_goal]),
            "value_divergence": float(np.mean(div_acc)),
            "theta_value": float(np.mean(theta_acc)),
            "recovered_beta": float(np.mean(beta_acc)),
            "correct": int(int(np.argmax(prior_final)) == source_goal),
        })
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    res_dir, fig_dir = C.ensure_dirs(out_dir)
    prereg_path = res_dir / "v4_5_preregistration.json"
    P45.write_preregistration_v4_5(cfg, prereg_path)
    P45.assert_prereg_locked_v4_5(prereg_path)

    base_seed = int(cfg.run.base_seed if seed is None else seed)
    n_obs = int(cfg.get("experiments.e29.n_observers", 60))
    n_seeds = int(cfg.get("experiments.e29.n_seeds", 20))
    n_enc = int(cfg.get("experiments.e29.n_encounters", 4))
    forced_k = int(cfg.get("experiments.e29.forced_deep_k", 10))
    n_timesteps = int(cfg.get("experiments.e29.n_timesteps", 24))

    payloads, ci = [], 0
    for kp in KAPPA_P_LEVELS:
        for bn in BETA_LEVELS:
            for tn in THETA_LEVELS:
                for s in range(n_seeds):
                    payloads.append((cfg.raw, ci, kp, bn, tn, s, base_seed, n_obs, n_enc,
                                     n_timesteps, forced_k))
                ci += 1
    recs = C.run_parallel(payloads, _e29_worker, workers)

    df = pd.DataFrame(recs)
    df.to_csv(res_dir / "e29_points.csv", index=False)
    stats = cell_stats(df)
    stats.to_csv(res_dir / "e29_cell_stats.csv", index=False)

    verdict = build_verdict(stats)
    (res_dir / "e29_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    if make_fig:
        make_figure(stats, verdict, fig_dir / "e29_gates.png")
    return stats


def cell_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (kp, bn, tn), sub in df.groupby(["kappa_p", "beta", "theta"]):
        rows.append({
            "kappa_p": kp, "beta": bn, "theta": tn, "n": int(len(sub)),
            "engaged_fraction": float(sub.engaged_fraction.mean()),
            "final_entropy": float(sub.final_entropy.mean()),
            "mean_entropy": float(sub.mean_entropy.mean()),
            "prior_drift": float(sub.prior_drift.mean()),
            "prior_drift_sd": float(sub.prior_drift.std()),
            "psi_per_encounter": float(sub.psi_per_encounter.mean()),
            "value_divergence": float(sub.value_divergence.mean()),
            "theta_value": float(sub.theta_value.mean()),
            "recovered_beta": float(sub.recovered_beta.mean()),
            "accuracy": float(sub.correct.mean()),
            "prior_mass_on_source_goal": float(sub.prior_mass_on_source_goal.mean()),
        })
    return pd.DataFrame(rows).sort_values(["kappa_p", "beta", "theta"]).reset_index(drop=True)


def _cell(stats: pd.DataFrame, kp: str, bn: str, tn: str) -> pd.Series | None:
    sub = stats[(stats.kappa_p == kp) & (stats.beta == bn) & (stats.theta == tn)]
    return sub.iloc[0] if len(sub) else None


def build_verdict(stats: pd.DataFrame) -> dict:
    ref = _cell(stats, *NAMED_CELLS["reference"])
    ref_update = float(ref["prior_drift"]) if ref is not None else float("nan")
    ref_div = float(ref["value_divergence"]) if ref is not None else float("nan")

    signatures = {}
    for name, coords in NAMED_CELLS.items():
        if name == "reference":
            continue
        c = _cell(stats, *coords)
        if c is None:
            continue
        signatures[name] = P45.e29_cell_signature(
            engaged=float(c["engaged_fraction"]),
            final_entropy=float(c["final_entropy"]),
            update=float(c["prior_drift"]),
            reference_update=ref_update,
            value_divergence=float(c["value_divergence"]),
            reference_divergence=ref_div)

    verdict = P45.e29_verdict(signatures)
    verdict["experiment"] = "E29 — do the three gates dissociate?"
    verdict["reference_cell"] = {
        "coords": NAMED_CELLS["reference"],
        "prior_drift": ref_update, "value_divergence": ref_div,
        "engaged_fraction": float(ref["engaged_fraction"]) if ref is not None else float("nan"),
        "final_entropy": float(ref["final_entropy"]) if ref is not None else float("nan"),
    }

    # THE HONESTY FLAG, and it should be read before the headline verdict.
    #
    # The divergence spike separates closed theta from low beta partly BY CONSTRUCTION:
    # divergence is how closed theta is defined in this design. If divergence is the only
    # thing that separates them, then the decomposition has not been shown to do behavioural
    # work, whatever the verdict string says. This is reported whether or not it is
    # flattering.
    dc = verdict.get("decisive_contrast", {})
    behavioural = bool(dc.get("differs_on_update") or dc.get("differs_on_engagement")
                       or dc.get("differs_on_resolution"))
    verdict["distinguishable_on_behaviour_alone"] = behavioural
    verdict["honesty_note"] = (
        "The divergence spike is definitional: closed theta is DEFINED by a value prior that "
        "diverges from the recovered goal, so a spike there is the manipulation and not a "
        "prediction. 'distinguishable_on_behaviour_alone' is the quantity that matters — it "
        "asks whether low beta and closed theta differ on engagement, resolution or the "
        "CARRIED-FORWARD update, none of which is a restatement of the manipulation. "
        + ("They do." if behavioural else
           "THEY DO NOT. Only the definitional measure separates them, so the theta gate has "
           "not been shown to do behavioural work that low beta could not account for, and "
           "V4.5 §5's instruction is to report that rather than keep the decomposition on "
           "theoretical grounds."))
    verdict["cell_table"] = stats.to_dict(orient="records")
    return verdict


def make_figure(stats: pd.DataFrame, verdict: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..figures import set_style
    set_style()

    order = ["reference", "low_kappa_p_goal_empty", "low_kappa_p_goal_foreign",
             "low_beta", "closed_theta"]
    nice = {"reference": "everything\nworking",
            "low_kappa_p_goal_empty": "nothing there\nto read",
            "low_kappa_p_goal_foreign": "structured, but\nnot for us",
            "low_beta": "they weren't\ntrying hard",
            "closed_theta": "read it, won't\ntake it on"}
    rows = []
    for name in order:
        c = _cell(stats, *NAMED_CELLS[name])
        rows.append(c)
    x = np.arange(len(order))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    for ax, col, title, ylab in (
            (axes[0], "engaged_fraction", "Do they keep looking?",
             "share of free time spent looking closely"),
            (axes[1], "final_entropy", "Do they work out what it was for?",
             "leftover uncertainty (nats, lower = worked it out)"),
            (axes[2], "prior_drift", "Does it change what they believe?",
             "how far their view moved over 4 encounters")):
        vals = [float(r[col]) if r is not None else np.nan for r in rows]
        colours = ["C7"] + ["C0", "C0", "C1", "C3"]
        ax.bar(x, vals, 0.6, color=colours)
        ax.set_xticks(x, [nice[n] for n in order], fontsize=7)
        ax.set(title=title, ylabel=ylab)

    axes[0].axhline(P45.E29_ENGAGED_HIGH, ls=":", color="k", lw=1)
    axes[1].axhline(P45.E29_RESOLVED_ENTROPY, ls=":", color="k", lw=1)
    ref_u = verdict["reference_cell"]["prior_drift"]
    axes[2].axhline(P45.E29_UPDATE_LOW * ref_u, ls=":", color="k", lw=1)
    axes[2].axhline(P45.E29_UPDATE_NONE * ref_u, ls="--", color="firebrick", lw=1)

    fig.suptitle(f"E29 — {verdict['verdict'].replace('_', ' ').lower()}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E29 — do the three gates dissociate?")
    args = ap.parse_args()
    cfg = load_v4_5_config(quick=args.quick)
    if args.seed is not None:
        cfg.set("run.base_seed", int(args.seed))
    out = Path(args.out) if args.out else None
    stats = run(cfg, out_dir=out, workers=args.workers or C.default_workers())
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
