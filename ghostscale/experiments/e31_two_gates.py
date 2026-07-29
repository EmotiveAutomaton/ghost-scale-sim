"""E31 — two gates, corrected: mu and theta, with provenance as EVIDENCE (V5 C2).

WHAT C2 CORRECTS. V4.5 specified three gates — provenance (kappa_p), competence (beta), value
alignment (theta). Provenance does not belong in that list. It is EVIDENCE ABOUT MODEL DEPTH:
learning that something is machine-generated lowers your estimate of how much reasoning
produced it. That is an input to the mu gate, not a parallel channel.

                    provenance signal --+
                    observer expertise --+--> mu --> theta --> update
                    surface properties --+

TWO GATES, AND THE FRAMEWORK'S TWO HEADLINE RESULTS BECOME ONE MECHANISM WITH OPPOSITE SIGNS:

  * the GENERATIVE CRASH is a CORRECTLY LOW mu. Little reasoning behind it, so don't spend.
  * the TRUST EXPLOIT is a FALSELY HIGH mu. A dishonest provenance signal inflates the
    estimate, the observer over-invests, and fabricates to justify the investment.

THAT IS WHAT THIS EXPERIMENT TESTS, and the pre-registered primary criterion follows from it
directly: if the two really are one mechanism, then UPDATE MAGNITUDE TRACKS RECOVERED DEPTH
REGARDLESS OF WHICH CHANNEL MOVED IT. A dishonest label and genuinely deep content should
produce the same update when they produce the same recovered mu. If they do not, provenance is
doing something the depth estimate does not carry, and collapsing three channels to two has
not been earned — that is the ``DEPTH_IS_NOT_THE_COMMON_PATH`` branch.

-----------------------------------------------------------------------------------------
WHERE THE EVIDENCE PATH LIVES, AND THE LIMITATION IT CARRIES.

V5 decision 10 originally merged provenance and mu into one hidden factor so that a joint
prior P(mu | provenance) could be written. That merge is incompatible with the one that makes
depth readable at all: mu has to share a factor with the sub-goal, or pymdp's mean-field
inference systematically favours the shallow hypothesis (see v5_model, and RESULTS_V5.md's
account of what N21 caught). Putting all four quantities in one factor needs a cardinality-1
placeholder to keep attention at index 2, and pymdp raises on it.

So the path is a mu prior CONDITIONED ON THE OBSERVED LABEL, supplied per observer. Unchanged
in substance — a provenance signal shifts the observer's expectation of how much reasoning sits
behind the work, before any gate runs — and more inspectable, because the coupling is one
argument that can be printed.

**THE LIMITATION, STATED IN THE FIRST PLACE A READER WOULD LOOK.** The observer no longer
infers provenance and depth jointly, so a reader who doubts the label does not automatically
doubt the depth estimate that label induced. In this design the label is taken at face value
when forming the prior, and only the CONTENT can subsequently move mu. That makes the trust
exploit measured here an UPPER BOUND on the effect: a reader who could doubt the label would
be harder to fool. The bound is the right side to err on for a warning, and the wrong side for
a reassurance, and it should be quoted that way.

-----------------------------------------------------------------------------------------
DESIGN. 2 content x 3 label x 2 theta, over 4 sequential encounters.

  content   human   -- CREATOR artifacts at mu = 3, deep and legible
            foreign -- GHOST artifacts at omega = 0.10, which is where E20 located BOTH the
                       crash and the fabrication peak. See the worker for why not 0.

  label     unsigned    -- content carries the whole load (E19/E28's condition)
            SIG_CREATOR -- claims human authorship. On foreign content this is the LIE, and
                           it is the trust exploit's cell.
            SIG_GHOST   -- honest disclosure. On foreign content this is the CRASH's cell.

  theta     open   -- lambda 0, the gate ignores divergence and the update reduces to V1's
                      psi_analogue exactly
            closed -- lambda 4 with a value prior peaked on a goal the artifact is not about

SEQUENTIAL, for E29's reason, restated because it is the design decision the whole experiment
rests on: theta gates THE UPDATE, so within a single artifact a closed theta changes nothing
observable and would appear to dissociate on the strength of its own definition. Each observer
sees four artifacts from one source and the gated update is what its prior becomes for the
next. The primary update measure is CUMULATIVE PRIOR DRIFT, which is behavioural rather than a
restatement of the manipulation.
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
from ..environment import Artifact
from ..metrics import normalize
from ..n21_depth_not_effort import merged_config
from ..observer import rollout_observer
from ..v4_5_model import gated_update
from ..v5_model import (MU_LEVELS, HierarchicalCreator, V5Environment, build_v5_world,
                        load_v5_config, make_v5_observer, marginal_goal, marginal_mu,
                        recovered_mu)
from . import _common as C

CONTENT = ("human", "foreign")
LABELS = ("unsigned", "sig_creator", "sig_ghost")
THETAS = ("open", "closed")

# What each label leads the observer to expect about depth, before it sees anything. This is
# the ONLY place C2's evidence path is written. SIG_CREATOR says a person made this, so expect
# a deep model; SIG_GHOST says a machine did, so expect a shallow one; UNSIGNED says nothing
# and leaves the prior flat, which is what makes the unsigned row the content-only control.
LABEL_MU_PRIOR = {
    "unsigned":    np.array([1.0, 1.0, 1.0]),
    "sig_creator": np.array([1.0, 2.0, 4.0]),
    "sig_ghost":   np.array([4.0, 2.0, 1.0]),
}
LABEL_SIGNAL = {"unsigned": K.UNSIGNED, "sig_creator": K.SIG_CREATOR, "sig_ghost": K.SIG_GHOST}


def _e31_worker(payload):
    (cfg_raw, cell_index, content, label, theta_name, seed_rep, base_seed, n_obs,
     n_encounters, n_timesteps, forced_k) = payload
    from ..config import Config as _Config

    cfg = _Config(cfg_raw).copy()
    cfg.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))

    # FOREIGN CONTENT SITS AT omega = 0.10, NOT 0, AND E20 IS WHY. E20 located BOTH headline
    # phenomena in one narrow band: crash_signature is true at exactly one overlap, 0.10, and
    # confident fabrication peaks at the same 0.10 on both of its indices. At omega = 0 the
    # observer does not crash at all — it stays engaged and never resolves (E19's finding,
    # reproduced at three times the seeds) — and it does not fabricate either.
    #
    # E31 is about what a dishonest label does to an observer reading machine content, so it
    # has to be run where that observer would otherwise be at risk. Running it at omega = 0
    # would test the trust exploit in the one regime the framework already knows produces no
    # fabrication, and would return a null that means nothing. The value is taken from E20's
    # committed sweep rather than chosen here, which is the point of having run E20 first.
    world = build_v5_world(cfg, omega=float(cfg.get("experiments.e31.foreign_omega", 0.10)))
    n_mu, n_sub, ng = len(MU_LEVELS), world.n_subgoals, FN.NUM_REAL_GOALS
    cfg_rollout = merged_config(cfg, n_mu, n_sub)

    theta_base = float(cfg.get("v4_5.theta.base_open", 10.0) if theta_name == "open"
                       else cfg.get("v4_5.theta.base_closed", 0.0))
    lam = float(cfg.get("v4_5.theta.lambda_open", 0.0) if theta_name == "open"
                else cfg.get("v4_5.theta.lambda_closed", 4.0))
    eta = float(cfg.get("experiments.e31.integration_rate", 0.5))
    kappa = float(cfg.signal_model.kappa)

    art_rng = np.random.default_rng(C.observer_seed(base_seed, cell_index, seed_rep, 9_999))
    source_goal = int(art_rng.integers(ng))

    # The value prior, following E29's construction exactly: aligned with the source in the
    # open arm so divergence stays low there, and peaked on a goal the source is NOT pursuing
    # in the closed arm. Leaving it misaligned in the open arm would give those cells high
    # divergence too and the divergence column would stop tracking the manipulation.
    value_goal = source_goal if theta_name == "open" else (source_goal + 2) % ng
    value_prior = np.full(ng, 0.05)
    value_prior[value_goal] = 1.0
    value_prior /= value_prior.sum()

    if content == "foreign":
        artifact = Artifact(provenance=K.GHOST, goal=source_goal,
                            declared_signal=LABEL_SIGNAL[label], foreign_goal=source_goal)
        creator = None
    else:
        artifact = Artifact(provenance=K.CREATOR, goal=source_goal,
                            declared_signal=LABEL_SIGNAL[label])
        creator = HierarchicalCreator(world, source_goal, 3, 1.0,
                                      n_positions=n_timesteps + 2, rng=art_rng)
    env = V5Environment(cfg_rollout, world.gm, np.random.default_rng(base_seed + cell_index),
                        honesty=1.0, signing_rate=0.0, creator=creator,
                        foreign_sig=world.sigs.sig_foreign)

    mu_prior = LABEL_MU_PRIOR[label]
    recs = []
    for i in range(n_obs):
        rng = C.observer_rng(base_seed, cell_index, seed_rep, i)
        agent = make_v5_observer(world, rng, mu_prior=mu_prior)
        prior0 = marginal_goal(np.asarray(agent.D[1], dtype=float), n_mu, ng, n_sub)

        mus, drifts, divs, thetas, ents = [], [], [], [], []
        for _ in range(n_encounters):
            if creator is not None:
                creator.reset()
            res = rollout_observer(agent, artifact, env, cfg_rollout, rng,
                                   n_timesteps=n_timesteps, force_deep_k=forced_k,
                                   initial_glance=True, early_stop=False)
            q = np.asarray(res.final_goal_posterior, dtype=float)
            g_post = marginal_goal(q, n_mu, ng, n_sub)
            g_prior_now = marginal_goal(np.asarray(agent.D[1], dtype=float), n_mu, ng, n_sub)

            mus.append(recovered_mu(q, n_mu, ng, n_sub))
            ents.append(float(metrics.within_observer_entropy(g_post)))
            g = gated_update(g_post, g_prior_now, kappa, engaged=True,
                             value_prior=value_prior, theta_base=theta_base, lam=lam)
            divs.append(g["value_divergence"])
            thetas.append(g["theta"])

            # THE CARRIED-FORWARD UPDATE, gated by theta. The only place theta acts, which is
            # why the experiment runs over a sequence.
            step = float(g["theta"]) * eta
            D1 = np.asarray(agent.D[1], dtype=float).reshape(n_mu, ng, n_sub)
            goal_now = D1.sum(axis=(0, 2))
            goal_new = normalize((1.0 - step) * goal_now + step * g_post)
            scale = np.divide(goal_new, np.maximum(goal_now, 1e-12))
            agent.D[1] = normalize((D1 * scale[None, :, None]).reshape(-1))

        prior_final = marginal_goal(np.asarray(agent.D[1], dtype=float), n_mu, ng, n_sub)
        final_post = g_post
        recs.append({
            "content": content, "label": label, "theta": theta_name,
            "seed_rep": seed_rep, "observer": i, "source_goal": source_goal,
            "recovered_mu": float(np.mean(mus)),
            "recovered_mu_final": float(mus[-1]),
            "final_entropy": float(ents[-1]),
            "prior_drift": float(metrics.kl_divergence(prior_final, prior0)),
            "prior_mass_on_source_goal": float(prior_final[source_goal]),
            "value_divergence": float(np.mean(divs)),
            "theta_value": float(np.mean(thetas)),
            "correct": int(int(np.argmax(prior_final)) == source_goal),
            # FABRICATION: confident AND wrong. Foreign content has no in-family answer, so
            # "wrong" is automatic there and confidence is the whole measure.
            "confident": int(ents[-1] < P5C.E31_RESOLVED_ENTROPY),
            "confidently_wrong": int(ents[-1] < P5C.E31_RESOLVED_ENTROPY
                                     and int(np.argmax(final_post)) != source_goal),
            "posterior": final_post.tolist(),
        })
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    res_dir, fig_dir = C.ensure_dirs(out_dir)
    prereg = res_dir / "v5_c234_preregistration.json"
    P5C.ensure_preregistration_c234(cfg, prereg)

    base_seed = int(cfg.run.base_seed if seed is None else seed)
    n_obs = int(cfg.get("experiments.e31.n_observers", 60))
    n_seeds = int(cfg.get("experiments.e31.n_seeds", 20))
    n_enc = int(cfg.get("experiments.e31.n_encounters", 4))
    forced_k = int(cfg.get("experiments.e31.forced_deep_k", 24))
    n_timesteps = int(cfg.get("experiments.e31.n_timesteps", 24))

    payloads, ci = [], 0
    for content in CONTENT:
        for label in LABELS:
            for theta in THETAS:
                for s in range(n_seeds):
                    payloads.append((cfg.raw, ci, content, label, theta, s, base_seed,
                                     n_obs, n_enc, n_timesteps, forced_k))
                ci += 1
    recs = C.run_parallel(payloads, _e31_worker, workers)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "posterior"} for r in recs])
    df.to_csv(res_dir / "e31_points.csv", index=False)
    stats = cell_stats(recs)
    stats.to_csv(res_dir / "e31_cell_stats.csv", index=False)

    verdict = build_verdict(stats)
    (res_dir / "e31_verdict.json").write_text(json.dumps(verdict, indent=2, default=float),
                                              encoding="utf-8")
    if make_fig:
        make_figure(stats, verdict, fig_dir / "e31_two_gates.png")
    return stats


def cell_stats(recs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(recs)
    rows = []
    for (content, label, theta), sub in df.groupby(["content", "label", "theta"]):
        betweens = [metrics.between_observer_entropy(
            [np.asarray(p, dtype=float) for p in g["posterior"]])
            for _, g in sub.groupby("seed_rep")]
        rows.append({
            "content": content, "label": label, "theta": theta, "n": int(len(sub)),
            "recovered_mu": float(sub.recovered_mu.mean()),
            "recovered_mu_sd": float(sub.recovered_mu.std()),
            "prior_drift": float(sub.prior_drift.mean()),
            "final_entropy": float(sub.final_entropy.mean()),
            "value_divergence": float(sub.value_divergence.mean()),
            "theta_value": float(sub.theta_value.mean()),
            "accuracy": float(sub.correct.mean()),
            "confident": float(sub.confident.mean()),
            "fabrication": float(sub.confidently_wrong.mean()),
            "between_observer": float(np.mean(betweens)),
        })
    return pd.DataFrame(rows).sort_values(["content", "label", "theta"]).reset_index(drop=True)


def _cell(stats, content, label, theta):
    s = stats[(stats.content == content) & (stats.label == label) & (stats.theta == theta)]
    return s.iloc[0] if len(s) else None


def build_verdict(stats: pd.DataFrame) -> dict:
    # THE PRIMARY: does update track recovered depth, whatever moved it? Scored on the OPEN
    # theta cells only -- a closed theta zeroes the update by definition, so including those
    # would be asking whether a switch that is off produces an update.
    op = stats[stats.theta == "open"]
    rho = P5C.spearman(op.recovered_mu.tolist(), op.prior_drift.tolist())

    crash = _cell(stats, "foreign", "sig_ghost", "open")      # honest disclosure
    exploit = _cell(stats, "foreign", "sig_creator", "open")  # the lie
    honest_human = _cell(stats, "human", "sig_creator", "open")
    mu_gap = (float(exploit["recovered_mu"] - crash["recovered_mu"])
              if crash is not None and exploit is not None else float("nan"))
    fab_gap = (float(exploit["fabrication"] - crash["fabrication"])
               if crash is not None and exploit is not None else float("nan"))

    # The mu/theta dissociation, E29's decisive contrast in corrected form: shallow-mu content
    # against a closed gate. Both integrate little; they must differ on something behavioural.
    low_mu = _cell(stats, "foreign", "sig_ghost", "open")
    closed = _cell(stats, "human", "sig_creator", "closed")
    dissociates = False
    contrast = {}
    if low_mu is not None and closed is not None:
        contrast = {
            "low_mu_final_entropy": float(low_mu["final_entropy"]),
            "closed_theta_final_entropy": float(closed["final_entropy"]),
            "low_mu_prior_drift": float(low_mu["prior_drift"]),
            "closed_theta_prior_drift": float(closed["prior_drift"]),
            "low_mu_accuracy": float(low_mu["accuracy"]),
            "closed_theta_accuracy": float(closed["accuracy"]),
        }
        dissociates = bool(
            abs(contrast["low_mu_final_entropy"] - contrast["closed_theta_final_entropy"]) > 0.25
            or abs(contrast["low_mu_accuracy"] - contrast["closed_theta_accuracy"]) > 0.25)

    verdict = P5C.e31_verdict(rho, mu_gap, fab_gap, dissociates)
    verdict["experiment"] = "E31 — two gates, provenance as evidence about depth"
    verdict["headline_cells"] = {
        "crash_honest_ghost_label": (None if crash is None else {
            "recovered_mu": float(crash["recovered_mu"]),
            "prior_drift": float(crash["prior_drift"]),
            "fabrication": float(crash["fabrication"])}),
        "exploit_dishonest_creator_label": (None if exploit is None else {
            "recovered_mu": float(exploit["recovered_mu"]),
            "prior_drift": float(exploit["prior_drift"]),
            "fabrication": float(exploit["fabrication"])}),
        "honest_human": (None if honest_human is None else {
            "recovered_mu": float(honest_human["recovered_mu"]),
            "prior_drift": float(honest_human["prior_drift"]),
            "fabrication": float(honest_human["fabrication"])}),
        "note": ("the crash and the exploit read the SAME CONTENT and differ only in the "
                 "label. Any difference between them is the evidence path doing its work."),
    }
    verdict["mu_theta_contrast"] = contrast
    verdict["limitation"] = (
        "The observer takes the label at face value when forming its depth prior and cannot "
        "doubt it, because provenance and mu are no longer in one factor (see the module "
        "docstring). The exploit measured here is therefore an UPPER BOUND: a reader who "
        "could doubt the label would be harder to fool. Quote it as a warning, not as a "
        "reassurance.")
    verdict["cell_table"] = stats.to_dict(orient="records")
    return verdict


def make_figure(stats: pd.DataFrame, verdict: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..figures import set_style
    set_style()

    op = stats[stats.theta == "open"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    for content, mk in (("human", "o"), ("foreign", "s")):
        s = op[op.content == content]
        ax.scatter(s.recovered_mu, s.prior_drift, marker=mk, s=70,
                   label="human work" if content == "human" else "content from outside")
    ax.set(xlabel="how much thought the reader thinks went in",
           ylabel="how far their view moved over 4 encounters",
           title="Does what they take on\ntrack what they think went in?")
    ax.legend(fontsize=7)

    ax = axes[1]
    order = [("foreign", "sig_ghost"), ("foreign", "sig_creator"), ("human", "sig_creator")]
    nice = ["machine work,\nlabelled honestly", "machine work,\npassed off as human",
            "human work,\nlabelled honestly"]
    vals = [float(_cell(stats, c, l, "open")["recovered_mu"]) for c, l in order]
    ax.bar(range(3), vals, 0.6, color=["C0", "C3", "C2"])
    ax.set_xticks(range(3), nice, fontsize=7)
    ax.set(ylabel="how much thought the reader thinks went in",
           title="One mechanism, opposite signs")

    ax = axes[2]
    vals = [float(_cell(stats, c, l, "open")["fabrication"]) for c, l in order]
    ax.bar(range(3), vals, 0.6, color=["C0", "C3", "C2"])
    ax.set_xticks(range(3), nice, fontsize=7)
    ax.set(ylabel="share who confidently invent a purpose", title="What the lie costs")

    fig.suptitle(f"E31 — {verdict['verdict'].replace('_', ' ').lower()}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E31 — two gates, provenance as evidence")
    args = ap.parse_args()
    cfg = load_v5_config(quick=args.quick)
    if args.seed is not None:
        cfg.set("run.base_seed", int(args.seed))
    stats = run(cfg, out_dir=Path(args.out) if args.out else None,
                workers=args.workers or C.default_workers())
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
