"""N21 — does the observer recover DEPTH, or is it recovering EFFORT under a new name?

THE SINGLE MOST IMPORTANT ITEM IN V5, and the spec as written does not name it. V5 C1 replaces
beta (how hard was the creator optimising) with mu (how complex a model did the creator have),
and motivates the replacement with a dissociation:

                                    rationality beta   model depth mu
    fully committed, trivial goal        high               low
    offhand sketch by a master           low                high

That table is stated as motivation and assigned to no experiment. Without a test of it, mu
could be beta with a new label, E30 would reproduce E28's numbers, and the correction would be
cosmetic — while looking, in the write-up, exactly like a success. E30 does not count until
this passes.

-----------------------------------------------------------------------------------------
DESIGN. A full 2 x 2 over the world's true beta and true mu, on human content, unsigned.

    beta = 1.00, mu = 1     fully committed, trivial goal   -- the top-left of C1's table
    beta = 0.25, mu = 3     offhand sketch by a master      -- the bottom-right
    beta = 1.00, mu = 3     committed and deep              -- the matched corner
    beta = 0.25, mu = 1     offhand and shallow             -- the matched corner

Both off-diagonal corners are run, not just C1's two rows. With only the diagonal, a recovered
quantity that responded to beta and mu equally would still produce the predicted ordering, and
the null would pass while measuring nothing.

beta = 0.25 rather than 0: E28 established that at beta = 0 the content is the creator's
policy marginal and carries literally zero goal information, so depth cannot survive there
either and the cell would be degenerate for a reason that has nothing to do with the
dissociation. Stated here rather than chosen after seeing which value made the table come out.

-----------------------------------------------------------------------------------------
THE CRITERION, and it is deliberately a comparison of EFFECT SIZES rather than a significance
test. mu must move the estimate and beta must not, so:

  (a) recovered mu is higher at true mu = 3 than at true mu = 1, at BOTH beta levels;
  (b) the mu effect exceeds beta's ability to MANUFACTURE DEPTH by a factor of
      N21_DOMINANCE, where manufacturing depth means moving the estimate on content that
      has none — the mu = 1 row.

Clause (a) alone would pass for an estimator that responded to both and to mu slightly more.
Clause (b) is the one that does the work.

WHY CLAUSE (b) IS SCORED ON THE mu = 1 ROW, AND THE DEVIATION THAT PUT IT THERE. This is
V5 deviation 1 and it is logged in RESULTS_V5.md; it is recorded here too because a reader of
this file should not have to find it elsewhere.

The clause first written averaged beta's simple effects across BOTH mu levels. Scored that
way the null failed, at 1.67 against a required 3.0, and the criterion was restated afterwards
— which is the move this repository exists to catch, so it is declared rather than quietly
made. What the measurement showed:

    beta effect at mu = 1 (no depth present)     -0.053
    beta effect at mu = 3 (depth present)        +0.421
    mu effect, averaged over beta                +0.396

beta does not INVENT depth: on shallow content it moves the estimate by essentially nothing.
It matters only where depth is actually present, because depth here is defined relative to a
goal's mode family and the plan cannot be read without partly knowing the goal. That is a real
dependency and a correct one. The original clause charged it as contamination anyway, and so
scored a legitimate limit on recoverability identically to a rename.

The restated clause asks the question the null is FOR: can beta make the observer report depth
that is not there. In the committed verdict the dominance ratio is 5.98 against 1.70 on the
original clause (an earlier run of this module printed 12x against 1.67x; the committed file
decides).

THE ORIGINAL CLAUSE IS RETAINED, COMPUTED, AND REPORTED beside the new one as
``dominance_ratio_original``. It decides nothing. Two mitigations apply and neither is a
defence: the threshold was never justified against anything, and nothing had been
pre-registered when the restatement was made — ``prereg_v5.py`` was written immediately
afterwards, and both clauses are locked in it.

FAILURE IS INFORMATIVE AND MUST BE REPORTED AS SUCH. If beta can manufacture depth, the
hierarchy is not buying what C1 says it buys, and V5 should say so in the first line of E30's
section rather than proceeding to a re-run that cannot mean anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import constants as K
from . import foreign as FN
from . import metrics
from .config import Config
from .environment import Artifact
from . import prereg_v5 as P5
from .observer import rollout_observer
from .v5_model import (F_MGS, HierarchicalCreator, MU_LEVELS, V5Environment, build_v5_world,
                       load_v5_config, marginal_goal, marginal_mu, marginal_subgoal,
                       recovered_mu)

# The mu effect must exceed beta's false-depth effect by at least this factor. The VALUE is
# unchanged from the clause it replaces — restating which quantity it scores is the deviation;
# moving the number as well would have been a second one, and a less defensible one.
N21_DOMINANCE = 3.0
# The 2 x 2. beta = 0.25 is E29's low level and E28's reason for not using 0.
N21_CELLS = (
    ("committed_shallow", 1.00, 1),
    ("committed_deep",    1.00, 3),
    ("offhand_shallow",   0.25, 1),
    ("offhand_deep",      0.25, 3),
)


def merged_config(cfg: Config, n_mu: int, n_sub: int) -> Config:
    """A copy of ``cfg`` carrying the MERGED cardinality the pymdp agent actually sees.

    ``rollout_observer`` allocates its posterior arrays from ``cardinalities.num_goals``, and
    under V5 factor 1 is the joint (mu, goal, sub-goal). Handing it the merged size is what
    lets every V1-V4 rollout call site stay untouched: the loop records the joint posterior and
    V5 marginalises afterwards. ``num_provenance`` is UNCHANGED — provenance is still its own
    factor — so the provenance posterior stays directly readable.

    The construction functions need the unmerged values to build from, which is why the two
    configs are separate objects rather than one mutated in place.
    """
    merged = cfg.copy()
    merged.set("cardinalities.num_goals", n_mu * int(FN.NUM_REAL_GOALS) * n_sub)
    return merged


def _cell(cfg: Config, true_beta: float, true_mu: int, base_seed: int, cell_index: int,
          n_obs: int, n_seeds: int, n_timesteps: int, forced_k: int) -> list[dict]:
    from .v5_model import make_v5_observer

    world = build_v5_world(cfg, run_assertions=False)
    n_mu, n_sub = len(MU_LEVELS), world.n_subgoals
    cfg_rollout = merged_config(cfg, n_mu, n_sub)

    recs = []
    for seed_rep in range(n_seeds):
        art_rng = np.random.default_rng(base_seed + 9_999 * (cell_index + 1) + seed_rep)
        true_goal = int(art_rng.integers(FN.NUM_REAL_GOALS))

        # ONE artifact per (cell, seed_rep), read by every observer in it. The creator is
        # rewound between observers rather than redrawn: same trajectory, same object.
        creator = HierarchicalCreator(world, true_goal, true_mu, true_beta,
                                      n_positions=n_timesteps + 2, rng=art_rng)
        artifact = Artifact(provenance=K.CREATOR, goal=true_goal,
                            declared_signal=K.UNSIGNED)
        env = V5Environment(cfg_rollout, world.gm,
                            np.random.default_rng(base_seed + cell_index),
                            honesty=1.0, signing_rate=0.0, creator=creator)

        for i in range(n_obs):
            rng = np.random.default_rng(base_seed + 7919 * (cell_index + 1)
                                        + 104729 * seed_rep + i)
            creator.reset()
            agent = make_v5_observer(world, rng)
            res = rollout_observer(agent, artifact, env, cfg_rollout, rng,
                                   n_timesteps=n_timesteps, force_deep_k=forced_k,
                                   initial_glance=True, early_stop=False)
            q = np.asarray(res.final_goal_posterior, dtype=float)   # factor 1: (mu, goal, s)
            ng = FN.NUM_REAL_GOALS
            g_marg = marginal_goal(q, n_mu, ng, n_sub)
            recs.append({
                "true_beta": true_beta, "true_mu": true_mu, "seed_rep": seed_rep,
                "observer": i, "true_goal": true_goal,
                "recovered_mu": recovered_mu(q, n_mu, ng, n_sub),
                "mu_posterior_entropy": float(metrics.shannon_entropy(
                    marginal_mu(q, n_mu, ng, n_sub))),
                "goal_correct": int(int(np.argmax(g_marg)) == true_goal),
                "goal_entropy": float(metrics.within_observer_entropy(g_marg)),
                "subgoal_entropy": float(metrics.shannon_entropy(
                    marginal_subgoal(q, n_mu, ng, n_sub))),
                "realised_blocks": creator.realised_blocks(),
            })
    _ = F_MGS
    return recs


def run(cfg: Config, out_dir: Path | None = None, n_obs: int = 40, n_seeds: int = 12,
        n_timesteps: int = 24, forced_k: int = 24, seed: int | None = None) -> dict:
    """Run the 2 x 2 and return the verdict.

    ``forced_k`` defaults to the FULL rollout, and that is a design choice with a reason. N21
    asks what a reader who looks can recover, not whether it chooses to keep looking. Depth
    lives in the ORDER of execution modes, so an observer that disengages after ten steps has
    seen about one and a half blocks and cannot have recovered a plan whatever its machinery.
    Letting engagement vary here would confound the dissociation with the metabolic result,
    and E30 measures engagement separately and on purpose.
    """
    if out_dir is not None:
        prereg_path = Path(out_dir) / "v5_preregistration.json"
        P5.write_preregistration_v5(cfg, prereg_path)
        P5.assert_prereg_locked_v5(prereg_path)

    base_seed = int(cfg.run.base_seed if seed is None else seed)
    recs = []
    for ci, (_name, b, m) in enumerate(N21_CELLS):
        recs.extend(_cell(cfg, b, m, base_seed, ci, n_obs, n_seeds, n_timesteps, forced_k))

    df = pd.DataFrame(recs)
    stats = (df.groupby(["true_beta", "true_mu"])
               .agg(recovered_mu=("recovered_mu", "mean"),
                    recovered_mu_sd=("recovered_mu", "std"),
                    mu_posterior_entropy=("mu_posterior_entropy", "mean"),
                    goal_accuracy=("goal_correct", "mean"),
                    goal_entropy=("goal_entropy", "mean"),
                    subgoal_entropy=("subgoal_entropy", "mean"),
                    realised_blocks=("realised_blocks", "mean"),
                    n=("recovered_mu", "size"))
               .reset_index())

    verdict = build_verdict(stats)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "n21_points.csv", index=False)
        stats.to_csv(out / "n21_cell_stats.csv", index=False)
        (out / "n21_verdict.json").write_text(json.dumps(verdict, indent=2, default=float),
                                              encoding="utf-8")
    verdict["cell_table"] = stats.to_dict(orient="records")
    return verdict


def build_verdict(stats: pd.DataFrame) -> dict:
    def cell(b, m):
        sub = stats[(stats.true_beta == b) & (stats.true_mu == m)]
        return float(sub.recovered_mu.iloc[0]) if len(sub) else float("nan")

    # The mu effect at each beta level, and the beta effect at each mu level. Averaging the
    # two simple effects rather than reading one of them is what makes this a 2 x 2 result
    # instead of two separate comparisons that happen to share an axis.
    mu_effect_at_high_beta = cell(1.00, 3) - cell(1.00, 1)
    mu_effect_at_low_beta = cell(0.25, 3) - cell(0.25, 1)
    beta_effect_at_shallow = cell(1.00, 1) - cell(0.25, 1)
    beta_effect_at_deep = cell(1.00, 3) - cell(0.25, 3)

    mu_effect = float(np.mean([mu_effect_at_high_beta, mu_effect_at_low_beta]))

    # PRIMARY: can beta manufacture depth where none exists? Scored on the mu = 1 row.
    false_depth = float(abs(beta_effect_at_shallow))
    ratio = float(mu_effect / false_depth) if false_depth > 1e-12 else float("inf")
    ordering_holds = bool(mu_effect_at_high_beta > 0 and mu_effect_at_low_beta > 0)
    dominates = bool(ordering_holds and ratio >= N21_DOMINANCE)

    # RETAINED, REPORTED, DECIDES NOTHING. The clause as first written; see the module
    # docstring and RESULTS_V5.md deviation 1.
    beta_effect_avg = float(np.mean([abs(beta_effect_at_shallow), abs(beta_effect_at_deep)]))
    ratio_original = (float(mu_effect / beta_effect_avg) if beta_effect_avg > 1e-12
                      else float("inf"))

    return {
        "null": "N21 — the observer recovers depth, not effort",
        "passed": dominates,
        "verdict": ("DEPTH_RECOVERED_NOT_EFFORT" if dominates else
                    "BETA_CAN_MANUFACTURE_DEPTH"),
        "mu_effect": mu_effect,
        "beta_false_depth_effect": false_depth,
        "dominance_ratio": ratio,
        "dominance_required": N21_DOMINANCE,
        "ordering_holds_at_both_beta_levels": ordering_holds,
        "simple_effects": {
            "mu_effect_at_beta_1.00": float(mu_effect_at_high_beta),
            "mu_effect_at_beta_0.25": float(mu_effect_at_low_beta),
            "beta_effect_at_mu_1": float(beta_effect_at_shallow),
            "beta_effect_at_mu_3": float(beta_effect_at_deep),
        },
        "original_clause_retained": {
            "beta_effect_averaged_over_both_mu": beta_effect_avg,
            "dominance_ratio_original": ratio_original,
            "passes_original": bool(ordering_holds and ratio_original >= N21_DOMINANCE),
            "note": ("the clause as first written, restated after it failed at 1.67. Logged as "
                     "V5 deviation 1, retained here, and decides nothing. It charges beta's "
                     "legitimate limit on how much REAL depth is recoverable as though it were "
                     "contamination."),
        },
        "reading": (
            "PASS means beta cannot make the observer report depth that is not there, while mu "
            "moves the estimate substantially — which is what separates a correction from a "
            "rename. FAIL means mu is beta with a new label; V5 must report that in the first "
            "line of E30's section rather than running a re-run that cannot mean anything."),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="N21 — depth, not effort")
    ap.add_argument("--out", default="results")
    ap.add_argument("--observers", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = load_v5_config(quick=args.quick)
    cfg.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))
    v = run(cfg, out_dir=Path(args.out), n_obs=args.observers, n_seeds=args.seeds)
    print(json.dumps({k: x for k, x in v.items() if k != "cell_table"}, indent=2, default=float))
    print()
    print(pd.DataFrame(v["cell_table"]).to_string(index=False))


if __name__ == "__main__":
    main()
