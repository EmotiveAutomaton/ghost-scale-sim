"""Uncertainty- and residual-triggered model-space expansion (spec §3.4.8, trunk M).

The reader starts with a *reduced* model space and may add latent variables, candidate policies or
earlier timesteps from a finite, predeclared symbolic proposal library. No language model is
involved: the point is to isolate hypothesis management from language generation, and to leave
Sounding Line a bridge it can later cross with a language-proposed library instead.

The two failures this module has to be able to exhibit, or the cards measuring them are theatre:

* **false expansion** -- firing on ordinary observation noise, which is what M04 separates from a
  genuinely missing variable, and what attack X20 tries to induce;
* **endless probing** -- expanding forever on a misspecified but learnable-looking item, which is
  what M11's expected-predictive-value selector is supposed to stop and F06 scores.

Two selectors are implemented and compared, never blended:

``residual``
    expand when the predictive surprise of the current model exceeds a threshold. Cheap, and the
    one that fires on noise.
``expected_value``
    expand when the *expected held-out predictive gain* of the proposal exceeds its search cost.
    Costlier per decision, and the one M11 predicts is worth the cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C
from . import exact as EX
from .ontology import COMPONENTS

#: The finite predeclared library. Every proposal is symbolic and enumerable; nothing is generated.
PROPOSAL_LIBRARY = (
    {"id": "add_tendency", "kind": "variable", "component": "tendency", "cost": 2.0},
    {"id": "add_goal", "kind": "variable", "component": "goal", "cost": 2.0},
    {"id": "add_process", "kind": "variable", "component": "process", "cost": 2.0},
    {"id": "add_distractor_a", "kind": "variable", "component": "distractor_a", "cost": 2.0},
    {"id": "add_distractor_b", "kind": "variable", "component": "distractor_b", "cost": 2.0},
    {"id": "add_timesteps_2", "kind": "timesteps", "k": 2, "cost": 1.0},
    {"id": "add_timesteps_4", "kind": "timesteps", "k": 4, "cost": 2.0},
    {"id": "add_policy_variant", "kind": "policy", "cost": 1.5},
)
#: Which proposals can never help, by construction. Accepting one is a false expansion.
DISTRACTORS = frozenset({"add_distractor_a", "add_distractor_b", "add_policy_variant"})


@dataclass
class ExpansionRecord:
    proposed: list = field(default_factory=list)
    accepted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    components: tuple = ("process", "goal")
    timesteps: int = 0
    likelihood_calls: int = 0

    def summary(self, truly_missing=()) -> dict:
        acc = set(self.accepted)
        missing = {f"add_{c}" for c in truly_missing}
        true_pos = sorted(acc & missing)
        false_pos = sorted(a for a in acc if a in DISTRACTORS
                           or (a.startswith("add_") and a not in missing and
                               a.replace("add_", "") in COMPONENTS))
        return {"proposed": list(self.proposed), "accepted": sorted(acc),
                "rejected": sorted(set(self.rejected)),
                "n_proposed": len(self.proposed), "n_accepted": len(acc),
                "true_expansions": true_pos, "false_expansions": false_pos,
                "true_expansion_rate": float(len(true_pos) / max(len(missing), 1)),
                "false_expansion_count": len(false_pos),
                "recall": float(len(true_pos) / max(len(missing), 1)),
                "precision": float(len(true_pos) / max(len(acc), 1)) if acc else float("nan"),
                "final_components": list(self.components), "added_timesteps": int(self.timesteps),
                "likelihood_calls": int(self.likelihood_calls)}


def _restricted_posterior(F, world, ep, upto, components, budget=None, model=None) -> np.ndarray:
    """Posterior over the full grid from a reader whose model contains only ``components``.

    A component the reader does not model is marginalized under its marginal prior rather than
    inferred -- that is exactly what "my model has no such variable" means, and it is why a missing
    variable shows up as residual structure rather than as a wrong value.
    """
    lg = EX.loglik_table(F, world, ep, upto, budget=budget, model=model)
    fp = EX.factorized_prior(F, world, model)
    full = EX.log_prior_table(F, world, model)
    keep = [i for i, c in enumerate(COMPONENTS) if c in components]
    post = C.softmax((lg + full).ravel()).reshape(lg.shape)
    for i, c in enumerate(COMPONENTS):
        if c in components:
            continue
        # replace the inferred marginal of an unmodelled component with its prior marginal
        axes = tuple(a for a in range(3) if a != i)
        prior_m = fp.sum(axis=axes)
        cur = post.sum(axis=axes)
        sh = [1, 1, 1]
        sh[i] = post.shape[i]
        post = post * (prior_m / np.maximum(cur, 1e-300)).reshape(sh)
    return C.normalize(post.ravel()).reshape(post.shape)


def _predictive_score(F, world, ep, post, endpoint, truth, budget=None) -> float:
    d = EX.predictive(F, world, ep, post, endpoint, budget)
    return C.log_score(d, truth)


def run_expansion(F, world, ep, upto: int, endpoint: str, *, selector: str = "expected_value",
                  start=("process", "goal"), truly_missing=(), rng=None, budget=None,
                  residual_threshold: float = 0.35, value_margin: float = 0.02,
                  cost_per_evaluation: float = 0.004, calibration=None) -> dict:
    """Run one expansion episode and return the record plus the final prediction.

    ``calibration`` is a held-out set of ``(episode, truth)`` pairs from the *same* world used to
    estimate a proposal's expected predictive gain. It never contains the scored episode and never
    contains latent labels, so an accepted proposal is earned on held-out prediction, not on fit.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    rec = ExpansionRecord(components=tuple(start))
    b0 = budget.likelihood if budget is not None else 0

    post = _restricted_posterior(F, world, ep, upto, rec.components, budget)
    base_pred = EX.predictive(F, world, ep, post, endpoint, budget)

    for prop in PROPOSAL_LIBRARY:
        if prop["kind"] != "variable":
            continue
        comp = prop["component"]
        if comp in rec.components:
            continue
        rec.proposed.append(prop["id"])
        trial = tuple(list(rec.components) + [comp]) if comp in COMPONENTS else rec.components

        if selector == "residual":
            # fire on surprise alone: the cheap rule, and the one that cannot tell a missing
            # variable from a noisy observation
            resid = C.entropy(base_pred) / np.log(len(base_pred))
            accept = bool(resid > residual_threshold and comp in COMPONENTS)
            if comp not in COMPONENTS:
                accept = bool(resid > residual_threshold)      # distractors look the same to it
        else:
            if trial == rec.components:
                accept = False                                  # a distractor adds no real variable
                gain = 0.0
            else:
                gains = []
                for cep, ctruth in (calibration or []):
                    p_old = _restricted_posterior(F, world, cep, upto, rec.components, budget)
                    p_new = _restricted_posterior(F, world, cep, upto, trial, budget)
                    gains.append(_predictive_score(F, world, cep, p_new, endpoint, ctruth, budget)
                                 - _predictive_score(F, world, cep, p_old, endpoint, ctruth, budget))
                gain = float(np.mean(gains)) if gains else 0.0
            spend = prop["cost"] * cost_per_evaluation
            accept = bool(gain - spend > value_margin)
            if budget is not None:
                budget.prop(1)

        if accept and comp in COMPONENTS:
            rec.components = trial
            rec.accepted.append(prop["id"])
            post = _restricted_posterior(F, world, ep, upto, rec.components, budget)
            base_pred = EX.predictive(F, world, ep, post, endpoint, budget)
        elif accept:
            rec.accepted.append(prop["id"])                     # a distractor was taken on board
        else:
            rec.rejected.append(prop["id"])

    rec.likelihood_calls = (budget.likelihood - b0) if budget is not None else 0
    out = rec.summary(truly_missing)
    out["prediction"] = base_pred
    out["posterior"] = post
    out["selector"] = selector
    return out
