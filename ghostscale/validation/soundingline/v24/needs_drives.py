"""Finite needs-versus-drives construction for the V24 design card.

This is a generator/reader implementation, not an admitted pilot.  Public
interventions name deficits and action consequences; planted mechanism labels
remain in evaluator truth only.
"""
import math

import numpy as np

from ..v23.common import normalize, rng

MECHANISMS = ("need_a", "need_b", "intrinsic_action", "cue_habit")
SATIATIONS = ("none", "a", "b")


def settings():
    return [dict(satiated=satiated, consequence_swap=swap, informative=informative)
            for satiated in SATIATIONS for swap in (False, True)
            for informative in (False, True)]


def action_probability(mechanism, row):
    if mechanism not in MECHANISMS:
        raise ValueError("unknown need/drive mechanism")
    if not row["informative"]:
        return np.array([.5, .5])
    if mechanism.startswith("need_"):
        need = mechanism[-1]
        if row["satiated"] == need:
            return np.array([.5, .5])
        utilities = np.array([row["consequences"][str(action)][need] for action in range(2)], float)
        return normalize(4 * utilities)
    if mechanism == "intrinsic_action":
        return np.array([.9, .1])
    return np.array([.85, .15]) if row["cue"] == 0 else np.array([.15, .85])


def generate(setting, mechanism, identity, *, trials=48):
    if setting not in settings() or mechanism not in MECHANISMS or trials < 1:
        raise ValueError("frozen setting, mechanism, and positive trials required")
    random = rng("v24-needs-drives", identity, setting, mechanism)
    normal = {"0": {"a": 1., "b": 0.}, "1": {"a": 0., "b": 1.}}
    swapped = {"0": normal["1"], "1": normal["0"]}
    history = []
    for trial in range(trials):
        row = dict(trial=trial, cue=trial % 2, satiated=setting["satiated"],
                   informative=setting["informative"],
                   consequences=swapped if setting["consequence_swap"] else normal)
        probability = action_probability(mechanism, row)
        row["action"] = int(random.choice(2, p=probability))
        history.append(row)
    public = dict(identity=identity, setting=setting, history=history)
    truth = dict(mechanism=mechanism)
    return public, truth


def posterior(public, prior=None):
    prior = np.ones(len(MECHANISMS)) / len(MECHANISMS) if prior is None else np.asarray(prior, float)
    if prior.shape != (len(MECHANISMS),) or np.any(prior <= 0) or not np.isclose(prior.sum(), 1):
        raise ValueError("positive normalized mechanism prior required")
    log_mass = np.log(prior)
    for row in public["history"]:
        for index, mechanism in enumerate(MECHANISMS):
            probability = action_probability(mechanism, row)
            if row["action"] not in (0, 1):
                raise ValueError("invalid public action")
            log_mass[index] += math.log(probability[row["action"]])
    return normalize(log_mass)


def controls():
    baseline = dict(satiated="none", consequence_swap=False, informative=False)
    public, _ = generate(baseline, "need_a", "v24-engineering-excluded-needs-alias")
    alias = posterior(public, prior=np.array([.5, .5, 1e-9, 1e-9]))
    informative = dict(satiated="a", consequence_swap=False, informative=True)
    planted, _ = generate(informative, "need_b", "v24-engineering-excluded-needs-recovery", trials=128)
    recovered = posterior(planted)
    public_keys = set(planted) | set(planted["history"][0])
    return dict(
        baseline_observational_alias=bool(np.isclose(alias[0], alias[1], atol=1e-12)),
        selective_satiation_known_answer=bool(np.argmax(recovered) == MECHANISMS.index("need_b")
                                               and recovered[MECHANISMS.index("need_b")] > .99),
        public_consequence_evaluator_separation="mechanism" not in public_keys,
        settings=len(settings()), mechanisms=len(MECHANISMS),
    )
