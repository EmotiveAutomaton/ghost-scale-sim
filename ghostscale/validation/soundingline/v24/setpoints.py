"""Finite decaying-set-point construction for the V24 design card."""
import math

import numpy as np

from ..v23.common import normalize, rng

MECHANISMS = ("state_maintenance", "intrinsic_act", "cue_habit")


def settings():
    return [dict(decay=decay, restored=restored, alternative=alternative)
            for decay in (0., .2) for restored in (False, True)
            for alternative in (False, True)]


def probability(mechanism, row):
    deficit = max(0., row["setpoint"] - row["state"])
    if mechanism == "state_maintenance":
        utilities = np.array([0., deficit * row["effects"][1], deficit * row["effects"][2]])
    elif mechanism == "intrinsic_act":
        utilities = np.array([0., 1., 1. if row["alternative"] else -1.])
    elif mechanism == "cue_habit":
        utilities = np.array([0., 1. if row["cue"] == 1 else -.5, -.5])
    else:
        raise ValueError("unknown set-point mechanism")
    return normalize(4 * utilities)


def generate(setting, mechanism, identity, *, steps=64):
    if setting not in settings() or mechanism not in MECHANISMS or steps < 1:
        raise ValueError("frozen setting, mechanism, and positive steps required")
    random = rng("v24-setpoints", identity, setting, mechanism)
    state = 1. if setting["restored"] else .5
    history = []
    effects = [0., .35, .35 if setting["alternative"] else 0.]
    for step in range(steps):
        if setting["restored"] and step == steps // 2:
            state = 1.
        state = max(0., state - setting["decay"])
        row = dict(step=step, state=state, setpoint=1., cue=step % 2,
                   alternative=setting["alternative"], effects=effects)
        p = probability(mechanism, row)
        action = int(random.choice(3, p=p))
        row["action"] = action
        state = min(1.5, state + effects[action])
        row["next_state"] = state
        history.append(row)
    return dict(identity=identity, setting=setting, history=history), dict(mechanism=mechanism)


def posterior(public):
    log_mass = np.zeros(len(MECHANISMS))
    for row in public["history"]:
        for index, mechanism in enumerate(MECHANISMS):
            log_mass[index] += math.log(probability(mechanism, row)[row["action"]])
    return normalize(log_mass)


def controls():
    zero = dict(decay=0., restored=True, alternative=False)
    maintained, _ = generate(zero, "state_maintenance", "v24-engineering-excluded-setpoint-zero")
    intervention = dict(decay=.2, restored=True, alternative=True)
    planted, _ = generate(intervention, "state_maintenance", "v24-engineering-excluded-setpoint-recovery", steps=128)
    recovered = posterior(planted)
    return dict(
        zero_decay_null_maintenance=all(row["state"] >= .8 for row in maintained["history"]),
        restoration_intervention_present=any(row["step"] == 64 and np.isclose(row["state"], .8)
                                             for row in planted["history"]),
        equivalent_means_available=intervention["alternative"],
        planted_maintenance_recovered=bool(np.argmax(recovered) == 0 and recovered[0] > .99),
        settings=len(settings()), mechanisms=len(MECHANISMS),
    )
