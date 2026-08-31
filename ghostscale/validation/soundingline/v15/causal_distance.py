"""The causal-distance audit (spec §6 I03, §8.3 clause 6).

Every V15 result has to answer one question before it is allowed to be called a discovery: *how far
did the information travel between the hidden state and the reader's answer?* Three distances, and
only the third earns a discovery claim:

``DIRECT_READOUT``
    the scored observation was generated from the hidden variable and read with the matching
    likelihood. V14's 90% fanatic/propagandist separation was this. It measures that the wiring is
    connected.
``PLANTED_SIGNATURE``
    the hidden variable was given a fixed, per-class surface marker that the reader can match
    against a table. V14's 99% acquisition-path classification was this. It measures that the
    marker survived the pipeline.
``INFERRED_THROUGH_BEHAVIOUR``
    the hidden variable shaped a *policy*, the policy produced behaviour, and the reader recovered
    it by modelling that behaviour. Only this is a claim about inference.

The audit is mechanical, not editorial. It is run against known-answer fixtures whose distance is
known by construction (``fixtures()``), and a card whose channel is not classified is refused.
"""
from __future__ import annotations

import numpy as np

from . import common as C

DISTANCES = ("DIRECT_READOUT", "PLANTED_SIGNATURE", "INFERRED_THROUGH_BEHAVIOUR")
#: A channel whose destroyed-by-shuffle drop is this large is doing more than carrying a marker.
BEHAVIOUR_FLOOR = 0.15


def classify(channel: dict) -> dict:
    """Classify one scored channel from three declared properties.

    ``generated_from_hidden``   the observation is a direct function of the hidden variable
    ``matching_likelihood``     the reader scores it with the generator's own likelihood
    ``fixed_class_marker``      the hidden variable maps to a constant per-class surface feature
    ``mediated_by_policy``      the hidden variable acts through a policy that produces behaviour
    """
    direct = bool(channel.get("generated_from_hidden")) and bool(channel.get("matching_likelihood"))
    marker = bool(channel.get("fixed_class_marker"))
    mediated = bool(channel.get("mediated_by_policy"))
    if direct and not mediated:
        d = "DIRECT_READOUT"
    elif marker and not mediated:
        d = "PLANTED_SIGNATURE"
    elif mediated:
        d = "INFERRED_THROUGH_BEHAVIOUR"
    else:
        d = "PLANTED_SIGNATURE"
    return {"channel": channel.get("name", "?"), "distance": d,
            "promotable_as_discovery": bool(d == "INFERRED_THROUGH_BEHAVIOUR"),
            "properties": {k: bool(channel.get(k)) for k in
                           ("generated_from_hidden", "matching_likelihood", "fixed_class_marker",
                            "mediated_by_policy")}}


def fixtures() -> list:
    """Known-answer fixtures. I03 passes only if ``classify`` reproduces every declared distance."""
    return [
        {"name": "v14_private_action", "generated_from_hidden": True, "matching_likelihood": True,
         "fixed_class_marker": False, "mediated_by_policy": False,
         "declared": "DIRECT_READOUT"},
        {"name": "v14_acquisition_signature", "generated_from_hidden": False,
         "matching_likelihood": False, "fixed_class_marker": True, "mediated_by_policy": False,
         "declared": "PLANTED_SIGNATURE"},
        {"name": "v15_next_action_from_policy", "generated_from_hidden": False,
         "matching_likelihood": False, "fixed_class_marker": False, "mediated_by_policy": True,
         "declared": "INFERRED_THROUGH_BEHAVIOUR"},
        {"name": "v15_probe_through_planner", "generated_from_hidden": True,
         "matching_likelihood": False, "fixed_class_marker": False, "mediated_by_policy": True,
         "declared": "INFERRED_THROUGH_BEHAVIOUR"},
        {"name": "v15_history_from_behaviour", "generated_from_hidden": False,
         "matching_likelihood": False, "fixed_class_marker": False, "mediated_by_policy": True,
         "declared": "INFERRED_THROUGH_BEHAVIOUR"},
        {"name": "leaky_route_token", "generated_from_hidden": True, "matching_likelihood": True,
         "fixed_class_marker": True, "mediated_by_policy": False,
         "declared": "DIRECT_READOUT"},
    ]


def audit_fixtures() -> dict:
    rows = []
    for f in fixtures():
        got = classify(f)
        rows.append({**got, "declared": f["declared"], "ok": got["distance"] == f["declared"]})
    return {"rows": rows, "n": len(rows), "n_ok": sum(r["ok"] for r in rows),
            "all_ok": all(r["ok"] for r in rows)}


def shuffle_probe(score_fn, shuffle_fn, rng, n: int = 24) -> dict:
    """The empirical half of the audit: destroy the *behavioural* structure and see what survives.

    A channel that is a marker keeps most of its score when the behaviour is shuffled but the
    marker is preserved. A channel that required modelling the behaviour collapses. Reported as the
    drop, so a card can show its distance rather than assert it.
    """
    intact = [float(score_fn(rng)) for _ in range(int(n))]
    broken = [float(shuffle_fn(rng)) for _ in range(int(n))]
    drop = float(np.mean(intact) - np.mean(broken))
    return {"intact": float(np.mean(intact)), "shuffled": float(np.mean(broken)),
            "drop": drop, "behaviour_dependent": bool(drop >= BEHAVIOUR_FLOOR),
            "floor": BEHAVIOUR_FLOOR, "n": int(n)}


def label_leak_probe(score_fn, permuted_score_fn, rng, n: int = 24, tol: float = 0.02) -> dict:
    """The no-label-leak gate: permuting a label the reader must not see may not move the score."""
    a = [float(score_fn(rng)) for _ in range(int(n))]
    b = [float(permuted_score_fn(rng)) for _ in range(int(n))]
    move = float(abs(np.mean(a) - np.mean(b)))
    return {"score": float(np.mean(a)), "permuted_score": float(np.mean(b)),
            "movement": move, "leaks": bool(move > tol), "tolerance": tol}


def distance_receipt(card_id: str, channels: list, empirical: dict | None = None) -> dict:
    """The block every substantive card writes into its verdict."""
    rows = [classify(c) for c in channels]
    worst = "INFERRED_THROUGH_BEHAVIOUR"
    for r in rows:
        if r["distance"] == "DIRECT_READOUT":
            worst = "DIRECT_READOUT"
            break
        if r["distance"] == "PLANTED_SIGNATURE":
            worst = "PLANTED_SIGNATURE"
    return {"card": card_id, "channels": rows, "limiting_distance": worst,
            "promotable_as_discovery": bool(worst == "INFERRED_THROUGH_BEHAVIOUR"),
            "empirical": empirical or {},
            "note": ("a card whose limiting distance is not INFERRED_THROUGH_BEHAVIOUR may still "
                     "land and may still be useful; it is capped at CONSTRUCTION_IDENTITY and "
                     "cannot be promoted as a simulator discovery")}
