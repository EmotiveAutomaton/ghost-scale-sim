"""Hierarchical and distributed production (trunks D and F).

D: a multi-part artifact produced under a named control ecology. Every part has a local
contributor with its own profile; a DIRECTOR may set the primary goal and the allocation of goals
to parts; an EDITOR may reorder parts; a RATIFIER may veto and re-draw a part; a SHARED BRIEF is a
common goal template with no live controller; INDEPENDENT contributors share nothing. Full logs
are kept so causal reach can be measured by intervention rather than inferred from the artifact.

F: layered vs flattened vs non-invertible decision topology. Layered: an upstream choice
constrains which downstream realization family is used (conditional dependency). Flattened: the
same marginals over upstream and downstream choices with no dependency. Non-invertible: the
surface is independent of the hidden hierarchy entirely. Matched on marginals and length so the
target difference is dependency topology, not more signal.
"""
from __future__ import annotations

import numpy as np

from .world import World, realization

ECOLOGIES = ("director", "shared_brief", "editor", "ratifier", "distributed", "independent")


def _part_emission(world: World, template: np.ndarray, g: int, slot: int) -> np.ndarray:
    base = template[g] / template[g].sum()
    return realization(world, base, g, slot)


def produce(world: World, ecology: str, contributors: list, rng, n_parts: int = 4,
            steps_per_part: int = 12, director_goal: int | None = None,
            intervene: dict | None = None) -> dict:
    """One multi-part artifact with a full decision log. ``intervene`` may override the
    director goal, one part's local choice, the final ratification, or the shared brief, so that
    causal reach can be measured as downstream change under intervention (D02)."""
    iv = intervene or {}
    ng = world.ng
    log = {"ecology": ecology, "parts": [], "director_goal": None, "brief": None,
           "editor_order": None, "vetoed": []}
    if ecology in ("director", "editor", "ratifier"):
        dg = int(rng.integers(ng)) if director_goal is None else int(director_goal)
        dg = int(iv.get("director_goal", dg))
        log["director_goal"] = dg
        # Director allocates goals to parts: the primary goal to most parts, a secondary to one.
        secondary = int(iv.get("secondary_goal", (dg + 1) % ng))
        alloc = [dg] * n_parts
        alloc[n_parts - 1] = secondary
    elif ecology == "shared_brief":
        brief = int(rng.integers(ng)) if director_goal is None else int(director_goal)
        brief = int(iv.get("brief", brief))
        log["brief"] = brief
        alloc = [brief] * n_parts
    else:
        alloc = [int(rng.integers(ng)) for _ in range(n_parts)]
    feats_parts, slots = [], []
    for i in range(n_parts):
        c = contributors[i % len(contributors)]
        g = alloc[i]
        if ecology == "distributed" and i > 0:
            # coordinate through the artifact: adopt the previous part's goal with prob 0.7
            g = alloc[i - 1] if rng.random() < 0.7 else g
            alloc[i] = g
        slot = int(rng.integers(len(world.family_names)))
        if "local_slot" in iv and iv.get("local_part", 0) == i:
            slot = int(iv["local_slot"])
        e = _part_emission(world, c.template, g, slot)
        a = world.alpha[c.tier]
        e = a * e + (1 - a) * world.synth
        e = e / e.sum()
        f = rng.choice(world.nf, size=int(steps_per_part), p=e)
        feats_parts.append(f)
        slots.append(slot)
        log["parts"].append({"contributor": c.id, "goal": int(g), "slot": slot})
    if ecology == "editor":
        order = list(rng.permutation(n_parts))
        log["editor_order"] = order
        feats_parts = [feats_parts[i] for i in order]
    if ecology == "ratifier":
        # veto parts whose goal is not the director's; redraw them under the director goal
        for i, p in enumerate(log["parts"]):
            if p["goal"] != log["director_goal"] and not iv.get("no_veto", False):
                log["vetoed"].append(i)
                c = contributors[i % len(contributors)]
                e = _part_emission(world, c.template, log["director_goal"], slots[i])
                a = world.alpha[c.tier]
                e = a * e + (1 - a) * world.synth
                e = e / e.sum()
                feats_parts[i] = rng.choice(world.nf, size=int(steps_per_part), p=e)
                p["goal"] = log["director_goal"]
                p["ratified"] = True
    return {"features": np.concatenate(feats_parts), "parts": feats_parts, "log": log,
            "alloc": alloc}


def part_goal_posteriors(world: World, template: np.ndarray, parts: list) -> np.ndarray:
    """Per-part posterior over goals from features alone (uniform prior)."""
    out = []
    for f in parts:
        ll = np.array([np.log(np.maximum(template[g][f], 1e-300)).sum() for g in range(world.ng)])
        v = np.exp(ll - ll.max())
        out.append(v / v.sum())
    return np.stack(out)


def coherence_features(world: World, template: np.ndarray, parts: list) -> dict:
    """Artifact-only statistics a coherence/style baseline would use (D03)."""
    P = part_goal_posteriors(world, template, parts)
    top = P.argmax(axis=1)
    return {"share_dominant_goal": float((top == np.bincount(top).argmax()).mean()),
            "mean_part_confidence": float(P.max(axis=1).mean()),
            "goal_entropy_across_parts": float(-(np.bincount(top, minlength=world.ng) / len(top)
                                                 * np.log(np.maximum(np.bincount(top, minlength=world.ng)
                                                                     / len(top), 1e-12))).sum())}


# --------------------------------------------------------------------------- #
# Layered / flattened / non-invertible emitters (trunk F).
# --------------------------------------------------------------------------- #
def layered_sequence(world: World, template: np.ndarray, rng, n_blocks: int = 6,
                     steps_per_block: int = 4, topology: str = "layered",
                     manipulation_on: bool = True) -> dict:
    """A sequence of blocks. Upstream choice u (the block's goal). Downstream choice v (the
    realization slot). Layered: v is drawn from a goal-specific conditional family. Flattened:
    v is drawn from the same marginal independently of u. Non-invertible: features do not
    depend on (u, v) at all beyond the marginal. The upstream/downstream marginals are matched
    across topologies by construction, so only the dependency differs."""
    ng, ns = world.ng, len(world.family_names)
    us, vs, feats = [], [], []
    # goal-specific conditional families: goal g prefers slots {g, g+1} (mod ns)
    for _ in range(n_blocks):
        u = int(rng.integers(ng))
        if topology == "layered" and manipulation_on:
            v = int((u + rng.integers(2)) % ns)
        else:
            # flattened: same marginal over v as layered's marginal, independent of u
            v = int((rng.integers(ng) + rng.integers(2)) % ns)
        if topology == "noninvertible":
            e = world.synth
        else:
            e = _part_emission(world, template, u, v)
        f = rng.choice(world.nf, size=int(steps_per_block), p=e / e.sum())
        us.append(u)
        vs.append(v)
        feats.append(f)
    return {"features": np.concatenate(feats), "blocks": feats, "u": us, "v": vs,
            "topology": topology}


def dependency_statistic(world: World, template: np.ndarray, blocks: list) -> float:
    """Mutual information between inferred upstream goal and inferred downstream slot across
    blocks, from features alone: the topology ruler (F03)."""
    ng, ns = world.ng, len(world.family_names)
    joint = np.zeros((ng, ns)) + 1e-9
    for f in blocks:
        ll = np.full((ng, ns), -np.inf)
        for g in range(ng):
            for s in range(ns):
                e = _part_emission(world, template, g, s)
                ll[g, s] = np.log(np.maximum(e[f], 1e-300)).sum()
        p = np.exp(ll - ll.max())
        joint += p / p.sum()
    joint /= joint.sum()
    pu = joint.sum(axis=1, keepdims=True)
    pv = joint.sum(axis=0, keepdims=True)
    return float((joint * np.log(joint / (pu @ pv))).sum())
