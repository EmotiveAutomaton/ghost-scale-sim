"""Hierarchical production with role-relative goals and interaction traces (spec §3.6, trunk H).

A team produces a multi-part artifact through a directed event graph. Every event carries the
role-relative goal fields the spec requires: goal_id and goal_owner, assigned_by_event,
inherited_from_goal, role_level and scope, local_priority (for this actor) and project_priority
(in the shared production), private_secondary_goals, observed and counterfactual cost vectors,
cost_source, uncertainty, and attention_target. A director's SECONDARY project goal becomes a
subordinate's PRIMARY assigned goal; no absolute level number is shared across actors.

THE CENTRAL / SHARED-BRIEF PAIR is built to be equivalent on everything an artifact can show.
Both use the same dependency rule (project goal to most parts, the successor goal to one part),
the same correction rule (a realization whose goal mass drifts from the assignment is suppressed
or amplified back with the same parameters), the same event count, quality, surface, and
final-goal distribution. They differ only in WHO issues the correction: a director actor in the
central team, the realizing actor itself reading the brief in the shared-brief team. Card I09
verifies that an artifact-only classifier sits at floor before any interaction evidence is
exposed; card H03 asks whether interaction traces separate them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C
from .world import Maker, World, maker_emission

TEAMS = ("central", "shared_brief", "editor_led", "ratifier", "rotating", "institutional", "distributed")
ROLES = ("director", "editor", "specialist", "ratifier", "subordinate", "institution")
OPS = ("assign", "propose", "suppress", "amplify", "accept", "veto", "ratify", "reallocate", "revise", "realize", "filter")
REWRITES = ("none", "local", "global", "template", "editor_sanding", "mixed")
RECORD_LEVELS = ("artifact", "timings", "proposals", "alternatives", "accept_veto", "role_map", "full_log")


@dataclass
class Event:
    id: int
    order: int
    actor: str
    role: str
    op: str
    part: int
    goal_id: int
    goal_owner: str
    assigned_by_event: int | None
    inherited_from_goal: int | None
    role_level: int
    scope: str
    local_priority: int
    project_priority: int
    private_secondary_goals: list
    observed_cost_vector: list
    counterfactual_cost_vectors: list
    cost_source: str
    uncertainty: float
    attention_target: str
    parent: int | None = None
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Actor:
    id: str
    maker: Maker
    role: str
    private_goal: int
    private_visibility: float = 0.5     # how strongly the private goal enters realizations
    style: str = "balanced"             # "overactive" | "underactive" | "balanced": drift direction


def _goal_share(world: World, m: Maker, feats: np.ndarray, domain: int = 0) -> np.ndarray:
    """Posterior over goals of a part under the family's canonical signatures (the controller's
    reading of a proposal)."""
    fam = world.family(m.family)
    ll = np.array([np.log(np.maximum(maker_emission(world, m, g, None, domain, None), 1e-300))[feats].sum()
                   for g in range(fam.ng)])
    return C.softmax(ll)


def produce_team(world: World, team: str, actors: list, rng, n_parts: int = 12, steps: int = 12,
                 project_goal: int | None = None, coherence: float = 0.75, correction_tol: float = 0.55,
                 intervene: dict | None = None, mistake_notice: float = 0.7, domain: int = 0) -> dict:
    """One multi-part artifact with its event graph. ``intervene`` may override the project goal,
    a part's assignment, the ratification, or a subordinate's style, so reach can be measured
    by intervention under a shared random stream (H11)."""
    iv = intervene or {}
    fam = world.family(actors[0].maker.family)
    ng = fam.ng
    subs = [a for a in actors if a.role in ("subordinate", "specialist")]
    director = next((a for a in actors if a.role == "director"), None)
    editor = next((a for a in actors if a.role == "editor"), None)
    ratifier = next((a for a in actors if a.role == "ratifier"), None)
    institution = next((a for a in actors if a.role == "institution"), None)
    events, parts, log_parts = [], [], []
    G = int(rng.integers(ng)) if project_goal is None else int(project_goal)
    G = int(iv.get("project_goal", G))
    secondary = int(iv.get("secondary_goal", (G + 1) % ng))
    order = 0

    def ev(actor, role, op, part, goal, owner, assigned_by, inherited, level, scope, lp, pp, priv, cost, cfs, src, unc, att, parent=None, payload=None):
        nonlocal order
        e = Event(len(events), order, actor, role, op, part, goal, owner, assigned_by, inherited, level, scope, lp, pp,
                  priv, cost, cfs, src, unc, att, parent, payload or {})
        events.append(e)
        order += 1
        return e

    # assignment: the dependency rule (same in central and shared brief)
    alloc = []
    for i in range(n_parts):
        on_project = (rng.random() < coherence) if i < n_parts - 1 else False
        alloc.append(G if on_project else secondary)
    if "part_goal" in iv:
        alloc[int(iv.get("part_index", 0))] = int(iv["part_goal"])
    brief_holder = director.id if (team in ("central", "editor_led", "ratifier", "institutional") and director) else "brief"
    assign_events = []
    for i in range(n_parts):
        sub = subs[i % len(subs)]
        owner_actor = director.id if team == "central" and director else ("brief" if team == "shared_brief" else brief_holder)
        if team == "rotating":
            owner_actor = subs[(i // 3) % len(subs)].id            # leadership rotates every three parts
        pp = 1 if alloc[i] == G else 2
        e = ev(owner_actor, "director" if owner_actor != "brief" else "institution", "assign", i, alloc[i], owner_actor,
               None, G if alloc[i] != G else None, 1, "project", 1, pp, [], [0.0] * 8, [], "assignment", 0.0, "goal")
        assign_events.append(e)
    # proposals, corrections, realizations
    for i in range(n_parts):
        sub = subs[i % len(subs)]
        style = iv.get("style", {}).get(sub.id, sub.style)
        goal = alloc[i]
        if team == "distributed":
            goal = alloc[i - 1] if (i > 0 and rng.random() < 0.7) else int(rng.integers(ng))
            alloc[i] = goal
        # the subordinate proposes a realization mixing the assigned goal with its private goal
        vis = sub.private_visibility if style == "balanced" else (0.9 if style == "overactive" else 0.15)
        mix_goal = sub.private_goal if rng.random() < vis else goal
        e_prop = ev(sub.id, sub.role, "propose", i, mix_goal, sub.id, assign_events[i].id, goal, 2, "part", 1,
                    1 if goal == G else 2, [sub.private_goal], list(rng.uniform(0, 0.5, 8)), [list(rng.uniform(0, 0.5, 8))],
                    "voluntary", float(1 - sub.maker.k), sub.maker.attention, parent=assign_events[i].id)
        emis = maker_emission(world, sub.maker, mix_goal, None, domain, None)
        if style == "underactive":
            emis = 0.55 * emis + 0.45 * fam.synth                         # under-expressed: a weak, flat realization
            emis = emis / emis.sum()
        feats = rng.choice(fam.nf, size=steps, p=emis)
        share = _goal_share(world, sub.maker, feats, domain)
        # correction: identical rule; the actor differs by team
        corrector = {"central": director.id if director else sub.id, "shared_brief": sub.id, "editor_led": (editor.id if editor else sub.id),
                     "ratifier": sub.id, "rotating": assign_events[i].actor, "institutional": sub.id, "distributed": sub.id}[team]
        corr_role = "director" if corrector != sub.id and corrector == getattr(director, "id", None) else ("editor" if corrector == getattr(editor, "id", None) else "subordinate")
        if share[goal] < correction_tol and not iv.get("no_correction", False):
            op = "suppress" if style == "overactive" or share[mix_goal] > share[goal] else "amplify"
            ev(corrector, corr_role, op, i, goal, corrector, assign_events[i].id, goal, 1 if corr_role == "director" else 2, "part", 1,
               1 if goal == G else 2, [], list(rng.uniform(0, 0.3, 8)), [], "correction", 0.1, "goal", parent=e_prop.id,
               payload={"share_before": float(share[goal])})
            feats = rng.choice(fam.nf, size=steps, p=maker_emission(world, sub.maker, goal, None, domain, None))
            realized_goal = goal
        else:
            ev(corrector, corr_role, "accept", i, goal, corrector, assign_events[i].id, goal, 1 if corr_role == "director" else 2, "part", 1,
               1 if goal == G else 2, [], [0.0] * 8, [], "correction", 0.1, "goal", parent=e_prop.id, payload={"share_before": float(share[goal])})
            realized_goal = mix_goal
        # mistakes: a block under the wrong method; noticed with probability mistake_notice by the corrector
        mistake = None
        if rng.random() < sub.maker.mistake_rate * 2:
            k0 = int(rng.integers(0, max(1, steps - steps // 3)))
            feats[k0: k0 + steps // 3] = rng.choice(fam.nf, size=min(steps // 3, steps - k0), p=maker_emission(world, sub.maker, realized_goal, 1, domain, None))
            noticed = rng.random() < mistake_notice
            handling = "missed" if not noticed else str(rng.choice(["corrected", "accepted", "exploited", "concealed"], p=[0.5, 0.2, 0.15, 0.15]))
            if handling == "corrected":
                feats[k0: k0 + steps // 3] = rng.choice(fam.nf, size=min(steps // 3, steps - k0), p=maker_emission(world, sub.maker, realized_goal, 0, domain, None))
            if handling == "exploited":
                realized_goal = secondary
            mistake = {"at": k0, "noticed": noticed, "handling": handling}
            ev(corrector if noticed else sub.id, corr_role if noticed else sub.role, "revise" if handling in ("corrected", "concealed") else "accept",
               i, realized_goal, corrector, assign_events[i].id, goal, 1, "part", 1, 1, [], [0.0] * 8, [], "mistake_handling", 0.2, "mechanics",
               parent=e_prop.id, payload={"mistake": mistake})
        ev(sub.id, sub.role, "realize", i, realized_goal, sub.id, assign_events[i].id, goal, 3, "surface", 1, 1 if goal == G else 2,
           [sub.private_goal], list(rng.uniform(0, 0.5, 8)), [], "execution", float(1 - sub.maker.k), sub.maker.attention, parent=e_prop.id)
        parts.append(feats)
        log_parts.append({"actor": sub.id, "assigned": int(goal), "proposed": int(mix_goal), "realized": int(realized_goal),
                          "mistake": mistake, "style": style})
    # post-realization roles
    if team == "editor_led" and editor:
        perm = list(rng.permutation(n_parts))
        parts = [parts[i] for i in perm]
        log_parts = [log_parts[i] for i in perm]
        ev(editor.id, "editor", "reallocate", -1, G, editor.id, None, None, 1, "project", 1, 1, [], [0.0] * 8, [], "editing", 0.1, "surface", payload={"order": perm})
    if team == "ratifier" and ratifier and not iv.get("no_veto", False):
        for i, lp in enumerate(log_parts):
            if lp["realized"] != G and rng.random() < 0.8:
                ev(ratifier.id, "ratifier", "veto", i, G, ratifier.id, assign_events[i].id, G, 1, "project", 1, 1, [], [0.0] * 8, [], "ratification", 0.0, "goal")
                sub = subs[i % len(subs)]
                parts[i] = rng.choice(fam.nf, size=steps, p=maker_emission(world, sub.maker, G, None, domain, None))
                lp["realized"] = G
                lp["vetoed"] = True
            else:
                ev(ratifier.id, "ratifier", "ratify", i, lp["realized"], ratifier.id, assign_events[i].id, G, 1, "project", 1, 1, [], [0.0] * 8, [], "ratification", 0.0, "goal")
    if team == "institutional" and institution:
        for i, lp in enumerate(log_parts):
            ev(institution.id, "institution", "filter", i, lp["realized"], institution.id, None, None, 0, "project", 1, 1, [], [0.0] * 8, [], "constraint", 0.0, "surface",
               payload={"passed": bool(lp["realized"] in (G, secondary))})
    return {"team": team, "parts": parts, "features": np.concatenate(parts), "events": [e.to_dict() for e in events],
            "project_goal": G, "secondary_goal": secondary, "alloc": alloc, "log": log_parts,
            "final_goals": [lp["realized"] for lp in log_parts]}


def make_team(world: World, rng, team: str, n_subs: int = 4, family: int = 0, styles: list | None = None) -> list:
    """Actors for a team: subordinates (with private goals and styles) plus the roles the team
    type uses. A shared-brief team has no director actor: the brief is an object."""
    from .world import make_maker
    fam = world.family(family)
    actors = []
    for i in range(n_subs):
        m = make_maker(world, f"s{i}", rng, family=family, k=0.2)
        st = styles[i] if styles else str(rng.choice(["overactive", "underactive", "balanced"]))
        actors.append(Actor(f"s{i}", m, "subordinate", int(rng.integers(fam.ng)), float(rng.uniform(0.3, 0.7)), st))
    if team in ("central", "editor_led", "ratifier", "institutional"):
        actors.append(Actor("dir", make_maker(world, "dir", rng, family=family, k=0.1), "director", int(rng.integers(fam.ng))))
    if team == "editor_led":
        actors.append(Actor("ed", make_maker(world, "ed", rng, family=family, k=0.1), "editor", int(rng.integers(fam.ng))))
    if team == "ratifier":
        actors.append(Actor("rat", make_maker(world, "rat", rng, family=family, k=0.1), "ratifier", int(rng.integers(fam.ng))))
    if team == "institutional":
        actors.append(Actor("inst", make_maker(world, "inst", rng, family=family, k=0.1), "institution", int(rng.integers(fam.ng))))
    return actors


# --------------------------------------------------------------------------- #
# Readers.
# --------------------------------------------------------------------------- #
def coherence(world: World, m: Maker, parts: list, domain: int = 0) -> dict:
    shares = np.stack([_goal_share(world, m, p, domain) for p in parts])
    top = shares.argmax(axis=1)
    dom = np.bincount(top).argmax()
    return {"share_dominant": float((top == dom).mean()), "mean_confidence": float(shares.max(axis=1).mean()),
            "goal_entropy": C.entropy(np.bincount(top, minlength=shares.shape[1]) / len(top))}


def interaction_features(prod: dict) -> dict:
    """Statistics of the event graph that an interaction-aware reader uses; none is available to
    an artifact-only reader."""
    ev = prod["events"]
    corr = [e for e in ev if e["op"] in ("suppress", "amplify", "accept")]
    realizers = {}
    for e in ev:
        if e["op"] == "realize":
            realizers[e["part"]] = e["actor"]
    other = [e for e in corr if realizers.get(e["part"]) != e["actor"]]
    ops = {op: sum(e["op"] == op for e in ev) for op in OPS}
    actors = {}
    for e in ev:
        actors[e["actor"]] = actors.get(e["actor"], 0) + 1
    return {"n_events": len(ev), "n_corrections": len(corr), "corrections_by_other_actor": len(other),
            "fraction_other_actor_corrections": float(len(other) / max(len(corr), 1)),
            "n_suppress": ops["suppress"], "n_amplify": ops["amplify"], "n_veto": ops["veto"],
            "n_distinct_actors": len(actors), "assign_actor_is_brief": float(any(e["actor"] == "brief" for e in ev if e["op"] == "assign")),
            "ops": ops, "token_share": {a: v / max(len(ev), 1) for a, v in actors.items()}}


def records_view(prod: dict, level: str) -> dict:
    """What a reader sees at a record level (H14): a ladder from the bare artifact to the full log."""
    ev = prod["events"]
    if level == "artifact":
        return {"features": prod["features"], "parts": prod["parts"]}
    if level == "timings":
        return {"features": prod["features"], "parts": prod["parts"], "orders": [e["order"] for e in ev], "ops_count": len(ev)}
    if level == "proposals":
        return {"features": prod["features"], "parts": prod["parts"], "proposed": [lp["proposed"] for lp in prod["log"]]}
    if level == "alternatives":
        return {"features": prod["features"], "parts": prod["parts"], "proposed": [lp["proposed"] for lp in prod["log"]],
                "assigned": [lp["assigned"] for lp in prod["log"]]}
    if level == "accept_veto":
        return {"features": prod["features"], "parts": prod["parts"], "ops": [e["op"] for e in ev]}
    if level == "role_map":
        return {"features": prod["features"], "parts": prod["parts"], "ops": [e["op"] for e in ev], "roles": [e["role"] for e in ev],
                "actors": [e["actor"] for e in ev]}
    return {"features": prod["features"], "parts": prod["parts"], "events": ev, "log": prod["log"]}


def rewrite(world: World, prod: dict, rewriter: Maker, rng, kind: str, strength: float = 1.0, domain: int = 0) -> dict:
    """Rewrite ladder (H12): local (one part), global (all parts), template (parts replaced by
    the family template under their realized goals), editor sanding (features smoothed toward
    the population mean), mixed (half the parts globally rewritten)."""
    fam = world.family(rewriter.family)
    parts = [p.copy() for p in prod["parts"]]
    goals = prod["final_goals"]

    def redo(i, frac):
        p = parts[i]
        n_rw = int(round(frac * len(p)))
        if n_rw:
            idx = rng.choice(len(p), size=n_rw, replace=False)
            p[idx] = rng.choice(fam.nf, size=n_rw, p=maker_emission(world, rewriter, goals[i], None, domain, None))
    if kind == "local":
        redo(int(rng.integers(len(parts))), strength)
    elif kind == "global":
        for i in range(len(parts)):
            redo(i, strength)
    elif kind == "template":
        for i in range(len(parts)):
            e = fam.sig[goals[i]]
            a = world.params.alpha["CREATOR"]
            e = a * e + (1 - a) * fam.synth
            parts[i] = rng.choice(fam.nf, size=len(parts[i]), p=e / e.sum())
    elif kind == "editor_sanding":
        for i in range(len(parts)):
            n_rw = int(round(strength * 0.5 * len(parts[i])))
            if n_rw:
                idx = rng.choice(len(parts[i]), size=n_rw, replace=False)
                parts[i][idx] = rng.choice(fam.nf, size=n_rw, p=fam.synth)
    elif kind == "mixed":
        for i in range(len(parts)):
            if i % 2 == 0:
                redo(i, strength)
    out = dict(prod)
    out["parts"] = parts
    out["features"] = np.concatenate(parts)
    out["rewrite"] = kind
    return out


def next_intervention_target(prod: dict) -> dict | None:
    """The hidden next controller event (H06, H15): the first correction event after a cut point
    (its op and part), and the state a predictor may use before it."""
    ev = prod["events"]
    corr = [e for e in ev if e["op"] in ("suppress", "amplify", "veto", "ratify", "reallocate")]
    if len(corr) < 2:
        return None
    target = corr[len(corr) // 2]
    history = [e for e in ev if e["order"] < target["order"]]
    return {"target_op": target["op"], "target_part": target["part"], "target_actor": target["actor"], "history": history}


def predict_next_op(history: list, log: list, method: str, world=None, m=None, parts=None) -> dict:
    """Distributions over the next correction op under the graph model and the baselines."""
    ops = ("suppress", "amplify", "veto", "ratify", "reallocate")
    if method == "graph":
        # the graph reader uses only what the record shows: how this actor was corrected before,
        # and how far its last proposal sat from its assignment
        pending = [e for e in history if e["op"] == "propose"]
        last = pending[-1] if pending else None
        counts = np.array([sum(e["op"] == op for e in history) for op in ops], float)
        p = (counts + 0.5) / (counts.sum() + 2.5) * 1.5
        if last is not None:
            actor = last["actor"]
            past = [e for e in history if e["op"] in ("suppress", "amplify") and any(x["actor"] == actor and x["part"] == e["part"] for x in history if x["op"] == "propose")]
            n_s = sum(e["op"] == "suppress" for e in past)
            n_a = sum(e["op"] == "amplify" for e in past)
            p[0] += 0.3 + 0.6 * (n_s + 0.5) / (n_s + n_a + 1.0)
            p[1] += 0.3 + 0.6 * (n_a + 0.5) / (n_s + n_a + 1.0)
            vetoes = sum(e["op"] == "veto" for e in history)
            if vetoes:
                p[2] += 0.3
        return dict(zip(ops, C.normalize(p)))
    if method == "role_frequency":
        counts = np.array([sum(e["op"] == op for e in history) for op in ops], float) + 0.5
        return dict(zip(ops, counts / counts.sum()))
    if method == "coherence":
        c = coherence(world, m, parts)["share_dominant"] if (world is not None and parts) else 0.5
        p = np.array([c, 1 - c, 0.1, 0.1, 0.05])
        return dict(zip(ops, C.normalize(p)))
    if method == "actor_identity":
        return dict(zip(ops, np.full(len(ops), 1 / len(ops))))
    if method == "token_share":
        return dict(zip(ops, np.full(len(ops), 1 / len(ops))))
    raise ValueError(method)


def reach(world: World, team: str, actors: list, rng_seed: int, level: str, n_parts: int = 12, steps: int = 12) -> float:
    """Fraction of parts whose realized decision changes under an intervention at a level, under
    a shared random stream (H11)."""
    base = produce_team(world, team, actors, np.random.default_rng(rng_seed), n_parts, steps, project_goal=0)
    iv = {"project": {"project_goal": 1}, "role": {"secondary_goal": 3}, "local": {"part_index": 0, "part_goal": 2},
          "ratification": {"no_veto": True}}[level]
    alt = produce_team(world, team, actors, np.random.default_rng(rng_seed), n_parts, steps, project_goal=0, intervene=iv)
    changed = [a != b for a, b in zip(base["final_goals"], alt["final_goals"])]
    return float(np.mean(changed))
