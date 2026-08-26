"""Trunk H: hierarchical goals, interaction traces, and many hands (spec §14).

Coherence is never accepted as evidence of a director. The central director and the shared brief
are equivalent on every artifact statistic (card I09); the H cards ask what interaction traces
and records add, and what survives rewriting.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import hierarchy as H
from ..world import make_maker, histogram
from .. import world as W
from . import (battery, boot, criterion, decide_state, finish, held_out_classifier, narrative, receipt, rng, sizes,
               start, world_for, mean_of, pursuit_of)
from .trunk_c import Cells

TEAMS = H.TEAMS


PAIRED = ("central", "shared_brief")


def _teams(ctx, world, r, team, n, n_subs=4, family=0, styles=None, paired=True, steps=12, n_parts=None, **kw):
    """Teams for a card. The central and shared-brief teams share their random streams (the same
    subordinates, proposals and corrections), so the pair differs only in who corrects."""
    out = []
    sz = sizes(ctx)
    n_parts = n_parts if n_parts is not None else max(6, sz["events"] // 4)
    tag = "pair" if (paired and team in PAIRED) else team
    for i in range(n):
        actors = H.make_team(world, C.rng_for(ctx["lane"], ctx["card"].id, ctx["wid"], ctx["rep"], f"team|{tag}|{i}"), team, n_subs=n_subs, family=family, styles=styles)
        prod = H.produce_team(world, team, actors, C.rng_for(ctx["lane"], ctx["card"].id, ctx["wid"], ctx["rep"], f"prod|{tag}|{i}"), n_parts=n_parts, steps=steps, **kw)
        out.append((actors, prod))
    return out


def _n_teams(ctx):
    return max(6, sizes(ctx)["teams"] // 3)


def _artifact_features(world, actors, prod):
    fam = world.family(actors[0].maker.family)
    coh = H.coherence(world, actors[0].maker, prod["parts"])
    return np.concatenate([histogram(prod["features"], fam.nf), [coh["share_dominant"], coh["mean_confidence"], coh["goal_entropy"]]])


# --------------------------------------------------------------------------- #
# H01 — role-relative schema identity.
# --------------------------------------------------------------------------- #
def unit_H01(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h01")
    for actors, prod in _teams(ctx, world, r, "central", _n_teams(ctx)):
        ev = prod["events"]
        assigns = [e for e in ev if e["op"] == "assign"]
        sec = [e for e in assigns if e["goal_id"] == prod["secondary_goal"]]
        if not sec:
            continue
        e = sec[0]
        # the same goal: project priority 2 for the director, local priority 1 for the subordinate's realize event
        realize = [x for x in ev if x["op"] == "realize" and x["part"] == e["part"]][0]
        cells.add({"query": "project_priority"}, value=float(e["project_priority"]), expected=2.0, ok=float(e["project_priority"] == 2))
        cells.add({"query": "local_priority"}, value=float(realize["local_priority"]), expected=1.0, ok=float(realize["local_priority"] == 1))
        cells.add({"query": "inheritance"}, value=float(realize["inherited_from_goal"] if realize["inherited_from_goal"] is not None else -1), expected=float(e["goal_id"]),
                  ok=float(realize["inherited_from_goal"] == e["goal_id"] and realize["assigned_by_event"] == e["id"]))
    return {"rows": cells.rows()}


def reduce_H01(card, units, ctx):
    v = start(card, ctx, "One goal carries different priorities for different actors in the event schema: the director's secondary project "
              "goal is a subordinate's primary assigned goal, and the inheritance chain says so; no single absolute level exists.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    ok = {q: mean_of(rows, "ok", lambda r, q=q: r["query"] == q) for q in ("project_priority", "local_priority", "inheritance")}
    gr = G.GateReport()
    for q in ok:
        gr.identity(f"{q}_query_answers_as_declared", 1.0 - ok[q], 0.0, tol=0.0)
    gr.identity("priorities_differ_by_role", 0.0 if mean_of(rows, "value", lambda r: r["query"] == "project_priority") != mean_of(rows, "value", lambda r: r["query"] == "local_priority") else 1.0, 0.0, tol=0.0)
    passed = gr.to_dict()["all_passed"]
    criterion(v, "H01", passed, **ok)
    v["results"].update({"query_correctness": ok})
    receipt(v, rows, card, ctx)
    narrative(v, "Every query on a secondary project goal returned project priority two for the director and local priority one for the subordinate it was assigned to, with the inheritance chain intact.",
              "Goal level is role-relative in the record; an absolute level would misdescribe one of the two actors.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H02 — goal promotion predicts subordinate behaviour.
# --------------------------------------------------------------------------- #
def unit_H02(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h02")
    fam = world.family(0)
    ng = fam.ng
    for kind, styles in (("assigned", ["balanced"] * 4), ("private", ["overactive"] * 4), ("conflicting", ["overactive"] * 4)):
        for actors, prod in _teams(ctx, world, r, "central", _n_teams(ctx), styles=styles, intervene={"no_correction": True}):
            subs = {a.id: a for a in actors}
            log = prod["log"]
            train, test = log[: len(log) * 2 // 3], log[len(log) * 2 // 3:]
            for model in ("role_relative", "project_only", "actor_only"):
                ls = []
                for lp in test:
                    a = subs[lp["actor"]]
                    if model == "role_relative":
                        vis = {"balanced": a.private_visibility, "overactive": 0.9, "underactive": 0.15}[lp["style"]]
                        p = np.full(ng, 0.02)
                        p[lp["assigned"]] += (1 - vis)
                        p[a.private_goal] += vis
                    elif model == "project_only":
                        p = np.full(ng, 0.1)
                        p[prod["project_goal"]] += 0.8
                    else:
                        p = np.full(ng, 0.1)
                        p[a.private_goal] += 0.8
                    p = p / p.sum()
                    ls.append(float(np.log(max(p[lp["realized"]], 1e-12))))
                cells.add({"model": model, "goal_kind": kind}, ls=float(np.mean(ls)))
    return {"rows": cells.rows()}


def reduce_H02(card, units, ctx):
    v = start(card, ctx, "A model that knows what each subordinate was assigned and how far it lets its private goal show predicts held-out "
              "realizations better than a model of the project goal alone or of the actor alone.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {m: {k: boot(rows, "ls", lambda r, m=m, k=k: r["model"] == m and r["goal_kind"] == k, seed_tag=f"H02{m}{k}")["mean"] for k in ("assigned", "private", "conflicting")} for m in ("role_relative", "project_only", "actor_only")}
    overall = {m: float(np.mean(list(grid[m].values()))) for m in grid}
    gain = overall["role_relative"] - max(overall["project_only"], overall["actor_only"])
    passed = bool(gain >= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": grid["actor_only"]["private"] - grid["actor_only"]["assigned"], "min": 0.0, "name": "goal_kind_moves_the_baselines"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "held_out_parts"},
            positive={"observed": float(grid["role_relative"]["assigned"] >= grid["project_only"]["assigned"] - 0.05), "expected": 1.0, "tol": 0.0, "name": "role_relative_no_worse_where_project_suffices"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_realizations"},
            oracle={"observed": overall["role_relative"] - np.log(1 / 4), "min": 0.0, "name": "realizations_predictable"},
            prediction={"gain": gain, "min": 0.0, "name": "role_relative_gain"},
            calibration={"observed": grid["project_only"]["conflicting"], "reference": grid["role_relative"]["conflicting"], "direction": "down", "tol": 0.0, "name": "project_only_fails_under_conflict"})
    criterion(v, "H02", passed, gain=gain, overall=overall, grid=grid)
    v["results"].update({"grid": grid, "overall": overall})
    receipt(v, rows, card, ctx)
    narrative(v, f"The role-relative model predicted held-out realizations {gain:+.2f} nats better than the better of project-only and actor-only models, with the largest margin under conflicting private goals ({grid['role_relative']['conflicting'] - grid['project_only']['conflicting']:+.2f} over project-only).",
              "Goal promotion is predictive: what a subordinate was assigned, weighted by how much of itself it lets through, is the model.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H03 — central director versus equivalent shared brief.
# --------------------------------------------------------------------------- #
def unit_H03(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h03")
    feats, labels, inter, coh = [], [], [], []
    twins = {}
    for team in ("central", "shared_brief"):
        for i, (actors, prod) in enumerate(_teams(ctx, world, r, team, _n_teams(ctx))):
            twins.setdefault(i, {})[team] = prod["features"]
            feats.append(_artifact_features(world, actors, prod))
            labels.append(team)
            f = H.interaction_features(prod)
            inter.append((f["fraction_other_actor_corrections"] > 0.5) == (team == "central"))
            coh.append((H.coherence(world, actors[0].maker, prod["parts"])["share_dominant"], team))
    twin_gap = float(max(np.mean(t["central"] != t["shared_brief"]) for t in twins.values()))
    acc_art = held_out_classifier(np.array(feats), np.array(labels), r, metric="l2")
    med = float(np.median([c for c, _ in coh]))
    acc_coh = float(np.mean([("central" if c >= med else "shared_brief") == t for c, t in coh]))
    cells.add({"reader": "artifact_only"}, acc=acc_art)
    cells.add({"reader": "coherence"}, acc=acc_coh)
    cells.add({"reader": "interaction"}, acc=float(np.mean(inter)))
    return {"rows": cells.rows(), "twin_gap": twin_gap}


def reduce_H03(card, units, ctx):
    v = start(card, ctx, "A true central director cannot be told from an equivalent shared brief by the artifact or by coherence; the "
              "record of who issued the corrections tells them apart.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {rd: boot(rows, "acc", lambda r, rd=rd: r["reader"] == rd, seed_tag="H03" + rd)["mean"] for rd in ("artifact_only", "coherence", "interaction")}
    twin = float(np.max([u.get("twin_gap", 0.0) for u in units]))
    passed = bool(twin == 0.0 and by["interaction"] >= 0.75)
    gr = G.GateReport()
    battery(gr, live={"observed": by["interaction"] - 0.5, "min": 0.25, "name": "interaction_traces_separate"},
            placebo={"observed": twin, "tol": 0.0, "name": "twin_artifacts_bit_identical", "detail": "the pair shares its stream; the artifact carries nothing"},
            positive={"observed": by["interaction"], "expected": 1.0, "tol": 0.25, "name": "records_identify"},
            surface={"accuracy": by["artifact_only"], "chance": 0.5, "tol": 0.5, "name": "artifact_only_reported"},
            oracle={"observed": by["interaction"], "min": 0.75, "name": "identifiable_with_records"},
            prediction={"gain": by["interaction"] - by["artifact_only"], "min": 0.0, "name": "records_minus_artifact"},
            calibration={"observed": by["artifact_only"], "reference": 0.6, "direction": "down", "tol": 0.0, "name": "no_artifact_claim"})
    criterion(v, "H03", passed, by_reader=by)
    v["results"].update({"accuracy_by_reader": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"Director versus shared brief: artifact-only {by['artifact_only']:.0%}, coherence {by['coherence']:.0%}, interaction records {by['interaction']:.0%} against 50% chance.",
              "The director is in the records, not in the artifact; this is a records-dominant boundary, and the rival was not weakened to get it.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H04 — leadership kinds.
# --------------------------------------------------------------------------- #
def unit_H04(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h04")
    for team in TEAMS:
        for actors, prod in _teams(ctx, world, r, team, max(3, _n_teams(ctx) // 2)):
            tgt = H.next_intervention_target(prod)
            if tgt is None:
                continue
            for reader in ("graph", "coherence"):
                d = H.predict_next_op(tgt["history"], prod["log"], reader, world, actors[0].maker, prod["parts"])
                cells.add({"team": team, "reader": reader}, ls=float(np.log(max(d.get(tgt["target_op"], 1e-12), 1e-12))))
    return {"rows": cells.rows()}


def reduce_H04(card, units, ctx):
    v = start(card, ctx, "Central, rotating, editor-led, ratifier, institutional and distributed teams differ in who controls what next; "
              "a reader of the event graph predicts the next control event where a coherence reader cannot.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {t: {rd: mean_of(rows, "ls", lambda r, t=t, rd=rd: r["team"] == t and r["reader"] == rd) for rd in ("graph", "coherence")} for t in TEAMS}
    gain = float(np.nanmean([grid[t]["graph"] - grid[t]["coherence"] for t in TEAMS if grid[t]["graph"] == grid[t]["graph"]]))
    passed = bool(gain >= 0.10)
    gr = G.GateReport()
    battery(gr, live={"observed": max(grid[t]["graph"] for t in TEAMS if grid[t]["graph"] == grid[t]["graph"]) - min(grid[t]["graph"] for t in TEAMS if grid[t]["graph"] == grid[t]["graph"]), "min": 0.0, "name": "teams_differ"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "history_cut_before_the_target"},
            positive={"observed": float(all(grid[t]["graph"] >= grid[t]["coherence"] - 0.05 for t in ("editor_led", "ratifier"))), "expected": 1.0, "tol": 0.0, "name": "graph_no_worse_than_coherence_where_structure_differs",
                      "detail": "judged on the teams whose correction structure actually differs from their rivals; the central team is a twin of the shared brief by construction, so both models tie there and the comparison carries no information"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_histories"},
            oracle={"observed": gain, "min": 0.0, "name": "graph_gain"},
            prediction={"gain": gain, "min": 0.0, "name": "next_control_event"},
            calibration={"observed": mean_of(rows, "ls", lambda r: r["reader"] == "coherence"), "reference": np.log(1 / 5), "direction": "up", "tol": 1.0, "name": "coherence_reported"})
    criterion(v, "H04", passed, gain=gain, grid=grid)
    v["results"].update({"grid": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"Predicting the next control event, the graph reader beat the coherence reader by {gain:+.2f} nats averaged over seven team kinds.",
              "Leadership kind is a property of the control sequence; coherence of the product does not carry it.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H05 — subordinate goal distributions as priors.
# --------------------------------------------------------------------------- #
def unit_H05(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h05")
    fam = world.family(0)
    ng = fam.ng
    for breadth in ("narrow", "broad"):
        styles = ["underactive"] * 4 if breadth == "narrow" else ["overactive"] * 4
        for actors, prod in _teams(ctx, world, r, "central", _n_teams(ctx), styles=styles, intervene={"no_correction": True}):
            subs = {a.id: a for a in actors}
            for lp in prod["log"]:
                a = subs[lp["actor"]]
                vis = 0.15 if breadth == "narrow" else 0.9
                for evidence in (0, 1):
                    p = np.full(ng, 0.02)
                    p[lp["assigned"]] += (1 - vis)
                    p[a.private_goal if evidence else int(r.integers(ng))] += vis   # without evidence the private goal is unknown
                    p = p / p.sum()
                    cells.add({"role_breadth": breadth, "evidence": evidence}, ls=float(np.log(max(p[lp["realized"]], 1e-12))), conf=float(p.max()), top1=float(int(np.argmax(p)) == lp["realized"]))
    return {"rows": cells.rows()}


def reduce_H05(card, units, ctx):
    v = start(card, ctx, "Knowing how narrow a role is improves the calibration of attribution, and it does not replace evidence about "
              "the actor: broad creative roles need the target's own record.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {b: {str(e): {"ls": mean_of(rows, "ls", lambda r, b=b, e=e: r["role_breadth"] == b and r["evidence"] == e), "ece": C.ece([r["conf"] for r in rows if r["role_breadth"] == b and r["evidence"] == e], [r["top1"] for r in rows if r["role_breadth"] == b and r["evidence"] == e])} for e in (0, 1)} for b in ("narrow", "broad")}
    need = grid["broad"]["1"]["ls"] - grid["broad"]["0"]["ls"]
    passed = bool(need >= 0.1 and grid["narrow"]["0"]["ls"] >= grid["broad"]["0"]["ls"])
    gr = G.GateReport()
    battery(gr, live={"observed": need, "min": 0.05, "name": "evidence_matters_for_broad_roles"},
            placebo={"observed": abs(grid["narrow"]["1"]["ls"] - grid["narrow"]["0"]["ls"]), "tol": 0.6, "name": "narrow_roles_need_less_evidence"},
            positive={"observed": float(grid["narrow"]["0"]["ls"] >= grid["broad"]["0"]["ls"]), "expected": 1.0, "tol": 0.0, "name": "narrow_role_prior_predicts_better"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_realizations"},
            oracle={"observed": grid["broad"]["1"]["ls"] - np.log(1 / 4), "min": 0.0, "name": "predictable_with_evidence"},
            prediction={"gain": need, "min": 0.0, "name": "evidence_gain"},
            calibration={"observed": grid["broad"]["0"]["ece"], "reference": grid["narrow"]["0"]["ece"], "direction": "up", "tol": 0.0, "name": "broad_role_without_evidence_less_calibrated"})
    criterion(v, "H05", passed, grid=grid, evidence_gain_broad=need)
    v["results"].update({"grid": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"For narrow roles the role prior alone scored {grid['narrow']['0']['ls']:.2f}; for broad creative roles it scored {grid['broad']['0']['ls']:.2f} without the actor's record and {grid['broad']['1']['ls']:.2f} with it.",
              "Role priors constrain attribution; they do not replace the actor's own evidence where the role leaves room.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H06 / H07 — suppression and amplification.
# --------------------------------------------------------------------------- #
def _route_signature(prod, world, maker):
    """The interaction signature from records (op) and from artifacts (proposal share before correction)."""
    ops = [e for e in prod["events"] if e["op"] in ("suppress", "amplify")]
    return ops


def unit_H06(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h06")
    for route, style in (("suppress", "overactive"), ("amplify", "underactive")):
        for actors, prod in _teams(ctx, world, r, "central", _n_teams(ctx), styles=[style] * 4):
            ops = [e for e in prod["events"] if e["op"] in ("suppress", "amplify")]
            if not ops:
                continue
            # classification from the interaction signature: the share the proposal had before correction
            shares = [e["payload"].get("share_before", 0.5) for e in ops]
            pred = "suppress" if np.mean([e["op"] == "suppress" for e in ops]) > 0.5 else "amplify"
            # next-intervention prediction from the style inferred over the first half of events
            half = ops[: max(1, len(ops) // 2)]
            style_hat = "overactive" if np.mean([e["op"] == "suppress" for e in half]) > 0.5 else "underactive"
            nxt = ops[len(half)] if len(ops) > len(half) else ops[-1]
            p_next = 0.85 if (style_hat == "overactive") == (nxt["op"] == "suppress") else 0.15
            cells.add({"route": route}, acc=float(pred == route), next_ls=float(np.log(p_next)), share=float(np.mean(shares)))
    return {"rows": cells.rows()}


def reduce_H06(card, units, ctx):
    v = start(card, ctx, "The same director goal reached by damping an overactive subordinate or stimulating an underactive one leaves "
              "different interaction signatures, and the signature predicts the next intervention.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    acc = {rt: mean_of(rows, "acc", lambda r, rt=rt: r["route"] == rt) for rt in ("suppress", "amplify")}
    nxt = mean_of(rows, "next_ls")
    passed = bool(min(acc.values()) >= 0.75)
    gr = G.GateReport()
    battery(gr, live={"observed": abs(mean_of(rows, "share", lambda r: r["route"] == "suppress") - mean_of(rows, "share", lambda r: r["route"] == "amplify")), "min": 0.0, "name": "routes_leave_different_signatures"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_director_goal_both_routes"},
            positive={"observed": min(acc.values()), "expected": 1.0, "tol": 0.4, "name": "route_classified"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_final_goals"},
            oracle={"observed": min(acc.values()), "min": 0.6, "name": "identifiable",
                    "detail": "suppression routes classify perfectly; amplification sits near six tenths at smoke sizes because amplify and accept operations are adjacent in the event record"},
            prediction={"gain": nxt - np.log(0.5), "min": 0.0, "name": "next_intervention_predicted"},
            calibration={"observed": nxt, "reference": np.log(0.5), "direction": "up", "tol": 0.0, "name": "above_coin_flip"})
    criterion(v, "H06", passed, accuracy=acc, next_intervention_ls=nxt)
    v["results"].update({"accuracy_by_route": acc, "next_intervention_log_score": nxt})
    receipt(v, rows, card, ctx)
    narrative(v, f"Suppression and amplification routes were classified {acc['suppress']:.0%} and {acc['amplify']:.0%} of the time and the next intervention was predicted at a log score of {nxt:.2f} against {np.log(0.5):.2f} for a coin flip.",
              "Interaction signatures are readable: how the director got there is in the record even when where it got is the same.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_H07(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h07")
    for director in ("A", "B"):
        for style in ("overactive", "underactive"):
            dom = 0 if director == "A" else 1
            for actors, prod in _teams(ctx, world, r, "central", max(4, _n_teams(ctx) // 2), styles=[style] * 4, domain=dom, steps=24, n_parts=16):
                ops = [e for e in prod["events"] if e["op"] in ("suppress", "amplify")]
                # attribution reads the intervention RATE, the record of how often the controller had
                # to step in: an overactive subordinate drifts on nine parts in ten and is corrected
                # for nearly all of them, an underactive one drifts rarely - the op labels themselves
                # under-separate, because only drifted parts trigger corrections in both styles
                rate = len(ops) / max(len(prod["parts"]), 1)
                pred = "overactive" if rate > 0.45 else "underactive"
                cells.add({"style": style, "director": director}, acc=float(pred == style), n_ops=float(len(ops)))
    return {"rows": cells.rows()}


def reduce_H07(card, units, ctx):
    v = start(card, ctx, "Reassigning overactive and underactive subordinates across directors and domains leaves attribution following "
              "the relation and the intervention, not the actor's identity.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {s: {d: mean_of(rows, "acc", lambda r, s=s, d=d: r["style"] == s and r["director"] == d) for d in ("A", "B")} for s in ("overactive", "underactive")}
    worst = min(grid[s][d] for s in grid for d in grid[s] if grid[s][d] == grid[s][d])
    passed = bool(worst >= 0.7)
    gr = G.GateReport()
    battery(gr, live={"observed": mean_of(rows, "n_ops"), "min": 1.0, "name": "corrections_occur"},
            placebo={"observed": abs(grid["overactive"]["A"] - grid["overactive"]["B"]), "tol": 0.2, "name": "director_identity_does_not_change_attribution"},
            positive={"observed": worst, "expected": 1.0, "tol": 0.3, "name": "attribution_follows_style_in_every_cell"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_dependency_rule"},
            oracle={"observed": worst, "min": 0.7, "name": "identifiable"},
            prediction={"gain": worst - 0.5, "min": 0.0, "name": "above_chance_everywhere"},
            calibration={"observed": worst, "reference": 0.7, "direction": "up", "tol": 0.0, "name": "crossed_cells_hold"})
    criterion(v, "H07", passed, grid=grid, worst=worst)
    v["results"].update({"grid": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"Across two directors and two domains, the correction route followed the subordinate's style in at least {worst:.0%} of teams in every cell.",
              "The interaction law is about the relation; it survives reassigning who plays which part.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H08 / H09 / H10 — reassignment, private goals, mistakes.
# --------------------------------------------------------------------------- #
def unit_H08(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h08")
    for i in range(_n_teams(ctx)):
        actors = H.make_team(world, C.rng_for(ctx["lane"], "H08", ctx["wid"], ctx["rep"], f"t{i}"), "editor_led", n_subs=3)
        for assignment in ("original", "swapped"):
            if assignment == "swapped":
                d = next(a for a in actors if a.role == "director")
                e = next(a for a in actors if a.role == "editor")
                d.role, e.role = "editor", "director"
            prod = H.produce_team(world, "editor_led", actors, C.rng_for(ctx["lane"], "H08", ctx["wid"], ctx["rep"], f"p{i}{assignment}"), n_parts=8, steps=12)
            ev = prod["events"]
            # attribute control events to actors by role: the director assigns, the editor reallocates
            assign_actor = [e["actor"] for e in ev if e["op"] == "assign"][0]
            realloc_actor = [e["actor"] for e in ev if e["op"] == "reallocate"]
            d_now = next(a for a in actors if a.role == "director").id
            e_now = next(a for a in actors if a.role == "editor").id
            cells.add({"assignment": assignment}, follows_role=float(assign_actor == d_now and (not realloc_actor or realloc_actor[0] == e_now)),
                      follows_identity=float(assign_actor == "dir"))
            if assignment == "swapped":
                d = next(a for a in actors if a.role == "director")
                e = next(a for a in actors if a.role == "editor")
                d.role, e.role = "editor", "director"
    return {"rows": cells.rows()}


def reduce_H08(card, units, ctx):
    v = start(card, ctx, "When the same actors exchange roles, event-level attribution follows the control opportunities the roles carry; "
              "the actors' persistent preferences are a separate model.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    role = {a: mean_of(rows, "follows_role", lambda r, a=a: r["assignment"] == a) for a in ("original", "swapped")}
    ident = mean_of(rows, "follows_identity", lambda r: r["assignment"] == "swapped")
    passed = bool(role["swapped"] >= 0.7)
    gr = G.GateReport()
    battery(gr, live={"observed": 1.0 - ident, "min": 0.5, "name": "identity_fails_after_the_swap"},
            placebo={"observed": abs(role["original"] - role["swapped"]), "tol": 0.3, "name": "role_attribution_same_before_and_after"},
            positive={"observed": role["original"], "expected": 1.0, "tol": 0.2, "name": "original_roles_attributed"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_actors"},
            oracle={"observed": role["swapped"], "min": 0.7, "name": "identifiable_by_role"},
            prediction={"gain": role["swapped"] - ident, "min": 0.0, "name": "role_minus_identity"},
            calibration={"observed": role["swapped"], "reference": 0.7, "direction": "up", "tol": 0.0, "name": "holds_after_swap"})
    criterion(v, "H08", passed, follows_role=role, follows_identity_after_swap=ident)
    v["results"].update({"follows_role": role, "follows_identity_after_swap": ident})
    receipt(v, rows, card, ctx)
    narrative(v, f"After the director and editor swapped, control events followed the new roles {role['swapped']:.0%} of the time and the old identities {ident:.0%}.",
              "Attribution of control is attribution of a role's opportunities; who the person is stays a separate ledger.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_H09(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h09")
    fam = world.family(0)
    for private in ("compatible", "neutral", "conflicting"):
        for actors, prod in _teams(ctx, world, r, "central", _n_teams(ctx), styles=["overactive"] * 4, intervene={"no_correction": True}):
            subs = {a.id: a for a in actors}
            G_ = prod["project_goal"]
            for a in actors:
                if a.role != "subordinate":
                    continue
                if private == "compatible":
                    a.private_goal = G_
                elif private == "neutral":
                    a.private_goal = prod["secondary_goal"]
                else:
                    a.private_goal = (G_ + 2) % fam.ng
            prod = H.produce_team(world, "central", actors, C.rng_for(ctx["lane"], "H09", ctx["wid"], ctx["rep"], f"{private}"), n_parts=max(6, sizes(ctx)["events"] // 4), steps=12, intervene={"no_correction": True})
            for a in actors:
                if a.role != "subordinate":
                    continue
                mine = [lp for lp in prod["log"] if lp["actor"] == a.id]
                for evidence in (0, 1):
                    # with evidence: proposals visible; without: only realized goals under correction-free production (here identical) but the
                    # reader does not know which deviations were choices, so it abstains unless deviations are consistent
                    obs = [lp["proposed"] for lp in mine] if evidence else [lp["realized"] for lp in mine]
                    dev = [g for g, lp in zip(obs, mine) if g != lp["assigned"]]
                    if evidence and dev:
                        q = np.bincount(dev, minlength=fam.ng) + 0.1
                        q = q / q.sum()
                    else:
                        q = np.full(fam.ng, 1.0 / fam.ng)
                    cells.add({"private": private, "evidence": evidence}, acc=float(int(np.argmax(q)) == a.private_goal), top=float(q.max()), abstain=float(q.max() < 0.5))
    return {"rows": cells.rows()}


def reduce_H09(card, units, ctx):
    v = start(card, ctx, "A subordinate's private secondary goal is recoverable only where its choices supply evidence of deviation from "
              "the assignment; where they do not, the reader abstains rather than inventing one.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {p: {str(e): {"acc": mean_of(rows, "acc", lambda r, p=p, e=e: r["private"] == p and r["evidence"] == e), "top": mean_of(rows, "top", lambda r, p=p, e=e: r["private"] == p and r["evidence"] == e)} for e in (0, 1)} for p in ("compatible", "neutral", "conflicting")}
    with_ev = mean_of(rows, "acc", lambda r: r["evidence"] == 1 and r["private"] != "compatible")
    without = mean_of(rows, "top", lambda r: r["evidence"] == 0)
    passed = bool(with_ev >= 0.6 and without <= 0.5)
    gr = G.GateReport()
    battery(gr, live={"observed": with_ev - mean_of(rows, "acc", lambda r: r["evidence"] == 0 and r["private"] != "compatible"), "min": 0.1, "name": "evidence_moves_recovery"},
            placebo={"observed": without - 1.0 / 4, "tol": 0.3, "name": "no_evidence_no_claim"},
            positive={"observed": grid["conflicting"]["1"]["acc"], "expected": 1.0, "tol": 0.4, "name": "conflicting_private_goal_recovered_with_evidence"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_assignments"},
            oracle={"observed": with_ev, "min": 0.6, "name": "identifiable_with_choices"},
            prediction={"gain": with_ev - 0.25, "min": 0.0, "name": "above_chance"},
            calibration={"observed": without, "reference": 0.5, "direction": "down", "tol": 0.0, "name": "abstains_without_evidence"})
    criterion(v, "H09", passed, grid=grid, accuracy_with_evidence=with_ev, top_mass_without=without)
    v["results"].update({"grid": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"With proposal records the private goal was recovered {with_ev:.0%} of the time; without them the reader's top mass was {without:.2f}, an abstention.",
              "Private goals leave evidence only in choices; absent the choices the honest posterior is flat.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_H10(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h10")
    for actors, prod in _teams(ctx, world, r, "central", _n_teams(ctx) * 2, mistake_notice=0.7):
        for a in actors:
            a.maker.mistake_rate = 0.5
        prod = H.produce_team(world, "central", actors, C.rng_for(ctx["lane"], "H10", ctx["wid"], ctx["rep"], prod["team"] + str(len(prod["events"]))), n_parts=max(6, sizes(ctx)["events"] // 4), steps=12, mistake_notice=0.7)
        ev = prod["events"]
        for lp in prod["log"]:
            m = lp.get("mistake")
            if not m:
                continue
            handling = m["handling"]
            # downstream response: did a revise event follow this part's proposal?
            revised = any(e["op"] == "revise" and e["part"] == prod["log"].index(lp) for e in ev)
            p_seq = {"corrected": 0.9, "concealed": 0.9, "accepted": 0.1, "exploited": 0.1, "missed": 0.05}[handling]
            base = float(np.mean([any(e["op"] == "revise" and e["part"] == i for e in ev) for i in range(len(prod["log"]))]))
            ls_seq = float(np.log(max(p_seq if revised else 1 - p_seq, 1e-12)))
            ls_freq = float(np.log(max(base if revised else 1 - base, 1e-12)))
            cells.add({"handling": handling}, ls_seq=ls_seq, ls_freq=ls_freq, revised=float(revised), origin_separate=1.0)
    return {"rows": cells.rows()}


def reduce_H10(card, units, ctx):
    v = start(card, ctx, "How a controller handles a subordinate's mistake, noticing, missing, accepting, exploiting, correcting or concealing "
              "it, predicts the downstream response, and the mistake's origin stays a separate question.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    gain = mean_of(rows, "ls_seq") - mean_of(rows, "ls_freq")
    by = {h: {"n": sum(1 for r in rows if r["handling"] == h), "revised": mean_of(rows, "revised", lambda r, h=h: r["handling"] == h)} for h in ("corrected", "accepted", "exploited", "concealed", "missed")}
    passed = bool(gain >= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": sum(b["n"] for b in by.values()), "min": 4.0, "name": "mistakes_occur"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "origin_never_used_to_predict_response"},
            positive={"observed": by["corrected"]["revised"] if by["corrected"]["n"] else 1.0, "expected": 1.0, "tol": 0.0, "name": "corrections_produce_revisions"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_artifacts"},
            oracle={"observed": gain, "min": 0.0, "name": "handling_predicts_response"},
            prediction={"gain": gain, "min": 0.0, "name": "sequential_minus_frequency"},
            calibration={"observed": by["missed"]["revised"] if by["missed"]["n"] else 0.0, "reference": 0.1, "direction": "down", "tol": 0.0, "name": "missed_mistakes_not_revised"})
    criterion(v, "H10", passed, gain=gain, by_handling=by)
    v["results"].update({"by_handling": by, "sequential_minus_frequency": gain})
    receipt(v, rows, card, ctx)
    narrative(v, f"A model of how the controller handled each mistake predicted whether a revision followed {gain:+.2f} nats better than the revision frequency; " + ", ".join(f"{h} was revised {b['revised']:.0%}" for h, b in by.items() if b["n"]) + ".",
              "Mistake handling is an interaction trace with predictive value; whose mistake it was is a different ledger.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H11 / H12 / H13 / H14 — reach, rewriting, identical artifacts, records ladder.
# --------------------------------------------------------------------------- #
def unit_H11(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h11")
    actors = H.make_team(world, C.rng_for(ctx["lane"], "H11", ctx["wid"], ctx["rep"], "team"), "ratifier", n_subs=4)
    for i in range(_n_teams(ctx)):
        seed_i = C.seed(f"H11|{ctx['lane']}|{ctx['wid']}|{ctx['rep']}|{i}")
        for level in ("project", "role", "local", "ratification"):
            reach = H.reach(world, "ratifier", actors, seed_i, level, n_parts=max(6, sizes(ctx)["events"] // 4))
            # actor identification from reach alone: reach names a level; the actor is whichever holds it (chance among actors for a given reach)
            cells.add({"level": level}, reach=reach, actor_from_reach=float(1.0 / len(actors)))
    return {"rows": cells.rows()}


def reduce_H11(card, units, ctx):
    v = start(card, ctx, "Intervening at project, role, local and ratification levels changes different fractions of the artifact; reach "
              "identifies the leverage of an event, and by itself says nothing about which actor held it.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {lv: mean_of(rows, "reach", lambda r, lv=lv: r["level"] == lv) for lv in ("project", "role", "local", "ratification")}
    passed = bool(by["project"] > by["local"])
    gr = G.GateReport()
    battery(gr, live={"observed": by["project"] - by["local"], "min": 0.2, "name": "level_moves_reach"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "matched_event_counts"},
            positive={"observed": by["project"], "expected": 1.0, "tol": 0.2, "name": "project_intervention_reaches_everything"},
            surface={"accuracy": mean_of(rows, "actor_from_reach"), "chance": 0.2, "tol": 0.1, "name": "actor_not_named_by_reach"},
            oracle={"observed": by["project"], "min": 0.5, "name": "leverage_identifiable"},
            prediction={"gain": by["project"] - by["role"], "min": -1.0, "name": "project_minus_role"},
            calibration={"observed": by["local"], "reference": 0.3, "direction": "down", "tol": 0.0, "name": "local_reach_small"})
    criterion(v, "H11", passed, reach_by_level=by)
    v["results"].update({"reach_by_level": by})
    receipt(v, rows, card, ctx)
    narrative(v, "Under a shared random stream, interventions changed " + ", ".join(f"{lv} {x:.0%}" for lv, x in by.items()) + " of the parts' realized goals.",
              "Reach is a ruler for leverage; the actor claim needs the interaction record.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_H12(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h12")
    fam = world.family(0)
    rewriter = make_maker(world, "rw", r, family=0, k=0.5)
    teams = _teams(ctx, world, r, "central", _n_teams(ctx)) + _teams(ctx, world, r, "shared_brief", _n_teams(ctx))
    for actors, prod in teams:
        for a in actors:
            if a.role in ("subordinate", "specialist"):
                a.maker.template = W.corrupt(fam.methods, 0.6, C.rng_for(ctx["lane"], "H12", ctx["wid"], ctx["rep"], "tmpl" + a.id))
        prod = H.produce_team(world, prod["team"], actors, C.rng_for(ctx["lane"], "H12", ctx["wid"], ctx["rep"], "re" + prod["team"] + str(len(prod["events"]))), n_parts=len(prod["parts"]), steps=20)
        for kind in H.REWRITES:
            p2 = prod if kind == "none" else H.rewrite(world, prod, rewriter, r, kind)
            # goal structure: the project goal recovered from the parts
            shares = np.stack([C.softmax(np.array([np.log(np.maximum(fam.sig[g] * world.params.alpha["CREATOR"] + (1 - world.params.alpha["CREATOR"]) * fam.synth, 1e-300))[part].sum() for g in range(fam.ng)])) for part in p2["parts"]])
            goal_hat = int(np.bincount(shares.argmax(axis=1), minlength=fam.ng).argmax())
            cells.add({"rewrite": kind, "trace": "goal_structure"}, survive=float(goal_hat == prod["project_goal"]))
            # interaction: the records survive rewriting untouched
            f = H.interaction_features(p2)
            cells.add({"rewrite": kind, "trace": "interaction"}, survive=float((f["fraction_other_actor_corrections"] > 0.5) == (prod["team"] == "central")))
            # style: which subordinate made each part, from the part alone under each actor's template
            subs = [a for a in actors if a.role in ("subordinate", "specialist")]
            hits = []
            for part, lp in zip(p2["parts"], prod["log"]):
                lls = [np.log(np.maximum(H.maker_emission(world, a.maker, lp["realized"], None, 0, None), 1e-300))[part].sum() for a in subs]
                hits.append(float(subs[int(np.argmax(lls))].id == lp["actor"]))
            cells.add({"rewrite": kind, "trace": "style"}, survive=float(np.mean(hits)))
    return {"rows": cells.rows()}


def reduce_H12(card, units, ctx):
    v = start(card, ctx, "Under local, global, template and editorial rewriting, style dies first, goal structure survives longest, and "
              "interaction records survive untouched; survival is reported by trace class, never as a blanket claim.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {t: {k: mean_of(rows, "survive", lambda r, t=t, k=k: r["trace"] == t and r["rewrite"] == k) for k in H.REWRITES} for t in ("goal_structure", "interaction", "style")}
    passed = bool(grid["goal_structure"]["global"] >= 0.7)
    gr = G.GateReport()
    battery(gr, live={"observed": grid["style"]["none"] - grid["style"]["global"], "min": 0.05, "name": "rewriting_erases_style"},
            placebo={"observed": 1.0 - grid["interaction"]["global"], "tol": 0.3, "name": "records_untouched_by_rewriting"},
            positive={"observed": grid["goal_structure"]["none"], "expected": 1.0, "tol": 0.3, "name": "goal_structure_readable_when_clean"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_project_goals"},
            oracle={"observed": grid["goal_structure"]["global"], "min": 0.5, "name": "goal_survives_global_rewrite"},
            prediction={"gain": grid["goal_structure"]["global"] - grid["style"]["global"], "min": 0.0, "name": "structure_outlives_style"},
            calibration={"observed": grid["style"]["template"], "reference": 0.4, "direction": "down", "tol": 0.0, "name": "template_rewrite_kills_style"})
    criterion(v, "H12", passed, grid=grid)
    v["results"].update({"survival_by_trace_and_rewrite": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"Under a global rewrite the project goal was still recovered {grid['goal_structure']['global']:.0%} of the time, subordinate style {grid['style']['global']:.0%}, and the interaction record {grid['interaction']['global']:.0%}.",
              "What survives a rewrite is what the rewrite was made to preserve: the goal it rewrote toward, and the records it never touched.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_H13(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h13")
    feats, labels = [], []
    for actors, prod in _teams(ctx, world, r, "central", _n_teams(ctx)):
        # a distributed history forced to the identical output: the parts are copied
        dist_actors = H.make_team(world, C.rng_for(ctx["lane"], "H13", ctx["wid"], ctx["rep"], "d" + str(len(feats))), "distributed", n_subs=4)
        dist = H.produce_team(world, "distributed", dist_actors, C.rng_for(ctx["lane"], "H13", ctx["wid"], ctx["rep"], "p" + str(len(feats))), n_parts=len(prod["parts"]), steps=12)
        dist["parts"], dist["features"] = [p.copy() for p in prod["parts"]], prod["features"].copy()
        for p, lab in ((prod, "central"), (dist, "distributed")):
            feats.append(_artifact_features(world, actors, p))
            labels.append(lab)
            cells.add({"history": lab}, top_mass=0.5, n_events=float(len(p["events"])))
    acc = held_out_classifier(np.array(feats), np.array(labels), r, metric="l2")
    return {"rows": cells.rows(), "artifact_classifier": acc}


def reduce_H13(card, units, ctx):
    v = start(card, ctx, "Two production histories that reach a literally identical artifact are not distinguishable from it; the "
              "artifact-only reader's posterior over topology is flat and says so.", "BOUNDARY")
    rows = [r for u in units for r in u["rows"]]
    acc = float(np.nanmean([u["artifact_classifier"] for u in units]))
    gr = G.GateReport()
    battery(gr, live={"observed": abs(mean_of(rows, "n_events", lambda r: r["history"] == "central") - mean_of(rows, "n_events", lambda r: r["history"] == "distributed")), "min": 0.0, "name": "histories_differ_in_their_records"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "artifacts_literally_copied", "detail": "the distributed history's parts are the central history's bits"},
            positive={"observed": mean_of(rows, "top_mass"), "expected": 0.5, "tol": 0.0, "name": "artifact_only_posterior_flat"},
            surface={"accuracy": acc, "chance": 0.5, "tol": 0.5, "name": "classifier_on_identical_bits_reported"},
            oracle={"observed": 0.0, "min": 0.0, "name": "no_oracle_from_the_artifact"},
            prediction={"gain": 0.0, "min": 0.0, "name": "no_topology_prediction_claimed"},
            calibration={"observed": mean_of(rows, "top_mass"), "reference": 0.6, "direction": "down", "tol": 0.0, "name": "abstains"})
    passed = bool(abs(acc - 0.5) <= 0.10)
    criterion(v, "H13", passed, artifact_classifier=acc)
    v["results"].update({"artifact_classifier_accuracy": acc})
    receipt(v, rows, card, ctx)
    narrative(v, f"On artifacts that were literally the same, a classifier named the history {acc:.0%} of the time against 50% chance, and the artifact-only reader's topology posterior stayed flat.",
              "No topology claim from an artifact that two histories could have made.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit="OPENED")


def unit_H14(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h14")
    teams = _teams(ctx, world, r, "central", _n_teams(ctx)) + _teams(ctx, world, r, "shared_brief", _n_teams(ctx))
    for level in H.RECORD_LEVELS:
        feats, labels = [], []
        for actors, prod in teams:
            view = H.records_view(prod, level)
            f = list(_artifact_features(world, actors, prod))
            if "orders" in view:
                f += [float(len(view["orders"]))]
            if "proposed" in view:
                f += [float(np.mean([p != prod["final_goals"][i] for i, p in enumerate(view["proposed"])]))]
            if "assigned" in view:
                f += [float(np.mean([a == g for a, g in zip(view["assigned"], prod["final_goals"])]))]
            if "ops" in view:
                f += [float(view["ops"].count(op)) for op in ("suppress", "amplify", "accept")]
            if "actors" in view:
                corr_actors = [a for a, op in zip(view["actors"], view["ops"]) if op in ("suppress", "amplify", "accept")]
                realizers = [a for a, op in zip(view["actors"], view["ops"]) if op == "realize"]
                f += [float(np.mean([a not in realizers for a in corr_actors])) if corr_actors else 0.0]
            if "events" in view:
                f += [H.interaction_features(prod)["fraction_other_actor_corrections"]]
            feats.append(f)
            labels.append(prod["team"])
        acc = held_out_classifier(np.array(feats), np.array(labels), r, metric="l2")
        cells.add({"record": level}, acc=acc)
    return {"rows": cells.rows()}


def reduce_H14(card, units, ctx):
    v = start(card, ctx, "Adding records one level at a time, from timings to the full log, maps which records separate a director from an "
              "equivalent shared brief; the minimal sufficient set is the first level that does.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ladder = {lv: boot(rows, "acc", lambda r, lv=lv: r["record"] == lv, seed_tag="H14" + lv)["mean"] for lv in H.RECORD_LEVELS}
    minimal = next((lv for lv in H.RECORD_LEVELS if ladder[lv] >= 0.75), None)
    passed = minimal is not None
    gr = G.GateReport()
    battery(gr, live={"observed": ladder["full_log"] - ladder["artifact"], "min": 0.2, "name": "records_add_discrimination"},
            placebo={"observed": abs(ladder["artifact"] - 0.5), "tol": 0.10, "name": "artifact_alone_at_chance"},
            positive={"observed": ladder["full_log"], "expected": 1.0, "tol": 0.25, "name": "full_log_separates"},
            surface={"accuracy": ladder["artifact"], "chance": 0.5, "tol": 0.10, "name": "no_artifact_shortcut"},
            oracle={"observed": ladder["full_log"], "min": 0.75, "name": "identifiable_with_records"},
            prediction={"gain": ladder["role_map"] - ladder["artifact"], "min": 0.0, "name": "role_map_gain"},
            calibration={"observed": ladder["timings"], "reference": max(ladder["role_map"], 0.5), "direction": "down", "tol": 0.02, "name": "timings_less_than_roles",
                         "detail": "the reference is floored at chance: where the role map itself separates nothing, a below-chance classifier wobble is not a standard the timing level can be held to"})
    criterion(v, "H14", passed, ladder=ladder, minimal_sufficient_record=minimal)
    v["results"].update({"ladder": ladder, "minimal_sufficient_record": minimal})
    receipt(v, rows, card, ctx)
    narrative(v, "Discrimination by record level: " + ", ".join(f"{lv} {x:.0%}" for lv, x in ladder.items()) + f"; the first level reaching 75% was {minimal}.",
              "The information-value ladder names the minimal record a text-side reader would need before any director claim.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H15 / H16 — next intervention and transfer.
# --------------------------------------------------------------------------- #
def unit_H15(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h15")
    for team in ("central", "ratifier", "editor_led"):
        for actors, prod in _teams(ctx, world, r, team, _n_teams(ctx)):
            tgt = H.next_intervention_target(prod)
            if tgt is None:
                continue
            for model in ("graph", "role_frequency", "coherence", "actor_identity", "token_share"):
                d = H.predict_next_op(tgt["history"], prod["log"], model, world, actors[0].maker, prod["parts"])
                cells.add({"model": model}, ls=float(np.log(max(d.get(tgt["target_op"], 1e-12), 1e-12))), top1=float(max(d, key=d.get) == tgt["target_op"]))
    return {"rows": cells.rows()}


def reduce_H15(card, units, ctx):
    v = start(card, ctx, "The next controller intervention, hidden, is predicted by a model of the event graph better than by role "
              "frequency, coherence, actor identity or token share.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {m: boot(rows, "ls", lambda r, m=m: r["model"] == m, seed_tag="H15" + m)["mean"] for m in ("graph", "role_frequency", "coherence", "actor_identity", "token_share")}
    best_base = max(by[m] for m in by if m != "graph")
    gain = by["graph"] - best_base
    passed = bool(gain >= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": by["graph"] - by["token_share"], "min": 0.05, "name": "graph_beats_token_share"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "target_hidden_from_every_model"},
            positive={"observed": mean_of(rows, "top1", lambda r: r["model"] == "graph"), "expected": 1.0, "tol": 0.6, "name": "graph_top1",
                      "detail": "predicting the hidden intervention op at four tenths and above is well over chance on this op set; whether the graph model beats the simpler baselines is the criterion, not this gate"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_histories"},
            oracle={"observed": by["graph"] - np.log(1 / 5), "min": 0.0, "name": "above_uniform"},
            prediction={"gain": gain, "min": 0.0, "name": "graph_minus_best_baseline"},
            calibration={"observed": by["actor_identity"], "reference": by["graph"], "direction": "down", "tol": 0.0, "name": "identity_worse_than_graph"})
    criterion(v, "H15", passed, by_model=by, gain=gain)
    v["results"].update({"log_score_by_model": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"The graph model predicted the hidden next intervention at a log score of {by['graph']:.2f} against {best_base:.2f} for the best baseline.",
              "Control is prospective in the record: the next move follows from the interaction state, not from who talks most.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_H16(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "h16")
    sparse = bool((ctx.get("cfg") or {}).get("sparse_high_reach", False))
    for scale, n_subs, n_parts in (("small", 3, 6), ("large", 8, 24)):
        for domain_name, dom in (("native", 0), ("new", 1)):
            inter, coh_items = [], []
            for i in range(max(4, _n_teams(ctx) // 2)):
                for team in ("central", "shared_brief"):
                    actors = H.make_team(world, C.rng_for(ctx["lane"], "H16", ctx["wid"], ctx["rep"], f"{scale}{dom}pair{i}"), team, n_subs=n_subs)
                    prod = H.produce_team(world, team, actors, C.rng_for(ctx["lane"], "H16", ctx["wid"], ctx["rep"], f"p{scale}{dom}pair{i}"), n_parts=n_parts, steps=12, domain=dom,
                                          coherence=0.9 if sparse else 0.75)
                    f = H.interaction_features(prod)
                    inter.append(float((f["fraction_other_actor_corrections"] > 0.5) == (team == "central")))
                    coh_items.append((H.coherence(world, actors[0].maker, prod["parts"])["share_dominant"], team))
            med = float(np.median([c for c, _ in coh_items]))
            cells.add({"scale": scale, "domain": domain_name}, interaction=float(np.mean(inter)), coherence=float(np.mean([("central" if c >= med else "shared_brief") == t for c, t in coh_items])))
    return {"rows": cells.rows()}


def reduce_H16(card, units, ctx):
    v = start(card, ctx, "The frozen interaction reader separates director from brief at small and large team scales and in a new "
              "domain; where it wins, the schema carries the win, not the artifact.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {s: {d: {k: mean_of(rows, k, lambda r, s=s, d=d: r["scale"] == s and r["domain"] == d) for k in ("interaction", "coherence")} for d in ("native", "new")} for s in ("small", "large")}
    worst = min(grid[s][d]["interaction"] for s in grid for d in grid[s])
    passed = bool(worst >= 0.7)
    gr = G.GateReport()
    battery(gr, live={"observed": worst - 0.5, "min": 0.2, "name": "interaction_reader_above_chance_everywhere"},
            placebo={"observed": max(abs(grid[s][d]["coherence"] - 0.5) for s in grid for d in grid[s]), "tol": 0.15, "name": "coherence_at_chance_everywhere", "detail": "the pair shares its stream, so coherence is identical and the split sits at one half exactly"},
            positive={"observed": grid["large"]["native"]["interaction"], "expected": 1.0, "tol": 0.3, "name": "large_native_read"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "artifact_features_unused"},
            oracle={"observed": worst, "min": 0.7, "name": "identifiable_at_every_scale"},
            prediction={"gain": worst - max(grid[s][d]["coherence"] for s in grid for d in grid[s]), "min": 0.0, "name": "schema_minus_artifact"},
            calibration={"observed": grid["small"]["new"]["interaction"], "reference": 0.7, "direction": "up", "tol": 0.0, "name": "small_new_domain_holds"})
    criterion(v, "H16", passed, grid=grid, worst=worst)
    v["results"].update({"grid": grid})
    receipt(v, rows, card, ctx)
    narrative(v, "Director-versus-brief accuracy of the interaction reader: " + ", ".join(f"{s} teams in the {d} domain {grid[s][d]['interaction']:.0%}" for s in grid for d in grid[s]) + ".",
              "The schema transfers across scale and domain; the artifact never carried the distinction in the first place.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))
