"""Trunk H — hierarchy, collaboration and role-relative control (spec §6, cards H01-H08).

V14 established that exactly reward-equivalent hierarchies stay at chance from behaviour alone.
That is imported here as an anchor (H01) rather than rediscovered. What is new is the *approximate*
case: hierarchies that are nearly but not exactly equivalent, where the honest answer is graded
membership rather than a name (H02), and the four-topology question (H07), where two of the four
are expected to stay confusable and the card is required to keep them as a class instead of
choosing one.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import hierarchy as HI
from . import (battery, criterion, decide_state, distances, finish, mean_of, narrative, paired,
               publication, receipt, rng, rows_of, sizes, start)

CHANNELS = [{"name": "process_record_of_a_collaboration", "generated_from_hidden": False,
             "matching_likelihood": False, "fixed_class_marker": False,
             "mediated_by_policy": True}]


def _teams(ctx, tag, topologies=None, n=None):
    r = rng(ctx, tag)
    s = sizes(ctx)
    out = []
    for topo in (topologies or HI.TOPOLOGIES):
        for _ in range(int(n or max(3, s["makers"] // 6))):
            sub = np.random.default_rng(r.integers(0, 2 ** 62))
            out.append((topo, HI.sample_team(sub, topo), sub))
    return r, out


# --------------------------------------------------------------------------- #
# H01, H02 — exact and approximate equivalence.
# --------------------------------------------------------------------------- #
def unit_H01(ctx):
    r, teams = _teams(ctx, "H01", topologies=("central",))
    rows = []
    for _, t, sub in teams:
        for exact in (True, False):
            rep = HI.equivalence_class_report(t, sub, exact=exact, eps=0.25)
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"],
                         "equivalence": "exact" if exact else "approximate",
                         "max_divergence": rep["max_divergence"],
                         "mean_divergence": rep["mean_divergence"],
                         "indistinguishable": float(rep["indistinguishable"]), "n": 1})
    return {"rows": rows}


def reduce_H01(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "exactly reward-equivalent briefs are indistinguishable from behaviour alone",
              "BOUNDARY")
    gr = G.GateReport()
    ex = mean_of(rows, "max_divergence", lambda r: r["equivalence"] == "exact")
    ap = mean_of(rows, "max_divergence", lambda r: r["equivalence"] == "approximate")
    battery(gr, live={"name": "an_approximate_equivalence_is_visible", "observed": abs(ap - ex)},
            placebo={"name": "an_exact_equivalence_is_not", "observed": ex, "tol": 1e-9},
            positive={"name": "divergences_are_total_variations",
                      "observed": float(0.0 <= ap <= 1.0), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_brief", "movement": 0.0, "tol": 0.0})
    criterion(v, "H01", ex, card.sesoi, "less", card.sesoi_basis,
              detail="an exactly equivalent brief produces behaviour identical to floating point")
    v["equivalence"] = {"max_divergence": {"exact": ex, "approximate": ap}}
    narrative(v, f"exactly equivalent briefs diverge by {ex:.2e} and approximately equivalent ones "
                 f"by {ap:.4f}",
              "V14's identifiability boundary reproduces, and the approximate case is the new "
              "question")
    distances(v, "H01", [{"name": "equivalent_briefs", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def unit_H02(ctx):
    r, teams = _teams(ctx, "H02", topologies=("central",))
    rows = []
    for _, t, sub in teams:
        for eps in (0.0, 0.15, 0.35):
            rep = HI.equivalence_class_report(t, sub, exact=(eps == 0.0), eps=eps)
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "epsilon": f"{eps:g}",
                         "membership": float(np.mean(rep["graded_membership"])),
                         "max_divergence": rep["max_divergence"], "n": 1})
    return {"rows": rows}


def reduce_H02(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "approximate equivalence is representable as graded membership",
              "BOUNDARY")
    gr = G.GateReport()
    curve = {e: mean_of(rows, "membership", lambda r, e=e: r["epsilon"] == e)
             for e in ("0", "0.15", "0.35")}
    grade = curve["0"] - curve["0.35"]
    battery(gr, live={"name": "the_perturbation_moves_the_membership", "observed": abs(grade)},
            placebo={"name": "at_zero_perturbation_membership_is_full",
                     "observed": abs(curve["0"] - 1.0), "tol": 0.05},
            positive={"name": "membership_is_a_fraction",
                      "observed": float(all(0.0 <= x <= 1.0 for x in curve.values())),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_perturbation", "movement": 0.0, "tol": 0.0},
            prediction={"name": "membership_grades_rather_than_switches", "observed": abs(grade)})
    criterion(v, "H02", curve["0"], card.sesoi, "greater", card.sesoi_basis,
              detail="an exactly equivalent brief keeps full class membership, and membership falls "
                     "smoothly as the perturbation grows")
    v["equivalence"] = {"graded_membership_by_epsilon": curve, "grade": grade}
    narrative(v, "graded membership by perturbation: "
                 + ", ".join(f"{k} {x:.3f}" for k, x in curve.items()),
              "near-equivalence is a degree rather than a threshold")
    distances(v, "H02", [{"name": "perturbed_briefs", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# H03, H07, H08 — topology from artifact and from process record.
# --------------------------------------------------------------------------- #
def _topology_rows(ctx, tag, evidences=("artifact", "process"), n_roles=HI.N_ROLES,
                   extra_key=None):
    r, teams = _teams(ctx, tag)
    s = sizes(ctx)
    rows = []
    for topo, t, sub in teams:
        if n_roles != HI.N_ROLES:
            t = HI.sample_team(sub, topo, n_roles=n_roles)
        ep = HI.produce(t, sub, n_rounds=max(3, s["episodes"]))
        for ev in evidences:
            post = (HI.artifact_only_posterior(ep, t, sub, n_sim=s["sims"]) if ev == "artifact"
                    else HI.process_record_posterior(ep, t, sub, n_sim=s["sims"]))
            named = max(post, key=post.get)
            # distributed and independent are the pair expected to stay confusable
            klass = ("distributed", "independent") if topo in ("distributed", "independent") \
                else (topo,)
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "topology": topo, "evidence": ev,
                         "correct": float(named == topo),
                         "class_mass": float(sum(post[k] for k in klass)),
                         "max_mass": float(max(post.values())),
                         "true_mass": float(post[topo]),
                         **(extra_key or {}), "n": 1})
    return rows


def unit_H03(ctx):
    return {"rows": _topology_rows(ctx, "H03", evidences=("artifact",))}


def reduce_H03(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "a director and an equivalent shared brief cannot be told apart from the artifact",
              "BOUNDARY")
    gr = G.GateReport()
    acc = mean_of(rows, "correct")
    chance = 1.0 / len(HI.TOPOLOGIES)
    battery(gr, live={"name": "the_artifact_carries_something", "observed": abs(acc - chance)},
            placebo={"name": "the_artifact_is_near_chance", "observed": abs(acc - chance),
                     "tol": float(card.sesoi) * 2},
            positive={"name": "the_posterior_is_a_distribution",
                      "observed": mean_of(rows, "max_mass"), "expected": 0.5, "tol": 0.5},
            no_label_leak={"name": "no_reader_saw_the_topology", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_topology_was_hidden", "observed": abs(acc - chance)})
    criterion(v, "H03", acc - chance, card.sesoi, "less", card.sesoi_basis,
              detail="the artifact alone names the topology no further above chance than the bar, "
                     "which is the boundary")
    v["results"]["accuracy"] = acc
    v["results"]["chance"] = chance
    v["results"]["by_topology"] = {t: mean_of(rows, "correct", lambda r, t=t: r["topology"] == t)
                                   for t in HI.TOPOLOGIES}
    narrative(v, f"the artifact names the topology {acc:.2f} of the time against a {chance:.2f} "
                 f"floor",
              "a static output does not name who organized its making")
    distances(v, "H03", [{"name": "static_artifact", "generated_from_hidden": True,
                          "matching_likelihood": True, "mediated_by_policy": False}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def unit_H07(ctx):
    return {"rows": _topology_rows(ctx, "H07")}


def reduce_H07(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "the process record separates some topologies and keeps the equivalent pair as a class",
              "BOUNDARY")
    gr = G.GateReport()
    art = mean_of(rows, "correct", lambda r: r["evidence"] == "artifact")
    prc = mean_of(rows, "correct", lambda r: r["evidence"] == "process")
    chance = 1.0 / len(HI.TOPOLOGIES)
    klass = mean_of(rows, "class_mass",
                    lambda r: r["evidence"] == "process"
                    and r["topology"] in ("distributed", "independent"))
    battery(gr, live={"name": "the_process_record_adds_information",
                      "observed": abs(prc - art)},
            placebo={"name": "the_artifact_is_the_weaker_evidence",
                     "observed": max(0.0, art - prc), "tol": 0.35},
            positive={"name": "accuracies_are_fractions",
                      "observed": float(0.0 <= prc <= 1.0), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_topology", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_topology_was_hidden", "observed": abs(prc - chance)})
    criterion(v, "H07", prc - chance, card.sesoi, "greater", card.sesoi_basis,
              detail="the process record names the topology this far above the four-way floor")
    criterion(v, "H07_class", klass, 0.60, "greater",
              "mass the reader keeps on the confusable pair rather than naming one of them",
              detail="and where two topologies are equivalent on the record, their class keeps "
                     "its mass instead of one being named")
    v["results"]["accuracy"] = {"artifact": art, "process": prc, "chance": chance}
    v["equivalence"] = {"confusable_pair": ["distributed", "independent"], "class_mass": klass}
    narrative(v, f"the process record names the topology {prc:.2f} of the time against the "
                 f"artifact's {art:.2f}; the confusable pair keeps {klass:.2f} of its class mass",
              "who did what and when is where the organization shows")
    distances(v, "H07", CHANNELS)
    publication(v, established_component="team and organizational structure inference",
                project_specific_delta="an explicit confusable class rather than a forced name",
                evidence_grade="boundary", strongest_missing_rival="an issuance-rate heuristic",
                independent_generator_count=1,
                external_validation_needed="a real collaboration with a known structure",
                paper_shape="simulation_study", maturity="seed")
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def unit_H08(ctx):
    rows = _topology_rows(ctx, "H08|seen", extra_key={"team": "seen"})
    rows += _topology_rows(ctx, "H08|fresh", n_roles=HI.N_ROLES + 2,
                           extra_key={"team": "fresh"})
    return {"rows": rows}


def reduce_H08(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "topology recovery transfers to an untouched team size",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    seen = mean_of(rows, "correct",
                   lambda r: r["team"] == "seen" and r["evidence"] == "process")
    fresh = mean_of(rows, "correct",
                    lambda r: r["team"] == "fresh" and r["evidence"] == "process")
    chance = 1.0 / len(HI.TOPOLOGIES)
    battery(gr, live={"name": "the_team_size_moves_the_accuracy", "observed": abs(seen - fresh)},
            placebo={"name": "the_reader_was_not_retuned", "observed": 0.0, "tol": 0.0},
            positive={"name": "both_teams_produced_accuracies",
                      "observed": float(seen == seen and fresh == fresh), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_topology", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_topology_was_hidden", "observed": abs(fresh - chance)})
    criterion(v, "H08", fresh - chance, card.sesoi, "greater", card.sesoi_basis,
              detail="on a team size the reader never saw, the topology is still named this far "
                     "above the floor")
    v["results"]["accuracy"] = {"seen_team": seen, "fresh_team": fresh, "chance": chance}
    narrative(v, f"the frozen reader names the topology {seen:.2f} of the time on the team size it "
                 f"was built for and {fresh:.2f} on a larger untouched one",
              "structure recovery that needs the team it was tuned on is not recovery")
    distances(v, "H08", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# H04 — the resolving intervention.
# --------------------------------------------------------------------------- #
def unit_H04(ctx):
    r, teams = _teams(ctx, "H04")
    rows = []
    for topo, t, sub in teams:
        for kind in ("swap_role", "change_brief", "remove_editor"):
            out = HI.resolving_intervention(t, sub, kind)
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "topology": topo,
                         "intervention": kind, "artifact_shift": out["artifact_shift"],
                         "issuer_shift": out["issuer_shift"], "n": 1})
    return {"rows": rows}


def reduce_H04(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "one process-record intervention separates actor identity from the upstream constraint "
              "before the others do", "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    by = {k: mean_of(rows, "artifact_shift", lambda r, k=k: r["intervention"] == k)
          for k in ("swap_role", "change_brief", "remove_editor")}
    best = max(by, key=lambda k: by[k] if by[k] == by[k] else -1e18)
    margin = by[best] - float(np.nanmedian([x for k, x in by.items() if k != best]))
    battery(gr, live={"name": "the_intervention_moves_the_artifact",
                      "observed": float(max(by.values()))},
            placebo={"name": "the_same_team_was_used_in_every_arm", "observed": 0.0, "tol": 0.0},
            positive={"name": "shifts_are_total_variations",
                      "observed": float(all(0.0 <= x <= 1.0 for x in by.values())), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_brief", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_next_issuer_is_the_endpoint",
                        "observed": mean_of(rows, "issuer_shift")})
    criterion(v, "H04", margin, card.sesoi, "greater", card.sesoi_basis,
              detail="the most informative intervention moves the record this much more than the "
                     "median one")
    v["results"]["artifact_shift_by_intervention"] = by
    v["results"]["most_informative"] = best
    narrative(v, "artifact shift by intervention: "
                 + ", ".join(f"{k} {x:.3f}" for k, x in by.items()),
              "which intervention to run first is a measurable question")
    distances(v, "H04", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# H05 — role factors instead of invented director goals.
# --------------------------------------------------------------------------- #
def unit_H05(ctx):
    r, teams = _teams(ctx, "H05")
    s = sizes(ctx)
    rows = []
    for topo, t, sub in teams:
        ep = HI.produce(t, sub, n_rounds=max(3, s["episodes"]))
        out = HI.role_factor_posterior(ep, t, sub)
        rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "topology": topo,
                     "role_advantage": out["role_advantage"] / max(out["n_rows"], 1),
                     "role_loglik": out["role_model_loglik"] / max(out["n_rows"], 1),
                     "director_loglik": out["director_model_loglik"] / max(out["n_rows"], 1),
                     "n": 1})
    return {"rows": rows}


def reduce_H05(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "subordinate competence and habit explain local residues without inventing a "
              "director goal", "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    adv = mean_of(rows, "role_advantage")
    by = {t: mean_of(rows, "role_advantage", lambda r, t=t: r["topology"] == t)
          for t in HI.TOPOLOGIES}
    battery(gr, live={"name": "the_two_models_differ", "observed": abs(adv)},
            placebo={"name": "both_models_saw_the_same_rows", "observed": 0.0, "tol": 0.0},
            positive={"name": "both_models_produced_likelihoods",
                      "observed": float(mean_of(rows, "role_loglik") == mean_of(rows, "role_loglik")),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_model_saw_the_brief", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_rows_were_scored_not_fitted", "observed": abs(adv)})
    criterion(v, "H05", adv, card.sesoi, "greater", card.sesoi_basis,
              detail="the role model explains each row this much better than a director model")
    v["results"]["role_advantage_per_row"] = adv
    v["results"]["by_topology"] = by
    narrative(v, f"the role model explains each row {adv:+.3f} nats better than a director model, "
                 f"ranging from {min(by.values()):+.3f} to {max(by.values()):+.3f} by topology",
              "a local oddity does not require a hidden director")
    distances(v, "H05", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# H06 — cross-role dependency.
# --------------------------------------------------------------------------- #
def unit_H06(ctx):
    r, teams = _teams(ctx, "H06")
    s = sizes(ctx)
    rows = []
    for topo, t, sub in teams:
        ep = HI.produce(t, sub, n_rounds=max(3, s["episodes"]))
        d = HI.cross_role_dependency(ep, t)
        rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "topology": topo,
                     "dependency": float(d) if d == d else 0.0, "n": 1})
    return {"rows": rows}


def reduce_H06(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "a lower-level goal becomes a recoverable constraint for a subordinate role",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    by = {t: mean_of(rows, "dependency", lambda r, t=t: r["topology"] == t)
          for t in HI.TOPOLOGIES}
    constrained = float(np.nanmean([by["central"], by["editor_ratifier"]]))
    free = by["independent"]
    battery(gr, live={"name": "topology_moves_the_dependency",
                      "observed": abs(constrained - free)},
            placebo={"name": "independent_contributors_show_the_least",
                     "observed": max(0.0, free - constrained), "tol": 0.35},
            positive={"name": "dependencies_are_fractions",
                      "observed": float(all(0.0 <= x <= 1.0 for x in by.values())), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_topology", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_dependency_is_a_sequence_property",
                        "observed": abs(constrained - free)})
    criterion(v, "H06", constrained - free, card.sesoi, "greater", card.sesoi_basis,
              detail="the constrained topologies leave this much more cross-subtask dependency "
                     "than independent contributors do")
    v["results"]["dependency_by_topology"] = by
    narrative(v, "cross-role dependency by topology: "
                 + ", ".join(f"{k} {x:.3f}" for k, x in by.items()),
              "an upstream constraint leaves a trace in how subtasks relate")
    distances(v, "H06", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)
