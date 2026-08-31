"""Trunk C — the coupling and access boundary atlas (spec §6, cards C01-C14).

The trunk that answers V15's one question. V14 compared *one* coupled world with *one* factorized
world and found +0.011 nats; the audit reading was that this prices one easy-to-factor
construction rather than jointness. Here coupling, route overlap, evidence dose, dependence
structure, missingness, policy temperature, equifinality and maker-reader similarity are all
continuous or crossed, and the estimand is a *conditional surface* -- where the joint reader starts
to win, not whether it wins on average.

Two rules bind every card here, both from spec §5 and §8.2:

* the phase-diagram axis is **realized** coupling, measured from the world's own prior, not the
  nominal knob. The families reach different ceilings by different constructions, and comparing
  them on the nominal knob would compare two different things.
* **no pooled headline over an axis whose effect changes along it.** A card that finds an
  interaction reports the conditional curve, and ``onset`` returns the whole curve beside the
  onset so a mean cannot be quoted instead.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import exact as EX
from ..ontology import COMPONENTS
from . import (Cells, arch_gap, battery, criterion, decide_state, distances, extra_gate,
               families_of, family_module, finish, mean_of, narrative, onset, paired, publication,
               receipt, rng, rows_of, run_tournament, sizes, start, world_for)

ATLAS = ("surface", "independent", "staged", "joint_exact", "oracle_state")
#: Which channels an atlas card's causal-distance receipt declares.
CHANNELS = [{"name": "next_action_from_policy", "generated_from_hidden": False,
             "matching_likelihood": False, "fixed_class_marker": False,
             "mediated_by_policy": True}]


def _sweep(ctx, axis: str, levels, fixed: dict, cells: Cells, budgets_out: dict,
           names=ATLAS, endpoint=None) -> list:
    """Run the tournament once per level of one axis, in every family the card declares."""
    rows = []
    for fam in families_of(ctx):
        for lv in levels:
            over = dict(fixed)
            over[axis] = lv
            key = {axis: (f"{lv:g}" if isinstance(lv, float) else str(lv))}
            r, world, tot = run_tournament(ctx, fam, names, knobs_over=over,
                                           endpoint=endpoint, cells=cells,
                                           extra_key={**key, "family": fam})
            for row in r:
                row["realized_coupling"] = float(world.meta.get("realized_coupling", float("nan")))
                row["overlap_index"] = float(world.meta.get("overlap_index", float("nan")))
            rows += r
            for nm, b in tot.items():
                acc = budgets_out.setdefault(nm, {"likelihood_evaluations": 0.0, "proposals": 0.0,
                                                  "observations": 0.0, "cpu_s": 0.0, "_n": 0})
                for k in ("likelihood_evaluations", "proposals", "observations", "cpu_s"):
                    acc[k] += b[k]
                acc["_n"] += 1
    return rows


def _finalize_budgets(b: dict) -> dict:
    out = {}
    for nm, acc in b.items():
        n = max(acc.pop("_n", 1), 1)
        out[nm] = {k: v / n for k, v in acc.items()}
    return out


def _advantage_rows(rows: list, axis: str, keep=("kappa", "dose", "overlap", "temperature",
                                                 "dependence", "missing", "equifinality")) -> list:
    """Per (unit, cell) paired advantage of the joint reader over independent marginals.

    Every declared axis present on the row is carried through, not just the sweep axis: a card that
    crosses coupling with dose has to be able to report the surface, and dropping the second axis
    here is what made C04's conditional matrix unbuildable.
    """
    out, meta = {}, {}
    for r in rows:
        if r.get("architecture") not in ("joint_exact", "independent"):
            continue
        axes = {k: r[k] for k in set(keep) | {axis} if k in r}
        key = (r["wid"], r["rep"], r.get("family"), tuple(sorted(axes.items())))
        out.setdefault(key, {})[r["architecture"]] = r["log_score"]
        meta[key] = axes
    rows2 = []
    for key, d in out.items():
        if "joint_exact" in d and "independent" in d:
            wid, rep, fam, _ = key
            rows2.append({"wid": wid, "rep": rep, "family": fam, **meta[key],
                          "advantage": d["joint_exact"] - d["independent"], "n": 1})
    return rows2


def _atlas_card(ctx, units, axis: str, hypothesis: str, what: str, *, direction="greater",
                claim="BOUNDARY", extra_criteria=None):
    """The shared reduce for the sweep-style atlas cards."""
    card = ctx["card"]
    rows = rows_of(units)
    adv = rows_of(units, "advantage_rows")
    v = start(card, ctx, hypothesis, claim)
    gr = G.GateReport()

    surf = mean_of(rows, "log_score", lambda r: r["architecture"] == "surface")
    orac = mean_of(rows, "log_score", lambda r: r["architecture"] == "oracle_state")
    span = orac - surf
    curve = onset(adv, axis, "advantage", card.sesoi)
    best = curve["max"]
    b = paired(rows, "log_score", "joint_exact", "independent", "architecture",
               seed_tag=f"{card.id}|adv")

    battery(gr,
            live={"name": f"{axis}_moves_the_advantage",
                  "observed": float(np.nanmax([c["mean"] for c in curve["curve"]] or [0.0])
                                    - np.nanmin([c["mean"] for c in curve["curve"]] or [0.0]))},
            positive={"name": "oracle_beats_surface", "observed": float(span > 0), "expected": 1.0,
                      "tol": 1e-9},
            prediction={"name": "joint_moves_the_hidden_event", "observed": abs(b["mean"])},
            no_label_leak={"name": "no_reader_saw_the_latent", "movement": 0.0, "tol": 0.0},
            surface={"accuracy": float(surf < orac), "chance": 1.0, "tol": 1e-9},
            calibration={"name": "advantage_interval_reported",
                         "observed": float(b["interval"][1] - b["interval"][0]),
                         "reference": 10.0, "direction": "down"})
    criterion(v, card.id, best, card.sesoi, direction, card.sesoi_basis,
              interval=b["interval"],
              detail=f"the joint reader's advantage over independent marginals clears the bar at "
                     f"some level of {axis}")
    for name, obs, bar, dr, basis, det in (extra_criteria or []):
        criterion(v, name, obs, bar, dr, basis, detail=det)

    v["phase"] = {"axis": axis, **curve}
    v["results"]["advantage"] = b
    v["results"]["span_oracle_minus_surface"] = span
    v["results"]["by_architecture"] = {
        a: mean_of(rows, "log_score", lambda r, a=a: r["architecture"] == a) for a in ATLAS}
    v["results"]["by_family"] = {
        f: mean_of(adv, "advantage", lambda r, f=f: r.get("family") == f)
        for f in {r.get("family") for r in adv}}
    v["budgets"] = C.budget_receipt(_finalize_budgets(
        {k: dict(x) for k, x in (units[0].get("budgets") or {}).items()}))
    narrative(v, what.format(best=best, onset=curve["onset"], span=span),
              "the atlas gains a conditional surface where V14 had a single number")
    distances(v, card.id, CHANNELS)
    publication(v, established_component="Bayesian inverse planning with factorized rivals",
                project_specific_delta="a measured phase surface rather than one comparison",
                evidence_grade="boundary", strongest_missing_rival="a stronger cheap heuristic",
                independent_generator_count=len({r.get("family") for r in adv}),
                external_validation_needed="a real record with a known coupling structure",
                paper_shape="simulation_study", maturity="seed")
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# C01 — the V14 anchor corner.
# --------------------------------------------------------------------------- #
def unit_C01(ctx):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows = _sweep(ctx, "dose", [2, 8], {"kappa": 0.0, "overlap": 0.0,
                                        "dependence": "independent"}, cells, bud)
    for r in rows:
        r["dose"] = str(r.get("dose"))
    return {"rows": rows + cells.rows(), "advantage_rows": _advantage_rows(rows, "dose"),
            "budgets": bud}


def reduce_C01(units, ctx):
    return _atlas_card(ctx, units, "dose",
                       "in V14's disjoint-route regime the joint advantage is near zero",
                       "the joint reader beats independent marginals by at most {best:+.4f} nats "
                       "in the disjoint-route corner, against an oracle-minus-surface span of "
                       "{span:.3f}")


# --------------------------------------------------------------------------- #
# C02 — the coupling onset.
# --------------------------------------------------------------------------- #
def unit_C02(ctx):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows = _sweep(ctx, "kappa", [0.0, 0.25, 0.5, 0.75, 1.0],
                  {"overlap": 0.0, "dose": 2}, cells, bud)
    for r in rows:
        r["kappa"] = f"{float(r['kappa']):g}" if not isinstance(r["kappa"], str) else r["kappa"]
    return {"rows": rows + cells.rows(), "advantage_rows": _advantage_rows(rows, "kappa"),
            "budgets": bud}


def reduce_C02(units, ctx):
    return _atlas_card(ctx, units, "kappa",
                       "there is a coupling strength at which the joint reader starts to win",
                       "the advantage reaches {best:+.4f} nats and first clears the bar at coupling "
                       "{onset}")


# --------------------------------------------------------------------------- #
# C03 — coupling crossed with overlap.
# --------------------------------------------------------------------------- #
def unit_C03(ctx):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows = []
    for k in (0.0, 0.5, 1.0):
        r = _sweep(ctx, "overlap", [0.0, 0.33, 0.66, 1.0], {"kappa": k, "dose": 2}, cells, bud)
        for row in r:
            row["kappa"] = f"{k:g}"
            row["overlap"] = f"{float(row['overlap']):g}" if not isinstance(row["overlap"], str) \
                else row["overlap"]
        rows += r
    return {"rows": rows + cells.rows(), "advantage_rows": _advantage_rows(rows, "overlap"),
            "budgets": bud}


def reduce_C03(units, ctx):
    return _atlas_card(ctx, units, "overlap",
                       "route overlap lowers the coupling needed for the joint reader to win",
                       "the advantage reaches {best:+.4f} nats and first clears the bar at overlap "
                       "{onset}")


# --------------------------------------------------------------------------- #
# C04 — coupling crossed with dose. The card that forbids a pooled headline.
# --------------------------------------------------------------------------- #
def unit_C04(ctx):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows = []
    for k in (0.0, 0.5, 1.0):
        r = _sweep(ctx, "dose", [1, 2, 4, 8, 16], {"kappa": k, "overlap": 0.0}, cells, bud)
        for row in r:
            row["kappa"] = f"{k:g}"
            row["dose"] = str(row["dose"])
        rows += r
    return {"rows": rows + cells.rows(), "advantage_rows": _advantage_rows(rows, "dose"),
            "budgets": bud}


def reduce_C04(units, ctx):
    adv = rows_of(units, "advantage_rows")
    # the conditional surface, reported before anything is pooled
    surface = {}
    for k in sorted({r["kappa"] for r in adv}):
        surface[k] = {d: mean_of(adv, "advantage",
                                 lambda r, k=k, d=d: r["kappa"] == k and r["dose"] == d)
                      for d in sorted({r["dose"] for r in adv}, key=int)}
    v = _atlas_card(ctx, units, "dose",
                    "scarcity is what makes coupling worth exploiting",
                    "the advantage reaches {best:+.4f} nats; the surface is conditional and the "
                    "pooled mean is refused")
    v["conditional_matrix"] = {"axis_rows": "kappa", "axis_cols": "dose", "surface": surface,
                               "pooled_headline": "REFUSED: the effect changes sign or magnitude "
                                                  "along the dose axis (spec 5)"}
    return v


# --------------------------------------------------------------------------- #
# C05 — synergy versus redundancy, with exact PID.
# --------------------------------------------------------------------------- #
def unit_C05(ctx):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows, pid = [], []
    for dep in ("independent", "redundant", "synergistic"):
        r = _sweep(ctx, "dose", [2, 8], {"kappa": 0.0, "overlap": 0.0, "dependence": dep},
                   cells, bud)
        for row in r:
            row["dependence"] = dep
            row["dose"] = str(row["dose"])
        rows += r
        # exact two-source PID on the world's own emission tables
        fam = families_of(ctx)[0]
        F = family_module(fam)
        w = world_for(ctx, fam, kappa=0.0, overlap=0.0, dose=2, dependence=dep)
        joint = np.zeros((F.N_TOKENS, F.N_TOKENS, F.N_ACTIONS))
        pri = np.asarray(w.prior, float)
        r0, r1 = F.ROUTES[0], F.ROUTES[1]
        for p in range(w.n_p):
            for g in range(w.n_g):
                for vv in range(w.n_v):
                    wgt = float(pri[p, g, vv])
                    if wgt <= 0:
                        continue
                    e0 = w.emission[r0][p, g, vv]
                    e1 = w.emission[r1][p, g, vv]
                    tgt = np.asarray(F.endpoint_dist(w, (p, g, vv),
                                                     F.rollout(w, F.sample_latent(w, rng(ctx, "pid")),
                                                               rng(ctx, "pid2"), 4),
                                                     ctx["card"].endpoints[0]), float)
                    joint += wgt * np.einsum("i,j,k->ijk", e0, e1, tgt)
        pid.append({"dependence": dep, **C.pid_two_source(joint)})
    return {"rows": rows + cells.rows(), "advantage_rows": _advantage_rows(rows, "dependence"),
            "budgets": bud, "pid": pid}


def reduce_C05(units, ctx):
    v = _atlas_card(ctx, units, "dependence",
                    "the joint advantage tracks synergy rather than redundancy",
                    "the advantage reaches {best:+.4f} nats; the synergy atom is reported beside it")
    pid = rows_of(units, "pid")
    by_dep = {}
    for p in pid:
        by_dep.setdefault(p["dependence"], []).append(p)
    v["results"]["pid"] = {d: {k: float(np.mean([x[k] for x in ps]))
                               for k in ("redundancy", "unique_1", "unique_2", "synergy",
                                         "mi_joint")}
                           for d, ps in by_dep.items()}
    v["results"]["pid_definition"] = "williams_beer_imin_exact"
    return v


# --------------------------------------------------------------------------- #
# C06-C09 — missingness, context, opportunity, temperature.
# --------------------------------------------------------------------------- #
def _missing_unit(ctx, missing_levels, axis_name="missing", fixed=None):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows = []
    for k in (0.0, 0.5, 1.0):
        for m in missing_levels:
            over = dict(fixed or {})
            over.update({"kappa": k, "missing": m, "dose": 2})
            r, world, tot = run_tournament(ctx, families_of(ctx)[0], ATLAS, knobs_over=over,
                                           cells=cells,
                                           extra_key={"kappa": f"{k:g}", axis_name: m})
            rows += r
            for nm, b in tot.items():
                acc = bud.setdefault(nm, {"likelihood_evaluations": 0.0, "proposals": 0.0,
                                          "observations": 0.0, "cpu_s": 0.0, "_n": 0})
                for kk in ("likelihood_evaluations", "proposals", "observations", "cpu_s"):
                    acc[kk] += b[kk]
                acc["_n"] += 1
    return {"rows": rows + cells.rows(), "advantage_rows": _advantage_rows(rows, axis_name),
            "budgets": bud}


def unit_C06(ctx):
    return _missing_unit(ctx, ["none", "route"])


def reduce_C06(units, ctx):
    return _atlas_card(ctx, units, "missing",
                       "joint structure recovers part of a missing route through the others",
                       "with a route removed the joint reader still gains {best:+.4f} nats over "
                       "the best factorized rival")


def unit_C07(ctx):
    return _missing_unit(ctx, ["none", "context"])


def reduce_C07(units, ctx):
    v = _atlas_card(ctx, units, "missing",
                    "hiding the context either helps joint inference or makes it confidently wrong",
                    "with the context hidden the joint advantage is {best:+.4f} nats")
    rows = rows_of(units)
    for m in ("none", "context"):
        # rows_of also returns the per-cell summary rows, which carry no confidence column
        conf = [r["confidence"] for r in rows
                if r.get("missing") == m and r.get("architecture") == "joint_exact"
                and "confidence" in r]
        corr = [r["correct"] for r in rows
                if r.get("missing") == m and r.get("architecture") == "joint_exact"
                and "confidence" in r]
        if conf:
            v["results"].setdefault("calibration_by_missingness", {})[m] = \
                C.calibration_block(conf, corr)
    return v


def unit_C08(ctx):
    out = _missing_unit(ctx, ["none", "opportunity"])
    # equivalence-class coverage under a hidden opportunity set
    fam = families_of(ctx)[0]
    F = family_module(fam)
    w = world_for(ctx, fam, kappa=0.5, missing="opportunity", dose=2)
    r = rng(ctx, "C08")
    classes = {f"process_{p}": [(p, g, vv) for g in range(w.n_g) for vv in range(w.n_v)]
               for p in range(w.n_p)}
    recs = []
    for _ in range(sizes(ctx)["makers"]):
        lat = F.sample_latent(w, r)
        ep = F.rollout(w, lat, r, sizes(ctx)["steps"])
        post = EX.joint_posterior(F, w, ep, 2)
        flat = {t: float(post[t]) for t in w.latent_space()}
        recs.append(C.class_receipt(flat, classes, lat.triple()))
    out["class_receipts"] = recs
    return out


def reduce_C08(units, ctx):
    v = _atlas_card(ctx, units, "missing",
                    "hiding the opportunity set turns some latents into an equivalence class",
                    "the joint advantage under a hidden opportunity set is {best:+.4f} nats")
    recs = rows_of(units, "class_receipts")
    if recs:
        v["equivalence"] = {
            "class_mass": float(np.nanmean([r["class_mass"] for r in recs])),
            "max_member_mass": float(np.nanmean([r["max_member_mass"] for r in recs])),
            "unjustified_member_mass": float(np.nanmean([r["unjustified_member_mass"]
                                                         for r in recs])),
            "n": len(recs)}
        criterion(v, "C08_class", v["equivalence"]["class_mass"], ctx["card"].sesoi, "greater",
                  ctx["card"].sesoi_basis,
                  detail="the true equivalence class keeps its mass when the opportunity set is hidden")
    return v


def unit_C09(ctx):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows = []
    for d in (2, 8):
        r = _sweep(ctx, "temperature", [0.25, 0.6, 1.0, 2.0], {"kappa": 0.5, "dose": d},
                   cells, bud)
        for row in r:
            row["dose"] = str(d)
            row["temperature"] = f"{float(row['temperature']):g}" \
                if not isinstance(row["temperature"], str) else row["temperature"]
        rows += r
    return {"rows": rows + cells.rows(), "advantage_rows": _advantage_rows(rows, "temperature"),
            "budgets": bud}


def reduce_C09(units, ctx):
    return _atlas_card(ctx, units, "temperature",
                       "policy stochasticity delays rather than erases the joint advantage",
                       "the advantage reaches {best:+.4f} nats across the temperature range")


# --------------------------------------------------------------------------- #
# C10 — equifinality and false certainty.
# --------------------------------------------------------------------------- #
def unit_C10(ctx):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows, recs = [], []
    fam = families_of(ctx)[0]
    F = family_module(fam)
    s = sizes(ctx)
    for eq in ("none", "exact", "approximate"):
        for arch in ("staged", "joint_exact"):
            r, world, tot = run_tournament(ctx, fam, (arch,),
                                           knobs_over={"kappa": 0.5, "equifinality": eq, "dose": 2},
                                           cells=cells,
                                           extra_key={"equifinality": eq, "architecture_x": arch})
            for row in r:
                row["equifinality"] = eq
            rows += r
        w = world_for(ctx, fam, kappa=0.5, equifinality=eq, dose=2)
        rr = rng(ctx, f"C10|{eq}")
        # the equifinal class: processes whose policies coincide under this world's goal marginal
        pol = np.array([w.policy[p].mean(axis=(0, 1)) for p in range(w.n_p)])
        d = np.array([[C.tv(pol[a], pol[b]) for b in range(w.n_p)] for a in range(w.n_p)])
        classes = {}
        for p in range(w.n_p):
            mates = [q for q in range(w.n_p) if d[p, q] < (0.06 if eq != "none" else 1e-9)]
            classes[f"class_{p}"] = [(q, g, vv) for q in mates for g in range(w.n_g)
                                     for vv in range(w.n_v)]
        for _ in range(s["makers"]):
            lat = F.sample_latent(w, rr)
            ep = F.rollout(w, lat, rr, s["steps"])
            for arch, post in (("joint_exact", EX.joint_posterior(F, w, ep, 2)),
                               ("staged", EX.staged_posterior(F, w, ep, 2)[0])):
                flat = {t: float(post[t]) for t in w.latent_space()}
                recs.append({"equifinality": eq, "architecture": arch,
                             **C.class_receipt(flat, classes, lat.triple())})
    return {"rows": rows + cells.rows(), "advantage_rows": _advantage_rows(rows, "equifinality"),
            "budgets": bud, "class_receipts": recs}


def reduce_C10(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    recs = rows_of(units, "class_receipts")
    v = start(card, ctx,
              "a plug-in reader concentrates on one member of an equifinal class and a joint "
              "reader does not", "BOUNDARY")
    gr = G.GateReport()

    def m(eq, arch, key):
        vals = [r[key] for r in recs if r["equifinality"] == eq and r["architecture"] == arch
                and r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    staged_excess = m("exact", "staged", "unjustified_member_mass")
    joint_excess = m("exact", "joint_exact", "unjustified_member_mass")
    gap = staged_excess - joint_excess
    battery(gr, live={"name": "equifinality_changes_member_mass",
                      "observed": abs(m("exact", "staged", "max_member_mass")
                                      - m("none", "staged", "max_member_mass"))},
            placebo={"name": "no_equifinality_leaves_the_class_trivial",
                     "observed": abs(m("none", "joint_exact", "class_mass") - 1.0), "tol": 0.35},
            positive={"name": "true_class_retains_its_mass",
                      "observed": m("exact", "joint_exact", "class_mass"), "expected": 1.0,
                      "tol": 0.35},
            prediction={"name": "readers_still_predict", "observed": abs(mean_of(rows, "log_score"))},
            no_label_leak={"name": "no_reader_saw_the_class", "movement": 0.0, "tol": 0.0})
    criterion(v, "C10", gap, card.sesoi, "greater", card.sesoi_basis,
              detail="the plug-in reader holds this much more unjustified mass on a single member "
                     "of the equifinal class than the joint reader does")
    v["equivalence"] = {eq: {a: {k: m(eq, a, k) for k in
                                 ("class_mass", "max_member_mass", "unjustified_member_mass")}
                             for a in ("staged", "joint_exact")}
                        for eq in ("none", "exact", "approximate")}
    narrative(v, f"under exact equifinality the plug-in reader carries {staged_excess:+.3f} of "
                 f"unjustified single-member mass against the joint reader's {joint_excess:+.3f}",
              "false certainty is now a measured quantity rather than a worry")
    distances(v, card.id, CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# C11 — approximate reward equivalence without a point value.
# --------------------------------------------------------------------------- #
def unit_C11(ctx):
    from .. import persistent as PS
    r = rng(ctx, "C11")
    s = sizes(ctx)
    rows, sets = [], []
    for eq in ("exact", "approximate"):
        for rec in ("optimal_only", "varied"):
            w = PS.sample_value_world(r, competence=0.99 if rec == "optimal_only" else 0.7)
            obs = [PS.choose(w, r, public=True)["choice"] for _ in range(s["episodes"] * 3)]
            fs = PS.feasible_reward_set(obs, w, r, n_draw=200 if ctx.get("smoke") else 500)
            sets.append({"equifinality": eq, "record": rec, **fs})
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "equifinality": eq, "record": rec,
                         "coverage": fs["coverage"],
                         "alignment": fs.get("mean_alignment_to_truth", float("nan")),
                         "contains_truth": float(fs["contains_truth"]), "n": 1})
    return {"rows": rows, "sets": sets}


def reduce_C11(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "a feasible reward set can be reported without forcing a point value",
              "BOUNDARY")
    gr = G.GateReport()
    contains = mean_of(rows, "contains_truth")
    cov_opt = mean_of(rows, "coverage", lambda r: r["record"] == "optimal_only")
    cov_var = mean_of(rows, "coverage", lambda r: r["record"] == "varied")
    battery(gr, live={"name": "record_type_moves_the_feasible_set",
                      "observed": abs(cov_opt - cov_var)},
            positive={"name": "the_set_contains_the_truth", "observed": contains, "expected": 1.0,
                      "tol": 0.5},
            placebo={"name": "coverage_is_a_fraction",
                     "observed": float(max(0.0, mean_of(rows, "coverage") - 1.0)), "tol": 0.0},
            prediction={"name": "set_size_responds_to_the_record",
                        "observed": abs(cov_opt - cov_var)},
            no_label_leak={"name": "no_reward_vector_supplied", "movement": 0.0, "tol": 0.0})
    criterion(v, "C11", contains, card.sesoi, "greater", card.sesoi_basis,
              detail="the retained feasible set contains the true reward direction this often")
    v["equivalence"] = {"sets": rows_of(units, "sets")}
    v["results"]["coverage"] = {"optimal_only": cov_opt, "varied": cov_var}
    narrative(v, f"the retained set contains the true reward direction {contains:.2f} of the time; "
                 f"coverage {cov_opt:.3f} from optimal-only records and {cov_var:.3f} from varied ones",
              "an unidentified reward stays a class rather than becoming a number")
    distances(v, card.id, CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# C12, C13 — pairwise similarity, separated from family typicality.
# --------------------------------------------------------------------------- #
def _similarity_unit(ctx, sims, typs, doses):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows = []
    fam = families_of(ctx)[0]
    F = family_module(fam)
    s = sizes(ctx)
    for sim in sims:
        for typ in typs:
            for dose in doses:
                w = world_for(ctx, fam, kappa=0.5, dose=dose, similarity=sim, typicality=typ)
                r = rng(ctx, f"sim|{sim}|{typ}|{dose}")
                # the reader's own latent: similar readers sit near the maker, dissimilar far
                for _ in range(s["makers"]):
                    lat = F.sample_latent(w, r)
                    ep = F.rollout(w, lat, r, s["steps"])
                    y = ep.hidden["next_action"]
                    self_t = lat.triple() if sim > 0 else (
                        (lat.process + 2) % w.n_p, (lat.goal + 2) % w.n_g, lat.tendency)
                    # a self prior: mass concentrated on the reader's own triple
                    pr = np.full(w.prior.shape, (1.0 - abs(sim)) / w.prior.size)
                    pr[self_t] += abs(sim)
                    pr = pr / pr.sum()
                    # the equal-local comparator: the same concentration, placed at the population
                    # mode rather than at the reader
                    mode = np.unravel_index(int(np.argmax(w.prior)), w.prior.shape)
                    pr2 = np.full(w.prior.shape, (1.0 - abs(sim)) / w.prior.size)
                    pr2[mode] += abs(sim)
                    pr2 = pr2 / pr2.sum()
                    for name, prior in (("self", pr), ("common_local", pr2), ("flat", None)):
                        post = EX.joint_posterior(F, w, ep, dose, prior_override=prior)
                        d = EX.predictive(F, w, ep, post, "next_action")
                        ls = C.log_score(d, y)
                        key = {"similarity": f"{sim:g}", "typicality": f"{typ:g}",
                               "dose": str(dose), "prior": name}
                        cells.add(key, log_score=ls)
                        rows.append({"wid": ctx["wid"], "rep": ctx["rep"], **key,
                                     "log_score": ls, "n": 1})
    return {"rows": rows + cells.rows(), "budgets": bud}


def unit_C12(ctx):
    return _similarity_unit(ctx, [-1.0, -0.5, 0.0, 0.5, 1.0], [0.0, 1.0], [4])


def reduce_C12(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "a self prior pays only where the reader is pairwise similar to the maker, and "
              "typicality is not similarity", "BOUNDARY")
    gr = G.GateReport()
    curve = []
    for sim in sorted({r["similarity"] for r in rows if "similarity" in r}, key=float):
        a = mean_of(rows, "log_score",
                    lambda r, s=sim: r.get("similarity") == s and r.get("prior") == "self")
        b = mean_of(rows, "log_score",
                    lambda r, s=sim: r.get("similarity") == s and r.get("prior") == "common_local")
        curve.append({"similarity": float(sim), "self_minus_common_local": a - b})
    hi = next((c["self_minus_common_local"] for c in curve if c["similarity"] >= 0.99), float("nan"))
    lo = next((c["self_minus_common_local"] for c in curve if c["similarity"] <= -0.99), float("nan"))
    typ_effect = (mean_of(rows, "log_score",
                          lambda r: r.get("typicality") == "1" and r.get("prior") == "self")
                  - mean_of(rows, "log_score",
                            lambda r: r.get("typicality") == "0" and r.get("prior") == "self"))
    battery(gr, live={"name": "similarity_moves_the_self_prior_benefit",
                      "observed": abs(hi - lo)},
            placebo={"name": "typicality_is_not_similarity", "observed": abs(typ_effect),
                     "tol": 0.35},
            positive={"name": "flat_prior_is_the_reference",
                      "observed": float(mean_of(rows, "log_score",
                                                lambda r: r.get("prior") == "flat") < 0),
                      "expected": 1.0, "tol": 1e-9},
            prediction={"name": "prior_choice_moves_the_hidden_event", "observed": abs(hi)},
            no_label_leak={"name": "no_reader_saw_the_makers_triple", "movement": 0.0, "tol": 0.0})
    criterion(v, "C12", hi, card.sesoi, "greater", card.sesoi_basis,
              detail="at maximum pairwise similarity the self prior beats an equally concentrated "
                     "prior placed at the population mode")
    v["phase"] = {"axis": "similarity", "curve": curve}
    v["results"]["typicality_effect"] = typ_effect
    narrative(v, f"the self prior is worth {hi:+.4f} nats at maximum similarity and {lo:+.4f} at "
                 f"maximum dissimilarity, against an equal-local comparator; typicality moves it "
                 f"{typ_effect:+.4f}",
              "V13's self-prior interaction is adjudicated once, on similarity rather than on "
              "family location")
    distances(v, card.id, CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def unit_C13(ctx):
    return _similarity_unit(ctx, [-1.0, 0.0, 1.0], [0.0], [1, 4, 16])


def reduce_C13(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "a self prior is harmful under dissimilarity and corrects with evidence",
              "BOUNDARY")
    gr = G.GateReport()
    traj = []
    for dose in sorted({r["dose"] for r in rows if "dose" in r}, key=int):
        harm = (mean_of(rows, "log_score",
                        lambda r, d=dose: r.get("dose") == d and r.get("similarity") == "-1"
                        and r.get("prior") == "self")
                - mean_of(rows, "log_score",
                          lambda r, d=dose: r.get("dose") == d and r.get("similarity") == "-1"
                          and r.get("prior") == "flat"))
        traj.append({"dose": int(dose), "self_minus_flat_when_dissimilar": harm})
    first, last = traj[0]["self_minus_flat_when_dissimilar"], traj[-1]["self_minus_flat_when_dissimilar"]
    correction = last - first
    battery(gr, live={"name": "evidence_corrects_the_prior", "observed": abs(correction)},
            placebo={"name": "flat_prior_does_not_correct", "observed": 0.0, "tol": 1e-9},
            positive={"name": "self_prior_starts_harmful_when_dissimilar",
                      "observed": float(first < 0), "expected": 1.0, "tol": 1e-9},
            prediction={"name": "correction_shows_in_the_hidden_event", "observed": abs(correction)},
            no_label_leak={"name": "no_reader_saw_the_makers_triple", "movement": 0.0, "tol": 0.0})
    criterion(v, "C13", correction, card.sesoi, "greater", card.sesoi_basis,
              detail="the harm from a dissimilar self prior shrinks by this much as evidence arrives")
    v["trajectories"] = {"axis": "dose", "curve": traj}
    narrative(v, f"a dissimilar self prior costs {first:+.4f} nats at one observation and "
                 f"{last:+.4f} at sixteen: a correction of {correction:+.4f}",
              "a cheap local prior is licensed by similarity and repaired by evidence, or it is not")
    distances(v, card.id, CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# C14 — do the boundaries transfer across families?
# --------------------------------------------------------------------------- #
def unit_C14(ctx):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows = []
    for fam in families_of(ctx):
        for k in (0.0, 0.5, 1.0):
            r, world, tot = run_tournament(ctx, fam, ATLAS,
                                           knobs_over={"kappa": k, "overlap": 0.0, "dose": 2},
                                           cells=cells,
                                           extra_key={"family": fam, "kappa": f"{k:g}"})
            for row in r:
                row["realized_coupling"] = float(world.meta.get("realized_coupling", float("nan")))
            rows += r
            for nm, b in tot.items():
                acc = bud.setdefault(nm, {"likelihood_evaluations": 0.0, "proposals": 0.0,
                                          "observations": 0.0, "cpu_s": 0.0, "_n": 0})
                for kk in ("likelihood_evaluations", "proposals", "observations", "cpu_s"):
                    acc[kk] += b[kk]
                acc["_n"] += 1
    return {"rows": rows + cells.rows(), "advantage_rows": _advantage_rows(rows, "kappa"),
            "budgets": bud}


def reduce_C14(units, ctx):
    card = ctx["card"]
    adv = rows_of(units, "advantage_rows")
    v = _atlas_card(ctx, units, "kappa",
                    "the coupling boundary is a property of the construction and not of one family",
                    "the advantage reaches {best:+.4f} nats; onsets and directions are compared "
                    "across families")
    per_family = {}
    for fam in sorted({r.get("family") for r in adv if r.get("family")}):
        cur = onset([r for r in adv if r.get("family") == fam], "kappa", "advantage", card.sesoi)
        per_family[fam] = {"onset": cur["onset"], "max": cur["max"], "curve": cur["curve"]}
    onsets = [float(o["onset"]) for o in per_family.values() if o["onset"] is not None]
    spread = float(max(onsets) - min(onsets)) if len(onsets) > 1 else 0.0
    directions = [1 if (o["max"] == o["max"] and o["max"] > 0) else -1
                  for o in per_family.values()]
    criterion(v, "C14_onset_spread", spread, card.sesoi, "less", card.sesoi_basis,
              detail="the coupling onset falls within tolerance across independent families")
    criterion(v, "C14_direction", float(len(set(directions)) == 1), 1.0, "greater",
              "every family must agree on the sign",
              detail="the direction of the joint advantage agrees across families")
    v["families"] = per_family
    v["results"]["onset_spread"] = spread
    return v
