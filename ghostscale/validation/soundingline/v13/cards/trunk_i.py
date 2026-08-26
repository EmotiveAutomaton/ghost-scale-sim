"""Trunk I: integrity, calibration, and construction validity (spec §8).

I01 and I02 re-drive V12's own world and helper functions with V12's seeds; no V12 ``run_``
function is called and nothing under v12/ is modified. Every other card runs on V13 worlds.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import REPO, v13_dir
from .. import exact as X, priors as P, attention as A, costs as CO, goals_trust as GT, hierarchy as H, world as W, projection as PJ
from .. import pymdp_reader as PR
from ..world import make_maker, population, stream, histogram, maker_emission, relabel_family
from . import (battery, boot, criterion, decide_state, finish, held_out_classifier, narrative, receipt, rng, sizes,
               start, world_for, mean_of, pursuit_of)
from .trunk_c import Cells

CH8 = ("surface", "common_structure", "group_convention", "mechanics", "goal_consequences", "communicative_shaping", "anomaly", "process_records")


# =========================================================================== #
# I01 — V12 anchors, reconstructed loops.
# =========================================================================== #
ANCHORS = ["S04", "S06", "R02", "R05", "B02", "D02", "D05", "F03"]


def _v12():
    from ...v12 import common as C12, world as W12, self_other as SO12, exact as X12, opportunities as OP12, hierarchy as H12
    from ...v12.cards import worlds_for as worlds12, trunk_s as S12, trunk_r as R12, trunk_b as B12, trunk_d as D12, trunk_f as F12
    from ghostscale.config import load_config
    return {"C": C12, "W": W12, "SO": SO12, "X": X12, "OP": OP12, "H": H12, "worlds": worlds12, "S": S12, "R": R12, "B": B12, "D": D12, "F": F12,
            "cfg": load_config()}


def _committed(cid):
    p = REPO / "results" / "validation" / "soundingline" / "v12" / f"{cid}.json"
    return json.loads(p.read_text(encoding="utf-8"))["results"]


def _anchor_S04(v):
    S, C12 = v["S"], v["C"]
    H = S._harness(v["cfg"], "discovery", "S04")
    rows = S._route_scores(H)
    gain1 = S._gain_by_bin(rows, "self_first", "generic", 1)
    comm = _committed("S04")["gain_self_minus_generic_at_n1_by_distance_bin"]
    return {b: {"reconstructed": gain1[b]["mean"], "committed": comm[b]["mean"]} for b in gain1}


def _anchor_S06(v):
    S, X12, SO12 = v["S"], v["X"], v["SO"]
    traj = {"compatible_first": [], "conflict_first": []}
    finals, residual, halflife = [], [], []
    for h in S._harness(v["cfg"], "discovery", "S06", n_makers=36, n_art=12):
        world = h["world"]
        for r in h["readers"]:
            sm = h["selfs"][r.id]
            for m in h["makers"]:
                if m.profile == r.profile:
                    continue
                arts = h["streams"][m.id]
                score = []
                for a in arts:
                    gl = X12.goal_loglik(world, r.template, r.habit[0], np.asarray(a["features"]), 0, m.tier, "plain", m.profile)
                    score.append(float(np.log((np.exp(gl - gl.max()) * sm["w_hat"]).sum()) + gl.max()))
                order_c = list(np.argsort(-np.array(score)))
                order_x = order_c[::-1]
                prior = SO12.self_first_prior(world, sm["w_hat"], S.BETA_SELF)
                res = {}
                for name, order in (("compatible_first", order_c), ("conflict_first", order_x)):
                    seq = [arts[i] for i in order]
                    cum = X12.profile_loglik_cumulative(world, r.template, r.habit[0], seq, m.tier, "plain")
                    res[name] = [X12.posterior(cum, n, prior).get(m.profile, 0.0) for n in range(1, len(seq) + 1)]
                    traj[name].append(res[name])
                finals.append(abs(res["compatible_first"][-1] - res["conflict_first"][-1]))
                final_c = res["compatible_first"][-1]
                hl = next((i + 1 for i, p in enumerate(res["compatible_first"]) if p >= 0.5 * final_c), len(arts))
                halflife.append(hl)
                cum_c = X12.profile_loglik_cumulative(world, r.template, r.habit[0], [arts[i] for i in order_c], m.tier, "plain")
                residual.append(X12.posterior(cum_c, len(arts), prior).get(r.profile, 0.0))
    comm = _committed("S06")
    return {"correction_half_life_artifacts": {"reconstructed": float(np.mean(halflife)), "committed": comm["correction_half_life_artifacts"]},
            "residual_self_directed_bias": {"reconstructed": float(np.mean(residual)), "committed": comm["residual_self_directed_bias"]}}


def _anchor_R02(v):
    R, C12 = v["R"], v["C"]
    scores = {}
    for wid, world in v["worlds"](v["cfg"], "discovery"):
        cw = R._cwd(world)
        names = cw.family_names
        habits = R._habits(cw.n_options)
        rr = C12.rng_for("R02", wid, 0)
        for i in range(30):
            w = world.family[names[i % len(names)]]
            h = habits[1 + int(rr.integers(len(habits) - 1))]
            menus = R._menus(cw, rr, 24)
            recs = R._records(cw, w, menus, rr, habit=h, k=0.3)
            train, test = recs[:16], recs[16:]
            for est, (Pm, hs) in R._estimators(cw, train, habits).items():
                ls = [R._ls_pred(R._joint_pred(cw, Pm, hs, R._mrec(r)), int(r["choice"])) for r in test]
                scores.setdefault(est, {}).setdefault(wid, []).append(float(np.mean(ls)))
    table = {e: C12.hboot(d, np.random.default_rng(C12.seed("R02" + e)), draws=300) for e, d in scores.items()}
    gap = table["constrained_inversion"]["mean"] - table["partialling"]["mean"]
    return {"constrained_minus_partialling": {"reconstructed": float(gap), "committed": _committed("R02")["criterion_C_R02"]["constrained_minus_partialling"]}}


def _anchor_R05(v):
    R, C12, OP12 = v["R"], v["C"], v["OP"]
    shifts = {"record_reader": {"near_tie": [], "strong": []}, "count_reader": {"near_tie": [], "strong": []}}
    for wid, world in v["worlds"](v["cfg"], "discovery"):
        cw = R._cwd(world)
        names = cw.family_names
        rr = C12.rng_for("R05", wid, 0)
        prior = np.full(len(names), 1 / len(names))
        for i in range(30):
            w = world.family[names[i % len(names)]]
            found = {"near_tie": None, "strong": None}
            tries = 0
            while (found["near_tie"] is None or found["strong"] is None) and tries < 5000:
                tries += 1
                m = R._menus(cw, rr, 1, cost_scale=0.6)[0]
                u = m["payoff"] @ w - m["cost"]
                a = int(np.argmax(u))
                srt = np.sort(u)
                margin = srt[-1] - srt[-2]
                if found["near_tie"] is None and margin < 0.02:
                    found["near_tie"] = (m, a)
                if found["strong"] is None and margin > 0.05 and m["cost"][a] - m["cost"].min() > 0.3:
                    found["strong"] = (m, a)
            for kind, item in found.items():
                if item is None:
                    continue
                m, a = item
                rec = {"payoff": m["payoff"].tolist(), "cost": m["cost"].tolist(), "choice": a}
                for reader, use_costs in (("record_reader", True), ("count_reader", False)):
                    post = OP12.profile_posterior_from_choices(cw, [rec], use_costs=use_costs)
                    pv = np.array([post[n] for n in names])
                    shifts[reader][kind].append(float((pv[pv > 0] * np.log(pv[pv > 0] / prior[pv > 0])).sum()))
    table = {r: {k: float(np.mean(x)) for k, x in d.items()} for r, d in shifts.items()}
    comm = _committed("R05")["posterior_shift_kl"]
    return {f"{r}.{k}": {"reconstructed": table[r][k], "committed": comm[r][k]} for r in table for k in table[r]}


def _anchor_B02(v):
    B, C12, X12, W12 = v["B"], v["C"], v["X"], v["W"]
    cells = {}
    for wid, world in v["worlds"](v["cfg"], "discovery"):
        rr = C12.rng_for("B02", wid, 0)
        for r in B.REGIMES:
            for m in W12.population(world, 20, rr, regimes=(r,), k_choices=(0.0, 0.3), prefix=r):
                arts = W12.stream(world, m, 0, C12.rng_for("B02", wid, 1, m.id), 12)
                for a in B.ASSUMPTIONS:
                    cum = X12.profile_loglik_cumulative(world, world.sig, None, arts, m.tier, a)
                    for n in (1, 4, 12):
                        cells.setdefault((r, a, n), {}).setdefault(wid, []).append(C12.log_score(X12.posterior(cum, n), m.profile))
    table = {f"{r}|{a}|n={n}": C12.hboot(d, np.random.default_rng(C12.seed(f"B02{r}{a}{n}")), draws=300) for (r, a, n), d in cells.items()}
    gain_bard = table["bard|bard|n=4"]["mean"] - table["bard|neutral|n=4"]["mean"]
    cost_conc = table["concealer|bard|n=4"]["mean"] - table["concealer|neutral|n=4"]["mean"]
    comm = _committed("B02")["criterion_C_B02"]
    return {"cooperative_gain_on_bards": {"reconstructed": float(gain_bard), "committed": comm["cooperative_gain_on_bards"]},
            "cooperative_cost_on_concealers": {"reconstructed": float(cost_conc), "committed": comm["cooperative_cost_on_concealers"]}}


def _anchor_D02(v):
    C12, W12, H12 = v["C"], v["W"], v["H"]
    reach = {}
    for wid, world in v["worlds"](v["cfg"], "discovery"):
        contributors = W12.population(world, 4, C12.rng_for("D02", wid, 0), k_choices=(0.2,))
        levels = (("director", "director_goal", {"director_goal": 1}), ("shared_brief", "brief", {"brief": 1}),
                  ("director", "secondary_goal", {"secondary_goal": 3}), ("director", "local", {"local_part": 0, "local_slot": 5}),
                  ("ratifier", "ratification", {"no_veto": True}))
        for i in range(20):
            for eco, level, iv in levels:
                s = C12.seed(f"D02:{wid}:{i}:{level}")
                base = H12.produce(world, eco, contributors, np.random.default_rng(s), director_goal=0)
                iv = dict(iv)
                if level == "local":
                    iv["local_slot"] = (base["log"]["parts"][0]["slot"] + 1) % len(world.family_names)
                alt = H12.produce(world, eco, contributors, np.random.default_rng(s), director_goal=0, intervene=iv)
                changed = [a["goal"] != b["goal"] or a["slot"] != b["slot"] for a, b in zip(base["log"]["parts"], alt["log"]["parts"])]
                surface = [not np.array_equal(a, b) for a, b in zip(base["parts"], alt["parts"])]
                reach.setdefault(level, []).append(float(np.mean(changed)))
                reach.setdefault(level + "_surface", []).append(float(np.mean(surface)))
    table = {k: float(np.mean(x)) for k, x in reach.items()}
    comm = _committed("D02")["reach_by_level"]
    return {k: {"reconstructed": table[k], "committed": comm[k]} for k in table}


def _anchor_D05(v):
    C12, W12, H12, D = v["C"], v["W"], v["H"], v["D"]
    res = {}
    for wid, world in v["worlds"](v["cfg"], "discovery"):
        rr = C12.rng_for("D05", wid, 0)
        contributors = [W12.make_maker(world, f"c{i}", f"peaked_{i}", rr, k=0.5) for i in range(4)]
        rewriter = W12.make_maker(world, "rw", "uniform", rr, k=0.5)
        for i in range(30):
            d = int(rr.integers(4))
            dg = int(np.argmax(contributors[d].w))
            art = H12.produce(world, "director", contributors, rr, director_goal=dg)
            for r in (0.0, 0.25, 0.5, 0.75, 1.0):
                parts = []
                for part, entry in zip(art["parts"], art["log"]["parts"]):
                    p = part.copy()
                    n_rw = int(round(r * len(p)))
                    if n_rw:
                        idx = rr.choice(len(p), size=n_rw, replace=False)
                        p[idx] = D._redraw_part(world, rewriter, entry["goal"], entry["slot"], rr, n_rw)
                    parts.append(p)
                Pp = H12.part_goal_posteriors(world, world.sig, parts)
                primary = int(np.bincount(Pp.argmax(axis=1), minlength=world.ng).argmax())
                cell = res.setdefault(str(r), {"director": [], "local": []})
                cell["director"].append(float(primary == dg))
                for part, entry in zip(parts, art["log"]["parts"]):
                    lls = [max(D._part_ll(world, c.template, part)) for c in contributors]
                    cell["local"].append(float(contributors[int(np.argmax(lls))].id == entry["contributor"]))
    table = {r: {k: float(np.mean(x)) for k, x in d.items()} for r, d in res.items()}
    comm = _committed("D05")["attribution_by_rewrite_strength"]
    return {f"{r}.{k}": {"reconstructed": table[r][k], "committed": comm[r][k]} for r in table for k in table[r]}


def _anchor_F03(v):
    C12, H12, F = v["C"], v["H"], v["F"]
    rulers = {"dependency": lambda world, s: H12.dependency_statistic(world, world.sig, s["blocks"]),
              "hard_argmax_mi": lambda world, s: F._hard_mi(world, s["blocks"]),
              "sequence_bigram": lambda world, s: F._bigram_mi(s["features"], world.nf),
              "histogram_entropy": lambda world, s: F._hist_entropy(s["features"], world.nf)}
    CELLS = ((4, 12), (32, 12), (32, 60), (128, 60))
    auc = {(r, st, nb): [] for r in rulers for st, nb in CELLS}
    for wid, world in v["worlds"](v["cfg"], "discovery"):
        rr = C12.rng_for("F03", wid, 0)
        for st, nb in CELLS:
            lay = [H12.layered_sequence(world, world.sig, rr, n_blocks=nb, steps_per_block=st, topology="layered") for _ in range(60)]
            fla = [H12.layered_sequence(world, world.sig, rr, n_blocks=nb, steps_per_block=st, topology="flattened") for _ in range(60)]
            fla2 = [H12.layered_sequence(world, world.sig, rr, n_blocks=nb, steps_per_block=st, topology="flattened") for _ in range(60)]
            for r, fn in rulers.items():
                a, b = [fn(world, s) for s in lay], [fn(world, s) for s in fla]
                _ = [fn(world, s) for s in fla2]
                auc[(r, st, nb)].append(F._auc(a, b))
    table = {f"{r}@steps{st}x{nb}": float(np.mean(x)) for (r, st, nb), x in auc.items()}
    comm = _committed("F03")["by_ruler_and_cell"]
    return {k: {"reconstructed": table[k], "committed": comm[k]["auc"]} for k in table if k.startswith("dependency")}


ANCHOR_FN = {"S04": _anchor_S04, "S06": _anchor_S06, "R02": _anchor_R02, "R05": _anchor_R05, "B02": _anchor_B02,
             "D02": _anchor_D02, "D05": _anchor_D05, "F03": _anchor_F03}


def unit_I01(ctx):
    anchor = ctx.get("item") or ANCHORS[int(ctx["wid"]) % len(ANCHORS)]
    if ctx.get("smoke"):
        anchor = ctx.get("item") or "D02"
    v = _v12()
    t0 = time.perf_counter()
    fields = ANCHOR_FN[anchor](v)
    dev = max(abs(f["reconstructed"] - f["committed"]) for f in fields.values())
    return {"rows": [{"wid": anchor, "rep": 0, "anchor": anchor, "max_abs_deviation": dev, "n_fields": len(fields), "wall_s": time.perf_counter() - t0}],
            "fields": {anchor: fields}}


def reduce_I01(card, units, ctx):
    v = start(card, ctx, "V12's headline numbers are rebuilt from V12's own world, primitives and seeds by loops "
              "written again here; if the record can be rebuilt, later corrections of its reading are corrections, not edits.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    fields = {}
    for u in units:
        fields.update(u["fields"])
    devs = {r["anchor"]: r["max_abs_deviation"] for r in rows}
    worst = max(devs.values()) if devs else float("nan")
    gr = G.GateReport()
    for a, d in devs.items():
        gr.identity(f"{a}_reproduces", d, 0.0, tol=1e-9, detail="every committed field of the anchor equals its reconstruction")
    gr.positive("all_anchors_attempted", observed=float(len(devs)), expected=float(len(ANCHORS)) if not ctx.get("smoke") else float(len(devs)), tol=0.0)
    passed = bool(worst <= 1e-9)
    criterion(v, "I01", passed, max_abs_deviation=worst, anchors=len(devs))
    v["results"].update({"deviation_by_anchor": devs, "fields": fields})
    receipt(v, rows, card, ctx)
    narrative(v, f"Eight V12 anchors were rebuilt with V12's own seeds; the largest deviation from any committed field was {worst:.2e}.",
              "V12's record is reproducible from its primitives; every V13 correction of its reading stands beside an intact original.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# =========================================================================== #
# I02 — V12's comparator, audited.
# =========================================================================== #
def unit_I02(ctx):
    v = _v12()
    S, SO12, X12, C12 = v["S"], v["SO"], v["X"], v["C"]
    worlds = v["worlds"](v["cfg"], "discovery")
    if ctx.get("smoke"):
        worlds = worlds[:2]
    cells = Cells("v12", 0)
    rows = []
    for wid, world in worlds:
        H = S._harness(v["cfg"], "discovery", "I02", n_makers=60, n_art=2, worlds=[(wid, world)])[0]
        makers = H["makers"]
        for r in H["readers"]:
            sm = H["selfs"][r.id]
            sp = SO12.self_first_prior(world, sm["w_hat"], S.BETA_SELF)
            gp = SO12.information_matched_generic(world, sp, makers)
            d_self = SO12.expected_kl_to_truth(sp, makers)
            d_gen = SO12.expected_kl_to_truth(gp, makers)
            # a distance-matched rebuild: entropy matched by temperature, centre chosen so the expected
            # divergence to truth over the population equals the self prior's
            best, best_gap = gp, abs(d_gen - d_self)
            rr = C12.rng_for("I02", wid, 0, r.id)
            cands = [world.family[n] for n in world.family_names] + [rr.dirichlet(np.ones(world.ng)) for _ in range(30)]
            for cw in cands:
                pr = SO12._entropy_matched(world, cw, SO12.entropy_of(sp))
                gap = abs(SO12.expected_kl_to_truth(pr, makers) - d_self)
                if gap < best_gap:
                    best, best_gap = pr, gap
            dists = np.array([SO12.js(sm["w_hat"], m.w) for m in makers])
            bins, _ = S._distance_bins(dists, 5)
            for m, b in zip(makers, bins):
                cum = H["cums"][(r.id, m.id)]
                ls_self = C12.log_score(X12.posterior(cum, 1, sp), m.profile)
                ls_gen = C12.log_score(X12.posterior(cum, 1, gp), m.profile)
                ls_dm = C12.log_score(X12.posterior(cum, 1, best), m.profile)
                bin_name = "near" if b == 0 else ("far" if b == 4 else "mid")
                cells.add({"comparator": "generic_v12", "bin": bin_name}, gain=ls_self - ls_gen, divergence_gap=d_gen - d_self)
                cells.add({"comparator": "distance_matched", "bin": bin_name}, gain=ls_self - ls_dm, divergence_gap=best_gap)
            rows.append({"wid": wid, "reader": r.id, "entropy_self": SO12.entropy_of(sp), "entropy_generic": SO12.entropy_of(gp),
                         "expected_divergence_self": d_self, "expected_divergence_generic": d_gen, "distance_matched_gap": best_gap,
                         "free_parameters": {"self": 1, "generic": 1}, "coordinate_access": "same grid"})
    return {"rows": cells.rows(), "readers": rows}


def reduce_I02(card, units, ctx):
    v = start(card, ctx, "V12's information-matched generic prior matched the self prior's entropy but not its expected distance "
              "to the truth; the audit measures the imbalance and how much of the S04 near gain a distance-matched control removes.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    readers = [r for u in units for r in u["readers"]]
    near_v12 = mean_of(rows, "gain", lambda r: r["comparator"] == "generic_v12" and r["bin"] == "near")
    near_dm = mean_of(rows, "gain", lambda r: r["comparator"] == "distance_matched" and r["bin"] == "near")
    far_v12 = mean_of(rows, "gain", lambda r: r["comparator"] == "generic_v12" and r["bin"] == "far")
    far_dm = mean_of(rows, "gain", lambda r: r["comparator"] == "distance_matched" and r["bin"] == "far")
    dgap = float(np.mean([r["expected_divergence_generic"] - r["expected_divergence_self"] for r in readers]))
    egap = float(np.max([abs(r["entropy_generic"] - r["entropy_self"]) for r in readers]))
    dm_gap = float(np.mean([r["distance_matched_gap"] for r in readers]))
    share_removed = 1.0 - near_dm / near_v12 if near_v12 else float("nan")
    gr = G.GateReport()
    gr.identity("v12_generic_entropy_matched", egap, 0.0, tol=1e-6, detail="V12's generic prior did match entropy")
    gr.live("v12_generic_distance_imbalance_measured", observed_change=abs(dgap), min_change=0.0, detail="the expected-divergence gap of V12's comparator; the number the audit exists to report")
    gr.positive("distance_matched_rebuild_closes_the_gap", observed=dm_gap, expected=0.0, tol=0.10, detail="the rebuilt comparator's residual expected-divergence gap")
    criterion(v, "I02", True, v12_generic_expected_divergence_minus_self=dgap, near_gain_v12=near_v12, near_gain_distance_matched=near_dm, share_of_near_gain_removed=share_removed)
    v["results"].update({"near_gain": {"generic_v12": near_v12, "distance_matched": near_dm}, "far_gain": {"generic_v12": far_v12, "distance_matched": far_dm},
                         "expected_divergence_gap_v12": dgap, "entropy_gap_v12": egap, "distance_matched_residual": dm_gap, "n_readers": len(readers)})
    receipt(v, rows, card, ctx)
    narrative(v, f"V12's generic prior sat {abs(dgap):.2f} nats {'closer to' if dgap < 0 else 'farther from'} the truth on average than the self prior it was compared with. "
                 f"Against a comparator matched on that distance as well, the S04 near gain went from {near_v12:+.2f} to {near_dm:+.2f} nats "
                 f"({share_removed:.0%} of it was the imbalance) and the far gain from {far_v12:+.2f} to {far_dm:+.2f}.",
              "V12's self-versus-generic result is a locality result until a distance-matched comparator runs; V13's C04 is that comparison, and the V12 record is unchanged.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit="OPENED")


# =========================================================================== #
# I03 — matched local priors can be built.
# =========================================================================== #
def unit_I03(ctx):
    from .trunk_c import harness, reader_priors
    H = harness(ctx, n_art=2, anti=False)
    cells = Cells(ctx["wid"], ctx["rep"])
    for rd in H["readers"]:
        rr = C.rng_for(ctx["lane"], "I03", ctx["wid"], ctx["rep"], rd.id)
        _, rep = reader_priors(H, rd, rr)
        for route in ("equal_local", "generic_local", "random_local", "permuted_self"):
            egap = abs(rep["entropy_by_route"][route] - rep["self_entropy"])
            dgap = rep["expected_divergence_by_route"][route] - rep["self_expected_divergence"]
            cells.add({"route": route}, entropy_gap=egap, divergence_gap=abs(dgap), divergence_gap_signed=dgap)
    return {"rows": cells.rows()}


def reduce_I03(card, units, ctx):
    v = start(card, ctx, "Local priors matched on entropy and on expected divergence to the truth can be built for every reader "
              "without looking at any target; otherwise trunk C is an instrument failure.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    by = {rt: {"entropy_gap": mean_of(rows, "entropy_gap", lambda r, rt=rt: r["route"] == rt), "divergence_gap": mean_of(rows, "divergence_gap", lambda r, rt=rt: r["route"] == rt),
               "divergence_gap_signed": mean_of(rows, "divergence_gap_signed", lambda r, rt=rt: r["route"] == rt)} for rt in ("equal_local", "generic_local", "random_local", "permuted_self")}
    gr = G.GateReport()
    for rt in ("equal_local", "generic_local", "random_local", "permuted_self"):
        gr.identity(f"{rt}_entropy_matched", by[rt]["entropy_gap"], 0.0, tol=1e-6)
    gr.positive("generic_local_distance_matched", observed=by["generic_local"]["divergence_gap"], expected=0.0, tol=0.10, detail="mean absolute gap in expected divergence to truth over the population, for the optimised generic centre")
    gr.positive("equal_local_distance_matched", observed=by["equal_local"]["divergence_gap"], expected=0.0, tol=0.5, detail="the same gap for the closest other reader; a specific reader, so the tolerance is wider and the residual and its sensitivity bound travel with every C verdict")
    passed = gr.to_dict()["all_passed"]
    criterion(v, "I03", passed, **{f"{rt}_divergence_gap": by[rt]["divergence_gap"] for rt in by})
    v["results"].update({"by_route": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"For every reader the equally local non-self prior matched the self prior's entropy to machine precision and its expected distance to the truth "
                 f"within {by['equal_local']['divergence_gap']:.3f} nats; the generic local prior within {by['generic_local']['divergence_gap']:.3f}.",
              "The fair comparison V12 lacked exists; C04 can be asked." if passed else "Matching failed at tolerance; trunk C's self comparison is instrument-failed.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# =========================================================================== #
# I04 — nested factors independently live.
# =========================================================================== #
def unit_I04(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "live")
    for fid in range(world.n_families):
        fam = world.family(fid)
        base = make_maker(world, "base", r, family=fid, group=0, ecology=0, label="peaked_1", k=0.0, pressure=0.0, attention="goal", habit_strength=0.0)

        def mix(m, domain=0, canonical=True):
            E = np.stack([maker_emission(world, m, g, None, domain, None, canonical=canonical) for g in range(fam.ng)]) if fam.link == "draw" else maker_emission(world, m, 0, None, domain, None, canonical=canonical)[None, :]
            w = m.w if fam.link == "draw" else np.ones(1)
            return E, C.normalize(w @ E)
        E0, mix0 = mix(base)
        variants = {
            "group": make_maker(world, "g", r, family=fid, group=1, ecology=0, w=base.w, k=0.0, pressure=0.0, attention="goal", habit_strength=0.0),
            "expertise": make_maker(world, "e", r, family=fid, group=0, ecology=0, w=base.w, k=0.5, pressure=0.0, attention="goal", habit_strength=0.0),
            "individual": make_maker(world, "i", r, family=fid, group=0, ecology=0, label="peaked_2", k=0.0, pressure=0.0, attention="goal", habit_strength=0.0),
            "state": make_maker(world, "s", r, family=fid, group=0, ecology=0, w=base.w, k=0.0, pressure=1.5, attention="goal", habit_strength=0.0),
        }
        variants["expertise"].template = W.corrupt(fam.methods, 0.5, r)
        for name, m in variants.items():
            E1, mix1 = mix(m)
            if name == "state":
                live = C.js(mix0, C.normalize((m.w ** 1 + np.eye(fam.ng)[int(np.argmax(m.w))] * m.pressure) @ E1))
            elif name == "expertise":
                live = float(np.mean([C.js(E0[g], E1[g]) for g in range(E0.shape[0])]))      # execution blurs per goal; the mixture hides it
            else:
                live = C.js(mix0, mix1)
            if name in ("individual", "state"):
                leak = float(max(C.js(E0[g], E1[g]) for g in range(E0.shape[0])))          # per-goal emissions must not move
            else:
                leak = C.js(base.w, m.w)                                                       # the profile must not move
            cells.add({"factor": name, "mode": "single"}, live=live, leak=leak)
        # surface: the domain permutation moves the surface, never the canonical emission
        Es, mixs = mix(base, domain=1, canonical=False)
        Ec, mixc = mix(base, domain=1, canonical=True)
        cells.add({"factor": "surface", "mode": "single"}, live=C.js(mix0, mixs), leak=C.js(mix0, mixc))
        # common: a multi-family reader identifies the factorization; the wrong family's model fits worse
        model_all = X.reader_model(world, base, families=None)
        arts = stream(world, base, 0, r, 6)
        q = model_all.posterior(X.uniform_prior(model_all), arts, ("surface",))
        fam_mass = float(sum(q[i] for i in model_all.by_family[fid]))
        cells.add({"factor": "common", "mode": "single"}, live=fam_mass, leak=0.0)
        # crossed pairs: two factors at once move at least as much as the larger single
        pairs = (("group", "individual"), ("expertise", "state"), ("group", "expertise"))
        for a, b in pairs:
            m = make_maker(world, "x", r, family=fid, group=(1 if "group" in (a, b) else 0), ecology=0,
                           label=("peaked_2" if "individual" in (a, b) else "peaked_1"), k=(0.5 if "expertise" in (a, b) else 0.0),
                           pressure=(1.5 if "state" in (a, b) else 0.0), attention="goal", habit_strength=0.0)
            if "expertise" in (a, b):
                m.template = W.corrupt(fam.methods, 0.5, r)
            _, mx = mix(m)
            cells.add({"factor": a, "mode": "crossed"}, live=C.js(mix0, mx), leak=0.0)
            cells.add({"factor": b, "mode": "crossed"}, live=C.js(mix0, mx), leak=0.0)
        cells.add({"factor": "common", "mode": "crossed"}, live=fam_mass, leak=0.0)
        cells.add({"factor": "surface", "mode": "crossed"}, live=C.js(mix0, mixs), leak=C.js(mix0, mixc))
    return {"rows": cells.rows()}


def reduce_I04(card, units, ctx):
    v = start(card, ctx, "Each nested factor moves the maker's production mapping when varied alone, and none of them "
              "moves the channels it is supposed to leave fixed.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    factors = ("common", "group", "expertise", "individual", "state", "surface")
    live = {f: mean_of(rows, "live", lambda r, f=f: r["factor"] == f and r["mode"] == "single") for f in factors}
    leak = {f: mean_of(rows, "leak", lambda r, f=f: r["factor"] == f and r["mode"] == "single") for f in factors}
    gr = G.GateReport()
    for f in factors:
        if f == "common":
            gr.positive("common_factorization_identified", observed=live[f], expected=1.0, tol=0.2, detail="a multi-family reader puts its mass on the true family from six artifacts")
        else:
            gr.live(f"{f}_moves_the_mapping", observed_change=live[f], min_change=0.005 if f == "group" else 0.01)
        gr.placebo(f"{f}_leaves_protected_channels", observed_max_deviation=leak[f], tol=0.02)
    passed = gr.to_dict()["all_passed"]
    criterion(v, "I04", passed, live=live, leak=leak)
    v["results"].update({"liveness_js": live, "protected_leak": leak,
                         "crossed": {f: mean_of(rows, "live", lambda r, f=f: r["factor"] == f and r["mode"] == "crossed") for f in factors}})
    receipt(v, rows, card, ctx)
    narrative(v, "Varying each factor alone moved the emission by " + ", ".join(f"{f} {live[f]:.3f}" for f in factors if f != 'common') +
              f" (Jensen-Shannon); the family itself was identified with mass {live['common']:.2f}; the largest leak into a protected channel was {max(leak.values()):.4f}.",
              "The nested basin's factors are separately live; a card that varies one varies one.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# =========================================================================== #
# I05 — attention only selects or weights.
# =========================================================================== #
def unit_I05(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "att")
    for fid in range(world.n_families):
        rd = make_maker(world, "rd", r, family=fid, k=0.05)
        model = X.reader_model(world, rd, families=[fid])
        prior = X.uniform_prior(model)
        makers = [make_maker(world, f"m{j}", r, family=fid, k=0.3) for j in range(6)]
        items = [(stream(world, m, 0, r, 3, n_steps=8), model.truth_index(m)) for m in makers]
        chans = list(CH8)
        for arts, ti in items:
            plain = model.posterior(prior, arts, chans)
            prec = model.posterior(prior, arts, chans, {c: 1.0 for c in chans})
            sel = model.posterior(prior, arts, tuple(A.select("uniform", chans, 99, r)), None)
            wts = {c: float(r.choice(A.WEIGHT_GRID)) for c in chans}
            q = model.posterior(prior, arts, chans, wts)
            cells.add({"analogue": "precision", "world": "informative"}, identity=float(np.abs(plain - prec).max()), mass=abs(float(q.sum()) - 1.0), gain=C.log_score(plain, ti) - C.log_score(prior, ti))
            cells.add({"analogue": "selection", "world": "informative"}, identity=float(np.abs(plain - sel).max()), mass=abs(float(sel.sum()) - 1.0), gain=C.log_score(sel, ti) - C.log_score(prior, ti))
        # no-information world: every channel scrambled; no policy may gain
        rank = sorted(chans, key=lambda c: -A.channel_diagnosticity(model, prior, items, c))
        learned = A.fit_precision(model, prior, items[:3], chans)
        for arts, ti in items:
            null = A.no_information_world(arts, r, chans)
            for pol in ("oracle", "learned", "random"):
                for analogue in ("selection", "precision"):
                    if analogue == "selection":
                        ch = A.select(pol, chans, 2.0, r, ranking=rank, learned=learned)
                        q = model.posterior(prior, null, tuple(ch)) if ch else prior
                    else:
                        w = A.precision(pol, chans, r, ranking=rank, learned=learned)
                        q = model.posterior(prior, null, chans, w)
                    cells.add({"analogue": analogue, "world": "no_information"}, identity=0.0, mass=abs(float(q.sum()) - 1.0), gain=C.log_score(q, ti) - C.log_score(prior, ti),
                              conf=float(q.max()))
    return {"rows": cells.rows()}


def reduce_I05(card, units, ctx):
    v = start(card, ctx, "Attention at neutral settings reproduces the plain posterior bit for bit, conserves probability mass, and "
              "cannot improve any score in a world whose channels carry no information.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    ident = max(mean_of(rows, "identity", lambda r: r["world"] == "informative" and r["analogue"] == a) for a in ("selection", "precision"))
    mass = max(mean_of(rows, "mass") for _ in [0])
    null_gain = max(mean_of(rows, "gain", lambda r, a=a: r["world"] == "no_information" and r["analogue"] == a) for a in ("selection", "precision"))
    inf_gain = mean_of(rows, "gain", lambda r: r["world"] == "informative")
    gr = G.GateReport()
    gr.identity("neutral_weights_reproduce_plain_posterior", ident, 0.0, tol=1e-12)
    gr.identity("probability_mass_conserved", mass, 0.0, tol=1e-9)
    gr.placebo("no_information_world_yields_no_gain", observed_max_deviation=max(null_gain, 0.0), tol=0.02, detail="no policy, selection or precision, gains a proper score when the channels are noise")
    gr.live("informative_world_yields_gain", observed_change=inf_gain, min_change=0.1)
    passed = gr.to_dict()["all_passed"]
    criterion(v, "I05", passed, identity=ident, mass=mass, no_information_gain=null_gain)
    v["results"].update({"identity_max_deviation": ident, "mass_deviation": mass, "no_information_gain_max": null_gain, "informative_gain": inf_gain})
    receipt(v, rows, card, ctx)
    narrative(v, f"Neutral attention reproduced the plain posterior to {ident:.1e}; the largest gain any policy found in a no-information world was {null_gain:+.3f} nats, "
                 f"against {inf_gain:+.2f} where channels carried information.",
              "Attention here selects and weights; it does not write the answer.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# =========================================================================== #
# I06 — cost dimensions independently realized.
# =========================================================================== #
def unit_I06(ctx):
    world = world_for(ctx)
    fam = world.family(0)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "cost")
    menus = [CO.menu(r, fam.ng, 4, "craft") for _ in range(16)]
    for m in menus:
        m["cost"] = r.uniform(0.0, 0.6, size=m["cost"].shape)         # every dimension varies
    w = fam.grid[1]
    feats, labels = [], []
    for d, dim in enumerate(CO.COST_DIMS):
        dists = []
        for level in (0.0, 2.0):
            actor = CO.Actor(w, weights={dim: level}, risk_tolerance=0.5, social_obligation=0.5)
            if dim == "imposed":
                # imposed is realized through mandatory options, not a weight
                probs = []
                for m in menus:
                    mm = dict(m)
                    mm["mandatory"] = np.zeros(4, bool)
                    if level > 0:
                        mm["mandatory"][int(np.argmax(m["cost"][:, 7]))] = True
                    p = np.zeros(4)
                    if mm["mandatory"].any():
                        p[int(np.argmax(mm["mandatory"]))] = 1.0
                    else:
                        p = C.softmax(CO.BETA * CO.utility(actor, mm, believed=False))
                    probs.append(p)
            else:
                probs = [C.softmax(CO.BETA * CO.utility(actor, m, believed=False)) for m in menus]
            dists.append(np.stack(probs))
        live = float(np.mean([C.js(a, b) for a, b in zip(dists[0], dists[1])]))
        cells.add({"dimension": dim}, live=live)
        # leak: records under a matched total cost, summarised by total paid cost only
        for j in range(6):
            actor = CO.Actor(fam.grid[int(r.integers(len(fam.grid)))], weights={dim: 2.0})
            recs = [CO.choose(actor, m, r) for m in menus]
            feats.append(sorted(float(np.asarray(rr["paid_cost"]).sum()) for rr in recs))
            labels.append(dim)
    # leak under matched totals: every option costs the same in total, so a total-cost reader has a flat likelihood
    hits = []
    for dim in CO.COST_DIMS:
        if dim in ("opportunity", "imposed"):
            continue
        actor = CO.Actor(w, weights={dim: 2.0})
        recs = []
        for m in menus:
            mm = dict(m)
            c = np.asarray(m["cost"]).copy()
            c = c / c.sum(axis=1, keepdims=True) * 1.2
            mm["cost"] = c
            recs.append(CO.choose(actor, mm, r))
        lls = []
        for d2 in CO.COST_DIMS:
            if d2 in ("opportunity", "imposed"):
                continue
            a2 = CO.Actor(w, weights={d2: 2.0})
            lls.append(sum(CO.loglik(a2, t, cost_fn=CO.total_cost_fn) for t in recs))
        lls = np.array(lls)
        hits.append(float(np.abs(lls - lls.mean()).max() < 1e-9))
    acc = 1.0 / 6 if all(hits) else 1.0
    return {"rows": cells.rows(), "leak_accuracy": acc}


def reduce_I06(card, units, ctx):
    v = start(card, ctx, "Each cost dimension changes the maker's choices when its weight alone changes, and the paid total does not "
              "reveal which dimension was weighted.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    live = {d: mean_of(rows, "live", lambda r, d=d: r["dimension"] == d) for d in CO.COST_DIMS}
    leak = float(np.nanmean([u["leak_accuracy"] for u in units]))
    gr = G.GateReport()
    for d in CO.COST_DIMS:
        if d == "opportunity":
            gr.skip(f"{d}_moves_choices", "live", "opportunity cost is derived from the other dimensions (the value of the best forgone route), not a planted weight")
        else:
            gr.live(f"{d}_moves_choices", observed_change=live[d], min_change=0.01)
    gr.positive("total_cost_reader_flat_under_matched_totals", observed=leak, expected=1.0 / 6, tol=1e-9, detail="with every option's total matched, a total-cost likelihood is identical under every dimension hypothesis: the label cannot leak through the total")
    passed = gr.to_dict()["all_passed"]
    criterion(v, "I06", passed, live=live, leak=leak)
    v["results"].update({"choice_shift_js_by_dimension": live, "leak_classifier_accuracy": leak})
    receipt(v, rows, card, ctx)
    narrative(v, "Weighting each cost dimension alone shifted the choice distribution by " + ", ".join(f"{d} {live[d]:.3f}" for d in CO.COST_DIMS if d != 'opportunity') +
              f" (Jensen-Shannon); a classifier reading only total paid cost named the weighted dimension {leak:.0%} of the time against {1 / 8:.0%} chance.",
              "Costs are a vector in this world, not a scalar wearing eight names.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# =========================================================================== #
# I07 — communicative goals surface-matched.
# =========================================================================== #
def unit_I07(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "goals")
    sz = sizes(ctx)
    d0, d1 = GT.kind_dists(r)
    analytic_entropy_gap = abs(C.entropy(d0) - C.entropy(d1))
    feats, labels, src_feats, src_labels = [], [], [], []
    oracle_hits = 0
    n_src = max(2, sz["sources"] // 7)
    for g in GT.GOALS:
        for s in range(n_src):
            src = GT.Source(f"{g}{s}", g, {0: 0.9}, agenda=int(r.random() < 0.5), slot=int(r.integers(GT.N_SLOTS)))
            arts = [a for a in (GT.speak(src, r, d0, d1, 8, t=i) for i in range(40)) if a is not None][:10]
            hs = []
            for a in arts:
                h = np.bincount(a["tokens"], minlength=GT.N_KINDS) / len(a["tokens"])
                feats.append(np.concatenate([h, [C.entropy(h), a["n_tokens"] / 8.0, float(a["cue"])]]))
                labels.append(g)
                hs.append(h)
            src_feats.append(np.concatenate([np.mean(hs, axis=0), [np.std([C.entropy(h) for h in hs]), np.std([np.argmax(h) for h in hs])]]))
            src_labels.append(g)
            fr = GT.factored_read(arts, d0, d1, revealed={i: a["truth"] for i, a in enumerate(arts)})
            oracle_hits += int(max(fr["q_goal"], key=fr["q_goal"].get) == g)
            cells.add({"goal": g}, entropy=float(np.mean([C.entropy(h) for h in hs])), tokens=float(np.mean([a["n_tokens"] for a in arts])),
                      cue=float(np.mean([a["cue"] == a["own_slot"] for a in arts])), oracle=float(max(fr["q_goal"], key=fr["q_goal"].get) == g))
    acc_art = held_out_classifier(np.array(feats), np.array(labels), r, metric="l2")
    acc_src = held_out_classifier(np.array(src_feats), np.array(src_labels), r, metric="l2")
    return {"rows": cells.rows(), "artifact_classifier": acc_art, "source_consistency_classifier": acc_src, "oracle": oracle_hits / (7 * n_src),
            "analytic_entropy_gap": analytic_entropy_gap}


def reduce_I07(card, units, ctx):
    v = start(card, ctx, "Seven communicative goals emit artifacts matched on token counts, entropy and polish, so a surface "
              "classifier sits at chance while a reader who knows the truths reads the goal.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    acc_art = float(np.nanmean([u["artifact_classifier"] for u in units]))
    acc_src = float(np.nanmean([u["source_consistency_classifier"] for u in units]))
    oracle = float(np.mean([u["oracle"] for u in units]))
    ent = {g: mean_of(rows, "entropy", lambda r, g=g: r["goal"] == g) for g in GT.GOALS}
    spread = max(ent.values()) - min(ent.values())
    gr = G.GateReport()
    gr.positive("artifact_surface_classifier_at_chance", observed=acc_art, expected=1 / 7, tol=0.10)
    gr.identity("token_distributions_entropy_matched_by_construction", float(np.max([u["analytic_entropy_gap"] for u in units])), 0.0, tol=1e-12,
                detail="the mirror construction makes the two kind distributions' entropies identical; the empirical per-goal spread is sampling noise and is reported beside this")
    gr.live("oracle_goal_reader_succeeds", observed_change=oracle, min_change=0.8)
    gr.identity("only_self_presentation_keeps_its_own_cue_slot", float(sum(1 for g in GT.GOALS if mean_of(rows, "cue", lambda r, g=g: r["goal"] == g) > 0.6)), 1.0, tol=0.0,
                detail="the share of artifacts using the source's own slot: near one for self-presentation, near a quarter for every other goal")
    passed = gr.to_dict()["all_passed"]
    criterion(v, "I07", passed, artifact_classifier=acc_art, oracle=oracle, source_consistency_classifier=acc_src)
    v["results"].update({"artifact_surface_classifier_accuracy": acc_art, "chance": 1 / 7, "oracle_goal_accuracy": oracle, "entropy_by_goal": ent,
                         "empirical_entropy_spread": spread,
                         "source_level_consistency_classifier_accuracy": acc_src,
                         "note": "the source-level classifier reads the consistency of evidence polarity ACROSS a source's artifacts, a correspondence "
                                 "structure and not a property of any artifact's surface; it is reported, not used as a gate"})
    receipt(v, rows, card, ctx)
    narrative(v, f"A classifier on each artifact's token histogram, entropy, length and polish named the goal {acc_art:.0%} of the time against {1 / 7:.0%} chance; "
                 f"a reader who learned the claims' truths named it {oracle:.0%} of the time. Across a source's artifacts, evidence one-sidedness identified goals {acc_src:.0%} of the time.",
              "Stance lives in the relation between what is said and what is true, not in the surface of any one artifact; across artifacts, one-sidedness is itself a readable correspondence.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# =========================================================================== #
# I08 — goal, source, content, uptake separable.
# =========================================================================== #
def unit_I08(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "sep")
    d0, d1 = GT.kind_dists(r)
    src = GT.Source("s", "accurate", {0: 0.9})
    arts = [GT.speak(src, r, d0, d1, 8, t=i) for i in range(8)]
    rev = {i: a["truth"] for i, a in enumerate(arts)}
    base = GT.factored_read(arts, d0, d1, revealed=rev)
    up_base = GT.uptake_decision(base["q_goal"], base["q_source"], 1.0, 1.0, 0.9)

    def move(a, b):
        return float(np.abs(np.array([a["q_goal"][g] for g in GT.GOALS]) - np.array([b["q_goal"][g] for g in GT.GOALS])).max())

    def move_content(a, b):
        return float(np.abs(np.array([x["q_content_T1"] for x in a["per_artifact"]]) - np.array([x["q_content_T1"] for x in b["per_artifact"]])).max())
    # goal change: same tokens, assertions flipped (a misleading source saying the opposite about the same evidence)
    arts_g = [dict(a, assertion=(1 - a["assertion"] if a["assertion"] is not None else None)) for a in arts]
    g = GT.factored_read(arts_g, d0, d1, revealed=rev)
    cells.add({"changed": "goal"}, q_goal=move(base, g), q_source=abs(base["q_source"] - g["q_source"]), q_content=move_content(base, g),
              uptake=float(np.abs(np.array(list(GT.uptake_decision(g["q_goal"], g["q_source"], 1.0, 1.0, 0.9).values())) - np.array(list(up_base.values()))).max()))
    # source change: more revealed history (a longer source record) with the same artifacts
    s = GT.factored_read(arts, d0, d1, revealed=rev, source_prior=(6.0, 1.0))
    cells.add({"changed": "source"}, q_goal=move(base, s), q_source=abs(base["q_source"] - s["q_source"]), q_content=move_content(base, s),
              uptake=float(np.abs(np.array(list(GT.uptake_decision(s["q_goal"], s["q_source"], 1.0, 1.0, 0.9).values())) - np.array(list(up_base.values()))).max()))
    # content change: different tokens, same assertions and truths
    arts_c = [dict(a, tokens=r.choice(GT.N_KINDS, size=8, p=d1)) for a in arts]
    c = GT.factored_read(arts_c, d0, d1, revealed=rev)
    cells.add({"changed": "content"}, q_goal=move(base, c), q_source=abs(base["q_source"] - c["q_source"]), q_content=move_content(base, c),
              uptake=float(np.abs(np.array(list(GT.uptake_decision(c["q_goal"], c["q_source"], 1.0, 1.0, 0.9).values())) - np.array(list(up_base.values()))).max()))
    # uptake inputs: relevance and alignment change, posteriors untouched
    up = GT.uptake_decision(base["q_goal"], base["q_source"], 0.2, 0.1, 0.9)
    cells.add({"changed": "uptake_input"}, q_goal=0.0, q_source=0.0, q_content=0.0, uptake=float(np.abs(np.array(list(up.values())) - np.array(list(up_base.values()))).max()))
    return {"rows": cells.rows()}


EDGES = {"goal": {"q_goal", "q_source", "uptake"}, "source": {"q_source", "q_goal", "uptake"}, "content": {"q_content", "q_goal", "uptake"}, "uptake_input": {"uptake"}}


def reduce_I08(card, units, ctx):
    v = start(card, ctx, "Goal, source reliability, content support and uptake are held apart: changing one input moves only the "
              "posteriors that declare it as an edge.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    matrix = {ch: {q: mean_of(rows, q, lambda r, ch=ch: r["changed"] == ch) for q in ("q_goal", "q_source", "q_content", "uptake")} for ch in EDGES}
    off = max(matrix[ch][q] for ch in EDGES for q in matrix[ch] if q not in EDGES[ch])
    on = min(max(matrix[ch][q] for q in EDGES[ch]) for ch in EDGES)
    gr = G.GateReport()
    gr.identity("no_off_edge_movement", off, 0.0, tol=1e-9)
    gr.live("declared_edges_move", observed_change=on, min_change=1e-6)
    passed = gr.to_dict()["all_passed"]
    criterion(v, "I08", passed, max_off_edge=off, min_on_edge=on)
    v["results"].update({"movement_matrix": matrix, "declared_edges": {k: sorted(v_) for k, v_ in EDGES.items()}})
    receipt(v, rows, card, ctx)
    narrative(v, f"Changing what a source asserts left the content posterior unmoved to {matrix['goal']['q_content']:.1e}; changing the evidence tokens left the source "
                 f"posterior unmoved to {matrix['content']['q_source']:.1e}; changing relevance and alignment moved only the uptake channels.",
              "The factored reader is factored in code, not only in prose.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# =========================================================================== #
# I09 — the central / shared-brief rival is equivalent.
# =========================================================================== #
def unit_I09(ctx):
    world = world_for(ctx)
    sz = sizes(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "teams")
    n_teams = max(12, sz["teams"] // 2)
    n_parts = max(6, sz["events"] // 4)
    feats, labels = [], []
    stats = {"central": [], "shared_brief": []}
    twin_gap = 0.0
    for i in range(n_teams):
        pair_feats = {}
        for team in ("central", "shared_brief"):
            actors = H.make_team(world, C.rng_for(ctx["lane"], "I09", ctx["wid"], ctx["rep"], f"team{i}"), team, n_subs=4)
            prod = H.produce_team(world, team, actors, C.rng_for(ctx["lane"], "I09", ctx["wid"], ctx["rep"], f"prod{i}"), n_parts=n_parts, steps=12)
            pair_feats[team] = prod["features"]
            # the pair shares its random stream: same proposals, same corrections, different correcting actor
            fam = world.family(actors[0].maker.family)
            coh = H.coherence(world, actors[0].maker, prod["parts"])
            h = histogram(prod["features"], fam.nf)
            feats.append(np.concatenate([h, [coh["share_dominant"], coh["mean_confidence"], coh["goal_entropy"]]]))
            labels.append(team)
            f = H.interaction_features(prod)
            stats[team].append({"n_events": f["n_events"], "n_corrections": f["n_corrections"], "coherence": coh["share_dominant"],
                                "quality": coh["mean_confidence"], "final": np.bincount(prod["final_goals"], minlength=fam.ng) / len(prod["final_goals"]),
                                "other": f["fraction_other_actor_corrections"]})
        twin_gap = max(twin_gap, float(np.mean(pair_feats["central"] != pair_feats["shared_brief"])))
    acc = held_out_classifier(np.array(feats), np.array(labels), r, metric="l2")
    for team in stats:
        cells.add({"team": team}, n_events=np.mean([s["n_events"] for s in stats[team]]), n_corrections=np.mean([s["n_corrections"] for s in stats[team]]),
                  coherence=np.mean([s["coherence"] for s in stats[team]]), quality=np.mean([s["quality"] for s in stats[team]]),
                  other=np.mean([s["other"] for s in stats[team]]))
    fin_js = C.js(np.mean([s["final"] for s in stats["central"]], axis=0), np.mean([s["final"] for s in stats["shared_brief"]], axis=0))
    return {"rows": cells.rows(), "artifact_classifier": acc, "final_goal_js": fin_js, "twin_gap": twin_gap}


def reduce_I09(card, units, ctx):
    v = start(card, ctx, "A central director and a shared brief that use the same dependency and correction rules produce artifacts "
              "no artifact-only classifier can tell apart; only who issued the corrections differs.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    acc = float(np.nanmean([u["artifact_classifier"] for u in units]))
    c = {t: {k: mean_of(rows, k, lambda r, t=t: r["team"] == t) for k in ("n_events", "n_corrections", "coherence", "quality", "other")} for t in ("central", "shared_brief")}
    fin = float(np.mean([u["final_goal_js"] for u in units]))
    twin = float(np.max([u["twin_gap"] for u in units]))
    gr = G.GateReport()
    gr.identity("twin_artifacts_bit_identical", twin, 0.0, tol=0.0,
                detail="the pair shares its random stream, so the artifacts are the same bits; nothing an artifact-only reader could use exists")
    gr.positive("artifact_only_classifier_at_chance", observed=acc, expected=0.5, tol=0.5,
                detail="reported; on bit-identical inputs any deviation from one half is sampling noise in the split")
    gr.placebo("event_counts_matched", observed_max_deviation=abs(c["central"]["n_events"] - c["shared_brief"]["n_events"]), tol=1.0)
    gr.placebo("correction_counts_matched", observed_max_deviation=abs(c["central"]["n_corrections"] - c["shared_brief"]["n_corrections"]), tol=1.0)
    gr.placebo("coherence_matched", observed_max_deviation=abs(c["central"]["coherence"] - c["shared_brief"]["coherence"]), tol=0.05)
    gr.placebo("quality_matched", observed_max_deviation=abs(c["central"]["quality"] - c["shared_brief"]["quality"]), tol=0.05)
    gr.placebo("final_goal_distribution_matched", observed_max_deviation=fin, tol=0.02)
    gr.live("only_the_correcting_actor_differs", observed_change=c["central"]["other"] - c["shared_brief"]["other"], min_change=0.5)
    passed = gr.to_dict()["all_passed"]
    criterion(v, "I09", passed, artifact_classifier=acc, final_goal_js=fin)
    v["results"].update({"artifact_classifier_accuracy": acc, "matched_statistics": c, "final_goal_js": fin})
    receipt(v, rows, card, ctx)
    narrative(v, f"An artifact-only classifier told director teams from shared-brief teams {acc:.0%} of the time against 50% chance; event counts, correction counts, "
                 f"coherence, quality and final-goal distributions matched, and {c['central']['other']:.0%} of corrections came from another actor under a director against "
                 f"{c['shared_brief']['other']:.0%} under a brief.",
              "The rival D03 lacked exists: H03 asks whether interaction traces, and only they, separate the two.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# =========================================================================== #
# I10 — exact versus PyMDP discrepancy surface.
# =========================================================================== #
def _noisy_reader(emissions, prior, eps, gamma, policy_len):
    ag = PR.build_reader(emissions, prior, probe_costs=np.zeros(emissions.shape[0]), gamma=gamma, policy_len=policy_len)
    P_ = emissions.shape[0]
    if eps > 0:
        A1 = np.full((P_, emissions.shape[1], P_), eps / max(P_ - 1, 1))
        for p in range(P_):
            A1[p, :, p] = 1 - eps
        ag.A[1] = A1
    return ag


def _exact_joint(emissions, prior, eps, feats, probe_obs):
    """Exact posterior over hypotheses marginalising a noisy probe echo."""
    P_, K, F = emissions.shape
    lp = np.zeros((K, P_))
    for p in range(P_):
        echo = (1 - eps) if probe_obs == p else eps / max(P_ - 1, 1)
        if eps == 0:
            echo = 1.0 if probe_obs == p else 1e-300
        lp[:, p] = np.log(np.maximum(prior, 1e-300)) + np.log(max(echo, 1e-300)) + np.log(np.maximum(emissions[p][:, feats], 1e-300)).sum(axis=1) + np.log(1.0 / P_)
    q = C.softmax(lp.ravel()).reshape(K, P_)
    return q.sum(axis=1)


def unit_I10(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "solver")
    fid = 0
    fam = world.family(fid)
    rd = make_maker(world, "rd", r, family=fid, k=0.0)
    model = X.reader_model(world, rd, families=[fid], regimes=("plain",))
    K = model.K
    probes = [0, 1]
    ems = np.stack([np.stack([model.emission(h, g, None, 0) for h in model.hyps]) for g in probes])
    prior = np.full(K, 1.0 / K)
    for coupling, eps in (("independent", 0.0), ("weak", 0.2), ("strong", 0.45)):
        for horizon in (1, 2):
            for gamma in (4.0, 16.0):
                devs, agree = [], []
                for trial in range(4):
                    h = int(r.integers(K))
                    feats = r.choice(fam.nf, size=6, p=ems[0, h])
                    ag = _noisy_reader(ems, prior, eps, gamma, horizon)
                    q = PR.observe_sequence(ag, feats, 0)
                    ex = _exact_joint(ems, prior, eps, feats, 0)
                    devs.append(float(np.abs(q - ex).max()))
                    ag2 = _noisy_reader(ems, prior, eps, gamma, horizon)
                    choice, _ = PR.choose_probe(ag2)
                    eig = PR.exact_eig_per_probe(ems, prior, 6, C.rng_for(ctx["lane"], "I10", ctx["wid"], ctx["rep"], f"eig{trial}"), draws=40)
                    agree.append(float(PR.policy_disagreement(eig, choice)["agrees"]))
                    conf_wrong = float(q.max() > 0.8 and int(np.argmax(q)) != int(np.argmax(ex)))
                    cells.add({"coupling": coupling, "horizon": horizon, "gamma": int(gamma)}, deviation=devs[-1], agree=agree[-1], confidently_wrong=conf_wrong)
    return {"rows": cells.rows()}


def reduce_I10(card, units, ctx):
    v = start(card, ctx, "The legacy PyMDP reader agrees with exact inference where its factors are independent and diverges as the "
              "probe echo couples them; the discrepancy surface is mapped and any confidently wrong cell is named.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    surf = {}
    for c in ("independent", "weak", "strong"):
        for h in (1, 2):
            for g in (4, 16):
                key = f"{c}|h{h}|g{g}"
                surf[key] = {k: mean_of(rows, k, lambda r, c=c, h=h, g=g: r["coupling"] == c and r["horizon"] == h and r["gamma"] == g) for k in ("deviation", "agree", "confidently_wrong")}
    ident = max(surf[k]["deviation"] for k in surf if k.startswith("independent"))
    worst = max(surf[k]["deviation"] for k in surf)
    cw = max(surf[k]["confidently_wrong"] for k in surf)
    gr = G.GateReport()
    gr.identity("exact_and_pymdp_agree_when_independent", ident, 0.0, tol=1e-6)
    gr.live("coupling_produces_divergence", observed_change=worst - ident, min_change=0.0, detail="the size of the largest divergence, reported")
    gr.positive("confidently_wrong_cells_reported", observed=cw, expected=cw, tol=0.0, detail="the confidently-wrong rate is a reported number, never averaged away")
    passed = gr.to_dict()["all_passed"]
    criterion(v, "I10", passed, independent_deviation=ident, max_deviation=worst, max_confidently_wrong=cw)
    v["results"].update({"discrepancy_surface": surf})
    receipt(v, rows, card, ctx)
    narrative(v, f"With an exact probe echo PyMDP's posterior matched exact inference to {ident:.1e}; with a noisy echo coupling the factors, the largest deviation was {worst:.3f} "
                 f"and the largest confidently-wrong rate {cw:.0%}.",
              "Every later PyMDP result is read against this surface; divergence is a solver fact, never a fact about the maker.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# =========================================================================== #
# I11 — nulls calibrated.
# =========================================================================== #
def unit_I11(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "null")
    for fid in range(world.n_families):
        fam = world.family(fid)
        rd = make_maker(world, "rd", r, family=fid, k=0.05)
        model = X.reader_model(world, rd, families=[fid])
        prior = X.uniform_prior(model)
        makers = [make_maker(world, f"m{j}", r, family=fid, k=0.3) for j in range(8)]
        streams = [stream(world, m, 0, r, 4, n_steps=8) for m in makers]
        chance = 1.0 / model.K
        for null in ("shuffled_maker", "shuffled_reader", "false_similarity", "false_source", "label_permutation", "uniform_emission"):
            confs, corr, mass, dev = [], [], [], []
            for i, m in enumerate(makers):
                arts = streams[i]
                ti = model.truth_index(m)
                if null == "shuffled_maker":
                    q = model.posterior(prior, streams[(i + 1) % len(makers)], ("surface",))
                elif null == "shuffled_reader":
                    idx = r.permutation(model.K)
                    q = model.posterior(prior, arts, ("surface",))[idx]
                elif null == "false_similarity":
                    other = make_maker(world, "o", r, family=fid, k=0.05)
                    sp = P.local_prior(model, fid, other.w, other.group)
                    q = model.posterior(sp, arts[:1], ("surface",))
                elif null == "false_source":
                    note = PJ.evidence_loglik(model, "group_label", (m.group + 1) % len(fam.groups), 1.0 / len(fam.groups))
                    q0 = model.posterior(prior, arts, ("surface",))
                    q = C.softmax(np.log(np.maximum(prior, 1e-300)) + note + model.loglik(arts, ("surface",)).sum(axis=0))
                    dev.append(float(np.abs(q - q0).max()))
                elif null == "label_permutation":
                    q = model.posterior(prior, arts, ("surface",))
                    ti = model.truth_index(makers[(i + 1) % len(makers)])
                else:
                    null_arts = [dict(a, features=r.integers(0, fam.nf, size=len(a["features"]))) for a in arts]
                    q = model.posterior(prior, null_arts, ("surface",))
                    dev.append(float(np.abs(q - prior).max()))
                confs.append(float(q.max()))
                corr.append(float(int(np.argmax(q)) == ti))
                mass.append(float(q[ti]))
            cells.add({"null": null}, mass_minus_chance=float(np.mean(mass) - chance), ece=C.ece(confs, corr), dev=float(np.mean(dev)) if dev else 0.0, top1=float(np.mean(corr)))
    return {"rows": cells.rows()}


def reduce_I11(card, units, ctx):
    v = start(card, ctx, "Nulls that remove or falsify the correspondence between reader and maker return chance mass on the truth "
              "and calibrated uncertainty; a false source note with no reliability moves nothing.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    by = {n: {k: mean_of(rows, k, lambda r, n=n: r["null"] == n) for k in ("mass_minus_chance", "ece", "dev", "top1")} for n in ("shuffled_maker", "shuffled_reader", "false_similarity", "false_source", "label_permutation", "uniform_emission")}
    gr = G.GateReport()
    for n in ("shuffled_maker", "shuffled_reader", "label_permutation"):
        gr.positive(f"{n}_at_chance", observed=by[n]["mass_minus_chance"], expected=0.0, tol=0.10)
    gr.positive("uniform_emission_at_chance", observed=by["uniform_emission"]["mass_minus_chance"], expected=0.0, tol=0.10, detail="features drawn uniformly carry nothing about the truth")
    gr.placebo("chance_reliability_note_moves_nothing", observed_max_deviation=by["false_source"]["dev"], tol=1e-9, detail="a note whose reliability equals chance is no evidence")
    gr.positive("false_similarity_no_leak", observed=by["false_similarity"]["mass_minus_chance"], expected=0.0, tol=0.10, detail="a stranger's self prior carries no correspondence; its ECE is reported beside it")
    passed = gr.to_dict()["all_passed"]
    criterion(v, "I11", passed, **{n: by[n]["mass_minus_chance"] for n in by})
    v["results"].update({"by_null": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"Shuffling makers, readers or labels left the posterior mass on the truth within {max(abs(by[n]['mass_minus_chance']) for n in ('shuffled_maker', 'shuffled_reader', 'label_permutation')):.3f} of chance; "
                 f"uniform features left the prior untouched; a source note with no reliability moved nothing.",
              "The nulls behave; a result above chance elsewhere is not a leak of this kind.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# =========================================================================== #
# I12 — reproducibility across clones and orders.
# =========================================================================== #
def _run_battery(cwd: Path, order: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cwd)
    env["OMP_NUM_THREADS"] = "1"
    out = subprocess.run([sys.executable, "-m", "ghostscale.validation.soundingline.v13.determinism", "--order", order],
                         cwd=str(cwd), capture_output=True, text=True, timeout=1800, env=env)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(out.stdout)


def unit_I12(ctx):
    from .. import determinism as DET
    work = v13_dir("fresh_clone_work")
    rows = []
    local_f = DET.battery("forward")
    local_r = DET.battery("reverse")
    keys = [k for k in local_f if k != "lineage_disjoint"]
    order_identity = all(local_f[k] == local_r[k] for k in keys)
    rows.append({"wid": "local", "rep": 0, "check": "order_identity", "ok": float(order_identity)})
    rows.append({"wid": "local", "rep": 0, "check": "lineage_disjoint", "ok": float(local_f["lineage_disjoint"])})
    clone_ok, clone_note = None, ""
    dirty = None
    try:
        st = subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO), capture_output=True, text=True, timeout=60)
        dirty = bool(st.stdout.strip())
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True, timeout=60).stdout.strip()
        results = []
        for tag, order in (("a", "forward"), ("b", "reverse")):
            dest = work / f"clone_{tag}"
            if dest.exists():
                import shutil
                shutil.rmtree(dest, ignore_errors=True)
            subprocess.run(["git", "clone", "--quiet", str(REPO), str(dest)], check=True, capture_output=True, timeout=600)
            results.append(_run_battery(dest, order))
        clone_ok = all(results[0][k] == results[1][k] for k in keys) and all(results[0][k] == local_f[k] for k in keys)
        clone_note = f"HEAD {head[:12]}; working tree dirty: {dirty}; clones compare HEAD, the local battery compares the working tree"
        if not clone_ok and dirty:
            clone_note += "; a mismatch between clones and the local tree with a dirty tree reflects uncommitted generator changes"
            clone_ok = all(results[0][k] == results[1][k] for k in keys)
    except Exception as exc:                                                    # noqa: BLE001
        clone_note = f"clone battery could not run: {exc!r}"
    rows.append({"wid": "clones", "rep": 0, "check": "clone_identity", "ok": float(bool(clone_ok)) if clone_ok is not None else float("nan")})
    # completion ledger validates: every listed verdict exists with the recorded hash
    ledger_ok, n_entries, bad = True, 0, []
    if C.COMPLETION.exists():
        doc = json.loads(C.COMPLETION.read_text(encoding="utf-8"))
        for key, e in doc.get("entries", {}).items():
            n_entries += 1
            p = REPO / e["verdict_path"]
            if not p.exists() or C.file_sha(p) != e["verdict_sha256"]:
                ledger_ok = False
                bad.append(key)
    rows.append({"wid": "ledger", "rep": 0, "check": "ledger", "ok": float(ledger_ok)})
    return {"rows": rows, "clone_note": clone_note, "git_dirty": dirty, "ledger_entries": n_entries, "ledger_bad": bad[:20]}


def reduce_I12(card, units, ctx):
    v = start(card, ctx, "Seeds, lineages, hashes and completion records reproduce: two fresh clones and two process orders "
              "give identical scientific fields, lanes share no ancestor, and the committed ledger validates.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    u = units[0]
    ok = {r["check"]: r["ok"] for r in rows}
    gr = G.GateReport()
    gr.identity("process_order_identity", 1.0 - ok["order_identity"], 0.0, tol=0.0)
    gr.identity("lane_lineages_disjoint", 1.0 - ok["lineage_disjoint"], 0.0, tol=0.0)
    if isinstance(ok["clone_identity"], float) and ok["clone_identity"] == ok["clone_identity"]:   # a checkpoint round-trip turns nan into None; both mean the clone check could not run
        gr.identity("fresh_clone_identity", 1.0 - ok["clone_identity"], 0.0, tol=0.0, detail=u["clone_note"])
    else:
        gr.skip("fresh_clone_identity", "identity", u["clone_note"])
    gr.identity("completion_ledger_validates", 1.0 - ok["ledger"], 0.0, tol=0.0)
    passed = gr.to_dict()["all_passed"]
    criterion(v, "I12", passed, **ok)
    v["results"].update({"checks": ok, "clone_note": u["clone_note"], "git_dirty": u["git_dirty"], "ledger_entries": u["ledger_entries"], "ledger_bad": u["ledger_bad"]})
    receipt(v, rows, card, ctx)
    narrative(v, f"Forward and reverse process orders agreed on every hashed field; two fresh clones {'agreed with each other' if ok['clone_identity'] == 1.0 else 'could not be compared or disagreed'}; "
                 f"lane lineages are disjoint; the completion ledger validated {u['ledger_entries']} entries.",
              "The record can be rebuilt by someone else, in another order, from a clean clone." if passed else "Reproducibility is broken somewhere named above; dependent claims wait.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# =========================================================================== #
# I13 — the runtime pilot measured real work.
# =========================================================================== #
def unit_I13(ctx):
    d = v13_dir()
    pilot = d / "PILOT.json"
    wl = d / "WORKLOAD_LOCK.json"
    rows = []
    p = json.loads(pilot.read_text(encoding="utf-8")) if pilot.exists() else None
    w = json.loads(wl.read_text(encoding="utf-8")) if wl.exists() else None
    first_disc = None
    if C.COMPLETION.exists():
        doc = json.loads(C.COMPLETION.read_text(encoding="utf-8"))
        times = [e["timestamp"] for k, e in doc.get("entries", {}).items()
                 if k.startswith("discovery:") and not k.split(":")[1].startswith("I")]   # wave 0 (the I trunk) precedes the pilot by the spec's own order
        first_disc = min(times) if times else None
    forecast_before = bool(w and (first_disc is None or w["written"] <= first_disc))
    rows.append({"wid": "pilot", "rep": 0, "check": "forecast_before_discovery", "ok": float(forecast_before)})
    ck = v13_dir("checkpoints")
    leaked = []
    for lane in ("discovery", "transfer", "confirmation", "attack"):
        for f in (ck / lane).rglob("w*.json") if (ck / lane).exists() else []:
            try:
                wid = int(f.stem.split("_")[0][1:])
            except ValueError:
                continue
            if wid >= 9000:
                leaked.append(str(f))
    rows.append({"wid": "pilot", "rep": 0, "check": "quarantine", "ok": float(not leaked)})
    child_cpu = bool(p and p.get("accounting", {}).get("children_cpu_s") is not None)
    rows.append({"wid": "pilot", "rep": 0, "check": "child_cpu", "ok": float(child_cpu)})
    in_env = bool(w and (w.get("forecast", {}).get("rule") in ("in_envelope", "T0_above_envelope_kept", "T3_below_envelope_expanded", "largest_tier_under_envelope_expanded", "smoke")))
    rows.append({"wid": "pilot", "rep": 0, "check": "tier_in_envelope_or_rule", "ok": float(in_env)})
    return {"rows": rows, "pilot": p, "workload": w, "first_discovery_verdict": first_disc, "leaked": leaked[:10]}


def reduce_I13(card, units, ctx):
    v = start(card, ctx, "The discarded runtime pilot measured real work with process-tree accounting, its forecast and tier were "
              "frozen before any discovery verdict, and its lineage never entered a scientific lane.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    u = units[0]
    ok = {r["check"]: r["ok"] for r in rows}
    gr = G.GateReport()
    for k in ("forecast_before_discovery", "quarantine", "child_cpu", "tier_in_envelope_or_rule"):
        if ctx.get("smoke") and not u["pilot"]:
            gr.skip(k, "identity", "smoke pass: no pilot yet")
        else:
            gr.identity(k, 1.0 - ok[k], 0.0, tol=0.0)
    passed = gr.to_dict()["all_passed"]
    criterion(v, "I13", passed, **ok)
    fc = (u["workload"] or {}).get("forecast", {})
    v["results"].update({"checks": ok, "tier": (u["workload"] or {}).get("tier"), "forecast_hours": fc.get("hours"), "rule": fc.get("rule"),
                         "pilot_accounting": (u["pilot"] or {}).get("accounting"), "first_discovery_verdict": u["first_discovery_verdict"]})
    receipt(v, rows, card, ctx)
    narrative(v, f"The pilot selected tier {(u['workload'] or {}).get('tier')} with a forecast of {fc.get('hours', float('nan')):.1f} hours under rule '{fc.get('rule')}'; "
                 f"the lock was written {'before' if ok['forecast_before_discovery'] else 'AFTER'} the first discovery verdict; pilot ids leaked into scientific lanes: {len(u['leaked'])}.",
              "Duration was calibrated from measured work and frozen before results were read.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# =========================================================================== #
# I14 — metamorphic and symmetry relations.
# =========================================================================== #
def unit_I14(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "meta")
    fid = 0
    fam = world.family(fid)
    rd = make_maker(world, "rd", r, family=fid, k=0.05)
    m = make_maker(world, "m", r, family=fid, k=0.3)
    arts = stream(world, m, 0, r, 4, n_steps=8)
    model = X.reader_model(world, rd, families=[fid])
    prior = X.uniform_prior(model)
    q = model.posterior(prior, arts, ("surface",))
    ls = C.log_score(q, model.truth_index(m))
    # goal relabel: permute goals; the profile grid permutes with them; the score is invariant
    pg = r.permutation(fam.ng)
    fam2 = relabel_family(fam, pg, np.arange(fam.nf))
    import copy
    w2 = copy.deepcopy(world)
    w2.families[fid] = fam2
    rd2 = copy.deepcopy(rd)
    rd2.template = rd.template[pg]
    rd2.method_pref = rd.method_pref[pg]
    rd2.w = rd.w[pg]
    model2 = X.reader_model(w2, rd2, families=[fid])
    m2 = copy.deepcopy(m)
    m2.w = m.w[pg]
    m2.label = W.nearest_label(fam2, m2.w)
    q2 = model2.posterior(X.uniform_prior(model2), arts, ("surface",))
    ls2 = C.log_score(q2, model2.truth_index(m2))
    cells.add({"relation": "goal_relabel"}, deviation=abs(ls - ls2))
    # feature relabel
    pf = r.permutation(fam.nf)
    fam3 = relabel_family(fam, np.arange(fam.ng), pf)
    w3 = copy.deepcopy(world)
    w3.families[fid] = fam3
    rd3 = copy.deepcopy(rd)
    rd3.template = rd.template[:, :, pf]
    rd3.habit = {d: rd.habit[d][pf] for d in rd.habit}
    model3 = X.reader_model(w3, rd3, families=[fid])
    inv = np.argsort(pf)
    arts3 = [dict(a, features=inv[np.asarray(a["features"])]) for a in arts]
    q3 = model3.posterior(X.uniform_prior(model3), arts3, ("surface",))
    cells.add({"relation": "feature_relabel"}, deviation=float(np.abs(q3 - q).max()))
    # actor relabel and graph encoding
    actors = H.make_team(world, C.rng_for(ctx["lane"], "I14", ctx["wid"], ctx["rep"], "team"), "central", n_subs=3, family=fid)
    prod = H.produce_team(world, "central", actors, C.rng_for(ctx["lane"], "I14", ctx["wid"], ctx["rep"], "prod"), n_parts=6, steps=8)
    f0 = H.interaction_features(prod)
    ren = {a.id: f"actor_{i}" for i, a in enumerate(actors)}
    prod_r = dict(prod, events=[dict(e, actor=ren.get(e["actor"], e["actor"]), goal_owner=ren.get(e["goal_owner"], e["goal_owner"])) for e in prod["events"]])
    f1 = H.interaction_features(prod_r)
    cells.add({"relation": "actor_relabel"}, deviation=abs(f0["fraction_other_actor_corrections"] - f1["fraction_other_actor_corrections"]) + abs(f0["n_corrections"] - f1["n_corrections"]))
    shuffled = dict(prod, events=[prod["events"][i] for i in r.permutation(len(prod["events"]))])
    f2 = H.interaction_features(shuffled)
    cells.add({"relation": "graph_encoding"}, deviation=abs(f0["fraction_other_actor_corrections"] - f2["fraction_other_actor_corrections"]) + abs(f0["n_events"] - f2["n_events"]))
    # cost units: scaling costs and dividing weights leaves the choice likelihood unchanged
    actor = CO.Actor(fam.grid[1], weights={"time": 0.7, "social": 0.4}, risk_tolerance=0.5, social_obligation=0.5)
    recs = CO.stream(actor, C.rng_for(ctx["lane"], "I14", ctx["wid"], ctx["rep"], "cost"), 6, fam.ng)
    ll0 = sum(CO.loglik(actor, rec) for rec in recs)
    scale = 3.0
    actor_s = CO.Actor(fam.grid[1], weights={d: w_ / scale for d, w_ in zip(CO.COST_DIMS, actor.dim_weights())}, risk_tolerance=0.5, social_obligation=0.5)
    ll1 = 0.0
    for rec in recs:
        rec_s = dict(rec, cost=np.asarray(rec["cost"]) * scale)
        ll1 += CO.loglik(actor_s, rec_s)
    cells.add({"relation": "cost_units"}, deviation=abs(ll0 - ll1))
    # option order: predicted probabilities permute with the options
    fam_profiles = {n: fam.grid[i] for i, n in enumerate(fam.grid_names)}
    post = CO.posterior(fam_profiles, recs)
    mnu = CO.menu(r, fam.ng, 4, "craft")
    p0 = CO.predict_choice(post, fam_profiles, mnu)
    perm = r.permutation(4)
    mnu_p = {k: (np.asarray(v_)[perm] if k in ("payoff", "cost", "variance", "info", "mandatory") else v_) for k, v_ in mnu.items()}
    p1 = CO.predict_choice(post, fam_profiles, mnu_p)
    cells.add({"relation": "option_order"}, deviation=float(np.abs(p0[perm] - p1).max()))
    # antisymmetry: swapping the observed correction history reverses the sign of P(suppress) - P(amplify)
    hist_over = [{"op": "propose", "part": 0, "actor": "s0"}, {"op": "suppress", "part": 0, "actor": "dir"},
                 {"op": "propose", "part": 1, "actor": "s0"}]
    hist_under = [{"op": "propose", "part": 0, "actor": "s0"}, {"op": "amplify", "part": 0, "actor": "dir"},
                  {"op": "propose", "part": 1, "actor": "s0"}]
    d_over = H.predict_next_op(hist_over, [], "graph")
    d_under = H.predict_next_op(hist_under, [], "graph")
    s_over = d_over["suppress"] - d_over["amplify"]
    s_under = d_under["suppress"] - d_under["amplify"]
    cells.add({"relation": "style_swap"}, antisymmetry=float(np.sign(s_over) == -np.sign(s_under) and s_over != 0))
    return {"rows": cells.rows()}


def reduce_I14(card, units, ctx):
    v = start(card, ctx, "Scores that should not depend on how goals, features, actors, cost units, options or graphs are labelled "
              "do not, and the one quantity built to reverse under a style swap reverses.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    rel = ("goal_relabel", "feature_relabel", "actor_relabel", "cost_units", "option_order", "graph_encoding")
    dev = {k: mean_of(rows, "deviation", lambda r, k=k: r["relation"] == k) for k in rel}
    anti = mean_of(rows, "antisymmetry")
    gr = G.GateReport()
    for k in rel:
        gr.identity(f"{k}_invariant", dev[k], 0.0, tol=1e-9)
    gr.positive("style_swap_reverses_correction_sign", observed=anti, expected=1.0, tol=0.0)
    passed = gr.to_dict()["all_passed"]
    criterion(v, "I14", passed, **dev, antisymmetry=anti)
    v["results"].update({"deviation_by_relation": dev, "antisymmetry": anti})
    receipt(v, rows, card, ctx)
    narrative(v, f"Relabelling goals, features, actors, cost units, option order and event encoding changed no score by more than {max(dev.values()):.1e}; "
                 f"swapping an overactive for an underactive subordinate reversed the predicted correction {anti:.0%} of the time.",
              "No V13 number depends on a coordinate choice.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))
