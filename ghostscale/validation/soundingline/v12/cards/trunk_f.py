"""Trunk F: layered versus flattened decision topology (spec section 14).

Blocks carry an upstream choice (goal) and a downstream choice (realization slot). Layered: the
slot depends on the goal. Flattened: same marginals, no dependency. Non-invertible: the surface
ignores both. Matched on marginals and length by construction, so only the dependency differs.
"""
from __future__ import annotations

import numpy as np

from .....generative_model import build_observer_signature
from .....methods import gates as G
from .. import common as C
from ..schemas import new_verdict
from ..hierarchy import layered_sequence, dependency_statistic, _part_emission
from . import finish, worlds_for, decide_state


def _seq(world, template, rng, n_blocks=6, steps=4, topology="layered", goal_set=None, fixed_u=None):
    ng, ns = world.ng, len(world.family_names)
    us, vs, feats = [], [], []
    for _ in range(n_blocks):
        if fixed_u is not None:
            u = int(fixed_u)
        elif goal_set is not None:
            u = int(rng.choice(goal_set))
        else:
            u = int(rng.integers(ng))
        v = int((u + rng.integers(2)) % ns) if topology == "layered" else int((rng.integers(ng) + rng.integers(2)) % ns)
        e = world.synth if topology == "noninvertible" else _part_emission(world, template, u, v)
        feats.append(rng.choice(world.nf, size=int(steps), p=e / e.sum()))
        us.append(u)
        vs.append(v)
    return {"features": np.concatenate(feats), "blocks": feats, "u": us, "v": vs}


def _block_posts(world, template, blocks):
    ng, ns = world.ng, len(world.family_names)
    out = []
    for f in blocks:
        ll = np.zeros((ng, ns))
        for g in range(ng):
            for s in range(ns):
                e = _part_emission(world, template, g, s)
                ll[g, s] = np.log(np.maximum(e[f], 1e-300)).sum()
        p = np.exp(ll - ll.max())
        out.append(p / p.sum())
    return out


def _auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def _bigram_mi(feats, nf):
    j = np.zeros((nf, nf)) + 1e-9
    for a, b in zip(feats[:-1], feats[1:]):
        j[a, b] += 1
    j /= j.sum()
    pa, pb = j.sum(axis=1, keepdims=True), j.sum(axis=0, keepdims=True)
    return float((j * np.log(j / (pa @ pb))).sum())


def _hist_entropy(feats, nf):
    h = np.bincount(feats, minlength=nf) / len(feats)
    h = h[h > 0]
    return float(-(h * np.log(h)).sum())


def _hard_mi(world, blocks):
    """Mutual information between per-block argmax upstream and downstream inferences."""
    ng, ns = world.ng, len(world.family_names)
    j = np.zeros((ng, ns)) + 1e-9
    for p in _block_posts(world, world.sig, blocks):
        g, s = np.unravel_index(int(np.argmax(p)), p.shape)
        j[g, s] += 1
    j /= j.sum()
    pu, pv = j.sum(axis=1, keepdims=True), j.sum(axis=0, keepdims=True)
    return float((j * np.log(j / (pu @ pv))).sum())


def _u_entropy(world, template, blocks):
    P = _block_posts(world, template, blocks)
    top = np.array([int(np.argmax(p.sum(axis=1))) for p in P])
    h = np.bincount(top, minlength=world.ng) / len(top)
    h = h[h > 0]
    return float(-(h * np.log(h)).sum())


def run_F01(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "The layered/flattened manipulation reaches the emitter: layered and "
                    "flattened sequences differ bit for bit under the same random stream, and the "
                    "manipulation switched off reproduces flattened exactly.", "METHOD")
    diff, placebo = [], []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            for i in range(20):
                s = C.seed(f"F01:{wid}:{i}")
                a = layered_sequence(world, world.sig, np.random.default_rng(s), topology="layered", manipulation_on=True)
                b = layered_sequence(world, world.sig, np.random.default_rng(s), topology="flattened")
                c = layered_sequence(world, world.sig, np.random.default_rng(s), topology="layered", manipulation_on=False)
                diff.append(float(np.mean(a["features"] != b["features"])))
                placebo.append(float(np.mean(c["features"] != b["features"])))
    gr = G.GateReport()
    gr.placebo("manipulation_off_equals_flattened", observed_max_deviation=float(max(placebo)), tol=0.0)
    gr.live("manipulation_changes_the_surface", observed_change=float(np.mean(diff)), min_change=0.05)
    v["results"] = {"bit_difference_layered_vs_flattened": float(np.mean(diff)), "placebo_difference": float(max(placebo))}
    v["what_must_hold_outside_the_simulation"] = "nothing; a construction check"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_F02(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Layered and flattened worlds share their upstream and downstream marginals "
                    "and their surface histogram; they differ in the mutual information between the two choices.",
                    "METHOD")
    rows = []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("F02", wid, 0)
            stats = {}
            for topo in ("layered", "flattened"):
                seqs = [layered_sequence(world, world.sig, rng, n_blocks=100, topology=topo) for _ in range(6)]
                us = np.concatenate([s["u"] for s in seqs])
                vs = np.concatenate([s["v"] for s in seqs])
                feats = np.concatenate([s["features"] for s in seqs])
                j = np.zeros((world.ng, len(world.family_names))) + 1e-9
                for a, b in zip(us, vs):
                    j[a, b] += 1
                j /= j.sum()
                stats[topo] = {"u": np.bincount(us, minlength=world.ng) / len(us), "v": np.bincount(vs, minlength=len(world.family_names)) / len(vs),
                               "f": np.bincount(feats, minlength=world.nf) / len(feats),
                               "mi": float((j * np.log(j / (j.sum(axis=1, keepdims=True) @ j.sum(axis=0, keepdims=True)))).sum())}

            def js(p, q):
                m = 0.5 * (p + q)
                s = lambda a: float((a[a > 0] * np.log(a[a > 0] / m[a > 0])).sum())
                return 0.5 * s(p) + 0.5 * s(q)
            rows.append({"wid": wid, "js_u": js(stats["layered"]["u"], stats["flattened"]["u"]), "js_v": js(stats["layered"]["v"], stats["flattened"]["v"]),
                         "js_f": js(stats["layered"]["f"], stats["flattened"]["f"]), "mi_layered": stats["layered"]["mi"], "mi_flattened": stats["flattened"]["mi"]})
    gr = G.GateReport()
    gr.placebo("upstream_marginals_matched", observed_max_deviation=float(max(r["js_u"] for r in rows)), tol=0.01)
    gr.placebo("downstream_marginals_matched", observed_max_deviation=float(max(r["js_v"] for r in rows)), tol=0.01)
    gr.placebo("surface_histograms_matched", observed_max_deviation=float(max(r["js_f"] for r in rows)), tol=0.02)
    gr.live("dependency_differs", observed_change=float(np.mean([r["mi_layered"] - r["mi_flattened"] for r in rows])), min_change=0.1)
    v["results"] = {"per_world": rows}
    v["what_must_hold_outside_the_simulation"] = "nothing; a construction check"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_F03(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Among candidate rulers, dependency recovery (mutual information between "
                    "inferred upstream and downstream choices) separates layered from flattened worlds "
                    "where sequence and histogram rulers do not, and sits at chance on null pairs.",
                    "CONSTRUCTED_MECHANISM")
    rulers = {"dependency": lambda world, s: dependency_statistic(world, world.sig, s["blocks"]),
              "hard_argmax_mi": lambda world, s: _hard_mi(world, s["blocks"]),
              "sequence_bigram": lambda world, s: _bigram_mi(s["features"], world.nf),
              "histogram_entropy": lambda world, s: _hist_entropy(s["features"], world.nf)}
    # (steps per block, blocks): the floor cell first, then a readability ladder. The ruler is validated
    # where the blocks are readable (slot accuracy about 0.8 at 128 steps) and read at the floor.
    CELLS = ((4, 12), (32, 12), (32, 60), (128, 60))
    auc = {(r, st, nb): [] for r in rulers for st, nb in CELLS}
    null = {(r, st, nb): [] for r in rulers for st, nb in CELLS}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("F03", wid, 0)
            for st, nb in CELLS:
                lay = [layered_sequence(world, world.sig, rng, n_blocks=nb, steps_per_block=st, topology="layered") for _ in range(60)]
                fla = [layered_sequence(world, world.sig, rng, n_blocks=nb, steps_per_block=st, topology="flattened") for _ in range(60)]
                fla2 = [layered_sequence(world, world.sig, rng, n_blocks=nb, steps_per_block=st, topology="flattened") for _ in range(60)]
                for r, fn in rulers.items():
                    a, b, c = [fn(world, s) for s in lay], [fn(world, s) for s in fla], [fn(world, s) for s in fla2]
                    auc[(r, st, nb)].append(_auc(a, b))
                    null[(r, st, nb)].append(_auc(c, b))
    table = {f"{r}@steps{st}x{nb}": {"auc": float(np.mean(x)), "null_auc": float(np.mean(null[(r, st, nb)]))} for (r, st, nb), x in auc.items()}
    gr = G.GateReport()
    gr.positive("rulers_at_chance_on_null_pairs", observed=float(max(abs(t["null_auc"] - 0.5) for t in table.values())), expected=0.0, tol=0.12,
                detail="two flattened samples must not be separable by any ruler; sixty against sixty sequences per cell")
    gr.live("dependency_ruler_separates_where_blocks_are_readable", observed_change=float(table["dependency@steps128x60"]["auc"] - 0.5), min_change=0.35,
            detail="with 128 steps per block and 60 blocks the upstream and downstream choices are inferred well enough that the "
                   "planted dependency must be recovered (AUC at least 0.85); the ruler is validated there before it is read at the floor")
    at_floor = {r: table[f"{r}@steps4x12"]["auc"] for r in rulers}
    v["results"] = {"by_ruler_and_cell": table, "auc_at_floor_steps4x12": at_floor,
                    "criterion_C_F03": {"passed": bool(at_floor["dependency"] >= max(at_floor["sequence_bigram"], at_floor["histogram_entropy"])),
                                        "dependency_auc_at_floor": at_floor["dependency"]}}
    v["what_must_hold_outside_the_simulation"] = "blocks are segmentable so that upstream and downstream choices can be inferred per block"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_F04(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Fewer goals, erased dependency and low effort are separable three ways "
                    "from artifact statistics (dependency, inferred-goal entropy, length).", "CONSTRUCTED_MECHANISM")
    feats, labels = [], []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("F04", wid, 0)
            for i in range(30):
                for cls in ("fewer_goals", "erased_dependency", "low_effort"):
                    if cls == "fewer_goals":
                        s = _seq(world, world.sig, rng, n_blocks=8, steps=4, topology="layered", goal_set=[0, 1])
                    elif cls == "erased_dependency":
                        s = _seq(world, world.sig, rng, n_blocks=8, steps=4, topology="flattened")
                    else:
                        s = _seq(world, world.sig, rng, n_blocks=8, steps=2, topology="layered")
                    feats.append([dependency_statistic(world, world.sig, s["blocks"]), _u_entropy(world, world.sig, s["blocks"]), float(len(s["features"]))])
                    labels.append(cls)
    Xm, L = np.array(feats), np.array(labels)
    Z = (Xm - Xm.mean(axis=0)) / np.maximum(Xm.std(axis=0), 1e-9)
    correct = []
    for i in range(len(Z)):
        cents = {c: Z[(L == c) & (np.arange(len(Z)) != i)].mean(axis=0) for c in set(L)}
        correct.append(min(cents, key=lambda c: np.linalg.norm(Z[i] - cents[c])) == L[i])
    gr = G.GateReport()
    gr.positive("length_separates_low_effort_exactly", observed=float(np.mean([(Xm[i, 2] < 32) == (L[i] == "low_effort") for i in range(len(L))])), expected=1.0, tol=1e-9,
                detail="low effort is shorter by construction; the known answer for that class")
    gr.live("classifier_above_chance", observed_change=float(np.mean(correct) - 1 / 3), min_change=0.2)
    v["results"] = {"three_way_accuracy": float(np.mean(correct)), "n": len(L)}
    v["what_must_hold_outside_the_simulation"] = "effort is visible as length"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_F05(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "A director's hand (a fixed upstream goal across blocks) survives local "
                    "flattening and partial rewriting, and dies only when the rewrite replaces the goal.",
                    "CONSTRUCTED_MECHANISM")
    res = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("F05", wid, 0)
            other = build_observer_signature(world.sig, 0.5, rng)
            for i in range(30):
                dg = int(rng.integers(world.ng))
                for present in (True, False):
                    for cond in ("clean", "local_flatten", "rewrite_0.5", "rewrite_1.0", "template_change"):
                        topo = "flattened" if cond == "local_flatten" else "layered"
                        s = _seq(world, world.sig, rng, n_blocks=8, steps=4, topology=topo, fixed_u=dg if present else None)
                        blocks = [b.copy() for b in s["blocks"]]
                        if cond.startswith("rewrite"):
                            r = float(cond.split("_")[1])
                            for b in blocks:
                                n_rw = int(round(r * len(b)))
                                if n_rw:
                                    b[rng.choice(len(b), size=n_rw, replace=False)] = rng.choice(world.nf, size=n_rw, p=world.synth)
                        if cond == "template_change":
                            blocks = _seq(world, other, np.random.default_rng(C.seed(f"F05t:{wid}:{i}:{present}")), n_blocks=8, steps=4, topology="layered", fixed_u=dg if present else None)["blocks"]
                        res.setdefault(cond, []).append((_u_entropy(world, world.sig, blocks), present))
    table = {}
    for cond, items in res.items():
        ents = np.array([e for e, _ in items])
        pres = np.array([p for _, p in items])
        thr = float(np.median(ents))
        table[cond] = {"attribution_accuracy": float(np.mean((ents < thr) == pres)), "entropy_present": float(ents[pres].mean()), "entropy_absent": float(ents[~pres].mean())}
    gr = G.GateReport()
    gr.live("director_visible_when_clean", observed_change=float(table["clean"]["attribution_accuracy"] - 0.5), min_change=0.2)
    gr.positive("full_rewrite_erases_the_hand", observed=table["rewrite_1.0"]["attribution_accuracy"], expected=0.5, tol=0.15,
                detail="a full rewrite from the surface prior carries no goal; attribution must fall to chance")
    v["results"] = {"by_condition": table}
    v["what_must_hold_outside_the_simulation"] = "a director's goal persists across blocks"
    return finish(card, v, gr, __file__, decide_state(gr))
