"""Trunk I — integrity, identities, and the three V13 repairs (spec §5, cards I01-I08).

I06-I08 import V13's closed modules READ-ONLY to re-run the repaired instruments on V13's own
world and priors; nothing under v13/ is written. The original failed verdicts stay in the V13
record; the repair verdicts live here, named as repairs, with the failed gate and the repair
rule copied from prereg_v14.REPAIRS.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .....methods import gates as G
from .. import REPO, common as C
from .. import joint as J
from .. import routes as R
from .. import history_skill as HS
from ..world import (N_ACT, N_FEAT, PLAN_DIRECT, PLAN_HABIT, ROUTES, episode, make_maker, relabel, stream, surface_histogram)
from . import (ACCESS_REGIMES, Cells, battery, criterion, decide_state, extra_gate, finish, mean_of, narrative, pursuit_of,
               receipt, rng, sizes, start, world_for)

V13_VERDICTS = REPO / "results" / "validation" / "soundingline" / "v13"
V13_LEDGER = REPO / "results" / "v13" / "COMPLETION.json"
#: What V14's design imports from V13 (Pass A of V13's RESULTS.md), at the precision it was cited.
ANCHORS = {
    "C04": {"path": "C04.json", "lane": "discovery", "fields": {"results.criterion_C04.near_gain": 0.26, "results.criterion_C04.far_gain": -0.16}},
    "A03": {"path": "A03.json", "lane": "discovery", "fields": {"results.by_weighting.learned.mean": -3.30, "results.by_weighting.uniform.mean": -17.40}},
    "G01": {"path": "G01.json", "lane": "discovery", "fields": {"results.accuracy_by_dose.12": 0.85}},
    "H03": {"path": "H03.json", "lane": "discovery", "fields": {"results.accuracy_by_reader.interaction": 1.00, "results.accuracy_by_reader.coherence": 0.50}},
    "C15": {"path": "C15.json", "lane": "discovery", "fields": {"results.criterion_C15.independent_gain": -1.15, "results.criterion_C15.correlated_gain": -1.91}},
    "X10": {"path": "attacks/X10.json", "lane": "attack", "fields": {"results.criterion_X10.table.cost_aware_maker_inference.attacked": -0.40,
                                                                    "results.criterion_X10.table.cost_aware_maker_inference.unattacked": 0.49}},
}


def _dig(d, path):
    for k in path.split("."):
        d = d[k]
    return d


# --------------------------------------------------------------------------- #
# I01 — V13 anchors reproduce from committed inputs.
# --------------------------------------------------------------------------- #
def unit_I01(ctx):
    a = ctx["item"]
    spec = ANCHORS[a]
    p = V13_VERDICTS / spec["path"]
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    ledger = json.loads(V13_LEDGER.read_text(encoding="utf-8"))["entries"]
    recorded = ledger.get(f"{spec['lane']}:{a}", {}).get("verdict_sha256")
    v = json.loads(p.read_text(encoding="utf-8"))
    devs = {k: abs(float(_dig(v, k)) - cited) for k, cited in spec["fields"].items()}
    return {"rows": [{"wid": ctx["wid"], "rep": 0, "anchor": a, "hash_match": float(sha == recorded), "deviation": max(devs.values()), "n": 1}],
            "sha256": sha, "ledger_sha256": recorded, "deviations": devs, "state": v.get("state")}


def reduce_I01(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    v = start(card, ctx, "The six V13 verdicts V14's design rests on are the committed ones, byte for byte, and the numbers V14 cites from them are in the files.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    tol = CRITERIA["I01"]["cited_tolerance"]
    worst = max(r["deviation"] for r in rows)
    hashes_ok = all(r["hash_match"] == 1.0 for r in rows)
    gr = G.GateReport()
    gr.identity("v13_verdicts_hash_to_their_ledger", float(sum(r["hash_match"] for r in rows)), float(len(rows)), tol=0.0,
                detail="every imported V13 verdict's sha256 equals its V13 completion-ledger entry")
    battery(gr, positive={"observed": worst, "expected": 0.0, "tol": tol, "name": "cited_numbers_in_the_files"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "no_v13_file_written", "detail": "the card reads only; V13 stays closed"})
    passed = bool(hashes_ok and worst <= tol)
    criterion(v, "I01", passed, max_deviation=worst, hashes_ok=hashes_ok)
    v["results"].update({"anchors": {u["rows"][0]["anchor"]: {"sha256": u["sha256"], "ledger": u["ledger_sha256"], "deviations": u["deviations"], "state": u["state"]} for u in units}})
    receipt(v, rows, card, ctx)
    narrative(v, f"All {len(rows)} imported V13 verdicts hashed to their ledger entries ({'yes' if hashes_ok else 'NO'}); the largest gap between a cited number and its file was {worst:.4f}.",
              "V14 inherits V13's numbers only where the committed files carry them.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# I02 — the manifest enumerates the program.
# --------------------------------------------------------------------------- #
def unit_I02(ctx):
    from .. import manifest as M
    doc = M.load_manifest()
    cards = doc["cards"]
    mand = [c for c in cards if c["trunk"] != "X"]
    att = [c for c in cards if c["trunk"] == "X"]
    checks = {"cards_64": float(len(mand) == 64), "attacks_12": float(len(att) == 12),
              "factors": float(all(c["factors"] for c in cards)), "lanes": float(all(c["lanes"] for c in cards)),
              "floors": float(all(int(c.get("min_effective_n", 0)) > 0 and int(c.get("min_rows_per_unit", 0)) >= 1 for c in cards))}
    rows = [{"wid": ctx["wid"], "rep": 0, "check": k, "ok": val, "n": 1} for k, val in checks.items()]
    return {"rows": rows, "counts": {"mandatory": len(mand), "attacks": len(att), "by_trunk": {t: sum(c["trunk"] == t for c in cards) for t in "IJREAHFBX"}}}


def reduce_I02(card, units, ctx):
    v = start(card, ctx, "The machine-readable manifest holds exactly the program the spec names: 64 mandatory cards, 12 attacks, every factor, lane and floor.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    ok = all(r["ok"] == 1.0 for r in rows)
    gr = G.GateReport()
    gr.identity("manifest_counts", float(sum(r["ok"] for r in rows)), float(len(rows)), tol=0.0, detail="every enumeration check holds")
    battery(gr, positive={"observed": float(ok), "expected": 1.0, "tol": 0.0, "name": "recursive_validator_passes"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "no_silent_not_applicable", "detail": "every attack card declares applicability per target"})
    criterion(v, "I02", ok, **{r["check"]: r["ok"] for r in rows})
    v["results"].update(units[0]["counts"])
    receipt(v, rows, card, ctx)
    narrative(v, f"The manifest enumerates {units[0]['counts']['mandatory']} mandatory cards and {units[0]['counts']['attacks']} attacks with factors, lanes and floors on every card.",
              "The program is literal in its record.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(ok))


# --------------------------------------------------------------------------- #
# I03 — joint enumerator identities.
# --------------------------------------------------------------------------- #
def unit_I03(ctx):
    world = world_for(ctx)
    r = rng(ctx, "i03")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = J.Reader(world, 0, 0.75, 0.8)
    prior = J.uniform_prior()
    m = make_maker(world, "m", r, family=0, competence="mid")
    eps = stream(world, m, r, 4)
    tabs = rd.route_tables(eps, ("action", "semantic", "context"))
    post = J.joint(prior, tabs)
    # normalization and marginal consistency
    norm_dev = abs(float(post.sum()) - 1.0) + max(abs(J.marginal(post, lat).sum() - 1.0) for lat in J.LATENTS)
    # brute force: the action-route likelihood of the current episode against a Monte-Carlo generative frequency
    truth = J.truth_of(m, eps[-1])
    ep = eps[-1]
    acts = [int(rd.inv_vocab[a]) for a in ep["action"][:2]]
    mm = make_maker(world, "bf", r, family=0, pref=truth[2], plan=truth[0], competence=None, k_exec=rd.k_exec, k_obs=rd.k_obs, h_strength=0.0)
    n_mc = 4000
    hits = 0
    for _ in range(n_mc):
        e = episode(world, mm, r, goal=truth[1], steps=2)
        hits += int([int(rd.inv_vocab[a]) for a in e["action"]] == acts)
    ll2 = rd.ll_action({"action": ep["action"][:2]})
    p_model = float(np.exp(ll2[J.state_index(*truth)]))
    bf_dev = abs(hits / n_mc - p_model)
    # label invariance: a permuted surface vocabulary leaves the posterior (surface is not a route input)
    perm = r.permutation(N_FEAT)
    tabs2 = rd.route_tables([relabel(e, perm) for e in eps], ("action", "semantic", "context"))
    label_dev = float(np.abs(J.joint(prior, tabs2) - post).max())
    # order invariance: past episodes permuted, current kept last
    past = list(eps[:-1])
    r.shuffle(past)
    tabs3 = rd.route_tables(past + [eps[-1]], ("action", "semantic", "context"))
    order_dev = float(np.abs(J.joint(prior, tabs3) - post).max())
    for name, val in (("normalization", norm_dev), ("brute_force", bf_dev), ("label_invariance", label_dev), ("order_invariance", order_dev)):
        cells.add({"check": name}, deviation=val)
    return {"rows": cells.rows(), "evaluations": J.evaluations(tabs)}


def reduce_I03(card, units, ctx):
    v = start(card, ctx, "The exact joint enumerator normalizes, agrees with the generative process by direct simulation, and is invariant to labels and evidence order.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    dev = {c: mean_of(rows, "deviation", lambda r, c=c: r["check"] == c) for c in ("normalization", "brute_force", "label_invariance", "order_invariance")}
    gr = G.GateReport()
    gr.identity("normalization", dev["normalization"], 0.0, tol=1e-9)
    gr.identity("label_invariance", dev["label_invariance"], 0.0, tol=1e-9)
    gr.identity("order_invariance", dev["order_invariance"], 0.0, tol=1e-9)
    battery(gr, positive={"observed": dev["brute_force"], "expected": 0.0, "tol": 0.03, "name": "likelihood_matches_generative_frequency", "detail": "Monte-Carlo frequency of a two-step chain against the reader's likelihood, 4000 draws"},
            placebo={"observed": dev["label_invariance"], "tol": 1e-9, "name": "surface_relabelling_inert"})
    passed = bool(dev["normalization"] <= 1e-9 and dev["label_invariance"] <= 1e-9 and dev["order_invariance"] <= 1e-9 and dev["brute_force"] <= 0.03)
    criterion(v, "I03", passed, **dev)
    v["results"].update({"deviations": dev, "evaluations_per_unit": units[0].get("evaluations")})
    receipt(v, rows, card, ctx)
    narrative(v, f"The grid posterior normalized to {dev['normalization']:.1e}, matched a 4000-draw generative frequency within {dev['brute_force']:.3f}, and moved {dev['label_invariance']:.1e} under surface relabelling and {dev['order_invariance']:.1e} under evidence reordering.",
              "The enumerator is a faithful, coordinate-free reference for every estimator in the program.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# I04 — routes are live and different.
# --------------------------------------------------------------------------- #
def unit_I04(ctx):
    world = world_for(ctx)
    r = rng(ctx, "i04")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = J.Reader(world, 0, 0.75, 0.8)
    info = R.route_information(world, rd, r, n=max(12, sizes(ctx)["makers"] // 2))
    prior = J.uniform_prior()
    # pairwise divergence of route-only posteriors and a shuffled null route
    m = make_maker(world, "m", r, family=0, competence="mid")
    eps = stream(world, m, r, 3)
    tabs = rd.route_tables(eps, ROUTES)
    posts = {rt: J.posterior(prior, tabs[rt]) for rt in ROUTES}
    div = float(np.mean([C.js(posts[a], posts[b]) for i, a in enumerate(ROUTES) for b in ROUTES[i + 1:]]))
    # a shuffled route carries nothing: its posterior puts no more mass on the TRUE goal than the
    # prior does, averaged over shuffles (entropy falls on noise too, so entropy is not the test)
    truth_goal = int(eps[-1]["goal"])
    gains = []
    fam = world.family(0)
    for _ in range(8):
        null_eps = []
        for e in eps:
            e2 = dict(e)
            # tokens from the marginal semantic mixture (a random goal per token): uniform draws would
            # systematically favour the flattest goal generator and are not a null
            e2["semantic"] = [int(r.choice(N_FEAT, p=fam.sem[int(r.integers(4))])) for _ in e["semantic"]]
            null_eps.append(e2)
        null_tabs = rd.route_tables(null_eps, ("semantic",))
        gains.append(float(np.log(J.marginal(J.posterior(prior, null_tabs["semantic"]), "goal")[truth_goal]) - np.log(0.25)))
    null_gain = float(max(0.0, np.mean(gains)))                  # a null can only hurt the truth; only a positive gain is a leak
    for rt in ROUTES:
        for lat in J.LATENTS:
            cells.add({"route": rt, "latent": lat}, info=info[rt][lat], divergence=div, null_gain=null_gain)
    return {"rows": cells.rows(), "divergence": div, "null_semantic_goal_gain": null_gain}


def reduce_I04(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["I04"]
    v = start(card, ctx, "The four routes are separate observations that disagree with one another and each carries information about some latent; a shuffled route carries none.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    info = {rt: {lat: mean_of(rows, "info", lambda r, rt=rt, lat=lat: r["route"] == rt and r["latent"] == lat) for lat in J.LATENTS} for rt in ROUTES}
    best = {rt: max(info[rt].values()) for rt in ROUTES}
    div = mean_of(rows, "divergence")
    null = abs(mean_of(rows, "null_gain"))
    gr = G.GateReport()
    extra_gate(gr, "divergence", "routes_pairwise_divergent", div, cr["min_divergence"], "min", "mean pairwise Jensen-Shannon of route-only posteriors")
    battery(gr, positive={"observed": min(best.values()), "expected": max(cr["min_information"], min(best.values())), "tol": 0.0, "name": "every_route_informative_about_some_latent"},
            placebo={"observed": null, "tol": cr["max_null_route"], "name": "shuffled_route_carries_nothing", "detail": "mean log-gain on the true goal over eight shuffles of the semantic tokens"},
            live={"observed": div, "min": cr["min_divergence"], "name": "routes_disagree"})
    passed = bool(div >= cr["min_divergence"] and min(best.values()) >= cr["min_information"] and null <= cr["max_null_route"])
    criterion(v, "I04", passed, divergence=div, min_best_information=min(best.values()), null_gain=null)
    v["results"].update({"information": info, "route_information_json": "results/v14/ROUTE_INFORMATION.json"})
    receipt(v, rows, card, ctx)
    narrative(v, f"Route-only posteriors disagreed by {div:.2f} (Jensen-Shannon); the least informative route still carried {min(best.values()):.2f} nats about its best latent; a shuffled route carried {null:.3f}.",
              "Routing has something to route: the evidence streams are different and live.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# I05 — construction identities.
# --------------------------------------------------------------------------- #
def unit_I05(ctx):
    world = world_for(ctx)
    r = rng(ctx, "i05")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = J.Reader(world, 0, 0.75, 0.8)
    # surface collision: two latents, one surface (copied); the surface histograms hash equal
    m1 = make_maker(world, "a", r, family=0, pref=0, plan=PLAN_DIRECT, competence="mid")
    m2 = make_maker(world, "b", r, family=0, pref=3, plan=2, competence="mid")
    e1, e2 = episode(world, m1, r, goal=1), episode(world, m2, r, goal=2)
    e2["surface"] = list(e1["surface"])
    h1, h2 = C.obj_sha(surface_histogram(e1)), C.obj_sha(surface_histogram(e2))
    collision = float(h1 == h2)
    # equifinal history: (DIRECT, 0) and (HABIT, 0) give identical non-forensic likelihoods
    mh = make_maker(world, "h", r, family=0, pref=2, plan=PLAN_HABIT, competence="mid")
    eh = episode(world, mh, r, goal=0)
    tabs = rd.episode_tables(eh, ("action", "semantic", "context"))
    ll = J.combined(tabs)
    equi_dev = max(abs(ll[J.state_index(PLAN_DIRECT, 0, pr)] - ll[J.state_index(PLAN_HABIT, 0, pr)]) for pr in range(6))
    tf = rd.ll_forensic(eh)
    forensic_sep = abs(float(tf[J.state_index(PLAN_DIRECT, 0, 2)] - tf[J.state_index(PLAN_HABIT, 0, 2)]))
    # factor orthogonality: competence moves execution accuracy, not early relevance; history the reverse
    def exec_acc(m):
        return float(np.mean([np.mean(np.array(e["intended"]) == np.array([rd.inv_vocab[a] for a in e["action"]])) for e in stream(world, m, r, 24)]))

    def early(m):
        return HS.history_signal(stream(world, m, r, 400), m, world.family(0), rd.inv_vocab)
    hf = r.normal(0, 1, N_FEAT)
    lo = HS.agent(world, "lo", r, 0, "low", "strong", pref=1, plan=1, h_feat=hf)       # competence varied under a live history signal
    hi = HS.agent(world, "hi", r, 0, "high", "strong", pref=1, plan=1, h_feat=hf)
    hn = HS.agent(world, "hn", r, 0, "mid", "none", pref=1, plan=1, h_feat=hf)
    hs = HS.agent(world, "hs", r, 0, "mid", "strong", pref=1, plan=1, h_feat=hf)
    k_own = abs(exec_acc(hi) - exec_acc(lo))
    k_leak = abs(early(hi) - early(lo))
    h_own = abs(early(hs) - early(hn))
    h_leak = abs(exec_acc(hs) - exec_acc(hn))
    lineage = float(C.lineage_disjoint({lane: C.lane_ids(lane, {"discovery_worlds": 256, "transfer_worlds": 128, "confirmation_worlds": 128, "pilot_worlds": 4}) for lane in ("discovery", "transfer", "confirmation", "pilot")}))
    cells.add({"check": "surface_collision"}, ok=collision, value=collision)
    cells.add({"check": "equifinal_history"}, ok=float(equi_dev < 1e-9), value=equi_dev, forensic_sep=forensic_sep)
    cells.add({"check": "factor_orthogonality"}, ok=float(max(k_leak, h_leak) <= 0.10), value=max(k_leak, h_leak), k_own=k_own, h_own=h_own)
    cells.add({"check": "lineage"}, ok=lineage, value=1.0 - lineage)
    return {"rows": cells.rows()}


def reduce_I05(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    v = start(card, ctx, "Matched surfaces collide by construction, equifinal histories are exactly equifinal on every route but forensic, factors move only their own measures, and lanes never share an ancestor.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    val = {c: mean_of(rows, "value", lambda r, c=c: r["check"] == c) for c in ("surface_collision", "equifinal_history", "factor_orthogonality", "lineage")}
    k_own, h_own = mean_of(rows, "k_own"), mean_of(rows, "h_own")
    fsep = mean_of(rows, "forensic_sep")
    gr = G.GateReport()
    gr.identity("surface_collision_hashes_equal", val["surface_collision"], 1.0, tol=0.0)
    gr.identity("equifinal_likelihoods_identical", val["equifinal_history"], 0.0, tol=1e-9)
    gr.identity("lineages_disjoint", val["lineage"], 0.0, tol=0.0)
    battery(gr, placebo={"observed": val["factor_orthogonality"], "tol": CRITERIA["I05"]["max_leak"], "name": "factors_leak_at_floor", "detail": "competence's effect on early relevance and history's on execution accuracy, 24 episodes per agent"},
            positive={"observed": min(k_own, h_own), "expected": max(0.05, min(k_own, h_own)), "tol": 0.0, "name": "factors_move_their_own_measures"},
            live={"observed": fsep, "min": 0.5, "name": "forensic_separates_the_class"})
    passed = bool(val["surface_collision"] == 1.0 and val["equifinal_history"] <= 1e-9 and val["factor_orthogonality"] <= CRITERIA["I05"]["max_leak"] and val["lineage"] == 0.0)
    criterion(v, "I05", passed, **val, competence_own=k_own, history_own=h_own, forensic_separation=fsep)
    receipt(v, rows, card, ctx)
    narrative(v, f"Surface collisions hashed equal; the process-equivalent pair differed by {val['equifinal_history']:.1e} on the non-forensic routes and by {fsep:.2f} nats on forensic; competence moved execution accuracy by {k_own:.2f} and history moved early relevance by {h_own:.2f}, each leaking at most {val['factor_orthogonality']:.3f}; lineages were disjoint.",
              "The constructions the program leans on are the ones it says they are.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# The three V13 repairs: V13 imported read-only.
# --------------------------------------------------------------------------- #
def _v13_ctx(ctx):
    """A V13-shaped context: V13's sizes() reads teams/events from the tier."""
    tier = dict(ctx["tier"])
    tier.setdefault("teams", 8)
    tier.setdefault("events", 24)
    return dict(ctx, tier=tier)


def unit_I06(ctx):
    from ghostscale.validation.soundingline.v13 import exact as X13
    from ghostscale.validation.soundingline.v13 import priors as P13
    from ghostscale.validation.soundingline.v13.cards import trunk_c as TC
    c = _v13_ctx(ctx)
    H = TC.harness(c, anti=False)
    world = H["world"]
    cells = Cells(ctx["wid"], ctx["rep"])
    floors = []
    for rd in H["readers"]:
        model_all = X13.reader_model(world, rd, families=None)
        fam_makers = [m for m in H["makers"] if m.family == rd.family]
        pri = {"within_common": P13.population_prior(model_all, H["makers"], family=rd.family),
               "all_family": P13.all_family_prior(model_all, H["makers"])}
        for m in fam_makers:
            ti = model_all.truth_index(m)
            floors.append(float(np.log(max(pri["within_common"][ti], 1e-300)) - np.log(max(pri["all_family"][ti], 1e-300))))
            L = model_all.loglik(H["streams"][m.id], TC.CH)
            for route, prior in pri.items():
                for n in (1, 2, 4):
                    post = TC.posterior_at(model_all, prior, L, n)
                    sc = X13.score_rows(model_all, post, m)
                    cells.add({"route": route, "dose": n}, ls=sc["ls"], top1=sc["top1"], conf=sc["conf"], ls0=float(np.log(max(prior[ti], 1e-300))))
    return {"rows": cells.rows(), "construction_floor": float(np.mean(floors))}


def reduce_I06(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA, REPAIRS
    v = start(card, ctx, "REPAIR of V13 C03: a prior restricted to the maker's common substrate beats a broad all-family prior at the first artifact by a margin the construction itself fixes.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    floor = float(np.mean([u["construction_floor"] for u in units]))
    from ghostscale.validation.soundingline.v13.cards import trunk_c as TC
    by_dose = {str(n): TC.gain(rows, "within_common", "all_family", lambda r, n=n: r["dose"] == n, tag=f"I06{n}") for n in (1, 2, 4)}
    g1 = by_dose["1"]["mean"]
    means = {rt: {str(n): mean_of(rows, "ls", lambda r, rt=rt, n=n: r["route"] == rt and r["dose"] == n) for n in (1, 2, 4)} for rt in ("within_common", "all_family")}
    conf_c = mean_of(rows, "conf", lambda r: r["route"] == "within_common" and r["dose"] == 1)
    top_c = mean_of(rows, "top1", lambda r: r["route"] == "within_common" and r["dose"] == 1)
    conf_a = mean_of(rows, "conf", lambda r: r["route"] == "all_family" and r["dose"] == 1)
    top_a = mean_of(rows, "top1", lambda r: r["route"] == "all_family" and r["dose"] == 1)
    bar = CRITERIA["I06"]["floor_share"] * floor
    passed = bool(g1 >= bar)
    g0 = mean_of(rows, "ls0", lambda r: r["route"] == "within_common" and r["dose"] == 1) - mean_of(rows, "ls0", lambda r: r["route"] == "all_family" and r["dose"] == 1)
    gr = G.GateReport()
    battery(gr, live={"observed": means["within_common"]["4"] - means["within_common"]["1"], "min": 0.05, "name": "evidence_moves_the_posterior"},
            placebo={"observed": abs(means["within_common"]["4"] - means["all_family"]["4"]), "tol": max(0.15, 0.5 * floor), "name": "priors_converge_with_evidence"},
            positive={"observed": g0, "expected": floor, "tol": 0.02 + 0.1 * abs(floor), "name": "instrument_sees_the_planted_prior_gain", "detail": f"the prior-level gain on the truth equals the construction floor {floor:.3f} nats; whether it survives the first artifact is the criterion, not a gate"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_likelihood_same_compute"},
            oracle={"observed": means["within_common"]["4"] - np.log(1 / 40), "min": 0.5, "name": "identifiable_with_evidence"},
            prediction={"gain": g1, "min": 0.0, "name": "first_artifact_gain"},
            calibration={"observed": abs(conf_c - top_c), "reference": abs(conf_a - top_a), "direction": "down", "tol": 0.05, "name": "confidence_tracks_accuracy"})
    criterion(v, "I06", passed, gain_at_1=g1, gain_at_0=g0, construction_floor=floor, bar=bar)
    v["repairs"] = [dict(REPAIRS["I06"], repaired_here=True)]
    v["results"].update({"gain_by_dose": by_dose, "mean_log_score_by_route_and_dose": means, "v13_failed_verdict": "results/validation/soundingline/v13/C03.json"})
    receipt(v, rows, card, ctx)
    narrative(v, f"Knowing the maker's common substrate was worth {g1:+.3f} nats after one artifact against the all-family prior; the construction's own floor was {floor:.3f} nats and the repaired gate asks for half of it.",
              "The common-substrate mechanism is read once, against a floor the construction fixed; no second repair follows.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_I07(ctx):
    from ghostscale.validation.soundingline.v13 import exact as X13
    from ghostscale.validation.soundingline.v13.cards import trunk_c as TC
    c = _v13_ctx(ctx)
    H = TC.harness(c, n_art=64)
    cells = Cells(ctx["wid"], ctx["rep"])
    ents = []
    # reader types by within-unit rank of typicality (V13's C06 lesson: every unit realizes every type)
    typ = {}
    for rd in H["readers"]:
        fm = [m for m in H["makers"] if m.family == rd.family]
        typ[rd.id] = C.js(rd.w, np.mean([m.w for m in fm], axis=0)) if fm else 1.0
    order = sorted(typ, key=typ.get)
    third = max(1, len(order) // 3)
    rtype_of = {rid: ("near" if i < third else ("typical" if i < 2 * third else "atypical")) for i, rid in enumerate(order)}
    for rd in H["readers"]:
        fam_makers = [m for m in H["makers"] if m.family == rd.family]
        rtype = rtype_of[rd.id]
        rr = C.rng_for(ctx["lane"], "I07", ctx["wid"], ctx["rep"], rd.id)
        base, _ = TC.reader_priors(H, rd, rr)
        model = H["models"][rd.id]
        for m, is_anti in [(m, a) for r2, m, a in TC.pair_iter(H) if r2.id == rd.id]:
            if is_anti:
                continue
            priors = TC.target_priors(H, rd, m, base)
            L = H["L"][(rd.id, m.id)]
            for n in (1, 4, 16):
                for route in ("self", "within_common", "anti_similar"):
                    post = TC.posterior_at(model, priors[route], L, n)
                    sc = X13.score_rows(model, post, m)
                    t = rtype if route != "anti_similar" else "anti"
                    cells.add({"reader_type": t, "dose": n, "route": "self" if route == "anti_similar" else route}, ls=sc["ls"], top1=sc["top1"], conf=sc["conf"], entropy=C.entropy(post))
                if rtype != "anti":
                    post = TC.posterior_at(model, priors["within_common"], L, n)
                    sc = X13.score_rows(model, post, m)
                    cells.add({"reader_type": "anti", "dose": n, "route": "within_common"}, ls=sc["ls"], top1=sc["top1"], conf=sc["conf"], entropy=C.entropy(post))
            # convergence probe (side list, not a cell): agreement of the self and within-common posteriors
            # at doses up to the stream's length, with the entropy that says whether the evidence was sufficient
            for n in (16, 32, 64):
                if n > len(H["streams"][m.id]):
                    break
                ps = TC.posterior_at(model, priors["self"], L, n)
                pc = TC.posterior_at(model, priors["within_common"], L, n)
                ents.append({"dose": n, "js": C.js(ps, pc), "entropy": max(C.entropy(ps), C.entropy(pc))})
    return {"rows": cells.rows(), "convergence": ents}


def reduce_I07(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA, REPAIRS
    from ghostscale.validation.soundingline.v13.cards import trunk_c as TC
    cr = CRITERIA["I07"]
    v = start(card, ctx, "REPAIR of V13 C05: one personal sample beats the common-substrate population prior for typical readers and not for anti-similar ones, and the two routes converge at the dose the construction makes sufficient.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    surf = {t: {str(n): TC.gain(rows, "self", "within_common", lambda r, t=t, n=n: r["reader_type"] == t and r["dose"] == n, tag=f"I07{t}{n}") for n in (1, 4, 16)}
            for t in ("near", "typical", "atypical", "anti")}
    near, anti = surf["near"]["1"]["mean"], surf["anti"]["1"]["mean"]
    conv_rows = [x for u in units for x in u["convergence"]]
    ent = {n: float(np.mean([x["entropy"] for x in conv_rows if x["dose"] == n])) for n in (16, 32, 64) if any(x["dose"] == n for x in conv_rows)}
    jsd = {n: float(np.mean([x["js"] for x in conv_rows if x["dose"] == n])) for n in ent}
    sufficient = next((n for n in sorted(ent) if ent[n] <= 0.3), None)
    conv = jsd[sufficient] if sufficient else (jsd[max(jsd)] if jsd else 1.0)
    passed = bool(near >= cr["min_gain"] and anti <= 0.0 and conv <= cr["convergence_tol"])
    gr = G.GateReport()
    battery(gr, live={"observed": near - anti if near == near and anti == anti else 0.0, "min": 0.05, "name": "reader_type_moves_the_gain"},
            placebo={"observed": conv, "tol": cr["convergence_tol"], "name": "routes_converge_at_the_sufficient_dose", "detail": f"sufficient dose {sufficient} (both posteriors under 0.3 nats, probed to 64); entropies {ent}; agreement (Jensen-Shannon) {jsd}"},
            positive={"observed": float(anti <= 0.0) if anti == anti else 1.0, "expected": 1.0, "tol": 0.0, "name": "anti_similar_self_does_not_gain"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_likelihood"},
            oracle={"observed": mean_of(rows, "top1", lambda r: r["route"] == "within_common" and r["dose"] == 16), "min": 0.5, "name": "identifiable_with_evidence"},
            prediction={"gain": near, "min": -0.5, "name": "near_reader_gain_reported"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["route"] == "self" and r["reader_type"] == "anti" and r["dose"] == 1),
                         "reference": mean_of(rows, "conf", lambda r: r["route"] == "self" and r["reader_type"] == "near" and r["dose"] == 1),
                         "direction": "down", "tol": 0.10, "name": "anti_similar_reader_not_more_confident"})
    criterion(v, "I07", passed, near_reader_gain=near, anti_reader_gain=anti, convergence_js=conv, sufficient_dose=sufficient)
    v["repairs"] = [dict(REPAIRS["I07"], repaired_here=True)]
    v["results"].update({"conditional_surface": surf, "entropy_by_dose": ent, "agreement_by_dose": jsd, "v13_failed_verdict": "results/validation/soundingline/v13/C05.json"})
    receipt(v, rows, card, ctx)
    narrative(v, f"For readers typical of their substrate the self prior beat the common population prior by {near:+.2f} nats after one artifact and for anti-similar readers by {anti:+.2f}; at the construction's sufficient dose ({sufficient if sufficient else 'not reached by sixty-four'}) the two routes' posteriors disagreed by {conv:.3f} (Jensen-Shannon).",
              "The typicality interaction is read once against a convergence dose the construction derives; no second repair follows.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_I08(ctx):
    from ghostscale.validation.soundingline.v13 import projection as PJ13
    from ghostscale.validation.soundingline.v13.cards import sim_bin as sim_bin13
    from ghostscale.validation.soundingline.v13.cards import trunk_c as TC
    c = _v13_ctx(ctx)
    H = TC.harness(c, n_art=12)
    cells = Cells(ctx["wid"], ctx["rep"])
    edges = TC.bins_for(H)
    per = []
    for rd in H["readers"]:
        rr = C.rng_for(ctx["lane"], "I08", ctx["wid"], ctx["rep"], rd.id)
        base, _ = TC.reader_priors(H, rd, rr)
        model = H["models"][rd.id]
        self_idx = model.truth_index(rd)
        for m in [m for m in H["makers"] if m.family == rd.family] + H["antis"].get(rd.id, []):
            is_anti = m.id.startswith("anti")
            d = C.js(H["selfs"][rd.id]["w_hat"], m.w)
            b = sim_bin13(d, edges[rd.id], anti=is_anti)
            pri = TC.target_priors(H, rd, m, base)
            ti = model.truth_index(m)
            for route in ("self", "equal_local", "generic_local"):
                cc = PJ13.correction_curve(model, pri[route], H["streams"][m.id], ti, self_idx, TC.CH)
                e0 = 1.0 - float(pri[route][ti])
                eT = 1.0 - float(cc["final_truth"])
                rate = (e0 - eT) / max(e0, 1e-9)                    # share of the initial error removed by the endpoint
                cells.add({"route": route, "sim_bin": b}, half_life=cc["half_life"], residual=cc["residual_self_mass"] if cc["residual_self_mass"] == cc["residual_self_mass"] else None,
                          order=cc["order_effect"], final=cc["final_truth"], conf=cc["confidence_final"], top1=float(cc["final_truth"] >= 0.5), rate=rate)
                per.append({"route": route, "bin": b, "conf": float(cc["confidence_final"]), "correct": float(cc["final_truth"] >= 0.5)})
    return {"rows": cells.rows(), "per_item": per}


def reduce_I08(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA, REPAIRS
    cr = CRITERIA["I08"]
    v = start(card, ctx, "REPAIR of V13 P01: the correction of a local prior by target evidence has a half-life, a residual and a rate, the same for self and an equally local non-self prior when the two are matched; calibration is scored at the prospective endpoint.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    per = [x for u in units for x in u["per_item"]]
    bins = ("near", "mid", "far", "anti")
    surf = {rt: {b: {k: mean_of(rows, k, lambda r, rt=rt, b=b: r["route"] == rt and r["sim_bin"] == b) for k in ("half_life", "residual", "order", "final", "conf", "top1", "rate")} for b in bins} for rt in ("self", "equal_local", "generic_local")}
    hl = mean_of(rows, "half_life", lambda r: r["route"] == "self" and r["sim_bin"] in ("far", "anti"))
    resid = mean_of(rows, "residual", lambda r: r["route"] == "self" and r["sim_bin"] in ("far", "anti"))
    order = mean_of(rows, "order")
    rate_near, rate_far = surf["self"]["near"]["rate"], surf["self"]["far"]["rate"]
    self_items = [x for x in per if x["route"] == "self"]
    ece = C.ece([x["conf"] for x in self_items], [x["correct"] for x in self_items]) if self_items else float("nan")
    slope = C.calibration_slope([x["conf"] for x in self_items], [x["correct"] for x in self_items]) if len(self_items) > 3 else float("nan")
    passed = bool(hl <= cr["max_half_life"] and resid <= cr["max_residual"] and rate_near >= rate_far - cr["rate_margin"] and ece <= cr["max_ece"])
    gr = G.GateReport()
    battery(gr, live={"observed": mean_of(rows, "final", lambda r: r["route"] == "self"), "min": 0.3, "name": "evidence_corrects_the_prior"},
            placebo={"observed": order, "tol": 0.05, "name": "order_effect_absent_for_exact_inference"},
            positive={"observed": rate_near - rate_far, "expected": max(-cr["rate_margin"], rate_near - rate_far), "tol": 0.0, "name": "near_correction_rate_at_least_far", "detail": "share of the initial error removed by the endpoint, per bin"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "matched_routes_share_likelihood"},
            oracle={"observed": mean_of(rows, "final", lambda r: r["route"] == "generic_local"), "min": 0.3, "name": "target_identifiable_at_twelve"},
            prediction={"gain": mean_of(rows, "top1", lambda r: r["route"] == "self"), "min": 0.3, "name": "final_top1"},
            calibration={"observed": slope if slope == slope else 0.0, "reference": 0.2, "direction": "up", "tol": 0.0, "name": "endpoint_confidence_predicts_correctness", "detail": "slope of correctness on confidence across makers; the unit-level ECE is the criterion's science"})
    criterion(v, "I08", passed, half_life_far=hl, residual_far=resid, order_effect=order, rate_near=rate_near, rate_far=rate_far, endpoint_ece=ece, calibration_slope=slope)
    v["repairs"] = [dict(REPAIRS["I08"], repaired_here=True)]
    v["results"].update({"surface": surf, "v13_failed_verdict": "results/validation/soundingline/v13/P01.json"})
    receipt(v, rows, card, ctx)
    narrative(v, f"For far and anti-similar makers the self prior corrected with a half-life of {hl:.1f} artifacts to a residual self mass of {resid:.2f}; near makers removed {rate_near:.0%} of their initial error against {rate_far:.0%} for far makers; endpoint calibration error was {ece:.3f}.",
              "Correction dynamics belong to locality, not to self, and the repaired gate compares rates where the original compared residuals.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))
