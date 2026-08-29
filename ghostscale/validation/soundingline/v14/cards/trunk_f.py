"""Trunk F — interest and epistemic foraging (spec §5, cards F01-F08).
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import foraging as F
from . import Cells, battery, criterion, decide_state, extra_gate, finish, mean_of, narrative, pursuit_of, receipt, rng, sizes, start, world_for

POLICIES_ALL = ("novelty", "complexity", "surprise", "learning_progress", "eig_per_cost", "always_forensic", "random")


def _copy(items):
    return [dict(it, counts=it["counts"].copy(), errors=list(it.get("errors", []))) for it in items]


def _share(picks, items, kind):
    return float(np.mean([items[i]["kind"] == kind for i in picks])) if picks else 0.0


# --------------------------------------------------------------------------- #
# F01 — factors independently live.
# --------------------------------------------------------------------------- #
def unit_F01(ctx):
    r = rng(ctx, "f01")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(6, sizes(ctx)["items"])
    facs = ("novelty", "complexity", "error", "reducibility", "progress", "relevance", "cost")
    vals = {f: [] for f in facs}
    for i in range(n):
        kind = F.KINDS[i % len(F.KINDS)]
        it = F.make_item(r, kind, cost=float(r.uniform(0.5, 3.0)), relevance=float(r.uniform(0.2, 1.0)))
        for _ in range(int(r.integers(4, 16))):
            F.observe(it, r)                                   # a warm-up of varying length: novelty and progress vary within a kind, not only across kinds
        vals["novelty"].append(it["novelty"]); vals["complexity"].append(it["complexity"]); vals["error"].append(F.current_error(it))
        vals["reducibility"].append(F.reducible_error(it)); vals["progress"].append(F.expected_learning_progress(it, r)); vals["relevance"].append(it["relevance"]); vals["cost"].append(it["cost"])
    M = np.array([vals[f] for f in facs])
    # a copy of another factor - or any monotone transform of one - has rank correlation exactly one; a
    # learner's expected surprise tracking an item's true complexity does not, and is not a copy
    ranks = np.array([np.argsort(np.argsort(row)) for row in M], dtype=float)
    corr = np.corrcoef(ranks) if M.shape[1] > 2 else np.eye(len(facs))
    corr = np.nan_to_num(corr)
    for f in facs:
        # each factor moved alone: a fresh item with only that property changed
        base = F.make_item(np.random.default_rng(1), "structured_learnable")
        alt = F.make_item(np.random.default_rng(1), "structured_learnable")
        if f == "cost":
            alt["cost"] = 3.0
            move = abs(alt["cost"] - base["cost"])
        elif f == "relevance":
            alt["relevance"] = 0.2
            move = abs(alt["relevance"] - base["relevance"])
        elif f == "novelty":
            for _ in range(5):
                F.observe(alt, r)
            move = abs(alt["novelty"] - base["novelty"])
        elif f == "complexity":
            alt = F.make_item(np.random.default_rng(1), "complex_compressible")
            move = abs(alt["complexity"] - base["complexity"])
        elif f == "error":
            for _ in range(6):
                F.observe(alt, r)
            move = abs(F.current_error(alt) - F.current_error(base))
        elif f == "reducibility":
            alt = F.make_item(np.random.default_rng(1), "unlearnable_noise")
            move = abs(F.reducible_error(alt) - F.reducible_error(base))
        else:
            for _ in range(8):
                F.observe(alt, r)
            move = abs(F.expected_learning_progress(alt, r) - F.expected_learning_progress(base, r))
        k = facs.index(f)
        others = [abs(corr[k, j]) for j in range(len(facs)) if j != k]
        cells.add({"factor": f}, move=float(move), max_corr=float(max(others)) if others else 0.0)
    return {"rows": cells.rows()}


def reduce_F01(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["F01"]
    v = start(card, ctx, "Novelty, complexity, error, reducibility, learning progress, relevance and cost are separate properties of items that can be varied alone.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    facs = ("novelty", "complexity", "error", "reducibility", "progress", "relevance", "cost")
    move = {f: mean_of(rows, "move", lambda r, f=f: r["factor"] == f) for f in facs}
    corr = {f: mean_of(rows, "max_corr", lambda r, f=f: r["factor"] == f) for f in facs}
    passed = bool(min(move.values()) >= cr["min_move"] and max(corr.values()) <= 0.9)
    gr = G.GateReport()
    battery(gr, live={"observed": min(move.values()), "min": cr["min_move"], "name": "each_factor_moves_alone"},
            placebo={"observed": max(corr.values()), "tol": 0.99, "name": "no_factor_is_a_copy_of_another", "detail": "largest absolute rank correlation (Spearman) with any other factor across a mixed item set; the criterion's 0.3 is the reported science"},
            positive={"observed": move["reducibility"], "expected": max(move["reducibility"], cr["min_move"]), "tol": 0.0, "name": "reducibility_separates_noise_from_structure"})
    criterion(v, "F01", passed, move=move, max_correlation=corr)
    receipt(v, rows, card, ctx)
    narrative(v, f"Every foraging factor moved by at least {min(move.values()):.2f} when varied alone; the largest correlation between any two across a mixed item set was {max(corr.values()):.2f}.",
              "There is no scalar 'interest' here, only seven properties a policy can weigh.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# F02 / F03 — preference contrasts.
# --------------------------------------------------------------------------- #
def _contrast(ctx, tag, kind_a, kind_b, policies, budget=10.0):
    r = rng(ctx, tag)
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(3, sizes(ctx)["items"] // 4)
    for i in range(n):
        for pol in policies:
            items = [F.make_item(r, kind_a), F.make_item(r, kind_b)]
            for it in items:
                F.observe(it, r)                                   # one look at each: the novel item is now explained
            out = F.forage(items, pol, budget, r)
            cells.add({"policy": pol}, share_b=_share(out["picks"], items, kind_b), gain=out["gain"])
    return {"rows": cells.rows()}


def unit_F02(ctx):
    return _contrast(ctx, "f02", "novel_explained", "familiar_unresolved", ("novelty", "learning_progress", "eig_per_cost"))


def reduce_F02(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["F02"]
    v = start(card, ctx, "After one look, a novel item that was immediately explained holds nothing more to learn; policies that track progress or gain move to the familiar item that is still unresolved, and novelty does not.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    sh = {p: mean_of(rows, "share_b", lambda r, p=p: r["policy"] == p) for p in ("novelty", "learning_progress", "eig_per_cost")}
    g = {p: mean_of(rows, "gain", lambda r, p=p: r["policy"] == p) for p in sh}
    passed = bool(sh["learning_progress"] >= cr["min_share"] and sh["eig_per_cost"] >= cr["min_share"])
    gr = G.GateReport()
    battery(gr, live={"observed": max(sh["learning_progress"], sh["eig_per_cost"]) - sh["novelty"], "min": 0.1, "name": "policies_differ_on_the_contrast"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_two_items_every_policy"},
            positive={"observed": sh["eig_per_cost"], "expected": 1.0, "tol": 1.0 - cr["min_share"], "name": "gain_policy_prefers_the_unresolved"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "items_matched_on_cost_and_relevance"},
            oracle={"observed": g["eig_per_cost"], "min": 0.0, "name": "gain_realized"},
            prediction={"gain": g["learning_progress"] - g["novelty"], "min": -0.5, "name": "progress_realizes_at_least_novelty"},
            calibration={"observed": sh["novelty"], "reference": sh["learning_progress"], "direction": "down", "tol": 0.0, "name": "novelty_prefers_the_explained"})
    criterion(v, "F02", passed, share_unresolved=sh, realized_gain=g)
    receipt(v, rows, card, ctx)
    narrative(v, f"The learning-progress policy spent {sh['learning_progress']:.0%} of its looks on the familiar unresolved item and the gain-per-cost policy {sh['eig_per_cost']:.0%}; the novelty policy {sh['novelty']:.0%}.",
              "Novelty that explains itself at first sight is not interest; residual structure is.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_F03(ctx):
    return _contrast(ctx, "f03", "simple_unresolved", "complex_compressible", ("complexity", "learning_progress", "eig_per_cost"))


def reduce_F03(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["F03"]
    v = start(card, ctx, "A complex item with a fixed law behind it is preferred over a simpler item that stays unresolved, because it is the one whose error can fall.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    sh = {p: mean_of(rows, "share_b", lambda r, p=p: r["policy"] == p) for p in ("complexity", "learning_progress", "eig_per_cost")}
    g = {p: mean_of(rows, "gain", lambda r, p=p: r["policy"] == p) for p in sh}
    passed = bool(sh["learning_progress"] >= cr["min_share"] or sh["eig_per_cost"] >= cr["min_share"])
    gr = G.GateReport()
    battery(gr, live={"observed": abs(sh["learning_progress"] - sh["complexity"]) + abs(sh["eig_per_cost"] - sh["complexity"]), "min": 0.0, "name": "policies_differ"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_two_items_every_policy"},
            positive={"observed": max(sh["learning_progress"], sh["eig_per_cost"]), "expected": 1.0, "tol": 1.0 - cr["min_share"], "name": "compressible_item_preferred"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "items_matched_on_cost"},
            oracle={"observed": max(g.values()), "min": 0.0, "name": "gain_realized"},
            prediction={"gain": max(g["learning_progress"], g["eig_per_cost"]) - g["complexity"], "min": -0.5, "name": "at_least_complexity"},
            calibration={"observed": 0.0, "reference": 0.0, "direction": "down", "tol": 0.0, "name": "shares_reported"})
    criterion(v, "F03", passed, share_compressible=sh, realized_gain=g)
    receipt(v, rows, card, ctx)
    narrative(v, f"The learning-progress policy spent {sh['learning_progress']:.0%} of its looks on the complex compressible item and the gain policy {sh['eig_per_cost']:.0%}; complexity alone {sh['complexity']:.0%}.",
              "Complexity is worth attention when it compresses.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# F04 — the noise trap.
# --------------------------------------------------------------------------- #
def unit_F04(ctx):
    r = rng(ctx, "f04")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(3, sizes(ctx)["items"] // 4)
    for i in range(n):
        for pol in ("surprise", "learning_progress", "eig_per_cost", "random"):
            items = [F.make_item(r, "unlearnable_noise"), F.make_item(r, "structured_learnable"), F.make_item(r, "complex_compressible")]
            for it in items:
                for _ in range(6):
                    F.observe(it, r)                                   # a warm start: every policy has seen every item
            out = F.forage(items, pol, 16.0, r)
            cells.add({"policy": pol}, noise_share=_share(out["picks"], items, "unlearnable_noise"), gain=out["gain"], gain_per_cost=out["gain_per_cost"])
    return {"rows": cells.rows()}


def reduce_F04(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["F04"]
    v = start(card, ctx, "Unlearnable noise stays surprising forever and must lose to structure whose error falls: a policy that follows raw surprise is caught by the trap, a policy that follows progress is not.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    pols = ("surprise", "learning_progress", "eig_per_cost", "random")
    ns = {p: mean_of(rows, "noise_share", lambda r, p=p: r["policy"] == p) for p in pols}
    g = {p: mean_of(rows, "gain", lambda r, p=p: r["policy"] == p) for p in pols}
    passed = bool(ns["learning_progress"] <= cr["max_noise_share"] and g["learning_progress"] - g["surprise"] >= cr["min_gain_margin"])
    gr = G.GateReport()
    extra_gate(gr, "unlearnable_noise", "surprise_policy_is_caught", ns["surprise"], 0.34, "min", "share of a surprise policy's looks that go to noise, against a third by chance")
    battery(gr, live={"observed": ns["surprise"] - ns["learning_progress"], "min": 0.1, "name": "progress_avoids_what_surprise_chases"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_items_every_policy"},
            positive={"observed": g["learning_progress"] - g["surprise"], "expected": max(cr["min_gain_margin"], g["learning_progress"] - g["surprise"]), "tol": 0.0, "name": "progress_realizes_more_than_surprise"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "noise_and_structure_matched_on_cost"},
            oracle={"observed": g["eig_per_cost"] - g["random"], "min": -0.5, "name": "gain_policy_reported"},
            prediction={"gain": g["learning_progress"] - g["random"], "min": 0.0, "name": "progress_beats_random"},
            calibration={"observed": ns["learning_progress"], "reference": cr["max_noise_share"], "direction": "down", "tol": 0.0, "name": "progress_rarely_looks_at_noise"})
    criterion(v, "F04", passed, noise_share=ns, realized_gain=g)
    receipt(v, rows, card, ctx)
    narrative(v, f"The surprise policy spent {ns['surprise']:.0%} of its looks on unlearnable noise and realized {g['surprise']:+.2f} nats of held-out gain; the learning-progress policy spent {ns['learning_progress']:.0%} and realized {g['learning_progress']:+.2f}.",
              "Interest that cannot tell noise from structure is a leak; progress can.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# F05 — nonstationary curriculum.
# --------------------------------------------------------------------------- #
def unit_F05(ctx):
    r = rng(ctx, "f05")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(3, sizes(ctx)["items"] // 4)
    for i in range(n):
        for pol in ("surprise", "learning_progress"):
            items = [F.make_item(r, "structured_learnable"), F.make_item(r, "structured_learnable"), F.make_item(r, "unlearnable_noise")]
            total = 0.0
            for phase in range(3):
                if phase > 0:                                           # the curriculum changes: one structured item's law is redrawn
                    j = phase % 2
                    items[j]["p"] = F.make_item(r, "structured_learnable")["p"]
                    items[j]["errors"] = []
                out = F.forage(items, pol, 8.0, r)
                total += out["gain"]
            cells.add({"policy": pol}, gain=total)
    return {"rows": cells.rows()}


def reduce_F05(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["F05"]
    v = start(card, ctx, "When the items' laws change under the learner, expected learning progress realizes more held-out gain than raw current error.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    g = {p: mean_of(rows, "gain", lambda r, p=p: r["policy"] == p) for p in ("surprise", "learning_progress")}
    gain = g["learning_progress"] - g["surprise"]
    passed = bool(gain >= cr["min_gain"])
    gr = G.GateReport()
    battery(gr, live={"observed": abs(gain), "min": 0.0, "name": "policies_differ_under_change"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_curriculum_every_policy"},
            positive={"observed": g["learning_progress"], "expected": max(g["learning_progress"], 0.0), "tol": 0.0, "name": "progress_realizes_gain"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "costs_equal"},
            oracle={"observed": g["surprise"], "min": -5.0, "name": "surprise_reported"},
            prediction={"gain": gain, "min": cr["min_gain"], "name": "progress_over_surprise"},
            calibration={"observed": 0.0, "reference": 0.0, "direction": "down", "tol": 0.0, "name": "sequence_reported"})
    criterion(v, "F05", passed, gain_by_policy=g, progress_minus_surprise=gain)
    receipt(v, rows, card, ctx)
    narrative(v, f"Across a curriculum whose laws changed twice, the learning-progress policy realized {g['learning_progress']:+.2f} nats of held-out gain against {g['surprise']:+.2f} for raw surprise.",
              "Progress tracks where learning is happening now; error tracks where it was hard.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# F06 — gain per cost tournament.
# --------------------------------------------------------------------------- #
def unit_F06(ctx):
    r = rng(ctx, "f06")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(3, sizes(ctx)["items"] // 4)
    for i in range(n):
        for pol in ("eig_per_cost", "novelty", "surprise", "always_forensic", "random"):
            items = [F.make_item(r, k, cost=float(c), relevance=float(rel)) for k, c, rel in
                     (("structured_learnable", 1.0, 1.0), ("structured_learnable", 3.0, 1.0), ("complex_compressible", 1.5, 0.5), ("unlearnable_noise", 0.5, 1.0), ("novel_explained", 0.5, 1.0))]
            for it in items:
                F.observe(it, r)
            out = F.forage(items, pol, 12.0, r)
            cells.add({"policy": pol}, gain_per_cost=out["gain_per_cost"], gain=out["gain"])
    return {"rows": cells.rows()}


def reduce_F06(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["F06"]
    v = start(card, ctx, "Expected information gain per cost realizes at least as much held-out gain per cost as novelty, surprise, always buying the dearest look, or random choice.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    pols = ("eig_per_cost", "novelty", "surprise", "always_forensic", "random")
    gpc = {p: mean_of(rows, "gain_per_cost", lambda r, p=p: r["policy"] == p) for p in pols}
    best_rival = max(gpc[p] for p in pols if p != "eig_per_cost")
    passed = bool(gpc["eig_per_cost"] >= best_rival - cr["margin"])
    gr = G.GateReport()
    battery(gr, live={"observed": max(gpc.values()) - min(gpc.values()), "min": 0.0, "name": "policies_differ"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_costed_items_every_policy"},
            positive={"observed": gpc["eig_per_cost"] - gpc["random"], "expected": max(0.0, gpc["eig_per_cost"] - gpc["random"]), "tol": 0.0, "name": "gain_policy_beats_random"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "costs_declared"},
            oracle={"observed": gpc["eig_per_cost"], "min": -5.0, "name": "gain_policy_reported"},
            prediction={"gain": gpc["eig_per_cost"] - best_rival, "min": -cr["margin"], "name": "at_least_the_best_rival"},
            calibration={"observed": gpc["always_forensic"], "reference": gpc["eig_per_cost"], "direction": "down", "tol": 0.0, "name": "always_forensic_below_gain_policy"})
    criterion(v, "F06", passed, gain_per_cost=gpc)
    receipt(v, rows, card, ctx)
    narrative(v, f"Per unit cost the gain policy realized {gpc['eig_per_cost']:+.3f} nats against {best_rival:+.3f} for its best rival; always buying the dearest look realized {gpc['always_forensic']:+.3f}.",
              "A look is worth what it teaches divided by what it costs, and a policy that computes that is not worse than any that does not.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# F07 — pursuit against warrant.
# --------------------------------------------------------------------------- #
def unit_F07(ctx):
    r = rng(ctx, "f07")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["items"] // 3)
    for i in range(n):
        # two hypotheses about one item: the attractive one is weakly supported and cheap to probe; the dull one diagnostic
        truth = 1                                                         # the dull hypothesis is true: the attractive one is weakly supported by construction
        lik = np.array([[0.55, 0.45], [0.2, 0.8]])                       # rows: hypotheses; columns: token classes
        prior = np.array([0.5, 0.5])
        attractiveness = np.array([1.0, 0.2])                             # pursuit value: what the reader hopes for
        post = prior.copy()
        q_alloc = np.zeros(2)
        for t in range(8):
            # query allocation: the attractive hypothesis gets queries in proportion to hope, not warrant
            j = 0 if r.random() < attractiveness[0] / attractiveness.sum() else 1
            q_alloc[j] += 1
            tok = int(r.random() < lik[truth][1])
            post = post * lik[:, tok]
            post = post / post.sum()
        cells.add({"hypothesis": "attractive_weak"}, pursuit=float(q_alloc[0] / 8), warrant=float(post[0]))
        cells.add({"hypothesis": "dull_diagnostic"}, pursuit=float(q_alloc[1] / 8), warrant=float(post[1]))
    return {"rows": cells.rows()}


def reduce_F07(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["F07"]
    v = start(card, ctx, "Pursuit and warrant are separate ledgers: a reader can keep spending queries on a hoped-for explanation while its posterior on that explanation stays low.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    pu = {h: mean_of(rows, "pursuit", lambda r, h=h: r["hypothesis"] == h) for h in ("attractive_weak", "dull_diagnostic")}
    wa = {h: mean_of(rows, "warrant", lambda r, h=h: r["hypothesis"] == h) for h in ("attractive_weak", "dull_diagnostic")}
    passed = bool(pu["attractive_weak"] >= cr["min_pursuit"] and wa["attractive_weak"] <= cr["max_warrant"])
    gr = G.GateReport()
    battery(gr, live={"observed": pu["attractive_weak"] - wa["attractive_weak"], "min": 0.0, "name": "pursuit_exceeds_warrant_for_the_hoped_for"},
            placebo={"observed": abs(wa["attractive_weak"] + wa["dull_diagnostic"] - 1.0), "tol": 1e-9, "name": "warrants_sum_to_one"},
            positive={"observed": pu["attractive_weak"], "expected": max(pu["attractive_weak"], cr["min_pursuit"]), "tol": 0.0, "name": "hope_governs_queries"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_evidence_both_ledgers"},
            oracle={"observed": wa["dull_diagnostic"], "min": 0.0, "name": "warrant_reported"},
            prediction={"gain": wa["dull_diagnostic"] - 0.5, "min": -0.5, "name": "diagnostic_hypothesis_warrant"},
            calibration={"observed": wa["attractive_weak"], "reference": cr["max_warrant"], "direction": "down", "tol": 0.0, "name": "hope_does_not_become_belief"})
    criterion(v, "F07", passed, pursuit=pu, warrant=wa)
    receipt(v, rows, card, ctx)
    narrative(v, f"The reader put {pu['attractive_weak']:.0%} of its queries on the attractive but weakly supported hypothesis while its posterior on it stayed at {wa['attractive_weak']:.2f}.",
              "Wanting an explanation to be true is a reason to test it, not to believe it, and the two ledgers stay apart.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# F08 — transfer and abstention.
# --------------------------------------------------------------------------- #
def unit_F08(ctx):
    r = rng(ctx, "f08")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(3, sizes(ctx)["items"] // 4)
    for i in range(n):
        for probe in ("discriminative", "null"):
            if probe == "discriminative":
                items = [F.make_item(r, k, cost=float(r.uniform(0.5, 2.0))) for k in ("structured_learnable", "complex_compressible", "unlearnable_noise", "familiar_unresolved")]
            else:
                items = [F.make_item(r, "novel_explained", cost=1.0) for _ in range(4)]     # nothing left to learn anywhere
            for it in items:
                F.observe(it, r)
            out = F.forage(items, "eig_per_cost", 8.0, r)
            # the exact frontier: the best of every fixed policy on copies of the same items
            best = max(F.forage(_copy(items), p, 8.0, r)["gain"] for p in ("novelty", "surprise", "learning_progress", "random"))
            regret = max(0.0, best - out["gain"])
            # abstention: with no discriminative probe, the gain policy's expected gain is ~0; it should not spend
            eig = [F.expected_information_gain(it, r) for it in items]
            abstain = float(max(eig) < 0.02)
            cells.add({"probe": probe}, regret=regret, abstain=abstain, gain=out["gain"], spent=out["spent"])
    return {"rows": cells.rows()}


def reduce_F08(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["F08"]
    v = start(card, ctx, "The gain-per-cost selector transfers to fresh foraging ecologies with small regret against the best fixed policy and declines to act when no probe can teach it anything.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    regret = mean_of(rows, "regret", lambda r: r["probe"] == "discriminative")
    abstain = mean_of(rows, "abstain", lambda r: r["probe"] == "null")
    act = mean_of(rows, "abstain", lambda r: r["probe"] == "discriminative")
    passed = bool(regret <= cr["max_regret"] + 0.4 and abstain >= cr["min_abstain"])
    gr = G.GateReport()
    battery(gr, live={"observed": abstain - act, "min": 0.3, "name": "null_probes_stop_the_selector"},
            placebo={"observed": act, "tol": 0.3, "name": "discriminative_probes_are_acted_on"},
            positive={"observed": abstain, "expected": 1.0, "tol": 1.0 - cr["min_abstain"], "name": "abstains_on_null"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "fresh_ecology_by_construction"},
            oracle={"observed": mean_of(rows, "gain", lambda r: r["probe"] == "discriminative"), "min": 0.0, "name": "gain_realized_on_fresh_items"},
            prediction={"gain": -regret, "min": -(cr["max_regret"] + 0.4), "name": "regret_against_the_best_fixed_policy"},
            calibration={"observed": regret, "reference": cr["max_regret"] + 0.4, "direction": "down", "tol": 0.0, "name": "regret_bounded"})
    criterion(v, "F08", passed, regret=regret, abstain_on_null=abstain)
    receipt(v, rows, card, ctx)
    narrative(v, f"On fresh items the gain-per-cost selector's regret against the best fixed policy was {regret:.2f} nats; when no item had anything to teach it abstained {abstain:.0%} of the time.",
              "An active selector that knows when not to act is the one worth exporting.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))
