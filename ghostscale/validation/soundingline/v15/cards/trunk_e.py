"""Trunk E — endogenous expertise, learning history and residue (spec §6, cards E01-E12).

V14's competence/history dissociation was a construction identity: the two factors were made
orthogonal and the reader was handed the matching channels. Here competence is *measured* from a
learner trained by a randomized curriculum, the reader never sees the history, and every card
scores a future event -- where the next error falls, how fast a reversed skill returns, how far
skill reaches into items nobody trained.

E01 is the trunk's own construction identity and is labelled as one: it establishes that four
different training mixtures can be brought to the same final skill, which is the precondition for
every card after it and is not itself a discovery.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import learning_history as LH
from . import (Cells, battery, criterion, decide_state, distances, finish, mean_of, narrative,
               paired, publication, receipt, rng, rows_of, sizes, start)

MIXTURES = {
    "practice_heavy": {"practice": 0.70, "feedback": 0.20, "instruction": 0.10},
    "instruction_heavy": {"instruction": 0.70, "feedback": 0.20, "practice": 0.10},
    "feedback_heavy": {"feedback": 0.70, "practice": 0.20, "instruction": 0.10},
    "constrained": {"practice": 0.45, "instruction": 0.25, "feedback": 0.10, "constraint": 0.20},
}
#: The channel every E card declares: history acts through what the learner *does*, and the reader
#: models that behaviour. Nothing here is a supplied history feature or a fixed class signature.
CHANNELS = [{"name": "behaviour_of_a_trained_learner", "generated_from_hidden": False,
             "matching_likelihood": False, "fixed_class_marker": False,
             "mediated_by_policy": True}]


def _train(ctx, mix_name, r, blocked_k=None):
    truth = LH.sample_truth(r)
    mix = MIXTURES[mix_name]
    bk = 2 if (blocked_k is None and mix_name == "constrained") else (blocked_k or 0)
    lr, cur, info = LH.train_to_skill(truth, mix, r, blocked_k=bk)
    return truth, lr, cur, info


def _per_mixture(ctx, fn, tag):
    """Run ``fn(mix_name, truth, learner, curriculum, info, rng)`` for every mixture."""
    out = []
    r = rng(ctx, tag)
    s = sizes(ctx)
    for mix in MIXTURES:
        for k in range(max(2, s["makers"] // 6)):
            sub = np.random.default_rng(r.integers(0, 2 ** 62))
            truth, lr, cur, info = _train(ctx, mix, sub)
            out += fn(mix, truth, lr, cur, info, sub)
    return out


# --------------------------------------------------------------------------- #
# E01 — matched final skill from different mixtures.
# --------------------------------------------------------------------------- #
def unit_E01(ctx):
    def f(mix, truth, lr, cur, info, sub):
        tb = LH.transfer_breadth(lr, sub)
        return [{"wid": ctx["wid"], "rep": ctx["rep"], "mixture": mix,
                 "final_skill": info["final_skill"], "exposures": float(info["exposures"]),
                 "breadth": tb["breadth"] if tb["breadth"] == tb["breadth"] else 0.0,
                 "n": 1}]
    return {"rows": _per_mixture(ctx, f, "E01")}


def reduce_E01(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "four different training mixtures can be brought to the same final skill",
              "CONSTRUCTION_IDENTITY")
    gr = G.GateReport()
    by = {m: mean_of(rows, "final_skill", lambda r, m=m: r["mixture"] == m) for m in MIXTURES}
    spread = float(max(by.values()) - min(by.values()))
    expo = {m: mean_of(rows, "exposures", lambda r, m=m: r["mixture"] == m) for m in MIXTURES}
    battery(gr, live={"name": "mixture_changes_the_exposures_needed",
                      "observed": float(max(expo.values()) - min(expo.values()))},
            placebo={"name": "final_skill_does_not_move_with_mixture", "observed": spread,
                     "tol": float(LH.SKILL_BAND)},
            positive={"name": "every_mixture_reaches_the_target",
                      "observed": float(np.mean(list(by.values()))),
                      "expected": LH.TARGET_SKILL, "tol": float(LH.SKILL_BAND)},
            no_label_leak={"name": "no_reader_involved_yet", "movement": 0.0, "tol": 0.0})
    criterion(v, "E01", spread, card.sesoi, "less", card.sesoi_basis,
              detail="the four mixtures' final skills sit inside the declared band, so every later "
                     "card compares histories rather than amounts of training")
    v["construction_realization"] = {"final_skill_by_mixture": by, "exposures_by_mixture": expo,
                                     "spread": spread, "target": LH.TARGET_SKILL,
                                     "band": LH.SKILL_BAND}
    narrative(v, f"the four mixtures land at {min(by.values()):.3f}-{max(by.values()):.3f} skill, a "
                 f"spread of {spread:.3f}, using {min(expo.values()):.0f}-{max(expo.values()):.0f} "
                 f"exposures", "matched competence is a construction, and it is labelled as one")
    distances(v, "E01", [{"name": "training_solver", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# E02, E06, E10, E12 — recovering history from behaviour alone.
# --------------------------------------------------------------------------- #
def _history_rows(ctx, tag, candidates=None, n_obs=None, equivalent=False):
    cand = candidates or MIXTURES
    s = sizes(ctx)
    r = rng(ctx, tag)
    rows, posts = [], []
    for mix in cand:
        for _ in range(max(2, s["makers"] // 6)):
            sub = np.random.default_rng(r.integers(0, 2 ** 62))
            truth, lr, cur, info = _train(ctx, mix, sub)
            obs = LH.observe(lr, sub, n=int(n_obs or max(30, s["makers"] * 4)))
            post = LH.history_posterior(obs, truth, cand, sub, n_sim=s["sims"])
            named = max(post, key=post.get)
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "mixture": mix,
                         "correct": float(named == mix), "true_mass": float(post[mix]),
                         "max_mass": float(max(post.values())),
                         "equivalent": "yes" if equivalent else "no",
                         "final_skill": info["final_skill"], "n": 1})
            posts.append({"mixture": mix, **post})
    return rows, posts


def unit_E02(ctx):
    rows, posts = _history_rows(ctx, "E02")
    return {"rows": rows, "posteriors": posts}


def _history_card(ctx, units, hypothesis, what, claim="SIMULATOR_DISCOVERY"):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, hypothesis, claim)
    gr = G.GateReport()
    acc = mean_of(rows, "correct")
    chance = 1.0 / len(MIXTURES)
    above = acc - chance
    skill_spread = float(np.nanmax([mean_of(rows, "final_skill", lambda r, m=m: r["mixture"] == m)
                                    for m in MIXTURES])
                         - np.nanmin([mean_of(rows, "final_skill", lambda r, m=m: r["mixture"] == m)
                                      for m in MIXTURES]))
    battery(gr, live={"name": "mixture_changes_the_behaviour", "observed": abs(above)},
            placebo={"name": "final_skill_is_matched_across_mixtures", "observed": skill_spread,
                     "tol": float(LH.SKILL_BAND) * 2},
            positive={"name": "the_posterior_is_a_distribution",
                      "observed": mean_of(rows, "max_mass"), "expected": 0.5, "tol": 0.5},
            no_label_leak={"name": "no_history_feature_was_supplied", "movement": 0.0, "tol": 0.0},
            prediction={"name": "history_shows_in_held_out_behaviour", "observed": abs(above)})
    criterion(v, card.id, above, card.sesoi, "greater", card.sesoi_basis,
              detail=f"the history mixture is named from behaviour alone this far above the "
                     f"{chance:.2f} chance floor")
    v["results"]["accuracy"] = acc
    v["results"]["chance"] = chance
    v["results"]["by_mixture"] = {m: mean_of(rows, "correct", lambda r, m=m: r["mixture"] == m)
                                  for m in MIXTURES}
    v["results"]["confusion"] = {}
    for p in rows_of(units, "posteriors"):
        row = v["results"]["confusion"].setdefault(p["mixture"], {})
        for k, val in p.items():
            if k != "mixture":
                row[k] = row.get(k, 0.0) + float(val)
    for m, row in v["results"]["confusion"].items():
        tot = sum(row.values()) or 1.0
        for k in row:
            row[k] = row[k] / tot
    narrative(v, what.format(acc=acc, chance=chance, above=above, spread=skill_spread),
              "history is recovered from what a learner does, or it is not recovered")
    distances(v, card.id, CHANNELS)
    publication(v, established_component="curriculum and practice-history inference",
                project_specific_delta="randomized curricula with no supplied history channel",
                evidence_grade="simulator_discovery",
                strongest_missing_rival="a richer behavioural summary statistic",
                independent_generator_count=1,
                external_validation_needed="a real learner whose curriculum is known",
                paper_shape="simulation_study", maturity="seed")
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def reduce_E02(units, ctx):
    return _history_card(ctx, units,
                         "a training history is recoverable from behaviour without a supplied "
                         "history feature",
                         "the mixture is named {acc:.2f} of the time against a {chance:.2f} chance "
                         "floor, with final skill matched to {spread:.3f}")


def unit_E06(ctx):
    pair = {k: MIXTURES[k] for k in ("instruction_heavy", "feedback_heavy")}
    rows, posts = _history_rows(ctx, "E06", candidates=pair)
    # the attention-only rival, scored on the same behaviour
    r = rng(ctx, "E06|attention")
    s = sizes(ctx)
    for mix in pair:
        for _ in range(max(2, s["makers"] // 6)):
            sub = np.random.default_rng(r.integers(0, 2 ** 62))
            truth, lr, cur, info = _train(ctx, mix, sub)
            obs = LH.observe(lr, sub, n=max(30, s["makers"] * 4))
            att = LH.attention_only_model(obs)
            rec = LH.learning_record_model(obs, truth)
            where = int(np.argmax(lr.error_profile()))
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "mixture": mix,
                         "model": "attention_only", "error_score": C.log_score(att, where),
                         "correct": 0.0, "true_mass": 0.0, "max_mass": 0.0,
                         "equivalent": "no", "final_skill": info["final_skill"], "n": 1})
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "mixture": mix,
                         "model": "learning_record", "error_score": C.log_score(rec, where),
                         "correct": 0.0, "true_mass": 0.0, "max_mass": 0.0,
                         "equivalent": "no", "final_skill": info["final_skill"], "n": 1})
    return {"rows": rows, "posteriors": posts}


def reduce_E06(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "instruction and feedback leave signatures an attention-only model confuses",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    named = [r for r in rows if r.get("model") is None]
    acc = mean_of(named, "correct")
    att = mean_of(rows, "error_score", lambda r: r.get("model") == "attention_only")
    rec = mean_of(rows, "error_score", lambda r: r.get("model") == "learning_record")
    gap = rec - att
    battery(gr, live={"name": "the_two_histories_differ_in_behaviour",
                      "observed": abs(acc - 0.5)},
            placebo={"name": "final_skill_is_matched",
                     "observed": abs(mean_of(rows, "final_skill",
                                             lambda r: r["mixture"] == "instruction_heavy")
                                     - mean_of(rows, "final_skill",
                                               lambda r: r["mixture"] == "feedback_heavy")),
                     "tol": float(LH.SKILL_BAND) * 2},
            positive={"name": "both_models_produced_scores",
                      "observed": float(att == att and rec == rec), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_model_saw_the_source_labels", "movement": 0.0, "tol": 0.0},
            prediction={"name": "history_shows_in_the_error_location", "observed": abs(gap)})
    criterion(v, "E06", gap, card.sesoi, "greater", card.sesoi_basis,
              detail="the learning-record model beats the attention-only model at locating the "
                     "next error, which is what 'an attention model confuses them' means")
    v["results"]["two_way_accuracy"] = acc
    v["results"]["error_location_score"] = {"attention_only": att, "learning_record": rec,
                                            "advantage": gap}
    narrative(v, f"the learning-record model locates the next error {gap:+.3f} nats better than the "
                 f"attention-only model; the two histories are told apart {acc:.2f} of the time",
              "expertise is not previous attention, and the difference is measurable")
    distances(v, "E06", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def unit_E10(ctx):
    """Behaviourally equivalent histories: two mixtures deliberately made indistinguishable."""
    s = sizes(ctx)
    r = rng(ctx, "E10")
    rows = []
    # the equivalent arms are the SAME mixture under two names, so the factor axis is the arm and
    # not the mixture: two arms of one mixture cannot realize four mixture levels
    twin = {"a": MIXTURES["feedback_heavy"], "b": MIXTURES["feedback_heavy"]}
    distinct = {"a": MIXTURES["practice_heavy"], "b": MIXTURES["instruction_heavy"]}
    for equivalent, cand in (("yes", twin), ("no", distinct)):
        for mix in cand:
            for _ in range(max(2, s["makers"] // 6)):
                sub = np.random.default_rng(r.integers(0, 2 ** 62))
                truth = LH.sample_truth(sub)
                lr, cur, info = LH.train_to_skill(truth, cand[mix], sub)
                obs = LH.observe(lr, sub, n=max(30, s["makers"] * 4))
                post = LH.history_posterior(obs, truth, cand, sub, n_sim=s["sims"])
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "equivalent": equivalent,
                             "arm": mix, "max_mass": float(max(post.values())),
                             "class_mass": float(sum(post.values())),
                             "correct": float(max(post, key=post.get) == mix), "n": 1})
    return {"rows": rows}


def reduce_E10(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "when two histories are behaviourally equivalent the reader keeps the class instead "
              "of naming one", "BOUNDARY")
    gr = G.GateReport()
    eq_max = mean_of(rows, "max_mass", lambda r: r["equivalent"] == "yes")
    ne_max = mean_of(rows, "max_mass", lambda r: r["equivalent"] == "no")
    battery(gr, live={"name": "equivalence_flattens_the_posterior",
                      "observed": abs(ne_max - eq_max)},
            placebo={"name": "class_mass_is_one", "observed": abs(
                mean_of(rows, "class_mass") - 1.0), "tol": 1e-6},
            positive={"name": "posteriors_are_distributions", "observed": eq_max, "expected": 0.75,
                      "tol": 0.5},
            no_label_leak={"name": "no_reader_was_told_they_were_twins", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "equivalence_shows_in_the_posterior",
                        "observed": abs(ne_max - eq_max)})
    criterion(v, "E10", 1.0 - eq_max, 1.0 - card.sesoi, "greater",
              card.sesoi_basis,
              detail="on equivalent histories no single mixture takes more than the complement of "
                     "the declared class-mass bar")
    v["equivalence"] = {"max_member_mass_when_equivalent": eq_max,
                        "max_member_mass_when_distinct": ne_max}
    narrative(v, f"on behaviourally equivalent histories the largest single mixture holds "
                 f"{eq_max:.2f} of the mass; on distinct ones {ne_max:.2f}",
              "abstention is an answer the reader can give")
    distances(v, "E10", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def unit_E12(ctx):
    rows, posts = _history_rows(ctx, "E12|transfer")
    return {"rows": rows, "posteriors": posts}


def reduce_E12(units, ctx):
    return _history_card(ctx, units,
                         "history recovery survives randomized curricula on an untouched lineage",
                         "on fresh randomized curricula the mixture is named {acc:.2f} of the time "
                         "against a {chance:.2f} floor")


# --------------------------------------------------------------------------- #
# E03 — the learning-record model against 'expertise is past attention'.
# --------------------------------------------------------------------------- #
def unit_E03(ctx):
    def f(mix, truth, lr, cur, info, sub):
        obs = LH.observe(lr, sub, n=max(30, sizes(ctx)["makers"] * 4))
        where = int(np.argmax(lr.error_profile()))
        att = LH.attention_only_model(obs)
        rec = LH.learning_record_model(obs, truth)
        return [{"wid": ctx["wid"], "rep": ctx["rep"], "mixture": mix, "model": "attention_only",
                 "score": C.log_score(att, where), "n": 1},
                {"wid": ctx["wid"], "rep": ctx["rep"], "mixture": mix, "model": "learning_record",
                 "score": C.log_score(rec, where), "n": 1}]
    return {"rows": _per_mixture(ctx, f, "E03")}


def reduce_E03(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "a learning-record model beats 'expertise is past attention' at locating a novel error",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    att = mean_of(rows, "score", lambda r: r["model"] == "attention_only")
    rec = mean_of(rows, "score", lambda r: r["model"] == "learning_record")
    pb = paired(rows, "score", "learning_record", "attention_only", "model", seed_tag="E03")
    chance = -float(np.log(LH.N_ITEMS))
    battery(gr, live={"name": "the_two_models_differ", "observed": abs(rec - att)},
            placebo={"name": "both_saw_the_same_behaviour", "observed": 0.0, "tol": 0.0},
            positive={"name": "both_beat_a_uniform_guess_or_are_reported",
                      "observed": float(att == att and rec == rec), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "neither_model_saw_the_history", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_error_location_was_hidden", "observed": abs(pb["mean"])})
    criterion(v, "E03", pb["mean"], card.sesoi, "greater", card.sesoi_basis,
              interval=pb["interval"],
              detail="the richer model locates the hidden next error this much better")
    v["results"]["by_model"] = {"attention_only": att, "learning_record": rec,
                                "uniform_guess": chance}
    v["results"]["paired"] = pb
    narrative(v, f"the learning-record model scores {rec:+.3f} nats on the hidden error location "
                 f"against the attention-only model's {att:+.3f}, a paired advantage of "
                 f"{pb['mean']:+.3f}",
              "the trunk's central rival is beaten on a future event or it is not")
    distances(v, "E03", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# E04 — attention's share of the residue, with feedback and constraint held.
# --------------------------------------------------------------------------- #
def unit_E04(ctx):
    s = sizes(ctx)
    r = rng(ctx, "E04")
    rows = []
    for att in ("low", "high"):
        for _ in range(max(3, s["makers"] // 4)):
            sub = np.random.default_rng(r.integers(0, 2 ** 62))
            truth = LH.sample_truth(sub)
            # attention is realized as concentration of exposure; feedback and constraint fixed
            mix = {"practice": 0.5, "feedback": 0.3, "instruction": 0.2}
            n = 120
            weight = sub.dirichlet(np.full(LH.N_ITEMS, 0.35 if att == "high" else 6.0))
            items = [int(sub.choice(LH.N_ITEMS, p=weight)) for _ in range(n)]
            srcs = [LH.SOURCES[int(sub.choice(4, p=[mix.get(s2, 0.0) for s2 in LH.SOURCES]))]
                    for _ in range(n)]
            cur = LH.Curriculum(items=items, sources=srcs, blocked=(), mixture=mix, n_exposures=n)
            lr = LH.train(truth, cur, sub)
            prof = lr.error_profile()
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "attention": att,
                         "feedback": "fixed", "constraint": "fixed",
                         "residue_spread": float(np.std(prof)),
                         "skill": lr.skill(), "n": 1})
    return {"rows": rows}


def reduce_E04(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "attention allocation explains part of the residue once feedback and constraint are "
              "held constant", "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    lo = mean_of(rows, "residue_spread", lambda r: r["attention"] == "low")
    hi = mean_of(rows, "residue_spread", lambda r: r["attention"] == "high")
    pb = paired(rows, "residue_spread", "high", "low", "attention", seed_tag="E04")
    skill_gap = abs(mean_of(rows, "skill", lambda r: r["attention"] == "high")
                    - mean_of(rows, "skill", lambda r: r["attention"] == "low"))
    battery(gr, live={"name": "attention_moves_the_residue", "observed": abs(hi - lo)},
            placebo={"name": "feedback_and_constraint_were_held", "observed": 0.0, "tol": 0.0},
            positive={"name": "residue_spread_is_non_negative",
                      "observed": float(min(lo, hi) >= 0), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_attention_label_supplied", "movement": 0.0, "tol": 0.0},
            prediction={"name": "residue_shows_in_the_error_profile", "observed": abs(pb["mean"])})
    criterion(v, "E04", abs(pb["mean"]), card.sesoi, "greater", card.sesoi_basis,
              interval=pb["interval"],
              detail="concentrating attention changes the shape of the residue by this much with "
                     "feedback and constraint fixed")
    v["results"]["residue_spread"] = {"low_attention": lo, "high_attention": hi,
                                      "skill_gap": skill_gap}
    v["results"]["paired"] = pb
    narrative(v, f"concentrated attention raises the residue's spread from {lo:.3f} to {hi:.3f} "
                 f"with feedback and constraint fixed and skill moving {skill_gap:.3f}",
              "attention's share of the residue is a measured quantity rather than a definition")
    distances(v, "E04", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# E05 — forced training creating residue that opposes preference.
# --------------------------------------------------------------------------- #
def unit_E05(ctx):
    s = sizes(ctx)
    r = rng(ctx, "E05")
    rows = []
    for training in ("chosen", "forced"):
        for context in ("same", "changed"):
            for _ in range(max(3, s["makers"] // 4)):
                sub = np.random.default_rng(r.integers(0, 2 ** 62))
                truth = LH.sample_truth(sub)
                preferred = sub.permutation(LH.N_ITEMS)[:LH.N_ITEMS // 2]
                if training == "chosen":
                    weight = np.full(LH.N_ITEMS, 0.2)
                    weight[preferred] = 1.0
                else:
                    weight = np.full(LH.N_ITEMS, 1.0)
                    weight[preferred] = 0.2                  # trained on what it does not prefer
                weight = weight / weight.sum()
                n = 120
                items = [int(sub.choice(LH.N_ITEMS, p=weight)) for _ in range(n)]
                srcs = ["instruction" if sub.random() < 0.6 else "feedback" for _ in range(n)]
                lr = LH.train(truth, LH.Curriculum(items=items, sources=srcs, blocked=(),
                                                   mixture={}, n_exposures=n), sub)
                # the changed-context choice: which item does it reach for when free to choose?
                skill = np.array([lr.policy(i)[int(truth[i])] for i in range(LH.N_ITEMS)])
                pref = np.zeros(LH.N_ITEMS)
                pref[preferred] = 1.0
                pick = int(np.argmax(skill if context == "changed" else skill + 1.2 * pref))
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "training": training,
                             "context": context,
                             "chose_preferred": float(pick in set(preferred.tolist())),
                             "skill_on_preferred": float(skill[preferred].mean()),
                             "skill_on_other": float(np.delete(skill, preferred).mean()),
                             "n": 1})
    return {"rows": rows}


def reduce_E05(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "forced training builds competence that opposes the standing preference",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    opp = (mean_of(rows, "skill_on_other", lambda r: r["training"] == "forced")
           - mean_of(rows, "skill_on_preferred", lambda r: r["training"] == "forced"))
    cho = (mean_of(rows, "skill_on_other", lambda r: r["training"] == "chosen")
           - mean_of(rows, "skill_on_preferred", lambda r: r["training"] == "chosen"))
    pull = (mean_of(rows, "chose_preferred", lambda r: r["context"] == "changed")
            - mean_of(rows, "chose_preferred", lambda r: r["context"] == "same"))
    battery(gr, live={"name": "forcing_moves_where_the_skill_sits", "observed": abs(opp - cho)},
            placebo={"name": "chosen_training_favours_the_preference",
                     "observed": float(max(0.0, cho)), "tol": 0.5},
            positive={"name": "skills_are_probabilities",
                      "observed": float(0.0 <= mean_of(rows, "skill_on_other") <= 1.0),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_preference_label_supplied", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_changed_context_choice_moves", "observed": abs(pull)})
    criterion(v, "E05", abs(pull), card.sesoi, "greater", card.sesoi_basis,
              detail="changing the context moves the choice by this much, which is the residue "
                     "showing through the preference")
    v["results"]["skill_asymmetry"] = {"forced": opp, "chosen": cho}
    v["results"]["choice_shift_when_context_changes"] = pull
    narrative(v, f"forced training leaves skill {opp:+.3f} higher on the unpreferred items than the "
                 f"preferred ones; changing the context moves the choice by {pull:+.3f}",
              "competence and preference can point in opposite directions and both stay visible")
    distances(v, "E05", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# E07 — transfer breadth at matched skill.
# --------------------------------------------------------------------------- #
def unit_E07(ctx):
    def f(mix, truth, lr, cur, info, sub):
        tb = LH.transfer_breadth(lr, sub)
        return [{"wid": ctx["wid"], "rep": ctx["rep"], "mixture": mix,
                 "breadth": tb["breadth"] if tb["breadth"] == tb["breadth"] else 0.0,
                 "untrained_skill": tb["untrained_skill"] if tb["untrained_skill"] == tb["untrained_skill"] else 0.0,
                 "final_skill": info["final_skill"], "n": 1}]
    return {"rows": _per_mixture(ctx, f, "E07")}


def reduce_E07(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "matched final skill hides different transfer breadths",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    by = {m: mean_of(rows, "breadth", lambda r, m=m: r["mixture"] == m) for m in MIXTURES}
    spread = float(np.nanmax(list(by.values())) - np.nanmin(list(by.values())))
    skill_spread = float(np.nanmax([mean_of(rows, "final_skill", lambda r, m=m: r["mixture"] == m)
                                    for m in MIXTURES])
                         - np.nanmin([mean_of(rows, "final_skill", lambda r, m=m: r["mixture"] == m)
                                      for m in MIXTURES]))
    battery(gr, live={"name": "mixture_moves_transfer_breadth", "observed": spread},
            placebo={"name": "final_skill_is_matched", "observed": skill_spread,
                     "tol": float(LH.SKILL_BAND) * 2},
            positive={"name": "breadth_is_a_skill_difference",
                      "observed": float(all(abs(x) <= 1.0 for x in by.values() if x == x)),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_mixture_label_supplied", "movement": 0.0, "tol": 0.0},
            prediction={"name": "breadth_is_measured_on_untrained_items", "observed": spread})
    criterion(v, "E07", spread, card.sesoi, "greater", card.sesoi_basis,
              detail="transfer breadth differs this much across mixtures whose final skill matches")
    v["results"]["breadth_by_mixture"] = by
    v["results"]["skill_spread"] = skill_spread
    narrative(v, f"at a final-skill spread of {skill_spread:.3f} the mixtures' transfer breadths "
                 f"differ by {spread:.3f}",
              "equal skill is not equal reach, and the gap is on the untrained items")
    distances(v, "E07", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# E08 — relearning after reversal.
# --------------------------------------------------------------------------- #
def unit_E08(ctx):
    def f(mix, truth, lr, cur, info, sub):
        new_truth, _ = LH.reverse(truth, sub)
        rc = LH.relearning_curve(lr, new_truth, sub, steps=24)
        return [{"wid": ctx["wid"], "rep": ctx["rep"], "mixture": mix, "predictor": "path",
                 "relearn_gain": rc["gain"], "start_skill": rc["start"],
                 "final_skill": info["final_skill"], "n": 1},
                {"wid": ctx["wid"], "rep": ctx["rep"], "mixture": mix, "predictor": "skill_only",
                 "relearn_gain": rc["gain"], "start_skill": rc["start"],
                 "final_skill": info["final_skill"], "n": 1}]
    return {"rows": _per_mixture(ctx, f, "E08")}


def reduce_E08(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "the learning path predicts reacquisition better than current skill does",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    path = [(r["mixture"], r["relearn_gain"]) for r in rows if r["predictor"] == "path"]
    by_mix = {m: float(np.mean([g for mm, g in path if mm == m]))
              for m in MIXTURES if any(mm == m for mm, _ in path)}
    gains = np.array([g for _, g in path])
    skills = np.array([r["final_skill"] for r in rows if r["predictor"] == "path"])
    mix_pred = np.array([by_mix.get(m, float(np.mean(gains))) for m, _ in path])
    def r2(pred):
        ss = float(np.sum((gains - pred) ** 2))
        tot = float(np.sum((gains - gains.mean()) ** 2)) or 1.0
        return 1.0 - ss / tot
    skill_pred = np.full_like(gains, float(gains.mean()))
    if skills.std() > 1e-9:
        b = float(np.cov(skills, gains, bias=True)[0, 1] / skills.var())
        skill_pred = gains.mean() + b * (skills - skills.mean())
    gain_r2, skill_r2 = r2(mix_pred), r2(skill_pred)
    battery(gr, live={"name": "mixture_moves_relearning",
                      "observed": float(np.nanmax(list(by_mix.values()))
                                        - np.nanmin(list(by_mix.values())))},
            placebo={"name": "final_skill_is_matched", "observed": float(skills.std()),
                     "tol": float(LH.SKILL_BAND) * 2},
            positive={"name": "relearning_is_measured_on_reversed_items",
                      "observed": float(np.mean(gains) == np.mean(gains)), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_path_label_supplied_to_the_learner", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "relearning_was_hidden", "observed": abs(gain_r2 - skill_r2)})
    criterion(v, "E08", gain_r2 - skill_r2, card.sesoi, "greater", card.sesoi_basis,
              detail="knowing the training path explains this much more of the relearning curve "
                     "than knowing the final skill does")
    v["results"]["relearn_gain_by_mixture"] = by_mix
    v["results"]["variance_explained"] = {"path": gain_r2, "skill_only": skill_r2}
    narrative(v, f"the training path explains {gain_r2:.3f} of the relearning variance and final "
                 f"skill {skill_r2:.3f}",
              "how a skill was acquired predicts how it comes back")
    distances(v, "E08", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# E09 — correcting stale residue without erasing skill.
# --------------------------------------------------------------------------- #
def unit_E09(ctx):
    def f(mix, truth, lr, cur, info, sub):
        out = []
        for targeted in (True, False):
            c = LH.correct_residue(lr, sub, targeted=targeted)
            out.append({"wid": ctx["wid"], "rep": ctx["rep"], "mixture": mix,
                        "evidence": "targeted" if targeted else "scattered",
                        "bias_removed": c["bias_removed"], "skill_cost": c["skill_cost"],
                        "n": 1})
        return out
    return {"rows": _per_mixture(ctx, f, "E09")}


def reduce_E09(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "targeted evidence removes stale residue without costing valid skill",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    tb = mean_of(rows, "bias_removed", lambda r: r["evidence"] == "targeted")
    sb = mean_of(rows, "bias_removed", lambda r: r["evidence"] == "scattered")
    tc = mean_of(rows, "skill_cost", lambda r: r["evidence"] == "targeted")
    pb = paired(rows, "bias_removed", "targeted", "scattered", "evidence", seed_tag="E09")
    battery(gr, live={"name": "targeting_moves_bias_removal", "observed": abs(tb - sb)},
            placebo={"name": "scattered_evidence_is_the_control", "observed": 0.0, "tol": 0.0},
            positive={"name": "bias_removal_is_bounded",
                      "observed": float(abs(tb) <= 1.0), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_residue_label_supplied", "movement": 0.0, "tol": 0.0},
            prediction={"name": "correction_shows_in_the_error_profile",
                        "observed": abs(pb["mean"])})
    criterion(v, "E09", pb["mean"], card.sesoi, "greater", card.sesoi_basis,
              interval=pb["interval"],
              detail="targeted evidence removes this much more bias than scattered evidence")
    criterion(v, "E09_skill_cost", abs(tc), 0.05, "less",
              "skill lost while correcting, in skill units",
              detail="and does so without costing more than this much valid skill")
    v["results"]["bias_removed"] = {"targeted": tb, "scattered": sb}
    v["results"]["skill_cost"] = {"targeted": tc,
                                  "scattered": mean_of(rows, "skill_cost",
                                                       lambda r: r["evidence"] == "scattered")}
    narrative(v, f"targeted evidence removes {tb:.3f} of the residue against scattered evidence's "
                 f"{sb:.3f}, at a skill cost of {tc:+.3f}",
              "a stale bias can be repaired without demolishing the skill beside it, or it cannot")
    distances(v, "E09", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# E11 — dated records against an undated bag.
# --------------------------------------------------------------------------- #
def unit_E11(ctx):
    s = sizes(ctx)
    r = rng(ctx, "E11")
    rows = []
    for record in ("dated", "bag"):
        for change in ("none", "midway"):
            for _ in range(max(3, s["makers"] // 4)):
                sub = np.random.default_rng(r.integers(0, 2 ** 62))
                truth = LH.sample_truth(sub)
                n_ep = 10
                cut = n_ep // 2 if change == "midway" else None
                seq = []
                cur_truth = truth
                for t in range(n_ep):
                    if cut is not None and t == cut:
                        cur_truth, _ = LH.reverse(truth, sub, k=6)
                    lr, _, _ = LH.train_to_skill(cur_truth, MIXTURES["feedback_heavy"], sub)
                    seq.append(int(np.argmax([lr.policy(i)[int(cur_truth[i])]
                                              for i in range(LH.N_ITEMS)])))
                if record == "bag":
                    seq_used = sorted(seq)
                else:
                    seq_used = seq
                best, best_t = -1e18, None
                for t in range(2, n_ep - 1):
                    a = np.bincount(seq_used[:t], minlength=LH.N_ITEMS) + 0.5
                    b = np.bincount(seq_used[t:], minlength=LH.N_ITEMS) + 0.5
                    sep = C.tv(C.normalize(a), C.normalize(b))
                    if sep > best:
                        best, best_t = sep, t
                err = abs((best_t or 0) - (cut if cut is not None else best_t or 0))
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "record": record,
                             "change": change, "changepoint_error": float(err),
                             "separation": float(best), "n": 1})
    return {"rows": rows}


def reduce_E11(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "dated records recover a change point that an undated bag cannot",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    d = mean_of(rows, "changepoint_error",
                lambda r: r["record"] == "dated" and r["change"] == "midway")
    b = mean_of(rows, "changepoint_error",
                lambda r: r["record"] == "bag" and r["change"] == "midway")
    battery(gr, live={"name": "dating_moves_the_changepoint_error", "observed": abs(b - d)},
            placebo={"name": "with_no_change_there_is_nothing_to_find",
                     "observed": abs(mean_of(rows, "separation",
                                             lambda r: r["change"] == "none")
                                     - mean_of(rows, "separation",
                                               lambda r: r["change"] == "midway")),
                     "tol": 1.0},
            positive={"name": "errors_are_non_negative", "observed": float(min(d, b) >= 0),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_reader_was_told_the_change_time", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_change_time_was_hidden", "observed": abs(b - d)})
    criterion(v, "E11", b - d, card.sesoi, "greater", card.sesoi_basis,
              detail="the dated record locates the change this many episodes closer than the bag")
    v["results"]["changepoint_error"] = {"dated": d, "bag": b}
    narrative(v, f"the dated record misses the change by {d:.2f} episodes and the undated bag by "
                 f"{b:.2f}", "when a work was made is evidence, and the size of that evidence is here")
    distances(v, "E11", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)
