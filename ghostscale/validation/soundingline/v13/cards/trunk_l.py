"""Trunk L: exportable rulers for Sounding Line (spec §16).

L runs no language model and reads no text. Each card reads the frozen V13 verdicts (discovery,
transfer, confirmation) and composes one bounded text-side design: the Ghost card that licenses
it, the exact transfer gap, the Sounding Line target, the records it needs, the cheap baseline, the
positive gate, what failure would mean, and the human-evidence boundary. A shape whose licensing
card did not land is exported as deferred or killed, never as a design.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import REPO, v13_dir, verdict_dir
from . import criterion, decide_state, finish, narrative, receipt, start
from .trunk_c import Cells

BRIDGE_DIR = v13_dir("bridge")
REQUIRED = ("licensing_cards", "licensing_states", "transfer_gap", "sounding_line_target", "records_required", "cheap_baseline",
            "positive_gate", "failure_meaning", "human_evidence_boundary", "recommendation")

#: The Sounding Line process-record schema as read at snapshot a80a3e7 (soundingline/process_record.py).
SL_EVENT_FIELDS = ["event_id", "order", "actor_id", "operation", "target", "parent_event_ids", "primary_goal_id",
                   "secondary_goal_candidates", "constraint_ids", "alternatives", "perceptual_access", "noticed",
                   "visible_in_final", "ground_truth_source", "payload"]
SL_OPERATIONS = ["propose", "perceive", "notice", "select", "reject", "veto", "revise", "integrate", "repair", "conceal",
                 "retain", "exploit", "realize_surface", "external_perturbation", "outline", "critique"]
SL_CONTRIBUTION_ROLES = ["proposal", "recognition", "selection", "veto", "integration", "repair", "surface_realization", "downstream_leverage"]
SL_ANOMALY_AXES = ["perceptual_access", "awareness", "origin", "handling", "recurrence", "secondary_goal", "final_status", "reader_model"]
SL_CASE_FIELDS = ["case_id", "lineage_id", "domain", "medium", "brief_id", "declared_context", "participants", "route_family", "events",
                  "artifact_final", "artifact_versions", "exact_equivalence_group", "near_equivalence_group", "split", "construction_seed", "context_fields"]


def _load(cid, lane="discovery"):
    return C.load_verdict(cid, lane)


def _state(cid):
    v = _load(cid) or _load(cid, "transfer")          # transfer-only cards (A14, C16, G16, H16, Q12) resolve in their own lane
    if v is None:
        return "UNRUN", None
    passed = None
    for k, x in v.get("results", {}).items():
        if k.startswith("criterion_") and isinstance(x, dict) and "passed" in x:
            passed = bool(x["passed"])
    conf = _load(cid, "confirmation")
    cpass = None
    if conf is not None:
        for k, x in conf.get("results", {}).items():
            if k.startswith("criterion_") and isinstance(x, dict) and "passed" in x:
                cpass = bool(x["passed"])
    return v.get("state", "UNRUN"), {"criterion": passed, "confirmed": cpass, "ceiling": v.get("claim_ceiling"), "what_happened": v.get("record", {}).get("what_happened", "")}


def _recommend(cards):
    """pursue when every licensing card LANDED with its criterion held (and held on confirmation where run);
    defer when landed but unconfirmed or mixed; kill when closed, void, instrument-failed, or criterion failed."""
    states = {c: _state(c) for c in cards}
    if any(s in ("INSTRUMENT_FAILED", "VOID", "SCIENTIFIC_CLOSED") for s, _ in states.values()):
        return "kill", states
    if any(s == "UNRUN" for s, _ in states.values()):
        return "defer", states
    crits = [d["criterion"] for s, d in states.values() if d]
    confs = [d["confirmed"] for s, d in states.values() if d and d["confirmed"] is not None]
    if all(x is True for x in crits if x is not None) and (not confs or all(confs)):
        return "pursue", states
    if any(x is False for x in crits) or any(x is False for x in confs):
        return "kill" if all(x is False for x in crits if x is not None) else "defer", states
    return "defer", states


def _write(lid, item):
    item["written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if os.environ.get("GS_V13_SMOKE") != "1":                       # smoke passes must not touch the record
        (BRIDGE_DIR / f"{lid}.json").write_text(json.dumps(C.to_jsonable(item), indent=2), encoding="utf-8")


def _deliver(card, ctx, lid, item, hypothesis):
    v = start(card, ctx, hypothesis, "METHOD")
    rec, states = _recommend(item["licensing_cards"])
    item["recommendation"] = rec
    item["licensing_states"] = {c: {"state": s, **(d or {})} for c, (s, d) in states.items()}
    missing = [k for k in REQUIRED if k not in item or item[k] in (None, "", [], {})]
    _write(lid, item)
    rows = [{"wid": "bridge", "rep": 0, "item": "deliverable", "fields_present": float(len(REQUIRED) - len(missing)), "recommendation": rec}]
    gr = G.GateReport()
    gr.identity("every_required_field_present", float(len(missing)), 0.0, tol=0.0, detail=f"missing: {missing}" if missing else "all fields present")
    gr.identity("licensing_cards_resolved_or_marked", float(sum(1 for s, _ in states.values() if s == "UNRUN" and rec != "defer")), 0.0, tol=0.0)
    gr.positive("no_human_claim_in_the_boundary", observed=float(any(w in item["human_evidence_boundary"].lower() for w in ("does not", "do not", "cannot", "no human", "not human", "never"))), expected=1.0, tol=0.0)
    criterion(v, lid, not missing, recommendation=rec, licensing=item["licensing_states"])
    v["results"].update({"deliverable": item})
    receipt(v, rows, card, ctx)
    narrative(v, f"{item['title']}: {rec}. Licensed by {', '.join(item['licensing_cards'])}; the transfer gap is {item['transfer_gap'][:160]}.",
              "A text-side design is exported with its gap, its records, its baseline and its failure meaning attached; nothing here is human evidence.",
              rival=item["failure_meaning"])
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit="OPENED")


def _single(fn):
    def unit(ctx):
        return {"rows": [{"wid": "bridge", "rep": 0, "item": "deliverable"}]}
    return unit


# --------------------------------------------------------------------------- #
unit_L01 = _single("L01")


def reduce_L01(card, units, ctx):
    o = {c: _state(c) for c in ("O02", "O04", "O07", "O10", "O16", "I06")}
    have = set(SL_EVENT_FIELDS)
    proposed_recorded = ["cost_vector (time, execution, cognitive, epistemic, social, risk, imposed) per alternative", "alternatives_available (the full menu, not the shown subset)",
                         "alternatives_believed_available (the actor's own view, when recorded)", "actor_control (voluntary | imposed | no_alternative)", "cost_timing (anticipated | sunk | discovered_late)",
                         "competence_at_decision (a recorded skill level or proxy)", "counterfactual_consequence per alternative (where a record or simulation supplies it)"]
    proposed_inferred = ["opportunity_cost (value of the best forgone route; derived from the recorded vector)", "menu_completeness (a probability, treated as evidence, never as truth)",
                         "goal_strength (identified only with competence, knowledge, constraint and risk held or modelled)"]
    item = {"id": "L01", "title": "A cost-aware decision record", "licensing_cards": ["O02", "O04", "O07", "O10", "O16", "I06"],
            "audit": {"sounding_line_event_fields": SL_EVENT_FIELDS, "already_present": ["alternatives (shown or available)", "constraint_ids", "primary_goal_id", "secondary_goal_candidates"],
                      "absent": ["a cost vector per alternative", "believed alternatives", "actor control flags", "cost timing", "competence", "counterfactual consequences"]},
            "proposed_recorded_fields": proposed_recorded, "proposed_inferred_fields": proposed_inferred,
            "validation_rules": ["an imposed decision contributes no preference evidence (O07)", "sunk and late-discovered cost are excluded from the decision-time vector (O10)",
                                 "a menu shown to the reader is a claim about the menu; completeness is a probability (O16)", "dimensions are recorded separately; the total is derived, never the primitive (O02, I06)"],
            "transfer_gap": "V13 costs are planted vectors with known causes; a text corpus records at most what a maker reported paying, with no counterfactuals and no competence measure. The record fields are the transferable object; the identification of causes is not.",
            "sounding_line_target": "the process-record schema (ProcessEvent.alternatives, constraint_ids, payload) extended with the fields above",
            "records_required": "per-decision menus with per-alternative costs; actor-control and timing flags; a competence proxy",
            "cheap_baseline": "a scalar effort field and a count of alternatives",
            "positive_gate": "a planted-cost corpus (constructed cases) in which the vector reader recovers the planted tradeoff and the scalar reader does not",
            "failure_meaning": "if text records cannot carry per-alternative costs, the cost-aware reader has no input and the O results stay simulation-bound",
            "human_evidence_boundary": "the fields record what a maker reported or what a construction planted; they do not establish that people compute costs this way, and no field may be read as a motive without the rival causes held"}
    return _deliver(card, ctx, "L01", item, "The O trunk's record fields, audited against the Sounding Line schema, become a proposed cost-aware decision record with validation rules.")


unit_L02 = _single("L02")


def reduce_L02(card, units, ctx):
    item = {"id": "L02", "title": "Communicative goals in the reader", "licensing_cards": ["G01", "G04", "G05", "G06", "G13", "G15", "I07", "I08"],
            "interface": {"q_goal": "posterior over accurate, comprehension_support, persuasion, self_presentation, concealment, misleading, neutral",
                          "q_source": "reliability by source and domain, updated only from revealed outcomes", "q_content": "support from the artifact alone",
                          "q_process": "the maker's process posterior", "uptake": "prediction_use, process_imitation, belief_update, preference_movement, refusal as separate outputs"},
            "declared_edges": {"goal": ["q_goal", "q_source", "uptake"], "source history": ["q_source", "q_goal", "uptake"], "content": ["q_content", "q_goal", "uptake"], "relevance and alignment": ["uptake"]},
            "strongest_failures": ["a stance readable only through an unmatched surface feature (I07 guards it)", "trust collapsed into one scalar (G04, G05)", "a false note taken as truth (G13)"],
            "transfer_gap": "V13 goals shape which evidence a source emits about claims whose truth the reader later learns; in text the truth of claims is rarely revealed, so q_goal will lean on q_source and content far more than in the simulation.",
            "sounding_line_target": "the probe schema's reader posterior, extended from an audience distribution to a goal distribution with separate reliability and content fields",
            "records_required": "claims with eventually revealed truth values, or at least a verifiable subset; source histories by domain",
            "cheap_baseline": "a helpfulness or valence score",
            "positive_gate": "a constructed corpus where sources with planted goals are surface-matched and the goal is recovered only when truths are revealed",
            "failure_meaning": "if goal recovery in text needs an unmatched surface cue, the reader is reading polish, and the export is withdrawn",
            "human_evidence_boundary": "the interface represents goals a constructed source held; it does not establish what a human writer intended, and no output may be labelled as a person's honesty"}
    return _deliver(card, ctx, "L02", item, "The G trunk's factorization becomes a goal-posterior interface that does not collapse reliability, evidence or uptake.")


unit_L03 = _single("L03")


def reduce_L03(card, units, ctx):
    item = {"id": "L03", "title": "Testable attention manipulations for a text reader", "licensing_cards": ["A01", "A02", "A03", "A10", "A12", "A13", "I05"],
            "protocols": {"same_information_salience": "present the same passages with one cue family made conspicuous; the reader's proper score may not change if attention only allocates (A13)",
                          "cue_selection_under_budget": "restrict the reader to k passages or cue families; compare learned, salience and random selection on held-out prediction (A02)",
                          "precision_weighting": "the same passages, different declared weights on cue families; learned weights beat uniform, wrong weights produce confident error (A03)",
                          "dynamic_reallocation": "a compatible prefix followed by a diagnostic conflict; an adaptive reader that shifts weight on surprise is compared with static readers (A10)"},
            "nulls": ["no-information passages (scrambled cue families) where no policy may gain", "duplicated cues that must not add confidence", "adversarial salience where a conspicuous weak cue is planted (A12)"],
            "transfer_gap": "V13 channels are disjoint by construction; text cue families overlap, so a precision reader on text can double count. The identity and no-information nulls are the transferable rulers; the gains are not.",
            "sounding_line_target": "the reading-profile machinery: which spans a probe is given, and with what declared weights",
            "records_required": "cue-family annotations on spans; held-out continuations or responses to score",
            "cheap_baseline": "uniform weights over everything the probe is shown",
            "positive_gate": "the identity test (all weights one reproduces the plain reading) and a no-information corpus where every policy scores at chance",
            "failure_meaning": "a text reader that gains in the no-information null is reading the manipulation, not the text; the protocol is then invalid, not the theory",
            "human_evidence_boundary": "these are operational readers; they do not measure human attention"}
    return _deliver(card, ctx, "L03", item, "The A policies become same-information salience, cue-selection and dynamic-reallocation protocols with their nulls.")


unit_L04 = _single("L04")


def reduce_L04(card, units, ctx):
    item = {"id": "L04", "title": "Nested similarity interactions for Sounding Line", "licensing_cards": ["C03", "C04", "C05", "C07", "C08", "C14", "C16", "P01", "P03", "I03"],
            "predictions": {"model_family": "a reader model from the same family as the writer model gains over a broad prior in proportion to shared factorization (C03)",
                            "expertise": "a reader that shares the writer's production competence reads mechanics better; preference similarity does not substitute (C08)",
                            "culture_or_convention": "a matched convention helps; a matched label without the convention costs (C07)",
                            "individual_history": "a writer's own earlier work improves continuation beyond group and expertise (C09)",
                            "misleading_similarity": "nuisance matches raise confidence without accuracy in a vulnerable reader; correction is measurable (C11, P03)",
                            "self_vs_local": "whatever a self prior buys, an equally local non-self prior buys too unless C04 found otherwise; report the conditional surface, never a pooled mean (C04, C14)"},
            "transfer_gap": "V13 similarity levels are planted and measurable; in text, family, convention and expertise similarity must be estimated, and the matched-prior construction (I03) has no text analogue yet.",
            "sounding_line_target": "reader-model families and their priors over writer models; the ordering of reader models by similarity to the writer",
            "records_required": "writer models with known family, convention and competence; reader models likewise; held-out continuations",
            "cheap_baseline": "a broad population prior with no similarity structure",
            "positive_gate": "a planted-similarity corpus in which the conditional interaction (near gain, far harm) is recovered before any pooling",
            "failure_meaning": "if only the pooled mean transfers, the mechanism did not; if the near gain transfers without the far harm, look for a leak",
            "human_evidence_boundary": "the predictions are about constructed readers of constructed writers; they do not describe human self-projection"}
    return _deliver(card, ctx, "L04", item, "The C and P conditional surfaces become similarity predictions Sounding Line can test, with no human claim.")


unit_L05 = _single("L05")


def reduce_L05(card, units, ctx):
    item = {"id": "L05", "title": "Testing purpose-first", "licensing_cards": ["A06", "Q05"],
            "comparison": ["goal-first", "mechanics-first", "anomaly-first", "context-first", "adaptive (EIG-chosen first cue)", "compute-matched (the same number of cue reads)"],
            "readers": ["novice (no execution model)", "expert (an execution model of the writer's craft)"],
            "measures": ["early proper score after the first cue", "final score after every cue (convergence expected)", "cost per cue"],
            "transfer_gap": "V13 entry points are cue families with declared costs; text entry points are prompts or span orders, whose costs must be declared before the test.",
            "sounding_line_target": "the order in which a probe is given the artifact's purpose, technique, mechanics, anomalies and context",
            "records_required": "artifacts with separable purpose, mechanics and anomaly spans; expert and novice reader models",
            "cheap_baseline": "fixed purpose-first",
            "positive_gate": "final posteriors converge across entries on a planted corpus; the early advantage of purpose-first exists for novices",
            "failure_meaning": "if purpose-first wins finally as well as early, the artifact is not separable by entry and the arrow claim is untestable there",
            "human_evidence_boundary": "an entry-point result describes readers with declared cue costs; it does not say how people begin"}
    return _deliver(card, ctx, "L05", item, "A06 and Q05 become an entry-policy comparison with fixed purpose-first as the baseline.")


unit_L06 = _single("L06")


def reduce_L06(card, units, ctx):
    item = {"id": "L06", "title": "Partitioning unexplained activity", "licensing_cards": ["A07", "H10", "O07", "P14"],
            "alternatives": ["unknown goal (the activity serves a goal the reader lacks a hypothesis for)", "unknown method or constraint (a goal the reader knows, reached by a route or under a constraint it cannot model)",
                             "mistake or accident (a deviation the maker did not choose)"],
            "rules": ["abstain until a handling or a later event separates the alternatives (P14)", "handling evidence separates mistakes from choices (A07, H10)", "constraint records separate imposed activity from preference (O07)"],
            "transfer_gap": "V13 anomalies carry planted origin and handling; text anomalies carry neither unless the record does. The abstention rule transfers; the recovery rates do not.",
            "sounding_line_target": "the anomaly axes of the process record (origin, handling, awareness) and the reader-model consequence field",
            "records_required": "handling events after anomalies; constraint records; held-out responses",
            "cheap_baseline": "label every unexplained span a mistake",
            "positive_gate": "a constructed corpus with planted origins in which the reader abstains on unhandled anomalies and separates handled ones",
            "failure_meaning": "if the reader names an origin without handling evidence, it is inventing a route",
            "human_evidence_boundary": "the partition is a reader's bounded hypothesis set; it does not say what a person did or meant"}
    return _deliver(card, ctx, "L06", item, "Unexplained activity is partitioned into bounded alternatives with abstention and held-out response.")


unit_L07 = _single("L07")


def reduce_L07(card, units, ctx):
    item = {"id": "L07", "title": "A ruler for controller-subordinate interaction", "licensing_cards": ["H06", "H07", "H10", "H15", "H03"],
            "protocol": {"crossed_roles": "the same subordinates under different controllers and domains; attribution must follow the correction relation (H07)",
                         "hidden_intervention": "hide the next correction; a graph model must beat role frequency, coherence, identity and token share (H15)",
                         "signatures": "suppression versus amplification from proposal-to-realization change (H06)", "mistakes": "handling sequences predict downstream revisions (H10)"},
            "rivals": ["coherence of the product", "actor style", "token share"],
            "transfer_gap": "V13 records who corrected whom; a text or log corpus needs correction events with actors. Without them the ruler has no input and only the artifact-floor result (H03) transfers.",
            "sounding_line_target": "collaboration logs and revision histories with actor-attributed operations (the contribution network's per-actor relations)",
            "records_required": "proposal, correction and realization events with actor ids; a hidden next intervention to predict",
            "cheap_baseline": "coherence of the final artifact",
            "positive_gate": "a constructed log corpus with paired central and shared-brief teams where the artifact classifier is at floor and the interaction reader is not",
            "failure_meaning": "if coherence separates the teams, the corpus leaked the director into the artifact",
            "human_evidence_boundary": "the ruler reads records of operations; it does not attribute authorship or intent to people"}
    return _deliver(card, ctx, "L07", item, "H06, H07, H10 and H15 become a crossed-role, hidden-intervention protocol with coherence and style as rivals.")


unit_L08 = _single("L08")


def reduce_L08(card, units, ctx):
    ladder = {}
    try:
        f03 = json.loads((REPO / "results/validation/soundingline/v12/F03.json").read_text(encoding="utf-8"))["results"]["by_ruler_and_cell"]
        ladder["v12_F03"] = {k: v["auc"] for k, v in f03.items() if k.startswith("dependency")}
    except Exception:                                                                # noqa: BLE001
        ladder["v12_F03"] = "unavailable"
    h14 = _load("H14")
    ladder["v13_H14"] = h14["results"].get("ladder") if h14 else "unrun"
    item = {"id": "L08", "title": "Resolution required for topology inference", "licensing_cards": ["H14", "H03", "H13", "Q08"],
            "ladder": ladder,
            "minimum": {"layered_vs_flat (V12 F03)": "at floor at 4 steps x 12 blocks; moderate at 32 x 12; strong at 32 x 60; near-perfect at 128 x 60",
                        "director_vs_brief (V13 H03, H14)": "artifact-only at floor at every resolution; separable only with correction records; the first sufficient record level is reported by H14",
                        "probes (Q08)": "a probe set below the divergence floor invalidates any active-selection claim"},
            "transfer_gap": "V12 blocks and V13 events are segmented by construction; text segmentation adds error the ladder does not include.",
            "sounding_line_target": "the admissibility check before any topology score is reported",
            "records_required": "segmentable blocks or events at the stated counts",
            "cheap_baseline": "report topology at any resolution",
            "positive_gate": "the ruler reaches its validated accuracy at the stated resolution on a planted corpus",
            "failure_meaning": "a topology score below the admissible resolution is a number without a ruler",
            "human_evidence_boundary": "resolution requirements are properties of these rulers on constructed data; they do not transfer to human evidence without remeasurement"}
    return _deliver(card, ctx, "L08", item, "The V12 F03 and V13 H and Q ladders become an admissibility rule for topology scores.")


unit_L09 = _single("L09")


def reduce_L09(card, units, ctx):
    item = {"id": "L09", "title": "The trust gate a model reader should expose", "licensing_cards": ["G04", "G05", "G07", "G08", "G09", "G10", "G12", "G13", "G16", "A09"],
            "outputs": ["q_source by source and domain", "q_content per claim", "q_goal per source", "uptake channels", "conflict flag or abstention when a note contradicts strong evidence"],
            "tests": ["false context at several evidence strengths and orders (G13)", "reversal of a source's reliability (G16)", "repair after failure by evidence versus wording (G10)",
                      "source-domain transfer against a shared label (G12)", "trust-default frontier (G07)"],
            "transfer_gap": "V13 reveals claim truths on schedule; in text, verification is sparse and delayed, which stretches every recovery time and weakens q_goal.",
            "sounding_line_target": "the probe schema's constrained output, extended with the separate posteriors above",
            "records_required": "claims with revealed truths, source histories by domain, notes with their own provenance",
            "cheap_baseline": "one trust scalar per source",
            "positive_gate": "a constructed corpus where the factored reader keeps strong contradicting evidence against a false note and the scalar reader does not",
            "failure_meaning": "if the factored reader never beats the scalar on fresh sources, keep the scalar and say so",
            "human_evidence_boundary": "the gate is a reader architecture; it does not measure anyone's trust"}
    return _deliver(card, ctx, "L09", item, "G04-G16 and A09 become the separate posteriors a model reader should expose, with false-context and reversal tests.")


unit_L10 = _single("L10")


def reduce_L10(card, units, ctx):
    table = {"recoverable_from_artifact": ["the maker's goal and profile at sufficient evidence (C trunk)", "goal structure under rewriting (H12)", "communicative goal when claim truths are known (G01)"],
             "conditionally_recoverable": ["expertise, when the reader shares the execution model (C08)", "a private secondary goal, when proposals are recorded (H09)", "the cost vector, when menus are recorded (O02)"],
             "underdetermined_or_records_dominant": ["director versus shared brief (H03, H14): records only", "identical artifacts from different histories (H13, P14): abstain", "hidden or false menus (O16): calibrated uncertainty, never a menu claim"]}
    item = {"id": "L10", "title": "Artifact-only versus record-dominant variables", "licensing_cards": ["H13", "H14", "P14", "O16", "H03", "H09"], "table": table,
            "transfer_gap": "the boundary between recoverable and record-dominant is set by V13's constructions; in text the artifact side shrinks (surface entanglement) and the record side depends on what was logged.",
            "sounding_line_target": "the claim ceiling attached to each target before a probe is run",
            "records_required": "the minimal record sets named by H14 for record-dominant targets",
            "cheap_baseline": "treat every target as artifact-recoverable",
            "positive_gate": "on a planted corpus, artifact-only readers sit at floor on the record-dominant targets and above it on the recoverable ones",
            "failure_meaning": "a record-dominant target reported from an artifact is a leak until shown otherwise",
            "human_evidence_boundary": "the table classifies constructed targets; it does not claim what can be known about people from their work"}
    return _deliver(card, ctx, "L10", item, "The H13, H14, P14 and O16 ladders become a table of recoverable, conditionally recoverable and underdetermined targets.")


unit_L11 = _single("L11")


def reduce_L11(card, units, ctx):
    item = {"id": "L11", "title": "What standardized author-purpose questions can validate", "licensing_cards": ["G01", "A06", "H13", "P14"],
            "targets": {"normative_rhetorical_purpose": "what a competent reader is expected to say the text is for; validated against reader agreement, not against the author",
                        "reader_interpretation": "what a given reader took the text to be for; a reading-profile measurement",
                        "private_process": "what the author actually did and wanted; recoverable only where records exist (L10) and never from a standardized answer"},
            "blueprint": ["a corpus with expert-adjudicated normative purpose labels", "reader-model answers scored against those labels", "a separate constructed corpus with planted private process to show the two targets come apart"],
            "claim_ceiling": "useful as a measure of reader competence on normative purpose; not creator-history ground truth",
            "transfer_gap": "V13 has ground truth for private process by construction; a standardized question has ground truth only for the normative target.",
            "sounding_line_target": "the author-purpose question set and its scoring",
            "records_required": "adjudicated normative labels; for private process, records, or nothing",
            "cheap_baseline": "agreement with the majority reader",
            "positive_gate": "on the constructed corpus, normative-purpose accuracy and private-process accuracy dissociate",
            "failure_meaning": "if they do not dissociate in construction, the question measures one thing and the distinction is idle",
            "human_evidence_boundary": "a standardized question validates reader competence on a normative target; it cannot certify what any author privately intended"}
    return _deliver(card, ctx, "L11", item, "Standardized author-purpose questions are given a corpus blueprint and a ceiling: reader competence, not creator-history truth.")


unit_L12 = _single("L12")


def reduce_L12(card, units, ctx):
    items = {}
    for k in range(1, 12):
        p = BRIDGE_DIR / f"L{k:02d}.json"
        if p.exists():
            items[f"L{k:02d}"] = json.loads(p.read_text(encoding="utf-8"))
    ranked = sorted(items.values(), key=lambda it: {"pursue": 0, "defer": 1, "kill": 2}.get(it.get("recommendation"), 3))
    packet = ["# V13 → Sounding Line bridge packet (generated by trunk L)", "",
              "Each item names the Ghost card that licenses it, the exact transfer gap, the Sounding Line target, the records it needs, the cheap baseline, the positive gate, what failure would mean, and the human-evidence boundary. "
              "Recommendations are mechanical: pursue when every licensing card landed with its criterion held (and held on confirmation where run); defer when landed but unconfirmed or mixed; kill when a licensing card closed, voided, failed its instrument, or failed its criterion.", ""]
    for it in ranked:
        packet += [f"## {it['id']} — {it['title']}: **{it['recommendation']}**", "",
                   f"- licensed by: {', '.join(f'{c} ({s['state']}{', criterion held' if s.get('criterion') else (', criterion failed' if s.get('criterion') is False else '')})' for c, s in it['licensing_states'].items())}",
                   f"- transfer gap: {it['transfer_gap']}", f"- target: {it['sounding_line_target']}", f"- records: {it['records_required']}",
                   f"- cheap baseline: {it['cheap_baseline']}", f"- positive gate: {it['positive_gate']}", f"- failure means: {it['failure_meaning']}",
                   f"- boundary: {it['human_evidence_boundary']}", ""]
    out = REPO / "docs" / "versions" / "v13-common-ground" / "BRIDGE_PACKET.md"
    if os.environ.get("GS_V13_SMOKE") != "1":                       # smoke passes must not touch the record
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(packet) + "\n", encoding="utf-8", newline="\n")
    item = {"id": "L12", "title": "Which Sounding Line branches deserve compute next", "licensing_cards": [c for it in items.values() for c in it["licensing_cards"]] or ["L01"],
            "ranking": [{"id": it["id"], "title": it["title"], "recommendation": it["recommendation"]} for it in ranked],
            "packet_path": str(out.relative_to(REPO).as_posix()),
            "transfer_gap": "every item's gap is inherited; this item only orders them",
            "sounding_line_target": "the next compute allocation",
            "records_required": "as per item",
            "cheap_baseline": "pursue everything",
            "positive_gate": "each pursued item's own positive gate",
            "failure_meaning": "a pursued item that fails its positive gate on the constructed corpus returns here as a kill",
            "human_evidence_boundary": "the ranking orders simulation-licensed designs; it does not import any simulation result into Sounding Line as data"}
    return _deliver(card, ctx, "L12", item, "Only validated, transferring, prediction-gated shapes are ranked for Sounding Line compute, with the controls each needs and the failure signatures to watch.")
