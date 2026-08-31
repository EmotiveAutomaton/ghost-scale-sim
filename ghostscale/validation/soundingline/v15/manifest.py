"""The V15 queue manifest: 112 mandatory cards in twelve trunks, plus 24 attacks (spec §6, §7).

Every card in the spec's inventory appears here literally, with the question the spec asked, the
endpoint it scores, the architectures it compares, the families it runs in, its declared factors
and its smallest effect of interest. ``runners/validate_v15_program.py`` checks this file against
the spec's counts and refuses a queue that does not enumerate all of them.

Where the effect sizes come from
--------------------------------
Spec §8.2 forbids recycling V14's 0.02-nat bar and asks for a fraction of a live positive control
on the same score. The construction's own spans were measured before anything was registered and
are recorded in ``prereg_v15.SESOI``: the distance from the cheapest reader to the state oracle is
0.30 nats at the atlas's reference settings and 0.54 nats under scarcity, so the architecture bar
is 0.015 -- five per cent of the smaller span. Accuracy bars are fractions above the construction's
own chance floor, abstention bars are rates, and each is named with its basis in the criterion.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import v15_dir
from .atomicio import write_json_atomic
from .schemas import TIERS, Card, card_from_dict, expected_cells

MANIFEST = v15_dir() / "QUEUE_MANIFEST.json"
CELLS_TEMPLATE = v15_dir() / "EXPECTED_CELLS_TEMPLATE.json"
CELLS = v15_dir() / "EXPECTED_CELLS.json"
CONSTRUCTION_GRAPH = v15_dir() / "CONSTRUCTION_GRAPH.json"
GENERATOR_FAMILIES = v15_dir() / "GENERATOR_FAMILIES.json"
ARCHITECTURE_BUDGETS = v15_dir() / "ARCHITECTURE_BUDGETS.json"
SOURCE_LINEAGES = v15_dir() / "SOURCE_LINEAGES.json"
ATTACK_MATRIX = v15_dir() / "ATTACK_MATRIX.json"
PUBLICATION_MAP_TEMPLATE = v15_dir() / "PUBLICATION_MAP_TEMPLATE.json"
AMENDMENTS = v15_dir() / "AMENDMENTS.json"
COVERAGE_FILE = v15_dir() / "COVERAGE.json"

ALL3 = ["chain", "composition", "communication"]
CORE_ARCH = ["surface", "label_only", "independent", "staged", "joint_exact", "particle",
             "oracle_state"]
FULL_ARCH = ["surface", "label_only", "independent", "independent_routed", "staged", "joint_exact",
             "factor_graph", "particle", "expand", "direct_predictor", "oracle_model_space",
             "oracle_state"]


def _c(cid, trunk, wave, question, construction, target, estimand, null, alt, rival, **kw) -> Card:
    return Card(id=cid, trunk=trunk, wave=wave, question=question, construction=construction,
                target=target, estimand=estimand, null_expectation=null,
                alternative_expectation=alt, strongest_rival=rival,
                module=f"ghostscale.validation.soundingline.v15.cards.trunk_{trunk.lower()}", **kw)


def build_cards() -> list:                                   # noqa: C901 - a literal inventory
    c = []

    # ---- I: integrity, V14 audit, construction distance (8) ------------------------------- #
    c += [
        _c("I01", "I", 0, "Do all V14 numeric anchors imported by V15 reproduce from the committed record?",
           "read V14's committed verdicts and ledger; hash and compare every cited number",
           "anchor reproduction", "hash match and cited-value deviation",
           "an anchor differs from its ledger entry", "every anchor reproduces exactly",
           "a stale or edited V14 verdict", claim_class="METHOD", sesoi=0.005,
           sesoi_basis="half a unit of the cited precision", unit_kind="list", causal=False,
           gates_required=["positive"], factors={"anchor": ["J04", "R02", "E01", "A06", "F08", "B01"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=0.2),
        _c("I02", "I", 0, "Does the manifest recursively enumerate all 112 cards, 24 attacks, families, factors, lanes and required cells?",
           "walk the manifest and the expected-cell template", "manifest completeness",
           "exact expected-cell validator result", "a card, attack, factor or lane is missing",
           "the enumeration is exact", "a card declared but never implemented",
           claim_class="METHOD", sesoi=0.0, sesoi_basis="exact", unit_kind="single", causal=False,
           gates_required=["positive"], factors={"check": ["cards", "attacks", "factors", "lanes", "cells"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=0.1),
        _c("I03", "I", 0, "Can a causal-distance audit distinguish a direct hidden-state readout, a planted class signature and inference through intervening behaviour?",
           "known-answer fixtures with declared distances", "causal-distance classifier",
           "fraction of fixtures classified as declared", "the classifier mislabels a fixture",
           "every fixture reproduces its declared distance", "a distance asserted rather than audited",
           claim_class="METHOD", sesoi=0.0, sesoi_basis="exact on known-answer fixtures",
           unit_kind="single", causal=False, gates_required=["positive", "placebo"],
           factors={"fixture_distance": ["DIRECT_READOUT", "PLANTED_SIGNATURE", "INFERRED_THROUGH_BEHAVIOUR"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=0.1),
        _c("I04", "I", 0, "Does a regenerated V14 bridge read F08 as landed while preserving the stale packet and its hash?",
           "regenerate the bridge from V14's committed verdicts; write an additive erratum only",
           "bridge regeneration", "agreement between regenerated and committed states",
           "the stale export is overwritten or the states still disagree",
           "the erratum is additive and the original file and hash are preserved",
           "silently rewriting history", claim_class="METHOD", sesoi=0.0,
           sesoi_basis="exact: no historical file may change", unit_kind="single", causal=False,
           gates_required=["positive", "placebo"], factors={"check": ["state_agreement", "original_preserved"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=0.1),
        _c("I05", "I", 0, "Do label-only readers fail when labels are shuffled while full-state readers retain only warranted prediction?",
           "shuffle the latent labels handed to the label reader; leave the world alone",
           "label versus state", "log-score drop under label shuffling",
           "shuffling labels does not change the label reader", "the label reader collapses and the state reader does not",
           "a side channel that carries the label anyway", claim_class="METHOD", sesoi=0.10,
           sesoi_basis="a third of the label reader's own advantage over surface",
           factors={"reader": ["label_only", "joint_exact", "surface"], "labels": ["true", "shuffled"]},
           families=ALL3, endpoints=["next_action"], architectures=["surface", "label_only", "joint_exact"],
           work_weight=1.0),
        _c("I06", "I", 0, "Are three generator families genuinely independent implementations with matched target semantics?",
           "audit code paths, emission mechanisms and metamorphic behaviour across families",
           "family independence", "shared-symbol overlap and realized-semantics agreement",
           "two families share a transition or emission path", "code paths are disjoint and semantics agree",
           "a family that relabels another's table", claim_class="METHOD", sesoi=0.0,
           sesoi_basis="exact: no shared generative symbol", unit_kind="single", causal=False,
           gates_required=["positive", "placebo"],
           factors={"family": ALL3, "check": ["code_path", "emission", "metamorphic"]},
           families=ALL3, endpoints=[], architectures=[], work_weight=0.3),
        _c("I07", "I", 0, "Do tiny worlds reproduce brute-force posteriors, equivalence classes and prospective scores under relabelling and row reordering?",
           "enumerate a tiny world exactly and compare with a naive linear-space product",
           "exactness", "maximum absolute deviation from brute force",
           "the fast path disagrees with brute force", "they agree to floating point",
           "a vectorisation bug hidden by normalisation", claim_class="METHOD", sesoi=1e-9,
           sesoi_basis="floating-point identity", causal=False, gates_required=["positive", "placebo"],
           factors={"family": ALL3, "check": ["brute_force", "relabel", "reorder"]},
           families=ALL3, endpoints=[], architectures=["joint_exact"], work_weight=0.4),
        _c("I08", "I", 0, "Does the runtime opening guard reject V14-sized and deliberately underfilled queues?",
           "feed the guard a fast-machine fixture and an empty-queue fixture",
           "opening guard", "guard verdict on each fixture", "the guard admits an underfilled queue",
           "both fixtures are refused", "a guard that only checks the happy path",
           claim_class="METHOD", sesoi=0.0, sesoi_basis="exact: both fixtures must be refused",
           unit_kind="single", causal=False, gates_required=["positive", "placebo"],
           factors={"fixture": ["v14_sized", "empty_queue", "fast_machine", "healthy"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=0.1),
    ]

    # ---- C: coupling and access boundary atlas (14) ---------------------------------------- #
    atlas_arch = ["surface", "independent", "staged", "joint_exact", "oracle_state"]
    c += [
        _c("C01", "C", 1, "Does V14's thin joint advantage reproduce in its original disjoint-route regime?",
           "coupling zero, overlap zero, generous evidence", "anchor", "joint minus independent, nats",
           "the advantage is near zero", "the advantage is near zero", "a coupled prior leaking in",
           claim_class="BOUNDARY", sesoi=0.015, sesoi_basis="5% of the 0.30-nat oracle-minus-surface span",
           factors={"dose": ["2", "8"], "architecture": ["independent", "joint_exact"]},
           families=ALL3, endpoints=["next_action"], architectures=atlas_arch, work_weight=1.0),
        _c("C02", "C", 1, "At what latent-coupling strength does joint prediction exceed independent marginals by the live-effect threshold?",
           "sweep coupling with overlap and dose held", "coupling onset",
           "first coupling level whose advantage clears the bar", "no level clears the bar",
           "an onset exists inside the swept range", "a marginal-prior artifact",
           claim_class="BOUNDARY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"kappa": ["0.0", "0.25", "0.5", "0.75", "1.0"], "architecture": ["independent", "joint_exact"]},
           families=ALL3, endpoints=["next_action"], architectures=atlas_arch, work_weight=1.6),
        _c("C03", "C", 1, "How does route overlap move that threshold?", "cross coupling with overlap",
           "coupling x overlap surface", "advantage per cell", "the surface is flat in overlap",
           "overlap lowers the coupling needed", "overlap measured under the world's own prior",
           claim_class="BOUNDARY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"kappa": ["0.0", "0.5", "1.0"], "overlap": ["0.0", "0.33", "0.66", "1.0"]},
           families=ALL3, endpoints=["next_action"], architectures=atlas_arch, work_weight=2.0),
        _c("C04", "C", 1, "How does evidence scarcity move it?", "cross coupling with dose",
           "dose x coupling surface", "advantage per cell", "the surface is flat in dose",
           "scarcity raises the advantage", "a pooled headline over a conditional surface",
           claim_class="BOUNDARY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"kappa": ["0.0", "0.5", "1.0"], "dose": ["1", "2", "4", "8", "16"]},
           families=ALL3, endpoints=["next_action"], architectures=atlas_arch, work_weight=2.2),
        _c("C05", "C", 1, "Does synergy rather than redundancy predict joint advantage?",
           "matched-information worlds with independent, redundant and synergistic emissions",
           "PID atoms versus advantage", "synergy atom and joint advantage per cell",
           "advantage is unrelated to the synergy atom", "advantage tracks synergy, not redundancy",
           "total information differing between conditions", claim_class="BOUNDARY", sesoi=0.015,
           sesoi_basis="5% of the oracle-minus-surface span",
           factors={"dependence": ["independent", "redundant", "synergistic"], "dose": ["2", "8"]},
           families=ALL3, endpoints=["next_action"], architectures=atlas_arch, work_weight=1.6),
        _c("C06", "C", 2, "When one route is missing, can joint structure recover useful information through the others?",
           "remove a route; compare joint with the best factorized reader", "recovery under missingness",
           "joint minus best factorized, nats", "no recovery", "joint recovers part of the loss",
           "oracle access to the missing route", claim_class="BOUNDARY", sesoi=0.015,
           sesoi_basis="5% of the oracle-minus-surface span",
           factors={"missing": ["none", "route"], "kappa": ["0.0", "0.5", "1.0"]},
           families=ALL3, endpoints=["next_action"], architectures=atlas_arch, work_weight=1.4),
        _c("C07", "C", 2, "When context is hidden, does joint inference help or amplify a wrong story?",
           "hide the context; score, calibrate and allow abstention", "context missingness",
           "score, calibration error and abstention rate", "hiding context is neutral",
           "joint either helps or becomes confidently wrong", "an abstention rule tuned after the fact",
           claim_class="BOUNDARY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"missing": ["none", "context"], "architecture": ["independent", "joint_exact"]},
           families=ALL3, endpoints=["next_action"], architectures=atlas_arch, work_weight=1.4),
        _c("C08", "C", 2, "When the opportunity set is hidden, which latents become an equivalence class?",
           "hide the opportunity set; report class coverage", "opportunity missingness",
           "equivalence-class mass and changed-opportunity prediction", "no class forms",
           "a declared class forms and is covered", "collapsing the class into one member",
           claim_class="BOUNDARY", sesoi=0.85, sesoi_basis="class coverage a reader must retain",
           factors={"missing": ["none", "opportunity"], "kappa": ["0.0", "0.5"]},
           families=ALL3, endpoints=["changed_context_choice"], architectures=atlas_arch, work_weight=1.2),
        _c("C09", "C", 2, "Does policy stochasticity erase or merely delay the joint advantage?",
           "cross temperature with dose", "temperature x dose interaction", "advantage per cell",
           "temperature has no effect", "higher temperature delays rather than erases",
           "a temperature that also changes the marginal", claim_class="BOUNDARY", sesoi=0.015,
           sesoi_basis="5% of the oracle-minus-surface span",
           factors={"temperature": ["0.25", "0.6", "1.0", "2.0"], "dose": ["2", "8"]},
           families=ALL3, endpoints=["next_action"], architectures=atlas_arch, work_weight=1.6),
        _c("C10", "C", 2, "Does process equifinality create false certainty in plug-in readers but not joint readers?",
           "equifinal processes; measure member and class mass before and after a resolving observation",
           "equifinality", "maximum single-member mass and class mass",
           "both readers hold the same member mass", "the plug-in reader concentrates and the joint does not",
           "an equifinality that is not actually exact", claim_class="BOUNDARY", sesoi=0.10,
           sesoi_basis="unjustified member mass above the within-class uniform share",
           factors={"equifinality": ["none", "exact", "approximate"], "architecture": ["staged", "joint_exact"]},
           families=ALL3, endpoints=["next_action"], architectures=atlas_arch, work_weight=1.4),
        _c("C11", "C", 3, "Can approximate reward equivalence be represented without forcing an arbitrary point value?",
           "sample the feasible reward set; measure coverage and downstream regret",
           "feasible-set representation", "feasible-set coverage and policy regret",
           "the set collapses to a point", "the set is retained and regret is bounded",
           "a point estimate dressed as a set", claim_class="BOUNDARY", sesoi=0.85,
           sesoi_basis="fraction of the true reward direction the retained set must cover",
           factors={"equifinality": ["exact", "approximate"], "record": ["optimal_only", "varied"]},
           families=["chain", "composition"], endpoints=["changed_context_choice"],
           architectures=["joint_exact"], work_weight=1.0),
        _c("C12", "C", 3, "Does pairwise maker-reader similarity reproduce the V13 near-maker self-prior benefit under an equal-local comparator?",
           "cross pairwise similarity against family typicality, once", "similarity-conditioned prior",
           "self-prior advantage by similarity", "no advantage at any similarity",
           "an advantage at high similarity only", "typicality standing in for similarity",
           claim_class="BOUNDARY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"similarity": ["-1.0", "-0.5", "0.0", "0.5", "1.0"], "typicality": ["0.0", "1.0"]},
           families=["chain", "communication"], endpoints=["next_action"],
           architectures=["independent", "joint_exact"], work_weight=1.4),
        _c("C13", "C", 3, "Does the self prior become harmful under dissimilarity and then correct under diagnostic evidence?",
           "dissimilar maker and reader; supply diagnostic evidence in doses", "self-prior correction",
           "score trajectory and calibration by prior distance", "the prior never corrects",
           "the prior corrects with evidence", "a correction that is really a prior reset",
           claim_class="BOUNDARY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"similarity": ["-1.0", "0.0", "1.0"], "dose": ["1", "4", "16"]},
           families=["chain", "communication"], endpoints=["next_action"],
           architectures=["independent", "joint_exact"], work_weight=1.4),
        _c("C14", "C", 3, "Do all phase boundaries transfer to two independent generator families?",
           "re-run the onset estimate on untouched family code", "cross-family transfer",
           "onset agreement and direction agreement", "the boundaries disagree across families",
           "direction agrees and onsets fall within tolerance", "a shared implementation detail",
           claim_class="BOUNDARY", sesoi=0.25, sesoi_basis="tolerated onset displacement in coupling units",
           factors={"family": ALL3, "kappa": ["0.0", "0.5", "1.0"]},
           families=ALL3, endpoints=["next_action"], architectures=atlas_arch,
           lanes=["discovery", "transfer"], work_weight=2.0),
    ]

    # ---- M: model space and inference architecture tournament (12) ------------------------- #
    c += [
        _c("M01", "M", 1, "Do factor-graph and particle readers agree with exact inference on small worlds?",
           "small enumerable worlds; compare posteriors and predictive scores", "approximation error",
           "KL from exact and predictive-score gap", "the approximations diverge from exact",
           "both agree within tolerance", "an approximation validated only on its own score",
           claim_class="METHOD", sesoi=0.05, sesoi_basis="KL from exact a reader may carry",
           factors={"architecture": ["factor_graph", "particle"], "dose": ["2", "8"]},
           families=ALL3, endpoints=["next_action"],
           architectures=["joint_exact", "factor_graph", "particle"], work_weight=1.4),
        _c("M02", "M", 1, "Which architecture best predicts under a correct complete model space at matched compute?",
           "full tournament at matched likelihood-evaluation budget", "architecture ranking",
           "hidden-event log score and calibration", "no architecture separates",
           "a ranking emerges and survives budget matching", "a winner that simply spent more",
           claim_class="METHOD", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"architecture": FULL_ARCH[:8], "dose": ["2", "8"]},
           families=ALL3, endpoints=["next_action"], architectures=FULL_ARCH, work_weight=2.4),
        _c("M03", "M", 1, "Which architecture survives a missing latent variable?",
           "remove a latent from the reader's model space", "misspecification robustness",
           "score under missing_latent relative to correct", "all architectures degrade equally",
           "expansion degrades least", "the oracle model space being compared as a rival",
           claim_class="METHOD", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"model_space": ["correct", "missing_latent"], "architecture": ["joint_exact", "expand", "oracle_model_space"]},
           families=ALL3, endpoints=["next_action"], architectures=FULL_ARCH, work_weight=1.8),
        _c("M04", "M", 1, "Can uncertainty-triggered expansion distinguish missing variables from ordinary observation noise?",
           "cross a genuinely missing variable against added observation noise", "expansion selectivity",
           "true and false expansion rates", "the two conditions expand equally",
           "expansion fires on the missing variable and not on noise", "a threshold fitted to the outcome",
           claim_class="METHOD", sesoi=0.30, sesoi_basis="separation between true and false expansion rates",
           factors={"condition": ["missing_latent", "noise_only"], "selector": ["residual", "expected_value"]},
           families=ALL3, endpoints=["next_action"], architectures=["expand"], work_weight=1.6),
        _c("M05", "M", 2, "Does adding earlier timesteps help only when the omitted history is causally relevant?",
           "relevant and irrelevant omitted history, matched in length", "timestep expansion",
           "score gain from added timesteps by relevance", "gain is the same either way",
           "gain only where the history is relevant", "a length cue standing in for relevance",
           claim_class="METHOD", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"history": ["relevant", "irrelevant"], "added_steps": ["0", "2", "4"]},
           families=["chain", "composition"], endpoints=["next_action"],
           architectures=["expand", "joint_exact"], work_weight=1.4),
        _c("M06", "M", 2, "Can particle hypothesis revision recover after an early wrong commitment?",
           "seed the filter on a wrong hypothesis; watch the true mass", "recovery",
           "steps to recover and final true mass", "no reader recovers",
           "the particle filter recovers where staged does not", "a jitter rate that hands back the answer",
           claim_class="METHOD", sesoi=0.50, sesoi_basis="true-state mass a recovered reader must reach",
           factors={"architecture": ["particle", "staged", "joint_exact"], "seeded": ["true", "wrong"]},
           families=ALL3, endpoints=["next_action"], architectures=["particle", "staged", "joint_exact"],
           work_weight=1.6),
        _c("M07", "M", 2, "Does a direct predictor outperform an interpretable maker model while failing changed-context transfer?",
           "train a direct predictor in domain; intervene on the context", "direct versus interpretable",
           "in-domain score and post-intervention score", "the direct predictor transfers as well",
           "it wins in domain and loses under intervention", "hiding the intervention failure",
           claim_class="METHOD", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"regime": ["in_domain", "intervened"], "architecture": ["direct_predictor", "joint_exact"]},
           families=ALL3, endpoints=["next_action", "changed_context_choice"],
           architectures=["surface", "direct_predictor", "joint_exact"], work_weight=1.6),
        _c("M08", "M", 2, "Do correct latent labels without a context-realized policy improve prediction?",
           "hand the reader the true labels, stripped of context", "pointer versus state",
           "label-only score against surface and full state", "labels add nothing over surface",
           "labels add something but less than context-realized state", "a label that carries context implicitly",
           claim_class="METHOD", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"architecture": ["surface", "label_only", "joint_exact", "oracle_state"], "dose": ["2", "8"]},
           families=ALL3, endpoints=["next_action"], architectures=CORE_ARCH, work_weight=1.4),
        _c("M09", "M", 3, "Does the expandable reader retain uncertainty over several semantically distinct but behaviorally equivalent hypotheses?",
           "behaviourally equivalent hypotheses; measure class mass", "class retention",
           "class mass and maximum member mass", "the reader concentrates on one member",
           "the class is retained and no member dominates", "an equivalence that is not exact",
           claim_class="BOUNDARY", sesoi=0.85, sesoi_basis="class mass a reader must retain",
           factors={"equifinality": ["exact", "approximate"], "architecture": ["expand", "joint_exact"]},
           families=ALL3, endpoints=["next_action"], architectures=["expand", "joint_exact"],
           work_weight=1.2),
        _c("M10", "M", 3, "How do proposal-library omissions and distractor hypotheses affect expansion?",
           "remove a needed proposal; add distractors", "library quality",
           "recall, precision, search cost and predictive regret", "library quality is irrelevant",
           "omissions cost recall and distractors cost precision", "counting a distractor as a success",
           claim_class="METHOD", sesoi=0.30, sesoi_basis="recall a usable expander must reach",
           factors={"library": ["complete", "omitted", "distractors"], "selector": ["residual", "expected_value"]},
           families=ALL3, endpoints=["next_action"], architectures=["expand"], work_weight=1.6),
        _c("M11", "M", 3, "Can model expansion be selected by expected predictive value rather than residual size alone?",
           "compare the two selectors on noise and on misspecification", "selector comparison",
           "gain net of search cost", "the selectors are equivalent",
           "expected value beats residual net of cost", "a cost that was never debited",
           claim_class="METHOD", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"selector": ["residual", "expected_value"], "condition": ["missing_latent", "noise_only"]},
           families=ALL3, endpoints=["next_action"], architectures=["expand"], work_weight=1.6),
        _c("M12", "M", 3, "Which architecture transfers across generator family and action vocabulary without retuning?",
           "freeze every reader; run on untouched lineages", "frozen transfer",
           "transfer regret by architecture", "all architectures transfer equally",
           "a ranking survives transfer", "retuning between domains",
           claim_class="METHOD", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"family": ALL3, "architecture": ["independent", "joint_exact", "particle", "direct_predictor"]},
           families=ALL3, endpoints=["next_action"], architectures=FULL_ARCH,
           lanes=["discovery", "transfer"], work_weight=2.0),
    ]

    # ---- E: endogenous expertise, learning history, residue (12) --------------------------- #
    e_fac = {"mixture": ["practice_heavy", "instruction_heavy", "feedback_heavy", "constrained"]}
    c += [
        _c("E01", "E", 1, "Can matched final competence arise from different mixtures of self-directed practice, instruction, feedback and constraint?",
           "solve each mixture's exposure count to a common final skill", "skill matching",
           "final-skill spread across mixtures", "the mixtures cannot be matched",
           "final skill matches within the band and histories differ", "a mixture that cannot reach the target at all",
           claim_class="CONSTRUCTION_IDENTITY", sesoi=0.05, sesoi_basis="the declared skill band",
           factors=e_fac, families=["chain"], family_bound=True, endpoints=[], architectures=[],
           work_weight=1.2),
        _c("E02", "E", 1, "Can any reader recover history above chance without a supplied history feature or fixed class signature?",
           "randomized curricula; permutation-invariant behaviour features only", "history recovery",
           "held-out history-mixture accuracy", "recovery sits at chance",
           "recovery is above chance", "a supplied history channel", claim_class="SIMULATOR_DISCOVERY",
           sesoi=0.10, sesoi_basis="accuracy above the four-way chance floor of 0.25",
           factors=e_fac, families=["chain"], family_bound=True, endpoints=[], architectures=[],
           work_weight=1.8),
        _c("E03", "E", 1, "Does a richer learning-record model beat 'expertise equals past attention' on novel errors?",
           "two rival readers on the same behaviour", "novel-error prediction",
           "log score on the hidden error location", "the two readers tie",
           "the learning-record model wins", "an attention model given less information",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.15,
           sesoi_basis="log-score gap on a twelve-way error location",
           factors={"model": ["attention_only", "learning_record"], **e_fac},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.4),
        _c("E04", "E", 1, "What portion of residue is attributable to attention allocation after feedback and constraints are held constant?",
           "intervene on attention with feedback and constraint fixed", "factor attribution",
           "residue change attributable to attention", "attention explains none of it",
           "attention explains a measurable share", "attention and exposure confounded",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="share of residue variance",
           factors={"attention": ["low", "high"], "feedback": ["fixed"], "constraint": ["fixed"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.2),
        _c("E05", "E", 2, "Can forced training create competence and residue that oppose current preference?",
           "train under constraint, then measure changed-context choice and devaluation",
           "opposed residue", "choice against the trained residue", "residue and preference agree",
           "residue opposes preference and both are visible", "a preference defined as the residue",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.15, sesoi_basis="choice-rate shift",
           factors={"training": ["chosen", "forced"], "context": ["same", "changed"]},
           families=["chain", "composition"], endpoints=["changed_context_choice"],
           architectures=[], work_weight=1.2),
        _c("E06", "E", 2, "Do instruction and feedback produce signatures that attention-only models misclassify?",
           "cross-history confusion under both readers", "confusion structure",
           "off-diagonal mass by reader", "both readers confuse them equally",
           "the attention-only reader confuses them more", "a signature planted by construction",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="off-diagonal mass difference",
           factors={"model": ["attention_only", "learning_record"],
                    "mixture": ["instruction_heavy", "feedback_heavy"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.4),
        _c("E07", "E", 2, "Does matched competence hide different transfer breadths?",
           "matched skill; measure performance on untrained neighbours", "transfer breadth",
           "untrained-item skill by mixture", "breadth is the same across mixtures",
           "breadth differs at matched skill", "a skill match that did not hold",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.08, sesoi_basis="breadth difference in skill units",
           factors=e_fac, families=["chain"], family_bound=True, endpoints=[], architectures=[],
           work_weight=1.2),
        _c("E08", "E", 2, "Does a learning path predict reacquisition after reversal better than current skill does?",
           "reverse a subset of items; measure relearning", "relearning prediction",
           "relearning gain predicted by path versus by skill", "skill predicts as well",
           "the path predicts better", "a path label that encodes skill",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="correlation gain over the skill-only predictor",
           factors={"predictor": ["skill_only", "path"], **e_fac},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.4),
        _c("E09", "E", 3, "Can target evidence correct stale residue without erasing valid skill?",
           "targeted versus scattered corrective evidence", "residue correction",
           "bias removed and skill cost", "targeting is no better than scattering",
           "targeting removes more bias at no more skill cost", "measuring bias removal alone",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="bias removed, in error-rate units",
           factors={"evidence": ["targeted", "scattered"], **e_fac},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.2),
        _c("E10", "E", 3, "When histories are behaviorally equivalent, does the reader abstain from naming one?",
           "construct behaviourally equivalent histories", "history-class abstention",
           "history-class mass and maximum member mass", "the reader names one",
           "the class is retained", "an equivalence that is not equivalent",
           claim_class="BOUNDARY", sesoi=0.85, sesoi_basis="class mass a reader must retain",
           factors={"equivalent": ["yes", "no"], "arm": ["a", "b"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.2),
        _c("E11", "E", 3, "Does dated multi-episode evidence recover a trajectory better than an undated bag of artifacts?",
           "same records, dated and shuffled", "dating value",
           "change-point error and future-choice score", "dating adds nothing",
           "dating recovers a change point a bag cannot", "a bag given fewer records",
           claim_class="SIMULATOR_DISCOVERY", sesoi=1.0, sesoi_basis="episodes of change-point error",
           factors={"record": ["dated", "bag"], "change": ["none", "midway"]},
           families=["chain", "composition"], endpoints=["next_episode_first_choice"],
           architectures=[], work_weight=1.2),
        _c("E12", "E", 3, "Do history results survive randomized curricula rather than two fixed transition matrices?",
           "fresh randomized curricula on an untouched lineage", "curriculum robustness",
           "history recovery on untouched curricula", "recovery collapses to chance",
           "recovery survives randomization", "a fixed curriculum leaking a signature",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="accuracy above the chance floor",
           factors=e_fac, families=["chain"], family_bound=True, endpoints=[], architectures=[],
           lanes=["discovery", "transfer"], work_weight=1.4),
    ]

    # ---- G: foreground control, switching, editing, stopping (10) -------------------------- #
    g_arch = ["single_switching", "simultaneous", "habitual", "hierarchical"]
    c += [
        _c("G01", "G", 1, "Can a single foreground goal with rapid switching be surface-matched to simultaneous weighted goals?",
           "rejection-sample worlds until the action marginals collide", "collision fixture",
           "residual total variation and oracle identifiability", "the surfaces cannot be matched",
           "they collide and the sequence still identifies", "a collision that was assumed, not measured",
           claim_class="CONSTRUCTION_IDENTITY", sesoi=0.03, sesoi_basis="declared collision tolerance",
           factors={"control": g_arch}, families=["chain"], family_bound=True,
           endpoints=[], architectures=[], work_weight=1.0),
        _c("G02", "G", 1, "Which architecture better predicts next edit under uninterrupted work?",
           "uninterrupted episodes; score the hidden next edit", "next-edit prediction",
           "log score by control architecture", "no architecture predicts better",
           "one architecture predicts better", "a surface cue that survives matching",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"control": g_arch}, families=["composition", "chain"],
           endpoints=["next_edit"], architectures=CORE_ARCH, work_weight=1.4),
        _c("G03", "G", 1, "Which predicts switch timing after an interruption or new constraint?",
           "interrupt at a known step; score the switch time", "switch timing",
           "event-time score and calibration", "switch timing is unpredictable",
           "one architecture predicts the timing", "the interruption itself being the cue",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="accuracy above the uniform floor",
           factors={"control": g_arch, "interrupt": ["none", "early", "late"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.4),
        _c("G04", "G", 1, "Does simultaneous control leave dependency patterns that rapid switching does not?",
           "matched surfaces; measure cross-goal dependency", "dependency signature",
           "cross-goal dependency by architecture", "the dependency is the same",
           "simultaneous control leaves a positive dependency", "a dependency driven by the marginal",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.15, sesoi_basis="correlation difference",
           factors={"control": g_arch}, families=["chain"], family_bound=True,
           endpoints=[], architectures=[], work_weight=1.0),
        _c("G05", "G", 2, "Can habit and expertise residue mimic a weak secondary foreground goal?",
           "habit at a strength matched to a weak second goal", "habit versus goal",
           "posterior separation and intervention response", "they are distinguishable from behaviour",
           "they are matched and separate only under intervention", "a habit that is a goal by another name",
           claim_class="BOUNDARY", sesoi=0.10, sesoi_basis="posterior separation above chance",
           factors={"control": ["habitual", "simultaneous"], "intervention": ["none", "devalue"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.2),
        _c("G06", "G", 2, "Does rereading reveal residue-driven deviations by changing the foreground goal to review?",
           "switch the foreground goal to review; score the next repair", "review mode",
           "repair prediction and stopping", "review changes nothing",
           "review exposes residue-driven deviations", "a review cue that marks the deviation",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="accuracy above the uniform floor",
           factors={"mode": ["work", "review"], "control": ["single_switching", "habitual"]},
           families=["composition"], endpoints=["next_edit", "stop_or_continue"],
           architectures=CORE_ARCH, work_weight=1.2),
        _c("G07", "G", 2, "Can the reader separate an ordinary mistake, deliberate exploration, hidden aesthetic goal and habit used out of context?",
           "four deviation kinds matched on the deviation itself", "four-way discrimination",
           "posterior accuracy and abstention on collisions", "all four are confusable",
           "the continuation separates them", "a deviation marker that differs by kind",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.15, sesoi_basis="accuracy above the four-way chance floor",
           factors={"deviation": ["mistake", "exploration", "hidden_aesthetic", "habit_out_of_context"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.4),
        _c("G08", "G", 3, "Does deliberate exploration require commitment sufficient to reveal an outcome?",
           "vary how long a departure is held", "exploration commitment",
           "subsequent method change by commitment length", "commitment length is irrelevant",
           "method change requires sufficient commitment", "a length cue read directly",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.15, sesoi_basis="method-change rate difference",
           factors={"commitment": ["1", "3", "5"]}, families=["chain"], family_bound=True,
           endpoints=[], architectures=[], work_weight=1.0),
        _c("G09", "G", 3, "Can a stopping rule be recovered independently of the content goal?",
           "matched local quality; vary the threshold", "stopping-rule recovery",
           "stop prediction with and without the rule", "quality alone predicts as well",
           "the rule adds accuracy over quality alone", "a threshold visible in the quality",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="accuracy gain over the quality-only reader",
           factors={"reader": ["quality_only", "with_rule"]}, families=["composition", "chain"],
           endpoints=["stop_or_continue"], architectures=CORE_ARCH, work_weight=1.2),
        _c("G10", "G", 3, "Do control-architecture conclusions transfer to composition and chain worlds?",
           "frozen readers on untouched family code", "cross-family transfer",
           "direction agreement and calibration", "the conclusions do not transfer",
           "direction agrees across families", "a family-specific cue",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="accuracy above the chance floor",
           factors={"family": ["chain", "composition"], "control": g_arch},
           families=["chain", "composition"], endpoints=["next_edit"], architectures=CORE_ARCH,
           lanes=["discovery", "transfer"], work_weight=1.6),
    ]

    # ---- V: persistent tendency, current value, change, concealment (10) ------------------- #
    v_riv = ["unchanged", "changed_preference", "changed_goal", "concealment", "stale_residue"]
    c += [
        _c("V01", "V", 1, "Can standing preference be recovered beyond current goal, competence and history in new contexts?",
           "hold goal and competence; change the context", "preference recovery",
           "changed-context choice score", "preference adds nothing beyond goal",
           "preference predicts the changed-context choice", "a goal that is the preference",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"context": ["same", "changed"], "architecture": ["independent", "joint_exact"]},
           families=ALL3, endpoints=["changed_context_choice"], architectures=CORE_ARCH,
           work_weight=1.4),
        _c("V02", "V", 1, "Which observations distinguish a changed preference from a changed foreground goal?",
           "cross the two interventions; observe further episodes", "change discrimination",
           "posterior separation across episodes", "the two are indistinguishable",
           "later episodes separate them", "an episode marker that differs",
           claim_class="BOUNDARY", sesoi=0.15, sesoi_basis="posterior separation above the two-way floor",
           factors={"rival": ["changed_preference", "changed_goal"], "episodes": ["1", "4"]},
           families=["chain", "composition"], endpoints=["next_episode_first_choice"],
           architectures=[], work_weight=1.2),
        _c("V03", "V", 1, "Which distinguish changed preference from lagging expertise residue?",
           "reversal, devaluation and relearning endpoints", "residue discrimination",
           "posterior separation and endpoint score", "the two are indistinguishable",
           "devaluation separates them", "a residue defined as a preference",
           claim_class="BOUNDARY", sesoi=0.15, sesoi_basis="posterior separation above the two-way floor",
           factors={"rival": ["changed_preference", "stale_residue"], "probe": ["none", "devalue", "relearn"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.2),
        _c("V04", "V", 1, "Which distinguish changed preference from better concealment?",
           "one planner; public and private choices at matched audience cost", "concealment discrimination",
           "public-private divergence and posterior separation", "the two are indistinguishable",
           "private choices separate them", "a concealment template visible in the public choice",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.15, sesoi_basis="posterior separation above the two-way floor",
           factors={"rival": ["changed_preference", "concealment"], "visibility": ["public", "private"]},
           families=["chain", "communication"], endpoints=["changed_context_choice"],
           architectures=[], work_weight=1.4),
        _c("V05", "V", 2, "Does paying avoidable cost identify preference, low competence, constraint, signalling or a different cost function?",
           "cross the five cost owners", "cost attribution", "cost-vector posterior and held-out choice",
           "cost is uninformative about the owner", "the posterior separates some owners and not others",
           "reading cost as preference", claim_class="BOUNDARY", sesoi=0.15,
           sesoi_basis="posterior mass above the five-way chance floor of 0.20",
           factors={"owner": ["preference", "competence", "constraint", "signalling", "cost_function"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.2),
        _c("V06", "V", 2, "Do forgone alternatives add information beyond selected action and scalar cost?",
           "vary the available option set", "opportunity information",
           "preference alignment with and without the option set", "forgone options add nothing",
           "the opportunity-aware reader aligns better", "an option set that encodes the answer",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="alignment gain, cosine units",
           factors={"reader": ["cost_only", "opportunity_aware"], "availability": ["full", "partial"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.0),
        _c("V07", "V", 2, "Can dated works recover directional change that undated expertise traces cannot?",
           "dated and undated records of the same trajectory", "trajectory recovery",
           "change-point error", "dating adds nothing", "dating recovers the direction",
           "a date that is also a content cue", claim_class="SIMULATOR_DISCOVERY", sesoi=1.0,
           sesoi_basis="episodes of change-point error",
           factors={"record": ["dated", "bag"], "change": ["none", "midway"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.0),
        _c("V08", "V", 3, "When several reward functions predict every observed choice, does the reader retain their equivalence class?",
           "sample the feasible reward set", "feasible-set retention",
           "coverage and downstream policy regret", "the set collapses",
           "the set is retained and covers the truth", "a point estimate reported as a set",
           claim_class="BOUNDARY", sesoi=0.85, sesoi_basis="coverage the retained set must reach",
           factors={"record": ["optimal_only", "varied_competence", "with_errors"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.0),
        _c("V09", "V", 3, "Can suboptimal or mistaken actions shrink the compatible preference set?",
           "records at three competence levels", "informative imperfection",
           "feasible-set size by record type", "mistakes do not shrink the set",
           "varied competence shrinks it", "a mistake that reveals the answer directly",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="feasible-set coverage difference",
           factors={"record": ["optimal_only", "varied_competence", "with_errors"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.0),
        _c("V10", "V", 3, "Does any persistent-tendency estimate improve next edit, stopping and changed-context choice across two families?",
           "three prospective endpoints, two families", "tendency value",
           "score gain on each endpoint", "tendency adds nothing anywhere",
           "tendency helps on at least the changed-context endpoint", "a label standing in for the estimate",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"endpoint": ["next_edit", "stop_or_continue", "changed_context_choice"],
                    "reader": ["label_only", "joint_exact"]},
           families=["chain", "composition"], endpoints=["next_edit", "stop_or_continue", "changed_context_choice"],
           architectures=CORE_ARCH, lanes=["discovery", "transfer"], work_weight=1.8),
    ]

    # ---- S: strategic sources, affect ownership, uptake (10) -------------------------------- #
    c += [
        _c("S01", "S", 1, "Can V14's exact artifact collision and chance abstention be reproduced without template leakage?",
           "motives grouped into surface profiles; artifact only", "static boundary anchor",
           "within-collision-pair accuracy from the artifact", "the artifact separates the pair",
           "the artifact sits at within-pair chance", "a template that marks the motive",
           claim_class="BOUNDARY", sesoi=0.08, sesoi_basis="deviation from the 0.50 within-pair floor",
           factors={"profile": ["0", "1"], "evidence": ["artifact_only"]},
           families=["communication"], family_bound=True, endpoints=[], architectures=[],
           work_weight=1.0),
        _c("S02", "S", 1, "Can sincere fear and strategic fear be generated by the same noisy planner with different higher-order objectives?",
           "one planner, two objectives, matched surface statistics", "matched generation",
           "surface, intensity and evidence divergence between motives", "the surfaces differ",
           "the surfaces match and the objectives differ", "an intensity difference doing the work",
           claim_class="CONSTRUCTION_IDENTITY", sesoi=0.05, sesoi_basis="tolerated surface divergence",
           factors={"motive": ["sincere", "strategic"], "channel": ["assertion", "evidence", "correction", "private"]},
           families=["communication"], family_bound=True, endpoints=[], architectures=[],
           work_weight=1.0),
        _c("S03", "S", 1, "Which counterfactual opportunity best separates them: audience already persuaded, private cost, correction or evidence choice?",
           "buy each probe; measure information and accuracy", "probe value",
           "information gain and accuracy per probe", "no probe separates them",
           "at least one probe separates them", "a probe that reads the belief directly",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.15, sesoi_basis="accuracy above the within-pair floor",
           factors={"probe": ["audience_persuaded", "private_cost", "correction", "evidence_choice"]},
           families=["communication"], family_bound=True, endpoints=["next_evidence_selection"],
           architectures=[], work_weight=1.4),
        _c("S04", "S", 2, "Does the discriminator survive when private action is only probabilistically related to belief?",
           "sweep the planner's noise", "noise curve", "accuracy and abstention by noise level",
           "accuracy is flat in noise", "accuracy falls with noise and abstention rises",
           "a noiseless private channel", claim_class="BOUNDARY", sesoi=0.15,
           sesoi_basis="accuracy above the within-pair floor",
           factors={"noise": ["0.0", "0.3", "0.6"], "probe": ["private_cost", "correction"]},
           families=["communication"], family_bound=True, endpoints=[], architectures=[],
           work_weight=1.2),
        _c("S05", "S", 2, "Can a fanatic strategically teach and a propagandist privately believe, defeating simple region labels?",
           "cross motive against private belief", "crossed motives",
           "posterior calibration under crossing", "the crossing is not expressible",
           "region labels fail under crossing", "a label that already encodes the crossing",
           claim_class="BOUNDARY", sesoi=0.10, sesoi_basis="calibration error a crossed reader may carry",
           factors={"motive": ["sincere", "strategic"], "belief": ["aligned", "opposed"]},
           families=["communication"], family_bound=True, endpoints=[], architectures=[],
           work_weight=1.2),
        _c("S06", "S", 2, "Does recursive audience modeling help when the maker is strategic and hurt when the assumed audience model is wrong?",
           "cross strategy against audience-model match", "strategy x model interaction",
           "score of the recursive reader relative to face value", "no interaction",
           "recursion helps when matched and hurts when wrong", "an oracle audience model",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"strategy": ["plain", "steering"], "model_match": ["1.0", "0.5", "0.0"]},
           families=["communication"], family_bound=True, endpoints=["next_evidence_selection"],
           architectures=[], work_weight=1.4),
        _c("S07", "S", 2, "Can the reader infer that evidence was selected strategically rather than being a random sample?",
           "score the evidence stream under every selection policy including the null one",
           "selection detection", "selected-policy posterior mass and next-evidence score",
           "selection is undetectable", "selection is detected above chance",
           "a stream length cue", claim_class="SIMULATOR_DISCOVERY", sesoi=0.15,
           sesoi_basis="posterior mass above the uniform floor over selection policies",
           factors={"selection": ["sample_all", "cherry_pick", "balanced", "escalate"]},
           families=["communication"], family_bound=True, endpoints=["next_evidence_selection"],
           architectures=[], work_weight=1.2),
        _c("S08", "S", 3, "Can source motive, content truth, reliability history, induced response and uptake remain factored under contradiction?",
           "intervene on each owner against a standing message", "owner factoring",
           "side-effect matrix off-diagonal mass", "the owners move together",
           "the factored gate stays near-diagonal and the scalar gate does not",
           "a gate that is diagonal because nothing moved", claim_class="SIMULATOR_DISCOVERY",
           sesoi=0.05, sesoi_basis="off-diagonal movement a factored gate may show",
           factors={"gate": ["factored", "scalar"],
                    "owner": ["source_motive", "content_truth", "reliability_history", "induced_response"]},
           families=["communication"], family_bound=True, endpoints=[], architectures=[],
           work_weight=0.8),
        _c("S09", "S", 3, "Does correct motive inference improve selective uptake without blanket distrust or copying?",
           "cross content truth against source strategy", "selective uptake",
           "true minus false uptake, and negative transfer", "the gates are equally selective",
           "the factored gate is more selective at less negative transfer",
           "a reader that can see the truth", claim_class="SIMULATOR_DISCOVERY", sesoi=0.05,
           sesoi_basis="selectivity gap in uptake units",
           factors={"gate": ["factored", "scalar"], "content": ["true", "false"],
                    "source": ["sincere", "strategic"]},
           families=["communication"], family_bound=True, endpoints=[], architectures=[],
           work_weight=1.0),
        _c("S10", "S", 3, "Do source results transfer to unseen strategies and fresh audience models?",
           "frozen reader, untouched selection policies and audience models", "frozen transfer",
           "accuracy on unseen strategies", "results do not transfer",
           "results transfer with a stated loss", "retuning on the transfer lineage",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.15, sesoi_basis="accuracy above the within-pair floor",
           factors={"strategy": ["seen", "unseen"], "audience": ["seen", "fresh"]},
           families=["communication"], family_bound=True, endpoints=["next_evidence_selection"],
           architectures=[], lanes=["discovery", "transfer"], work_weight=1.4),
    ]

    # ---- R: route reliability, shared causes, robust transfer (8) --------------------------- #
    c += [
        _c("R01", "R", 1, "When does learned route weighting beat equal weighting by a practically live amount?",
           "cross reliability dispersion with evidence dose", "reliability boundary",
           "learned minus equal, nats, per cell", "learned never beats equal",
           "an onset in dispersion", "a dispersion that also changes total information",
           claim_class="BOUNDARY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"dispersion": ["0.05", "0.2", "0.35"], "dose": ["2", "8"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.2),
        _c("R02", "R", 1, "Can reliability be learned from sparse predictive feedback without target labels at test?",
           "withhold target labels; vary feedback sparsity", "sparse-feedback learning",
           "held-out score and calibration by sparsity", "sparse feedback teaches nothing",
           "reliability is learnable to a stated sparsity", "target labels leaking at test",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"sparsity": ["0.0", "0.5", "0.8"], "weighter": ["equal", "learned"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.2),
        _c("R03", "R", 1, "Can the reader infer duplicate and shared-cause evidence rather than being told the correlation graph?",
           "duplicate one route; supply no graph", "shared-cause recovery",
           "pair recall and confidence inflation", "the duplication is undetectable",
           "the shared cause is recovered and inflation is corrected", "being handed the graph",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.60, sesoi_basis="recall of the true shared-cause pairs",
           factors={"duplicated": ["no", "yes"], "fuser": ["naive", "shared_cause"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.0),
        _c("R04", "R", 2, "Does robust or minimax weighting beat empirical reliability under domain shift?",
           "shift the reliabilities; compare weighters", "robust transfer",
           "transfer regret and calibration", "robust weighting never pays",
           "robust weighting pays after a shift and costs in domain", "a shift that also changes the task",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"shift": ["0.0", "0.5", "1.0"], "weighter": ["learned", "robust"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.2),
        _c("R05", "R", 2, "When should a reader reset, partially transfer, or retain old route weights?",
           "sweep shift size against the three policies", "shift phase diagram",
           "score by shift and policy", "the three policies tie everywhere",
           "the best policy changes with shift size", "a pooled mean over a sign change",
           claim_class="BOUNDARY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"shift": ["0.0", "0.25", "0.5", "1.0"], "policy": ["retain", "partial", "reset"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.4),
        _c("R06", "R", 2, "Can an adversarially easy route capture a learned reader without corrupting its held-out accuracy record?",
           "plant an easy uninformative route; vary feedback sparsity", "ease trap",
           "weight on the easy route and score cost", "the trap does not spring",
           "the ease-driven reader is captured and the learned one is not",
           "an ease-driven reader that is a straw rival", claim_class="SIMULATOR_DISCOVERY",
           sesoi=0.10, sesoi_basis="score cost in nats of being captured",
           factors={"weighter": ["learned", "ease_driven"], "sparsity": ["0.0", "0.8"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.2),
        _c("R07", "R", 3, "Is costly forensic access worth purchasing under prior ambiguity?",
           "five purchase policies against a cost", "purchase policy",
           "realized gain per cost and purchase rate", "no policy beats never buying",
           "a policy beats both never and always", "a cost that was never debited",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="gain per unit cost",
           factors={"policy": ["never", "always", "fixed", "eig", "robust_eig"], "cost": ["0.02", "0.10"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.4),
        _c("R08", "R", 3, "Do route weights help only as bounded processing shortcuts, or change the final exact posterior?",
           "compare full-budget and truncated-budget fusion", "shortcut or answer",
           "advantage at full and finite budget", "weights change the answer at full budget",
           "weights matter only under a finite budget", "a budget that was not actually bounded",
           claim_class="BOUNDARY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"budget": ["full", "finite"], "weighter": ["equal", "learned"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.0),
    ]

    # ---- F: change-aware epistemic foraging (10) -------------------------------------------- #
    f_pol = ["random", "surprise", "progress", "changepoint", "eig", "robust_eig", "gain_per_cost"]
    c += [
        _c("F01", "F", 1, "Does V14's learning-progress noise avoidance and silent-change failure reproduce?",
           "mixed ecology of learnable, familiar-noise and silent-change items", "anchor",
           "fraction of looks on noise and held-out gain", "progress does not avoid noise",
           "progress avoids noise and does not gain more", "a noise item that is actually learnable",
           claim_class="BOUNDARY", sesoi=0.10, sesoi_basis="difference in the fraction of looks on noise",
           factors={"policy": ["random", "surprise", "progress"], "ecology": ["mixed"]},
           families=["chain"], family_bound=True, endpoints=["realized_gain_per_cost"],
           architectures=[], work_weight=1.0),
        _c("F02", "F", 1, "Can a changepoint-aware progress rule re-engage after silent law changes?",
           "silent changes; a detector that discounts stale evidence", "changepoint repair",
           "held-out gain and detection count", "the detector never fires",
           "the detector fires and gain improves over plain progress", "a detector given the change time",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.05, sesoi_basis="held-out gain in nats over plain progress",
           factors={"policy": ["progress", "changepoint"], "ecology": ["silent_change", "mixed"]},
           families=["chain"], family_bound=True, endpoints=["realized_gain_per_cost"],
           architectures=[], work_weight=1.2),
        _c("F03", "F", 1, "Does raw surprise remain useful for change detection while harmful on unlearnable noise?",
           "cross noise against change", "surprise crossed", "gain and look allocation per ecology",
           "surprise behaves the same in both", "surprise helps on change and hurts on noise",
           "a noise ecology with hidden structure", claim_class="BOUNDARY", sesoi=0.10,
           sesoi_basis="difference in the fraction of looks on noise",
           factors={"policy": ["surprise", "progress", "gain_per_cost"],
                    "ecology": ["noise", "silent_change", "mixed"]},
           families=["chain"], family_bound=True, endpoints=["realized_gain_per_cost"],
           architectures=[], work_weight=1.2),
        _c("F04", "F", 2, "Does robust expected information gain improve decisions under prior ambiguity?",
           "ambiguity set of priors", "robust design", "regret and gain per cost against ordinary EIG",
           "robustness costs and never pays", "robustness pays under ambiguity",
           "an ambiguity set that contains the truth by construction",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.05, sesoi_basis="held-out gain in nats",
           factors={"policy": ["eig", "robust_eig"], "ambiguity": ["none", "wide"]},
           families=["chain"], family_bound=True, endpoints=["realized_gain_per_cost"],
           architectures=[], work_weight=1.2),
        _c("F05", "F", 2, "When should a probe target a latent value versus the reader's model structure?",
           "separate value information from structure information", "probe targeting",
           "information about state and about model class, separately", "the two are the same quantity",
           "they diverge and the better target depends on the ecology", "conflating the two",
           claim_class="METHOD", sesoi=0.10, sesoi_basis="divergence between the two information terms",
           factors={"target": ["value", "structure"], "ecology": ["learnable", "mixed"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.0),
        _c("F06", "F", 2, "Can model-expansion value prevent endless probing of a misspecified but learnable-looking item?",
           "a misspecified item that looks learnable", "endless probing",
           "looks spent and posterior predictive gain", "the controller probes forever either way",
           "expansion value stops the probing", "an item that is obviously unlearnable",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.15, sesoi_basis="reduction in the fraction of looks spent",
           factors={"selector": ["residual", "expected_value"], "item": ["learnable", "misspecified"]},
           families=["chain"], family_bound=True, endpoints=["realized_gain_per_cost"],
           architectures=["expand"], work_weight=1.2),
        _c("F07", "F", 3, "Does compressibility add anything after reducible prediction error and cost are known?",
           "regress gain on compressibility given error and cost", "unique contribution",
           "conditional contribution of compressibility", "compressibility adds nothing",
           "compressibility adds a measurable increment", "compressibility correlated with error",
           claim_class="METHOD", sesoi=0.05, sesoi_basis="conditional increment in explained gain",
           factors={"ecology": ["learnable", "noise", "mixed"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.0),
        _c("F08", "F", 3, "Can the forager distinguish hoped-for truth from warranted probability while still pursuing it?",
           "a hoped-for hypothesis pulls query allocation only", "pursuit versus warrant",
           "query share and posterior fidelity, separately", "the posterior follows the allocation",
           "allocation follows the hope and the posterior does not", "a posterior that never moves at all",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.20, sesoi_basis="query share above the uniform share",
           factors={"hoped": ["yes", "no"], "ecology": ["mixed"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.0),
        _c("F09", "F", 3, "Does gain-per-cost abstain under no-information, excessive-cost and already-resolved conditions?",
           "three null ecologies", "abstention", "abstention rate per null ecology",
           "the controller never abstains", "it abstains in all three",
           "an abstention floor tuned to the ecology", claim_class="SIMULATOR_DISCOVERY",
           sesoi=0.70, sesoi_basis="abstention rate required in a null ecology",
           factors={"ecology": ["noise", "expensive", "resolved"], "policy": ["gain_per_cost", "surprise"]},
           families=["chain"], family_bound=True, endpoints=["realized_gain_per_cost"],
           architectures=[], work_weight=1.0),
        _c("F10", "F", 3, "Which policy transfers across stationary, drifting, adversarial and sparse-feedback ecologies?",
           "frozen tournament on untouched ecologies", "frozen transfer",
           "worst-case regret across ecologies", "no policy transfers",
           "one policy has the smallest worst-case regret", "tuning per ecology",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.05, sesoi_basis="worst-case regret in nats",
           factors={"policy": f_pol, "ecology": ["learnable", "noise", "silent_change", "mixed"]},
           families=["chain"], family_bound=True, endpoints=["realized_gain_per_cost"],
           architectures=[], lanes=["discovery", "transfer"], work_weight=1.6),
    ]

    # ---- H: hierarchy, collaboration, role-relative control (8) ----------------------------- #
    h_top = ["central", "distributed", "editor_ratifier", "independent"]
    c += [
        _c("H01", "H", 1, "Do exact reward-equivalent hierarchies remain indistinguishable from behaviour alone?",
           "briefs differing by a constant shift", "equivalence anchor",
           "behavioural divergence between equivalent briefs", "they are distinguishable",
           "they are exactly indistinguishable", "an equivalence that is only approximate",
           claim_class="BOUNDARY", sesoi=1e-9, sesoi_basis="floating-point identity",
           factors={"equivalence": ["exact", "approximate"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=0.8),
        _c("H02", "H", 1, "Can approximate equivalence be represented as graded class membership?",
           "briefs perturbed by a declared epsilon", "graded membership",
           "class coverage and predictive regret", "membership is all or nothing",
           "membership grades with the perturbation", "a hard threshold reported as a grade",
           claim_class="BOUNDARY", sesoi=0.85, sesoi_basis="class coverage a reader must retain",
           factors={"epsilon": ["0.0", "0.15", "0.35"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=0.8),
        _c("H03", "H", 1, "Can a director and an equivalent shared brief remain matched on all local goal-allocation rules?",
           "central and distributed topologies, artifact only", "artifact boundary",
           "topology posterior from the artifact", "the artifact identifies the topology",
           "the artifact is near chance", "an artifact statistic that encodes issuance",
           claim_class="BOUNDARY", sesoi=0.15, sesoi_basis="posterior mass above the four-way floor of 0.25",
           factors={"topology": h_top, "evidence": ["artifact"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.0),
        _c("H04", "H", 2, "Which process-record intervention first separates actor identity from an upstream organizing constraint?",
           "swap a role, change the brief, remove the ratifier", "resolving intervention",
           "artifact and issuer shift per intervention", "no intervention separates them",
           "one intervention separates them first", "an intervention that changes the task",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="shift in total-variation units",
           factors={"intervention": ["swap_role", "change_brief", "remove_editor"]},
           families=["chain"], family_bound=True, endpoints=["next_intervention_issuer"],
           architectures=[], work_weight=1.0),
        _c("H05", "H", 2, "Can subordinate competence and habits explain local residues without inventing director goals?",
           "score a role model against a director model on the same rows", "role factors",
           "role-model advantage in log likelihood", "the director model wins",
           "the role model explains the residues", "a director model given less information",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="log-likelihood advantage per row",
           factors={"topology": h_top}, families=["chain"], family_bound=True,
           endpoints=[], architectures=[], work_weight=1.0),
        _c("H06", "H", 2, "Do lower-level goals become higher-order constraints for subordinate roles in a recoverable way?",
           "measure cross-role dependency and predict a fresh subtask", "cross-role dependency",
           "dependency by topology and fresh-subtask score", "no dependency exists",
           "dependency exists in the constrained topologies", "an autocorrelation from the action alphabet",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="dependency difference between topologies",
           factors={"topology": h_top}, families=["chain"], family_bound=True,
           endpoints=[], architectures=[], work_weight=1.0),
        _c("H07", "H", 3, "Can a central controller, distributed shared model, editor-ratifier and independent contributors be separated under surface matching?",
           "four topologies; artifact matched, process record available", "topology recovery",
           "topology posterior with abstention on equivalent pairs", "all four are confusable",
           "some separate and the equivalent pair is retained as a class",
           "naming a member of an equivalent pair", claim_class="BOUNDARY", sesoi=0.15,
           sesoi_basis="posterior mass above the four-way floor of 0.25",
           factors={"topology": h_top, "evidence": ["artifact", "process"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.2),
        _c("H08", "H", 3, "Does topology recovery transfer to an untouched team size and dependency graph?",
           "frozen reader; fresh team size", "frozen transfer", "accuracy on the untouched team",
           "recovery does not transfer", "recovery transfers with a stated loss",
           "retuning on the transfer team", claim_class="SIMULATOR_DISCOVERY", sesoi=0.15,
           sesoi_basis="accuracy above the four-way floor",
           factors={"topology": h_top, "team": ["seen", "fresh"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[],
           lanes=["discovery", "transfer"], work_weight=1.2),
    ]

    # ---- P: prospective synthesis and Sounding Line rulers (8) ------------------------------ #
    c += [
        _c("P01", "P", 4, "Does a context-realized maker state beat correct labels on next action?",
           "matched raw evidence; label reader given the true labels", "state versus label",
           "log-score gap on the hidden next action", "labels are sufficient",
           "context-realized state beats labels", "a label reader denied evidence the state reader has",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"reader": ["label_only", "oracle_state"], "dose": ["2", "8"]},
           families=ALL3, endpoints=["next_action"], architectures=CORE_ARCH, work_weight=1.4),
        _c("P02", "P", 4, "Does it beat labels on next edit and stopping?",
           "two endpoints in the composition family", "state versus label, two endpoints",
           "log-score gap per endpoint and calibration", "labels are sufficient on both",
           "state wins on at least one", "an endpoint that ignores context",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"endpoint": ["next_edit", "stop_or_continue"], "reader": ["label_only", "oracle_state"]},
           families=["composition", "chain"], endpoints=["next_edit", "stop_or_continue"],
           architectures=CORE_ARCH, work_weight=1.4),
        _c("P03", "P", 4, "Does it beat labels on changed-context choice?",
           "intervene on the context", "transfer intervention", "log-score gap after the intervention",
           "labels transfer as well", "state transfers better", "an intervention the label encodes",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.015, sesoi_basis="5% of the oracle-minus-surface span",
           factors={"context": ["same", "changed"], "reader": ["label_only", "oracle_state"]},
           families=ALL3, endpoints=["changed_context_choice"], architectures=CORE_ARCH,
           work_weight=1.4),
        _c("P04", "P", 4, "What unique, redundant and synergistic predictive information comes from process, foreground goal and persistent tendency?",
           "exact PID on two sources and exact Shapley on three", "information decomposition",
           "PID atoms and Shapley values", "one component carries everything",
           "the decomposition is non-trivial", "an assumed all-in-one result",
           claim_class="METHOD", sesoi=0.05, sesoi_basis="atom magnitude worth reporting, in nats",
           factors={"component": ["process", "goal", "tendency"], "dose": ["2", "8"]},
           families=ALL3, endpoints=["next_action"], architectures=["joint_exact"], work_weight=1.6),
        _c("P05", "P", 4, "Does removing any one component create a characteristic error rather than a generic score loss?",
           "ablate each component; look at the error topology", "error topology",
           "error-profile divergence between ablations", "all ablations look alike",
           "each ablation has its own error signature", "a score difference read as a signature",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="divergence between ablation error profiles",
           factors={"ablated": ["none", "process", "goal", "tendency"]},
           families=ALL3, endpoints=["next_action"], architectures=["joint_exact"], work_weight=1.6),
        _c("P06", "P", 4, "Can the reader predict epistemic exploration rather than using it as a residual label for unexplained action?",
           "explore and not-explore episodes with matched surprise", "exploration prediction",
           "explore/not-explore score and method-change score", "exploration is a residual category",
           "exploration is predicted before it happens", "labelling every unexplained action exploration",
           claim_class="SIMULATOR_DISCOVERY", sesoi=0.10, sesoi_basis="accuracy above the two-way floor",
           factors={"kind": ["exploration", "mistake", "habit_out_of_context"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=1.2),
        _c("P07", "P", 5, "Which rulers are licensed, partial, deferred or killed for Sounding Line Stage 6?",
           "walk every promoted flight against access, construction gate, cheap rival, endpoint and ceiling",
           "bridge disposition", "one disposition per ruler", "nothing is licensed",
           "a bounded set is licensed", "exporting a construction identity",
           claim_class="BRIDGE_CANDIDATE", sesoi=0.0, sesoi_basis="a disposition, not a magnitude",
           unit_kind="single", causal=False, gates_required=["positive"],
           factors={"disposition": ["license", "partial", "defer", "kill"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=0.3),
        _c("P08", "P", 5, "Can a compact public benchmark be regenerated from committed constructed worlds without oracle fields?",
           "regenerate splits from the committed record; audit for leaks", "benchmark regeneration",
           "split hashes, baseline reproduction and leak audit", "the benchmark leaks an oracle field",
           "it regenerates and no oracle field survives", "a benchmark that ships the answers",
           claim_class="BRIDGE_CANDIDATE", sesoi=0.0, sesoi_basis="exact: no oracle field may survive",
           unit_kind="single", causal=False, gates_required=["positive", "placebo"],
           factors={"check": ["hashes", "baseline", "leak_audit"]},
           families=ALL3, endpoints=[], architectures=[], work_weight=0.4),
    ]

    # ---- B: closure and program-level disposition (2) --------------------------------------- #
    c += [
        _c("B01", "B", 6, "Which results survived discovery, generator transfer, attacks and untouched confirmation?",
           "walk the committed record", "survival ledger", "one row per flight",
           "nothing survives", "a stated set survives", "a failed criterion quietly dropped",
           claim_class="METHOD", sesoi=0.0, sesoi_basis="a ledger, not a magnitude",
           unit_kind="single", causal=False, gates_required=["positive"],
           factors={"stage": ["discovery", "transfer", "attack", "confirmation"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=0.3),
        _c("B02", "B", 6, "Is the next justified action another Ghost version, a Sounding transfer, a human or model study, or pause?",
           "pursuit, warrant and publication ledgers", "program disposition", "a recommendation",
           "no recommendation is warranted", "a recommendation with its reasons",
           "a recommendation that ignores the nulls", claim_class="METHOD", sesoi=0.0,
           sesoi_basis="a disposition, not a magnitude", unit_kind="single", causal=False,
           gates_required=["positive"],
           factors={"branch": ["ghost_v16", "sounding_transfer", "model_study", "human_study", "pause"]},
           families=["chain"], family_bound=True, endpoints=[], architectures=[], work_weight=0.3),
    ]

    # ---- X: the cross-cutting adversarial matrix (24) --------------------------------------- #
    attacks = [
        ("X01", "Surface names, colors, token identities and action labels permuted."),
        ("X02", "Route overlap changed while marginal route information is matched."),
        ("X03", "One route removed and another duplicated."),
        ("X04", "Shared-cause evidence presented as independent paraphrases."),
        ("X05", "Likelihood family wrong but superficially well calibrated in training."),
        ("X06", "Missing true latent variable."),
        ("X07", "Irrelevant extra latent and distractor hypotheses."),
        ("X08", "Context and opportunity sets hidden."),
        ("X09", "Policy temperature and competence shifted."),
        ("X10", "Exact and approximate equifinality."),
        ("X11", "Maker-reader pairwise similarity reversed while family typicality is fixed."),
        ("X12", "Attention allocation swapped with imposed constraint."),
        ("X13", "Training path randomized with final skill rematched."),
        ("X14", "Current preference reversed while expertise residue persists."),
        ("X15", "Foreground goal interrupted and rapidly restored."),
        ("X16", "Source audience already persuaded or absent."),
        ("X17", "Private action and correction made noisy and costly."),
        ("X18", "Strategic speaker assumes the wrong audience model."),
        ("X19", "Silent changepoint inside an apparently settled item."),
        ("X20", "Unlearnable noise with high surprise and apparent short-run progress."),
        ("X21", "Aggregation constructed to hide a sign reversal."),
        ("X22", "Solver approximation, particle impoverishment and proposal-order changes."),
        ("X23", "Discovery, transfer and confirmation lineage or seed swap attempt."),
        ("X24", "Fast-machine, restart, orphan-kill, stale-checkpoint and clean-clone failure injection."),
    ]
    for xid, what in attacks:
        c.append(_c(xid, "X", 7, f"Does the targeted result survive: {what}", what,
                    "attacked result", "attacked minus unattacked, in the flight's own units",
                    "the attack destroys the result", "the result survives with a stated loss",
                    "an attack that changes the task rather than the reading",
                    claim_class="METHOD", sesoi=0.015,
                    sesoi_basis="5% of the oracle-minus-surface span, or the flight's own bar",
                    factors={"attacked": ["no", "yes"]}, families=ALL3,
                    endpoints=["next_action"], architectures=CORE_ARCH,
                    lanes=["attack"], work_weight=1.0))
    # Every causal card declares at least one hidden event. The assignments below name the
    # quantities the spec's card table describes in prose -- a held-out history, a relearning
    # curve, a switch time, a team topology -- so that no substantive card reaches the validator
    # with an empty endpoint list and gets read as retrospective.
    late = ENDPOINT_ASSIGNMENTS
    for card in c:
        if card.causal and not card.endpoints and card.id in late:
            card.endpoints = list(late[card.id])
    return c


ENDPOINT_ASSIGNMENTS = {'E01': ['transfer_breadth'], 'E02': ['held_out_history'], 'E03': ['hidden_error_location'], 'E04': ['hidden_error_location'], 'E06': ['held_out_history'], 'E07': ['transfer_breadth'], 'E08': ['relearning_curve'], 'E09': ['hidden_error_location'], 'E10': ['held_out_history'], 'E12': ['held_out_history'], 'G01': ['collision_residual'], 'G03': ['switch_time'], 'G04': ['cross_goal_dependency'], 'G05': ['deviation_continuation'], 'G07': ['deviation_continuation'], 'G08': ['method_change'], 'V02': ['next_episode_first_choice'], 'V03': ['relearning_curve'], 'V05': ['cost_owner'], 'V06': ['cost_owner'], 'V07': ['change_point'], 'V08': ['feasible_set'], 'V09': ['feasible_set'], 'S01': ['source_motive'], 'S02': ['collision_residual'], 'S04': ['source_motive'], 'S05': ['source_motive'], 'S07': ['selection_policy'], 'S08': ['source_motive'], 'S09': ['source_motive'], 'R01': ['route_weighting'], 'R02': ['route_weighting'], 'R03': ['route_weighting'], 'R04': ['route_weighting'], 'R05': ['route_weighting'], 'R06': ['route_weighting'], 'R07': ['realized_gain_per_cost'], 'R08': ['route_weighting'], 'F05': ['expansion_decision'], 'F07': ['held_out_gain'], 'F08': ['held_out_gain'], 'H01': ['team_topology'], 'H02': ['team_topology'], 'H03': ['team_topology'], 'H05': ['team_topology'], 'H06': ['team_topology'], 'H07': ['team_topology'], 'H08': ['team_topology'], 'M04': ['expansion_decision'], 'M09': ['expansion_decision'], 'M10': ['expansion_decision'], 'P06': ['deviation_continuation'], 'E05': ['changed_context_choice'], 'E11': ['next_episode_first_choice']}


# --------------------------------------------------------------------------- #
# Files.
# --------------------------------------------------------------------------- #
def mandatory(cards=None) -> list:
    return [c for c in (cards or build_cards()) if c.trunk != "X"]


def attacks(cards=None) -> list:
    return [c for c in (cards or build_cards()) if c.trunk == "X"]


def lineages(tier: dict | None = None) -> dict:
    from . import common as C
    t = tier or TIERS["T0"]
    ids = {ln: C.lane_ids(ln, t) for ln in ("discovery", "transfer", "confirmation", "coverage")}
    return {"ids": {k: [v[0], v[-1]] if v else [] for k, v in ids.items()},
            "counts": {k: len(v) for k, v in ids.items()},
            "disjoint": C.lineage_disjoint(ids)}


def write_manifest(cards=None, note: str = "") -> dict:
    cs = cards or build_cards()
    doc = {"program": "v15", "note": note, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "n_cards": len(mandatory(cs)), "n_attacks": len(attacks(cs)),
           "trunks": sorted({c.trunk for c in cs}),
           "cards": {c.id: c.to_dict() for c in cs}}
    write_json_atomic(MANIFEST, doc)
    return doc


def write_cells_template(cards=None) -> dict:
    cs = cards or build_cards()
    doc = {"program": "v15", "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "template": {c.id: {ln: expected_cells(c, TIERS["T0"], ln) for ln in c.lanes}
                        for c in cs}}
    write_json_atomic(CELLS_TEMPLATE, doc)
    return doc


def instantiate_cells(tier_name: str, tier: dict, cards=None) -> dict:
    cs = cards or build_cards()
    doc = {"program": "v15", "tier": tier_name, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "cells": {c.id: {ln: expected_cells(c, tier, ln) for ln in c.lanes} for c in cs}}
    write_json_atomic(CELLS, doc)
    return doc


def write_generator_families() -> dict:
    from . import world_chain, world_communication, world_composition
    doc = {"program": "v15", "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "families": {m.FAMILY: {"module": m.__name__, "routes": list(m.ROUTES),
                                   "n_tokens": m.N_TOKENS, "n_actions": m.N_ACTIONS,
                                   "home": dict(getattr(m, "HOME", {}))}
                        for m in (world_chain, world_composition, world_communication)},
           "independence_claim": ("shared ontology and a shared reader surface; independent "
                                  "latent priors, transitions and emissions. Audited by I06.")}
    write_json_atomic(GENERATOR_FAMILIES, doc)
    return doc


def write_construction_graph(cards=None) -> dict:
    cs = cards or build_cards()
    doc = {"program": "v15", "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "nodes": {c.id: {"trunk": c.trunk, "claim_class": c.claim_class,
                            "endpoints": c.endpoints, "families": c.families,
                            "depends_on": c.depends_on} for c in cs}}
    write_json_atomic(CONSTRUCTION_GRAPH, doc)
    return doc


def write_architecture_budgets() -> dict:
    from .architectures import ALL as ARCH
    doc = {"program": "v15", "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "architectures": list(ARCH),
           "non_promotable": ["oracle_state"],
           "matching_rule": ("every architecture comparison reports information-matched (same raw "
                             "observations) and budget-matched (likelihood evaluations within a "
                             "declared relative tolerance) results; an untouched budget counter is "
                             "refused at reduce time"),
           "tolerance": 0.25}
    write_json_atomic(ARCHITECTURE_BUDGETS, doc)
    return doc


def write_source_lineages(tier: dict | None = None) -> dict:
    doc = {"program": "v15", "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **lineages(tier)}
    write_json_atomic(SOURCE_LINEAGES, doc)
    return doc


def write_publication_template(cards=None) -> dict:
    from .schemas import PUBLICATION_FIELDS
    cs = cards or build_cards()
    doc = {"program": "v15", "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "fields": list(PUBLICATION_FIELDS),
           "rows": {c.id: {f: None for f in PUBLICATION_FIELDS} for c in mandatory(cs)}}
    write_json_atomic(PUBLICATION_MAP_TEMPLATE, doc)
    return doc


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def save_manifest(doc: dict) -> None:
    doc["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json_atomic(MANIFEST, doc)


def get_card(doc: dict, cid: str) -> Card:
    return card_from_dict(doc["cards"][cid])


def update_card(doc: dict, cid: str, **fields) -> None:
    doc["cards"][cid].update(fields)
    save_manifest(doc)


def coverage(doc: dict) -> dict:
    from .schemas import RESOLVED
    cards = doc["cards"]
    by_state, by_trunk, by_criterion = {}, {}, {}
    for cid, d in cards.items():
        by_state[d.get("status", "PLANNED")] = by_state.get(d.get("status", "PLANNED"), 0) + 1
        t = by_trunk.setdefault(d["trunk"], {"n": 0, "resolved": 0})
        t["n"] += 1
        if d.get("status") in RESOLVED:
            t["resolved"] += 1
        cs = d.get("criterion_status", "UNEVALUATED")
        by_criterion[cs] = by_criterion.get(cs, 0) + 1
    mand = [d for d in cards.values() if d["trunk"] != "X"]
    return {"by_state": by_state, "by_trunk": by_trunk, "by_criterion_status": by_criterion,
            "n_cards": len(cards), "n_mandatory": len(mand),
            "n_attacks": len(cards) - len(mand),
            "mandatory_resolved": sum(1 for d in mand if d.get("status") in RESOLVED),
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")}


def write_coverage(doc: dict) -> dict:
    cov = coverage(doc)
    write_json_atomic(COVERAGE_FILE, cov)
    return cov


def add_amendment(card: str, original: dict, replacement: dict, reason: str) -> None:
    doc = json.loads(AMENDMENTS.read_text(encoding="utf-8")) if AMENDMENTS.exists() \
        else {"program": "v15", "amendments": []}
    doc["amendments"].append({"card": card, "reason": reason,
                              "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "original": original, "replacement": replacement})
    write_json_atomic(AMENDMENTS, doc)
