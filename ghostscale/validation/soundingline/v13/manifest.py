"""The V13 queue manifest: every mandatory card, the attack matrix, tiers, lineages, expected
cells, ledgers (spec §5, §7, §8–§17). Written to results/v13/QUEUE_MANIFEST.json; the validator
compares it to the literal card ids in the spec (spec §7.5) and refuses a program with a card
missing, a factor absent, or a floor lowered without an amendment.

A card's ``factors`` are its cell axes: every row a card emits carries these keys, and the
expected-cell receipt counts the realized crossings per independent unit.
"""
from __future__ import annotations

import json
import time

from . import v13_dir
from .schemas import Card, EXPANSIONS, RESOLVED, STATES, TIERS, card_from_dict, expected_cells

MANIFEST = v13_dir() / "QUEUE_MANIFEST.json"
COVERAGE = v13_dir() / "COVERAGE.json"
RUNTIME = v13_dir() / "RUNTIME.json"
CELLS_TEMPLATE = v13_dir() / "EXPECTED_CELLS_TEMPLATE.json"
CELLS = v13_dir() / "EXPECTED_CELLS.json"
WORKLOAD = v13_dir() / "WORKLOAD_LOCK.json"
AMENDMENTS = v13_dir() / "AMENDMENTS.json"
PKG = "ghostscale.validation.soundingline.v13.cards"

ROUTES12 = ["self", "equal_local", "within_common", "within_group", "within_expertise", "all_family",
            "generic_local", "random_local", "permuted_self", "anti_similar", "target_learned", "oracle"]
SIM_BINS = ["near", "mid", "far", "anti"]
DOSES = [1, 2, 4, 8, 16]
LEVELS = ["common", "group", "expertise", "individual", "state", "surface"]
ATT_POL = ["uniform", "random", "salience", "oracle", "wrong", "learned", "adaptive", "narrow", "broad"]
COST_DIMS = ["time", "execution", "cognitive", "epistemic", "opportunity", "social", "risk", "imposed"]
GOALS7 = ["accurate", "comprehension_support", "persuasion", "self_presentation", "concealment", "misleading", "neutral"]
TEAMS = ["central", "shared_brief", "editor_led", "ratifier", "rotating", "institutional", "distributed"]
TRUST = ["bayes", "leaky", "asymmetric", "threshold", "change_point"]


def _c(cid, trunk, wave, question, construction, target, estimand, null, alt, rival, closure="",
       ceiling="CONSTRUCTED_MECHANISM", causal=True, factors=None, lanes=("discovery",), unit="world",
       unit_kind="world", weight=1.0, pilot=False, deps=(), paths=("exact",), prior_routes=(),
       attention=(), cost_ecologies=(), reader_policies=(), min_rows=1, domains=2, families=4):
    gates = ["live", "placebo", "positive", "surface", "oracle", "prediction", "calibration"] if causal else ["identity", "positive", "placebo"]
    return Card(id=cid, trunk=trunk, wave=wave, question=question, construction=construction, target=target,
                estimand=estimand, null_expectation=null, alternative_expectation=alt, strongest_rival=rival,
                claim_ceiling=ceiling, solver_paths=list(paths), depends_on=list(deps), closure=closure, causal=causal,
                gates_required=gates, factors=dict(factors or {}), domains=domains, world_families=families,
                prior_routes=list(prior_routes), attention_policies=list(attention), cost_ecologies=list(cost_ecologies),
                reader_policies=list(reader_policies), lanes=list(lanes), independent_unit=unit, unit_kind=unit_kind,
                min_rows_per_unit=min_rows, work_weight=weight, pilot=pilot,
                module=f"{PKG}.trunk_{trunk.lower()}:{cid}",
                output=f"results/validation/soundingline/v13/{cid}.json",
                checkpoint_key=f"results/v13/checkpoints/<lane>/{cid}/", completion_key=f"<lane>:{cid}")


def build_cards() -> list:
    C = []
    # ---- I: integrity, calibration, construction validity -------------------------------- #
    C += [
        _c("I01", "I", 0, "Do the V12 anchors reproduce without modifying V12?",
           "reconstructed loops over V12's own world and modules for S04, S06, R02, R05, B02, D02, D05, F03",
           "eight committed V12 numbers", "max absolute deviation from the committed verdict fields", "identity within tolerance",
           "a deviation means V12's record cannot be rebuilt", "provenance drift", causal=False, ceiling="METHOD",
           factors={"anchor": ["S04", "S06", "R02", "R05", "B02", "D02", "D05", "F03"]}, unit="anchor", unit_kind="list", weight=3.0),
        _c("I02", "I", 0, "Was V12's self comparator incompletely matched?",
           "V12 self and generic priors recreated on V12 worlds; entropy, expected divergence to truth, parameter count, coordinate access; a distance-matched rebuild",
           "comparator imbalance and the share of S04's near gain it explains", "expected-divergence gap and near-gain share under distance matching",
           "no imbalance", "generic sat closer to truth than self by construction", "the S04 gain was a locality artefact", causal=False, ceiling="METHOD",
           factors={"comparator": ["generic_v12", "distance_matched"]}, unit="v12_world", unit_kind="single", weight=2.0),
        _c("I03", "I", 0, "Can entropy-and-distance-matched local priors be built?",
           "routes 1, 2, 7, 8, 9 constructed for every reader without outcome access; residuals reported",
           "matching tolerances", "entropy gap and expected-divergence gap per route", "all within tolerance",
           "unmatchable priors make trunk C an instrument failure", "a residual that tracks the effect", causal=False, ceiling="METHOD",
           factors={"route": ["equal_local", "generic_local", "random_local", "permuted_self"]}, weight=1.0),
        _c("I04", "I", 0, "Are nested similarity factors independently live?",
           "vary common, group, expertise, individual, state and surface one at a time and in crossed pairs; JS on emission and on protected nuisance statistics",
           "factor liveness", "JS moved per factor; nuisance movement per factor", "every factor moves its mapping and not the protected channels",
           "a dead or leaking factor voids its cards", "a factor that only moves surface", causal=False, ceiling="METHOD",
           factors={"factor": LEVELS, "mode": ["single", "crossed"]}, weight=1.0),
        _c("I05", "I", 0, "Does attention only select or weight evidence?",
           "identity worlds for selection and precision attention; mass conservation; no-information worlds",
           "attention identities", "bit deviation at neutral weights; gain in no-information worlds", "zero and zero",
           "a nonzero would mean attention injects evidence", "a policy that adds a label", causal=False, ceiling="METHOD",
           factors={"analogue": ["selection", "precision"], "world": ["informative", "no_information"]}, weight=1.0),
        _c("I06", "I", 0, "Are cost dimensions independently realized?",
           "factorial time, execution, cognitive, epistemic, opportunity, social, risk and imposed costs; each varied alone",
           "cost-dimension liveness", "choice-distribution shift per dimension; label leakage under matched totals", "each moves utility; no leak",
           "a dead dimension voids its O cards", "dimensions that only move the total", causal=False, ceiling="METHOD",
           factors={"dimension": COST_DIMS}, weight=1.0),
        _c("I07", "I", 0, "Are communicative goals surface-matched?",
           "seven goals at matched token counts, entropy, polish and difficulty; cheap classifier and oracle reader",
           "goal surface matching", "surface classifier accuracy against chance; oracle goal accuracy", "chance and above",
           "a readable surface would make G an artefact", "a goal-specific token histogram", causal=False, ceiling="METHOD",
           factors={"goal": GOALS7}, weight=1.0),
        _c("I08", "I", 0, "Are goal, source reliability, content evidence, and uptake separable?",
           "orthogonal identity construction: change one input, hold the others", "factor separability",
           "movement of each posterior under each single change", "only the declared edge moves", "a coupling is a construction bug",
           "hidden coupling through a shared statistic", causal=False, ceiling="METHOD",
           factors={"changed": ["goal", "source", "content", "uptake_input"]}, weight=1.0),
        _c("I09", "I", 0, "Is the central/shared-brief rival genuinely equivalent?",
           "central and shared-brief teams under the same dependency and correction rules; artifact-only classifier",
           "rival equivalence", "artifact-only accuracy against chance; matching of coherence, counts, quality, surface, final goals", "chance and matched",
           "an unmatched rival is not a rival", "coherence that leaks the team", causal=False, ceiling="METHOD",
           factors={"team": ["central", "shared_brief"]}, weight=1.5),
        _c("I10", "I", 0, "Where do exact and PyMDP paths diverge?",
           "coupling x horizon x policy count x precision sweep with known exact posterior and EIG",
           "solver discrepancy surface", "max posterior deviation and probe disagreement by cell", "identity where factors are independent",
           "a confidently wrong region must be mapped", "the approximation", causal=False, ceiling="METHOD", paths=("exact", "pymdp"), pilot=True,
           factors={"coupling": ["independent", "weak", "strong"], "horizon": [1, 2], "gamma": [4, 16]}, weight=2.0),
        _c("I11", "I", 0, "Are no-information and false-correspondence nulls calibrated?",
           "shuffled maker, shuffled reader, false similarity, false source, label permutation, uniform emission",
           "null calibration", "posterior mass on truth against chance; ECE", "chance and calibrated",
           "structure in a null is a leak", "residual correspondence", causal=False, ceiling="METHOD",
           factors={"null": ["shuffled_maker", "shuffled_reader", "false_similarity", "false_source", "label_permutation", "uniform_emission"]}, weight=1.0),
        _c("I12", "I", 0, "Are seeds, lineages, hashes, caches and completion records reproducible?",
           "two fresh clones of HEAD run the determinism battery in opposite orders; lane ancestry compared; completion ledger validated",
           "reproducibility invariants", "field identity across clones and orders; lineage overlap; ledger validity", "identical, disjoint, valid",
           "a difference stops the program", "process-order dependence", causal=False, ceiling="METHOD", unit_kind="single",
           factors={"check": ["clone_identity", "order_identity", "lineage_disjoint", "ledger"]}, weight=1.0),
        _c("I13", "I", 0, "Does the runtime pilot measure real work?",
           "the discarded twelve-card pilot with parent and child CPU, memory, and forecast; pilot lineage quarantined",
           "pilot validity", "forecast written before discovery; pilot ids absent from scientific lanes; child CPU present", "all hold",
           "a pilot that measured nothing cannot select a tier", "wall time without CPU", causal=False, ceiling="METHOD", unit_kind="single",
           factors={"check": ["forecast_before_discovery", "quarantine", "child_cpu", "tier_in_envelope_or_rule"]}, weight=0.2),
        _c("I14", "I", 0, "Do metamorphic and symmetry relations hold?",
           "relabel goals, actors, families, cost units, option order and graph encodings",
           "invariances", "score change under relabelling; sign under antisymmetric relabelling", "invariant scores invariant; antisymmetries reverse",
           "a violation is a coordinate dependence", "hidden coordinate dependence", causal=False, ceiling="METHOD",
           factors={"relation": ["goal_relabel", "feature_relabel", "actor_relabel", "cost_units", "option_order", "graph_encoding"]}, weight=1.0),
    ]
    # ---- C: common prior and nested similarity -------------------------------------------- #
    C += [
        _c("C01", "C", 1, "Can each reader measure its own production model?",
           "reader produces in every family and two domains; held-out continuations and transitions", "self-model quality",
           "held-out continuation and transition log score against frequency and family priors", "no gain", "self model wins",
           "pooled frequency", factors={"domain": [0, 1], "baseline": ["frequency", "population"]}, unit="reader", weight=1.0),
        _c("C02", "C", 1, "Does the nested similarity ruler recover each planted level?",
           "one-factor and crossed differences at every level; surface similarity separately", "ruler recovery",
           "rank correlation of measured distance with planted difference per level", "no ordering", "monotone per level",
           "a ruler that reads labels", ceiling="METHOD", factors={"level": LEVELS}, weight=0.5),
        _c("C03", "C", 1, "What does a common-substrate prior buy over a broad population?",
           "same evidence and compute; within-common versus all-family priors", "common-prior gain",
           "log-score gain, calibration, evidence savings by target level", "no gain", "gain within the substrate",
           "a family label", factors={"route": ["within_common", "all_family"], "dose": DOSES}, weight=1.5, prior_routes=("within_common", "all_family")),
        _c("C04", "C", 1, "Does measured self beat an equally local non-self prior?",
           "entropy-, distance-, coordinate- and compute-matched local priors", "self privilege",
           "self minus equal-local log score by similarity bin and dose", "tie", "self wins near itself",
           "locality", factors={"route": ["self", "equal_local", "generic_local"], "sim_bin": SIM_BINS, "dose": DOSES}, weight=2.0, pilot=True,
           prior_routes=("self", "equal_local", "generic_local", "permuted_self", "random_local", "oracle"),
           closure="close self privilege if self never beats a truly equal local prior"),
        _c("C05", "C", 1, "Does self beat the within-common population prior?",
           "near, typical, atypical and anti-similar readers; matched information", "self versus common",
           "conditional surface of self minus within-common by reader type", "flat", "one personal sample helps when typical",
           "typicality", factors={"reader_type": ["near", "typical", "atypical", "anti"], "dose": DOSES}, weight=1.5, prior_routes=("self", "within_common")),
        _c("C06", "C", 2, "Does reader typicality govern the value of self-projection?",
           "readers at varying distance from their family mean; optimally weighted common prior", "typicality law",
           "self gain against typicality; optimal common-mixture comparison", "no relation", "typical readers benefit more",
           "an optimally weighted common prior", factors={"typicality_bin": ["low", "mid", "high"], "route": ["self", "optimal_mix"]}, weight=1.5),
        _c("C07", "C", 2, "What does group or convention matching add?",
           "same substrate; true and claimed group crossed", "group matching",
           "within-group gain by true and claimed match", "gain follows the label", "gain follows convention",
           "the label", factors={"true_match": [0, 1], "claimed_match": [0, 1]}, weight=1.5),
        _c("C08", "C", 2, "What does production expertise matching add?",
           "practitioners, observationally familiar readers, novices in one family", "expertise access",
           "log-score gain by reader kind; transition-model access separated from preference similarity", "no difference",
           "practitioners read mechanics", "preference similarity", factors={"reader_kind": ["practitioner", "familiar", "novice"], "target": ["profile", "method"]}, weight=1.5),
        _c("C09", "C", 2, "What does individual history add after group and expertise?",
           "repeated target-specific histories with fresh current goals; relabelled-source control", "target history",
           "held-out continuation gain of the target-learned prior; identity-shortcut control", "no gain", "history helps without identity",
           "source identity memorisation", factors={"history": [0, 2, 8], "control": ["real", "relabelled"]}, weight=1.5, prior_routes=("target_learned", "within_group")),
        _c("C10", "C", 2, "Which similarity axes matter to which inference target?",
           "observation, transition, profile, policy, cost-model and surface similarity crossed with targets", "axis x target map",
           "partial R2 of each axis for each target", "one scalar", "axis-specific", "a universal scalar",
           factors={"axis": ["observation", "transition", "profile", "policy", "cost_model", "surface"], "target": ["goal", "profile", "method", "cost_weights"]}, weight=1.5),
        _c("C11", "C", 2, "Does irrelevant similarity create false projection?",
           "high nuisance match with decision-relevant conflict", "false projection",
           "self weight and confidence against accuracy under nuisance match", "no rise", "confidence rises without accuracy",
           "a calibrated reader", factors={"nuisance_match": [0, 1], "relevant_conflict": [0, 1]}, weight=1.0),
        _c("C12", "C", 2, "Can anti-similarity be learned as a useful transform?",
           "systematic invertible maker transform; difference-from-self model", "learned transform",
           "log score of the transform model against broad population and raw self by dose", "never beats", "beats after learning",
           "broad population", factors={"model": ["raw_self", "difference_from_self", "all_family"], "dose": DOSES}, weight=1.5),
        _c("C13", "C", 2, "Does focusing on common structure improve common-shaped goal recovery?",
           "attention to common axes, individual surface, maker-diagnostic axes, uniform; by causal world", "focus value",
           "goal log score by focus and causal world", "no effect", "helps only when common axes are causal",
           "maker-diagnostic focus", factors={"focus": ["common", "surface", "diagnostic", "uniform"], "causal_world": ["common", "group", "individual", "nuisance"]}, weight=1.5),
        _c("C14", "C", 3, "Does locality predict a hidden continuation?",
           "near, intermediate, far and anti-similar makers; hidden next action", "conditional continuation",
           "self minus generic-local continuation log score by bin before pooling", "null everywhere", "gain near, harm far",
           "pooled mean", factors={"sim_bin": SIM_BINS, "dose": [1, 4, 12]}, weight=1.5, lanes=("discovery", "transfer")),
        _c("C15", "C", 3, "Can several readers correct one another's projection?",
           "independent and correlated readers; vote, Bayesian pool, single", "ensemble correction",
           "ensemble minus best single by correlation", "no help", "helps only with diverse errors",
           "agreement", factors={"correlation": ["independent", "correlated"], "method": ["vote", "bayes", "single"]}, weight=1.5),
        _c("C16", "C", 3, "Does the frozen nested-prior law transfer?",
           "fresh families, groups, expertise, coordinate relabelling, maker distributions", "transfer",
           "the C04/C05 interaction on fresh worlds", "collapse", "same qualitative interaction",
           "overfitting to discovery worlds", factors={"route": ["self", "equal_local", "within_common"], "sim_bin": SIM_BINS}, weight=2.0, pilot=True,
           lanes=("transfer",)),
    ]
    # ---- A: attention and entry ------------------------------------------------------------ #
    C += [
        _c("A01", "A", 1, "Are selection and precision attention distinct but calibrated?",
           "same cue families under finite selection and full-data tempering", "analogue calibration",
           "identity at neutral settings; budget curves of both", "identical curves", "distinct curves, identity at neutral",
           "assumed equivalence", ceiling="METHOD", factors={"analogue": ["selection", "precision"], "budget": [1, 2, 4]}, weight=1.0),
        _c("A02", "A", 1, "Can a reader select the known diagnostic cue?",
           "one diagnostic, several salient-weak, several irrelevant cues", "cue selection",
           "information per cost by policy", "random", "above random and salience", "salience",
           factors={"policy": ["random", "salience", "oracle", "learned"], "budget": [1, 2]}, weight=1.0, attention=ATT_POL),
        _c("A03", "A", 1, "Can precision weighting help with identical available evidence?",
           "all cues visible; learned, oracle, uniform, wrong precision", "precision value",
           "held-out log score by weighting", "no change", "learned improves; wrong exposes confident error",
           "uniform", factors={"weighting": ["uniform", "learned", "oracle", "wrong"]}, weight=1.0),
        _c("A04", "A", 1, "Does common-structure focus help locate common goals?",
           "common-, group-, individual-causal and nuisance worlds", "focus by cause",
           "goal log score of common focus against uniform and oracle by world", "no effect", "helps only in the common-causal world",
           "maker-diagnostic oracle", factors={"causal_world": ["common", "group", "individual", "nuisance"], "focus": ["common", "uniform", "oracle"]}, weight=1.0),
        _c("A05", "A", 2, "Does matching the maker's attention help reconstruct the maker?",
           "reader attention matched, orthogonal, opposite, or inferred", "attention matching",
           "log score by matching when maker attention controls emissions or not", "no effect", "matters only when maker attention controls decisions",
           "supplied attention", factors={"match": ["matched", "orthogonal", "opposite", "inferred"], "controls": [0, 1]}, weight=1.0),
        _c("A06", "A", 2, "Is purpose-first merely a frequent high-yield entry point?",
           "novice and expert readers; purpose, technique, mechanics, anomaly, random entry", "entry point",
           "early and final log score by entry and expertise", "no difference", "goal-first wins generically; mechanics-first for experts",
           "a universal arrow", factors={"entry": ["purpose", "technique", "mechanics", "anomaly", "random"], "expertise": ["novice", "expert"]}, weight=1.0),
        _c("A07", "A", 2, "Do anomalies and mistakes buy information because they expose alternatives?",
           "mistake, unfamiliar technique, forced defect, intentional deviation, none", "anomaly value",
           "classification and held-out response gain from attending to anomalies", "no gain", "gain only when handling differs",
           "surprise itself", factors={"anomaly": ["mistake", "unfamiliar", "forced", "intentional", "none"]}, weight=1.0),
        _c("A08", "A", 2, "Does attending to opportunity and cost correct outcome-only inference?",
           "same outcome under different menus and costs", "opportunity salience",
           "profile prediction gain from menu attention", "no gain", "gain when the menu is relevant",
           "outcome-only", factors={"menu_relevant": [0, 1], "attention": ["outcome_only", "menu"]}, weight=1.0),
        _c("A09", "A", 2, "Can source attention improve trust calibration without overwriting artifact evidence?",
           "true, false, ambiguous, irrelevant histories crossed with strong and weak artifact evidence", "source attention",
           "content and source posteriors by cell", "one overwrites the other", "separate updates; strong artifacts resist false assertions",
           "a scalar trust", factors={"history": ["true", "false", "ambiguous", "irrelevant"], "evidence": ["strong", "weak"]}, weight=1.0),
        _c("A10", "A", 2, "Does surprise trigger useful attention reallocation?",
           "local prior then diagnostic conflict at controlled time", "reallocation",
           "residual projection of adaptive against static readers", "no difference", "adaptive corrects faster",
           "static broad", factors={"reader": ["adaptive", "static_local", "static_broad"], "conflict_at": [2, 6]}, weight=1.0),
        _c("A11", "A", 2, "When does focused attention become tunnel vision?",
           "correct cue early, regime change later; narrow versus broad", "tunnel vision",
           "post-change error of narrow against broad", "no difference", "narrow persists in error",
           "narrow precision", factors={"monitoring": ["narrow", "broad"], "change": [0, 1]}, weight=1.0),
        _c("A12", "A", 2, "Can adversarial salience hijack the reader?",
           "concealer makes a weak cue conspicuous and the diagnostic cue quiet at matched information", "hijack",
           "log score of salience-only against learned and active", "salience survives", "salience fails; learned recovers",
           "salience", factors={"policy": ["salience", "learned", "oracle"], "adversarial": [0, 1]}, weight=1.0),
        _c("A13", "A", 2, "Does attention create apparent information in null worlds?",
           "no-information, cue duplication, label permutation, irrelevant high precision", "null confidence",
           "proper-score gain and confidence inflation by null", "zero", "any gain is a failure",
           "confidence", ceiling="METHOD", factors={"null": ["no_information", "duplication", "permutation", "high_precision"]}, weight=1.0),
        _c("A14", "A", 3, "Which attention policy best predicts and abstains across ecologies?",
           "frozen tournament over all policies on fresh worlds", "policy tournament",
           "log score, calibration, risk-coverage and compute by policy and ecology", "one policy universal", "policy x ecology surface",
           "one-policy universality", factors={"policy": ATT_POL, "ecology": ["common", "group", "individual", "nuisance"]}, weight=2.0, pilot=True,
           lanes=("transfer",)),
    ]
    # ---- O: opportunity and cost ----------------------------------------------------------- #
    C += [
        _c("O01", "O", 1, "Do the V12 opportunity anchors reproduce?",
           "scalar-cost world and readers independently rebuilt", "anchor reproduction",
           "constrained-inversion advantage and near-tie/strong-cost shift", "absent", "present", "the V12 build",
           ceiling="METHOD", factors={"anchor": ["R02", "R05"]}, weight=0.5),
        _c("O02", "O", 1, "Can the cost vector be recovered when dimensions vary independently?",
           "full factorial dimensions at matched scalar total", "dimension recovery",
           "dimension-weight posterior and held-out choice against total-cost reader", "no gain", "gain",
           "total cost", factors={"reader": ["factored", "total_cost"], "varied_dim": [d for d in COST_DIMS if d not in ("opportunity", "imposed")]}, weight=1.5),
        _c("O03", "O", 1, "What information lies in menu composition beyond menu size?",
           "same size, different quality, dominance, similarity, constraint", "composition value",
           "full-menu against size-only held-out score", "size suffices", "composition matters",
           "size", factors={"composition": ["quality", "dominance", "similarity", "constraint"], "reader": ["full_menu", "size_only"]}, weight=1.5, pilot=True),
        _c("O04", "O", 1, "Does the same choice mean more under a stronger forgone alternative?",
           "near tie, dominated, attractive alternatives, costly chosen action", "counterfactual utility",
           "posterior shift by alternative strength", "count-like", "tracks counterfactual utility",
           "the count", factors={"alternative": ["near_tie", "dominated", "attractive", "costly_chosen"]}, weight=1.0),
        _c("O05", "O", 2, "Which cost dimensions generalize across domains?",
           "infer in one domain, predict equivalent tradeoffs in another", "portability",
           "cross-domain gain by dimension", "none portable", "portable weighting separable from local competence",
           "domain-local competence", factors={"dimension": [d for d in COST_DIMS if d not in ("opportunity", "imposed")], "domain_pair": ["same", "cross"]}, weight=1.5),
        _c("O06", "O", 2, "Can motivation be separated from competence?",
           "high/low motivation x competence with matched behaviour cells", "motivation vs competence",
           "joint posterior accuracy and prospective action against effort-only", "confounded", "separable",
           "effort-only", factors={"motivation": ["low", "high"], "competence": ["low", "high"], "reader": ["joint", "effort_only"]}, weight=1.5),
        _c("O07", "O", 2, "Can motivation be separated from external constraint?",
           "voluntary effort, imposed work, no viable alternative, free choice at matched cost", "motivation vs constraint",
           "preference evidence attributed to imposed cost", "imposed reads as preference", "imposed carries none",
           "cost-as-preference", factors={"condition": ["voluntary", "imposed", "no_alternative", "free"], "reader": ["record", "cost_blind"]}, weight=1.0),
        _c("O08", "O", 2, "Does paid voluntary cost identify goal strength when rivals are held?",
           "motivation varied with competence, knowledge, constraint, risk fixed", "goal strength",
           "monotone calibrated inference and hidden persistence prediction", "flat", "monotone",
           "an uncalibrated monotone", factors={"motivation": ["0.6", "1.0", "1.6"]}, weight=1.0),
        _c("O09", "O", 2, "Does epistemic cost identify confidence, knowledge, or desire to know?",
           "information gain and acquisition cost crossed with prior knowledge and curiosity", "epistemic cost",
           "four-way discrimination and next-query prediction", "confounded", "distinguishable",
           "already-knew versus did-not-care", factors={"state": ["knew", "did_not_care", "too_costly", "explored"]}, weight=1.0),
        _c("O10", "O", 2, "Do sunk, wasted, and discovered-late costs carry the same evidence?",
           "identical realized expenditure with different anticipation and timing", "cost timing",
           "preference weight by cost timing; hindsight error", "same weight", "only anticipated cost counts",
           "hindsight reader", factors={"timing": ["anticipated", "sunk", "discovered_late"], "reader": ["decision_time", "hindsight"]}, weight=1.0),
        _c("O11", "O", 2, "Can social and risk costs be read without moralizing them?",
           "obligation, reputation, coordination, variance, downside crossed with private reward", "social and risk",
           "planted tradeoff recovery or abstention; no virtue label", "confounded with reward", "recovered",
           "a virtue label", factors={"cost": ["social", "risk"], "reward": ["low", "high"]}, weight=1.0),
        _c("O12", "O", 2, "Can V13 reproduce a human-like choice-set-size neglect curve?",
           "set sizes 2/4/6/8; outcome-only or low-salience reader against exact", "neglect curve",
           "sensitivity to set size by reader", "no curve", "mimic underweights size, direction kept",
           "the exact reader", ceiling="METHOD", factors={"set_size": [2, 4, 6, 8], "reader": ["mimic", "exact"]}, weight=1.0),
        _c("O13", "O", 2, "Do attention manipulations remove choice-set neglect?",
           "separate vs joint comparison, choose vs rank, recall first, explicit cue", "salience repair",
           "size sensitivity by condition without changing choice data", "no change", "sensitivity rises",
           "changed data", factors={"condition": ["separate", "joint", "rank", "recall", "explicit"]}, weight=1.0),
        _c("O14", "O", 2, "Which opportunity weighting family predicts best?",
           "linear, logarithmic, saturating, threshold, rank, resource-rational, learned monotone", "weighting tournament",
           "cross-validated held-out log score by family and ecology; equivalence classes", "one law", "equivalence classes",
           "a declared human law", factors={"family": ["linear", "logarithmic", "saturating", "threshold", "rank_based", "resource_rational", "learned_monotone"], "ecology": ["craft", "bureaucratic", "hazardous", "collegial"]},
           weight=2.0, pilot=True, cost_ecologies=("craft", "bureaucratic", "hazardous", "collegial")),
        _c("O15", "O", 2, "Can an idealized reader safely outperform the planted human-like heuristic?",
           "mimic, exact, misspecified exact, calibrated hybrid", "ideal-reader gain",
           "held-out gain and calibration under missing menus", "no safe gain", "bounded ideal-reader gain",
           "misspecification", factors={"reader": ["mimic", "exact", "misspecified", "hybrid"], "menu": ["complete", "missing"]}, weight=1.5),
        _c("O16", "O", 2, "What happens when the reader sees an incomplete or false choice set?",
           "missing, false, uncertain menus and later correction", "menu integrity",
           "calibration under menu corruption; treatment of menu claims as evidence", "menus taken as truth", "calibrated",
           "a naive complete-menu reader", factors={"menu": ["complete", "missing", "false", "uncertain"], "reader": ["naive", "calibrated"]}, weight=1.0),
        _c("O17", "O", 3, "Do recovered tradeoffs predict future and counterfactual choices?",
           "changed costs, new domain, new goal, new commission, new role", "prospective prediction",
           "proper-score gain over identity, frequency and habit by target", "retrospective only", "prospective gain",
           "retrospective fit", factors={"target": ["changed_costs", "new_domain", "new_goal", "new_commission", "new_role"], "baseline": ["identity", "frequency", "habit"]},
           weight=1.5, lanes=("discovery", "transfer")),
        _c("O18", "O", 3, "Does effort information become a misleading quality cue?",
           "same quality with claimed effort varied; true effort with quality varied; expert and novice readers", "effort heuristic",
           "quality judgement moved by claimed effort; motivation inference separated", "conflated", "separated",
           "the effort heuristic", factors={"claimed_effort": ["low", "high"], "quality": ["low", "high"], "reader": ["expert", "novice"]}, weight=1.0),
    ]
    # ---- P: projection and correction ------------------------------------------------------ #
    C += [
        _c("P01", "P", 1, "Does the V12 correction curve reproduce under a truly matched local control?",
           "near, intermediate, far, anti-similar makers; diagnostic conflict", "correction curve",
           "half-life, residual bias, calibration, order effect by route and bin", "no correction", "correction by route",
           "the unmatched generic", factors={"route": ["self", "equal_local", "generic_local"], "sim_bin": SIM_BINS}, weight=1.5, pilot=False,
           prior_routes=("self", "equal_local", "generic_local")),
        _c("P02", "P", 1, "Which target-specific evidence corrects projection?",
           "behaviour, process record, biography, group label, stated preference, source history with truth varied", "evidence validity",
           "correction by evidence type and truth", "vividness decides", "diagnostic validity decides",
           "vividness", factors={"evidence": ["behaviour", "process_record", "biography", "group_label", "stated_preference", "source_history"], "true": [0, 1]}, weight=1.5),
        _c("P03", "P", 1, "Is correction rationally conditioned on relevant similarity?",
           "actual and perceived similarity crossed with cue reliability", "conditioned correction",
           "self weight after evidence by cell", "unconditional", "falls when the relevant mapping differs",
           "perceived similarity", factors={"actual": [0, 1], "perceived": [0, 1], "reliability": ["low", "high"]}, weight=1.0),
        _c("P04", "P", 2, "Does outcome feedback improve future maker inference?",
           "accurate, noisy, delayed, absent feedback across repeated targets", "feedback learning",
           "calibration and prediction over targets; learning rate; overfitting", "no learning", "learning under accurate feedback",
           "overfitting", factors={"feedback": ["accurate", "noisy", "delayed", "absent"]}, weight=1.0),
        _c("P05", "P", 2, "Does correction generalize beyond the learned target?",
           "same target/new domain, new target/same group, entirely new target", "generalization",
           "gain by transfer kind", "over-generalises", "target-specific stays target-specific",
           "group certainty from one target", factors={"transfer": ["same_target_new_domain", "new_target_same_group", "new_target"]}, weight=1.0),
        _c("P06", "P", 2, "Do time, compute, and accuracy incentives change adjustment?",
           "fast, deliberative, compute-matched, accuracy-rewarded, confidence-rewarded readers", "incentives",
           "residual projection by reader", "no difference", "confidence reward worsens projection",
           "information absence", factors={"reader": ["fast", "deliberative", "compute_matched", "accuracy_rewarded", "confidence_rewarded"]}, weight=1.0),
        _c("P07", "P", 2, "Can a single decisive conflict override accumulated compatible evidence appropriately?",
           "conflict strength x history length x source reliability", "decisive conflict",
           "Bayesian and bounded correction surfaces", "binary", "graded", "a binary rule",
           factors={"strength": ["weak", "strong"], "history": [2, 8], "reliability": ["low", "high"]}, weight=1.5, pilot=True),
        _c("P08", "P", 2, "When does correction become overcorrection?",
           "noisy outlier, adversarial conflict, true regime change, stable maker", "overcorrection",
           "robust reader against reset and anchor by scenario", "no difference", "robust separates outlier from change",
           "reset", factors={"scenario": ["outlier", "adversarial", "regime_change", "stable"], "reader": ["robust", "reset", "anchor"]}, weight=1.5),
        _c("P09", "P", 2, "If equal-local non-self ties self, what remains distinctive?",
           "self, equal-local, group exemplar, learned local transform with acquisition costs", "cheap locality",
           "prediction and construction cost by route", "nothing", "cheap-locality claim",
           "an informational tie", factors={"route": ["self", "equal_local", "group_exemplar", "learned_transform"]}, weight=1.0),
        _c("P10", "P", 2, "Can independent readers debias one another?",
           "readers with different self priors and shared evidence exchange posteriors", "plurality",
           "ensemble log score and calibration against single", "agreement only", "diverse ensemble improves",
           "agreement", factors={"exchange": ["posteriors", "reasons", "none"], "n_readers": [2, 4]}, weight=1.0),
        _c("P11", "P", 2, "What if reader errors are correlated?",
           "shared misconception, common false context, same source bias", "correlated errors",
           "ensemble confidence against accuracy under correlation", "unpenalised", "penalised",
           "independence assumed", factors={"correlation": ["misconception", "false_context", "source_bias", "none"]}, weight=1.0),
        _c("P12", "P", 2, "Can cross-group interaction build a maker-specific bridge?",
           "distant groups with repeated target evidence and partial structural correspondence", "bridge",
           "target model gain without collapse into reader or stereotype", "collapse", "bridge forms",
           "stereotype", factors={"correspondence": ["partial", "none"], "dose": [2, 8, 16]}, weight=1.0),
        _c("P13", "P", 3, "Does correction improve hidden continuation and probe choice?",
           "freeze posterior before held-out action and active query", "prospective correction",
           "corrected against uncorrected self and reset on passive and active targets", "no gain", "gain on both",
           "broad reset", factors={"route": ["corrected", "uncorrected_self", "reset"], "target": ["passive", "active"]}, weight=1.5, lanes=("discovery", "transfer")),
        _c("P14", "P", 3, "Does the reader abstain under irreducible equifinality?",
           "distinct histories with identical artifacts and context, then a separating event", "equifinality",
           "abstention before separation; update after", "invents a route", "abstains then updates",
           "overclaiming", factors={"phase": ["before", "after"]}, weight=0.5),
    ]
    # ---- G: communicative goals and trust ---------------------------------------------------- #
    C += [
        _c("G01", "G", 1, "Can communicative stance be represented as an ordinary maker goal?",
           "one architecture with seven goals activated; no source-type label", "stance as goal",
           "goal posterior accuracy and log score; oracle and surface gates", "unreadable", "readable through correspondence",
           "a source-type label", factors={"goal": GOALS7, "dose": [2, 6, 12]}, weight=1.0),
        _c("G02", "G", 1, "Can helping, neutrality, and misleading be inferred jointly with task goals?",
           "communicative and task goals crossed, compatible and conflicting", "joint inference",
           "joint against independent shortcut by interaction", "independent suffices", "joint wins where goals interact",
           "the independent shortcut", factors={"pair": ["compatible", "conflicting"], "reader": ["joint", "independent"]}, weight=1.0),
        _c("G03", "G", 1, "Can competing communicative goals explain mixed signals?",
           "accuracy, kindness, persuasion, self-presentation tradeoffs", "multi-goal",
           "held-out token selection log score of multi-goal against three-class stance", "three-class suffices", "multi-goal wins",
           "three-class stance", factors={"model": ["multi_goal", "three_class"]}, weight=1.0),
        _c("G04", "G", 1, "Is source reliability distinct from cooperative intent?",
           "helpful-incompetent, neutral-reliable, adversarial-truthful, helpful-outdated", "reliability vs intent",
           "separate calibrated posteriors by source kind", "valence shortcut", "separate",
           "a valence shortcut", factors={"source": ["helpful_incompetent", "neutral_reliable", "adversarial_truthful", "helpful_outdated"]}, weight=1.0),
        _c("G05", "G", 1, "Is content evidence distinct from source reliability?",
           "strong/weak artifact evidence x strong/weak source history", "content vs source",
           "each posterior's update by cell", "one overwrites", "neither universally overwrites",
           "source dominance", factors={"evidence": ["strong", "weak"], "history": ["strong", "weak"]}, weight=1.0),
        _c("G06", "G", 2, "Is goal or value alignment distinct from reliability?",
           "aligned-false, divergent-true, matched controls", "alignment vs reliability",
           "prediction, belief and preference uptake by cell", "one channel", "separate channels",
           "alignment as trust", factors={"alignment": ["aligned", "divergent"], "truth": ["true", "false"]}, weight=1.0),
        _c("G07", "G", 2, "What does a high- or low-trust default change?",
           "distribution of initial source priors at equal evidence", "trust default",
           "efficiency-vulnerability frontier and calibration by default", "no effect", "frontier",
           "a personality label", factors={"default": ["low", "mid", "high"], "reliability": ["low", "high"]}, weight=1.0),
        _c("G08", "G", 2, "Which trust-update dynamics fit changing sources?",
           "Bayes, leaky, asymmetric, threshold, change-point on long histories", "dynamics tournament",
           "Brier of predicted reliability by model and history kind", "one universal", "history-dependent",
           "loss-faster-than-gain assumed", factors={"model": TRUST, "history": ["stable", "one_change", "many_changes"]}, weight=1.5, pilot=True),
        _c("G09", "G", 2, "Does detecting adversarial goals justify retrospective reinterpretation?",
           "early ambiguous acts, later diagnostic event, true and false revelation", "reinterpretation",
           "earlier-state posterior improvement when causally relevant", "rewrites everything", "reanalysis only where relevant",
           "global rewrite", factors={"revelation": ["true", "false"], "relevant": [0, 1]}, weight=1.0),
        _c("G10", "G", 2, "What evidence restores trust after a failure?",
           "apology-like claim, costly action, repeated reliability, third-party record, none", "repair",
           "recovery by repair kind", "wording repairs", "predictive evidence repairs",
           "mere wording", factors={"repair": ["apology", "costly_action", "repeated_reliability", "third_party", "none"]}, weight=1.0),
        _c("G11", "G", 2, "Does large goal divergence appropriately close the uptake gate?",
           "proximal conflict, standing-value conflict, competitive task, benign difference", "uptake gate",
           "belief update separable from imitation and coordination by conflict", "all-or-nothing", "separable",
           "one gate", factors={"conflict": ["proximal", "standing", "competitive", "benign"]}, weight=1.0),
        _c("G12", "G", 2, "Can source-specific reliability be learned and transferred selectively?",
           "multiple sources, domains, regime changes, shared labels", "selective transfer",
           "source x domain posterior transfer only where evidence supports it", "identity shortcut", "selective",
           "the identity shortcut", factors={"transfer": ["same_domain", "new_domain", "shared_label"]}, weight=1.0),
        _c("G13", "G", 2, "Can false context be treated as an assertion rather than truth?",
           "true, false, ambiguous, irrelevant notes at several evidence strengths and orders", "false context",
           "note influence against contradiction; conflict or abstention use", "note dominates", "note shrinks with contradiction",
           "the note as truth", factors={"note": ["true", "false", "ambiguous", "irrelevant"], "evidence": ["weak", "strong"], "order": ["note_first", "evidence_first"]}, weight=1.5),
        _c("G14", "G", 2, "Can teaching, persuasion, and deception be separated by active challenge?",
           "same surface artifact with different response policies under diagnostic query", "challenge",
           "stance log score after challenge against static polish; free-look and random controls", "static suffices", "challenge wins",
           "static polish", factors={"policy": ["challenge", "static", "free_look", "random"]}, weight=1.0, paths=("exact", "pymdp")),
        _c("G15", "G", 3, "Do reconstruction, trust, and four uptake channels remain separate?",
           "accurate/wrong posteriors x reliability x relevance x goal alignment", "uptake factorization",
           "each channel's response to each factor", "one output", "distinct outputs",
           "an aggregate score", factors={"accuracy": ["accurate", "wrong"], "reliability": ["low", "high"], "relevance": ["low", "high"], "alignment": ["low", "high"]}, weight=1.0),
        _c("G16", "G", 3, "Does the frozen trust architecture predict fresh sources and reversals?",
           "new identities, domains, base rates, reliability changes", "transfer",
           "proper score, calibration, risk-coverage, recovery time on fresh sources", "collapse", "transfer",
           "overfitting", factors={"fresh": ["identity", "domain", "base_rate", "reversal"]}, weight=1.5, pilot=True, lanes=("transfer",)),
    ]
    # ---- H: hierarchy and many hands --------------------------------------------------------- #
    C += [
        _c("H01", "H", 1, "Does the role-relative event schema represent the same goal at different levels?",
           "director's secondary project goal becomes a subordinate's primary task", "schema identity",
           "query answers by role; absence of a single absolute level", "one number suffices", "role-relative",
           "an absolute level", ceiling="METHOD", causal=False, factors={"query": ["project_priority", "local_priority", "inheritance"]}, weight=0.5),
        _c("H02", "H", 1, "Does goal promotion predict subordinate behavior?",
           "assigned, self-generated, private, conflicting subordinate goals", "promotion prediction",
           "held-out realization log score of role-relative against project-only and actor-only", "no difference", "role-relative wins",
           "actor-only", factors={"model": ["role_relative", "project_only", "actor_only"], "goal_kind": ["assigned", "private", "conflicting"]}, weight=1.0),
        _c("H03", "H", 1, "Can a true central director be distinguished from an equivalent shared brief?",
           "same dependency, coherence, quality, surface, goal distribution; interaction history differs", "director vs brief",
           "artifact-only and interaction-aware accuracy", "both at floor", "interaction-aware succeeds",
           "coherence", factors={"reader": ["artifact_only", "coherence", "interaction"]}, weight=1.5, pilot=True,
           closure="if only records distinguish them, retain a records-dominant boundary"),
        _c("H04", "H", 1, "Can centralized, rotating, and distributed leadership be distinguished?",
           "central, rotating, emergent, editor-led, institution-filtered teams", "leadership kinds",
           "next-control-event prediction of the graph reader against coherence", "coherence suffices", "graph reader wins",
           "coherence", factors={"team": TEAMS, "reader": ["graph", "coherence"]}, weight=1.5),
        _c("H05", "H", 2, "Do subordinate goal distributions constrain attribution?",
           "narrow role goals, broad creative roles, private goals, different expertise", "role priors",
           "calibration gain from role priors; dependence on target evidence", "priors replace evidence", "priors help, evidence needed",
           "role priors alone", factors={"role_breadth": ["narrow", "broad"], "evidence": [0, 1]}, weight=1.0),
        _c("H06", "H", 2, "Are suppression and amplification interactions readable?",
           "same director goal by damping an overactive or stimulating an underactive subordinate", "interaction signatures",
           "route classification and next-intervention prediction", "unreadable", "readable",
           "goal coherence", factors={"route": ["suppress", "amplify"]}, weight=1.0),
        _c("H07", "H", 2, "Does the interaction law survive crossed subordinate styles?",
           "overactive and underactive profiles reassigned across directors and domains", "crossed styles",
           "attribution accuracy by crossing", "follows style", "follows relation and intervention",
           "actor style", factors={"style": ["overactive", "underactive"], "director": ["A", "B"]}, weight=1.0),
        _c("H08", "H", 2, "What happens when roles are reassigned?",
           "same actors exchange director, editor, specialist, ratifier roles", "role reassignment",
           "event attribution follows control opportunities; personal preferences separately modeled", "follows identity", "follows role",
           "identity", factors={"assignment": ["original", "swapped"]}, weight=1.0),
        _c("H09", "H", 2, "Can private secondary goals be separated from project goals?",
           "compatible, neutral, conflicting private goals with controlled visibility", "private goals",
           "recovery with and without choice or cost evidence; abstention", "always recovered", "recovered only with evidence",
           "overclaiming", factors={"private": ["compatible", "neutral", "conflicting"], "evidence": [0, 1]}, weight=1.0),
        _c("H10", "H", 2, "Do mistakes expose controller-subordinate interaction?",
           "mistake noticed, missed, accepted, exploited, corrected, concealed", "mistake handling",
           "downstream-response prediction of the sequential model; origin kept separate", "unpredictable", "predictable",
           "origin confusion", factors={"handling": ["corrected", "accepted", "exploited", "concealed", "missed"]}, weight=1.0),
        _c("H11", "H", 2, "Does causal reach identify event leverage without identifying the actor?",
           "interventions at project, role, local, ratification levels with matched counts", "reach",
           "reach by level; actor identification requires separate evidence", "reach names the actor", "reach names the level only",
           "actor from reach", factors={"level": ["project", "role", "local", "ratification"]}, weight=1.0),
        _c("H12", "H", 2, "Which interaction traces survive rewriting and flattening?",
           "local, global, template, editor sanding, mixed rewrites", "trace survival",
           "survival by trace class and resolution", "all die", "survival by class",
           "a blanket director-survives", factors={"rewrite": ["none", "local", "global", "template", "editor_sanding", "mixed"], "trace": ["goal_structure", "interaction", "style"]}, weight=1.5),
        _c("H13", "H", 2, "Are literally identical final artifacts historically non-identifying?",
           "central and distributed trajectories forced to the same output and context", "non-identifiability",
           "artifact-only abstention on identical outputs", "topology claimed", "abstention",
           "latent truth leaking", factors={"history": ["central", "distributed"]}, weight=0.5),
        _c("H14", "H", 2, "Which minimal records separate equivalent production graphs?",
           "timings, proposals, alternatives, accept/veto, role map, full logs added one at a time", "record ladder",
           "discrimination by record level; minimal sufficient set", "artifact suffices", "records dominate",
           "the artifact", factors={"record": ["artifact", "timings", "proposals", "alternatives", "accept_veto", "role_map", "full_log"]}, weight=1.5),
        _c("H15", "H", 3, "Can the reader predict the next controller intervention?",
           "hidden next suppression, amplification, veto, ratification, reallocation", "next intervention",
           "log score of the graph model against role frequency, coherence, identity, token share", "no gain", "gain",
           "role frequency", factors={"model": ["graph", "role_frequency", "coherence", "actor_identity", "token_share"]}, weight=1.5, lanes=("discovery", "transfer")),
        _c("H16", "H", 3, "Does a frozen hierarchy reader transfer across team scale and domain?",
           "small/large teams, new domains, new role labels, sparse high-reach events", "transfer",
           "conditional transfer and calibration; schema against artifact", "collapse", "transfer",
           "schema carries it", factors={"scale": ["small", "large"], "domain": ["native", "new"]}, weight=2.0, pilot=True, lanes=("transfer",)),
    ]
    # ---- Q: active epistemic foraging ------------------------------------------------------ #
    C += [
        _c("Q01", "Q", 1, "Does PyMDP select a known information-maximizing probe?",
           "distinct likelihoods, exact EIG ranking, utility-only and random agents; identity world first", "probe agreement",
           "agreement with the exact best probe across priors and horizons", "random", "high agreement",
           "utility-only", ceiling="METHOD", paths=("exact", "pymdp", "baseline"), factors={"prior": ["uniform", "peaked"], "horizon": [1, 2]}, weight=1.5),
        _c("Q02", "Q", 1, "Can the failed V12 commission instrument be repaired honestly?",
           "commission actions with pairwise divergence and realized information above a frozen gate", "commission repair",
           "live gate then policy test: exact and PyMDP against random and free look", "no divergence", "live and useful",
           "free look", paths=("exact", "pymdp", "baseline"), factors={"policy": ["exact", "pymdp", "random", "free_look"]}, weight=1.5,
           closure="one repair only; after that the present PyMDP reader closes"),
        _c("Q03", "Q", 2, "Does the reader query missing opportunity information?",
           "menu size, composition, costs, actor control at different prices", "opportunity queries",
           "field bought against expected decision value; polish control", "buys polish", "buys decision value",
           "polish", factors={"field": ["size", "composition", "costs", "control"], "price": ["low", "high"]}, weight=1.0),
        _c("Q04", "Q", 2, "Can the reader choose which cost cause to disambiguate?",
           "competing motivation, competence, knowledge, constraint, risk explanations", "cause probing",
           "probe targets the most confused pair; held-out choice gain", "random targeting", "targets confusion",
           "uncertainty sampling", factors={"confused_pair": ["motivation_competence", "motivation_constraint", "knowledge_risk"], "policy": ["eig", "uncertainty", "random"]}, weight=1.0),
        _c("Q05", "Q", 2, "Does goal-first probing dominate only under generic uncertainty?",
           "purpose, mechanics, context, anomaly, source probes crossed with reader expertise", "entry probing",
           "first probe chosen by adaptive EIG by expertise; fixed purpose-first as baseline", "purpose-first always", "entry changes with expertise",
           "fixed purpose-first", factors={"expertise": ["novice", "expert"], "policy": ["adaptive", "purpose_first"]}, weight=1.0),
        _c("Q06", "Q", 2, "Can active observation infer maker attention?",
           "inspect regions shaped by different maker attention allocations", "attention inference",
           "attention and goal posterior gain of selected observation over uncertainty sampling", "no gain", "gain",
           "uncertainty sampling", factors={"maker_attention": ["goal", "mechanics", "surface"], "policy": ["eig", "uncertainty"]}, weight=1.0),
        _c("Q07", "Q", 2, "Can a challenge distinguish teaching from strategic shaping?",
           "helpful, persuasive, misleading makers respond to a challenge; adversary may anticipate", "challenge",
           "goal log score of goal-aware challenge against passive and random; anticipating adversary", "no gain", "gain except under anticipation",
           "passive", paths=("exact", "pymdp"), factors={"policy": ["goal_aware", "passive", "random"], "anticipates": [0, 1]}, weight=1.0),
        _c("Q08", "Q", 2, "Are candidate probes genuinely non-equivalent?",
           "pairwise JS/EIG, outcome variance, realized information, surface audit per action", "probe audit",
           "minimum pairwise divergence and discriminativeness against the floor", "below floor", "above floor",
           "equivalent probes", ceiling="METHOD", causal=False, factors={"probe_set": ["commission", "channel", "query"]}, weight=0.5),
        _c("Q09", "Q", 2, "When should the reader stop?",
           "inspection, integration, error cost and abstention value sweeps", "stopping",
           "regret against exact; premature and unnecessary probes", "large regret", "near the frontier",
           "premature stopping", paths=("exact", "pymdp"), factors={"cost": ["low", "mid", "high"], "policy": ["exact", "pymdp", "fixed"]}, weight=1.5),
        _c("Q10", "Q", 2, "Can exploration recover after a misleading high-salience cue?",
           "concealer plants an attractive decoy; quieter probes remain diagnostic", "decoy recovery",
           "robust or challenge policy gain; naive salience failure", "no recovery", "recovery",
           "naive salience", factors={"policy": ["salience", "robust", "challenge"]}, weight=1.0),
        _c("Q11", "Q", 3, "Does a nested local prior improve active selection?",
           "self, equal-local, common, broad, corrected, anti-similar priors at matched budgets", "prior x action",
           "information per cost by prior and bin; no pooled headline", "no effect", "conditional surface",
           "a pooled self headline", factors={"prior": ["self", "equal_local", "within_common", "all_family", "corrected", "anti_similar"], "sim_bin": SIM_BINS}, weight=1.5, lanes=("discovery", "transfer")),
        _c("Q12", "Q", 3, "Do exact and PyMDP policies transfer to fresh active worlds?",
           "new families, cost ecologies, source goals, probe vocabularies", "solver transfer",
           "agreement, regret, confidently-wrong rate by coupling", "collapse", "transfer with a mapped boundary",
           "solver artefact", paths=("exact", "pymdp"), factors={"coupling": ["independent", "weak", "strong"]}, weight=2.0, pilot=True, lanes=("transfer",)),
    ]
    # ---- L: Sounding Line bridge ------------------------------------------------------------- #
    L = [("L01", "What should a cost-aware decision record contain?", "O fields against the Sounding Line process-record schema"),
         ("L02", "How should communicative goals enter the reader?", "G factorization and its strongest failures"),
         ("L03", "What is a testable attention manipulation for text readers?", "A policies without neural language"),
         ("L04", "What nested similarity interactions should Sounding Line test?", "C and P conditional surfaces"),
         ("L05", "How should purpose-first be tested?", "A06 and Q05 entry-policy results"),
         ("L06", "How should unexplained activity be partitioned?", "goal, process, constraint and mistake factorials"),
         ("L07", "What ruler can detect controller-subordinate interaction?", "H06, H07, H10, H15"),
         ("L08", "What resolution does topology inference require?", "V12 F03 and V13 H and Q ladders"),
         ("L09", "What trust gate should a model reader expose?", "G04-G16 and A09"),
         ("L10", "Which variables are artifact-only versus record-dominant?", "H13, H14, P14, O16 information ladders"),
         ("L11", "What can standardized author-purpose questions validate?", "normative purpose, reader interpretation, private process"),
         ("L12", "Which Sounding Line branches deserve compute next?", "validated, transferring, prediction-gated shapes")]
    for lid, q, src in L:
        C.append(_c(lid, "L", 5, q, src, "a bridge deliverable", "a bounded text-side design with licensing card, gap, target, baseline, positive gate, failure meaning, human boundary",
                    "no licensed shape", "a shape licensed by a landed card", "an unlicensed export", ceiling="METHOD", causal=False,
                    factors={"item": ["deliverable"]}, unit_kind="single", weight=0.1))
    # ---- X: adversarial matrix --------------------------------------------------------------- #
    X = [("X01", "surface and source equalization", "Does the effect survive removal of cheap identity cues?"),
         ("X02", "policy preserved, labels changed", "Does it follow the causal policy rather than naming?"),
         ("X03", "labels preserved, policy shuffled", "Does claimed similarity or role alone create the answer?"),
         ("X04", "entropy-and-distance prior rematching", "Was the prior comparison actually fair?"),
         ("X05", "group-composition shift", "Does a typicality or population result survive a new mixture?"),
         ("X06", "false and irrelevant similarity", "Does perceived commonality create projection without causal relevance?"),
         ("X07", "false biography and source context", "Can an assertion override artifact evidence?"),
         ("X08", "adversarial salience", "Can attention be hijacked by conspicuous weak evidence?"),
         ("X09", "hidden and false choice sets", "Does opportunity inference remain calibrated when records are incomplete?"),
         ("X10", "misspecified cost function", "Does a wrong cost model become confident motive attribution?"),
         ("X11", "competence-cost reversal", "Does effort still get mislabeled as motivation when experts make hard acts cheap?"),
         ("X12", "equifinal history", "Does the reader abstain when two processes produce the same observables?"),
         ("X13", "central/shared exact topology match", "Does a director result survive its strongest distributed rival?"),
         ("X14", "correlated reader ensemble", "Does agreement masquerade as independent confirmation?"),
         ("X15", "source or goal regime switch", "Can trust and attention recover after change?"),
         ("X16", "random valid architecture family", "How often does the pattern arise accidentally in constructed families?"),
         ("X17", "exact, PyMDP, and cheap-solver substitution", "Is the result a solver artifact?"),
         ("X18", "seed, order, and coordinate relabeling", "Is it stable under nuisance transformations?"),
         ("X19", "fresh domain and role vocabulary", "Does it transfer beyond the discovery semantics?"),
         ("X20", "adaptive maker adversary", "Can a maker exploit the frozen reader after observing its policy?")]
    for xid, attack, q in X:
        C.append(_c(xid, "X", 4, q, f"attack: {attack}, applied to every relevant promotion candidate, one method positive and one null",
                    "survival", "effect under attack against unattacked; surviving region", "survives", "dies or narrows", attack,
                    ceiling="CONSTRUCTED_MECHANISM", causal=False, factors={"target": ["candidate", "positive", "null"]}, lanes=("attack",), weight=1.0))
    for c in C:
        c.status = "PLANNED"
    return C


MANDATORY_IDS = [c.id for c in build_cards() if c.trunk != "X"]
ATTACK_IDS = [c.id for c in build_cards() if c.trunk == "X"]
PILOT_IDS = [c.id for c in build_cards() if c.pilot]
assert len(MANDATORY_IDS) == 132, len(MANDATORY_IDS)
assert len(ATTACK_IDS) == 20
assert len(PILOT_IDS) == 12, PILOT_IDS


def lineages(tier: dict | None = None) -> dict:
    from .common import lane_ids
    t = tier or TIERS["T3"]
    return {"discovery": lane_ids("discovery", t), "transfer": lane_ids("transfer", t),
            "confirmation": lane_ids("confirmation", t), "pilot": lane_ids("pilot", {**t, "pilot_worlds": 4})}


def write_manifest(cards: list | None = None, note: str = "") -> dict:
    cards = cards if cards is not None else build_cards()
    doc = {"program": "V13 — Common Ground", "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "spec": "V13_SPEC.md", "allowed_states": list(STATES), "resolved_states": list(RESOLVED),
           "tiers": TIERS, "expansions": EXPANSIONS, "lineages": lineages(), "selected_tier": None,
           "mandatory": MANDATORY_IDS, "attacks": ATTACK_IDS, "pilot_cards": PILOT_IDS, "note": note,
           "cards": [c.to_dict() for c in cards]}
    MANIFEST.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def write_cells_template(cards: list | None = None) -> dict:
    cards = cards if cards is not None else build_cards()
    doc = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "by_tier": {}}
    for tname, t in TIERS.items():
        doc["by_tier"][tname] = {c.id: {lane: expected_cells(c, t, lane) for lane in c.lanes} for c in cards}
    CELLS_TEMPLATE.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def instantiate_cells(tier_name: str, tier: dict, cards: list | None = None) -> dict:
    cards = cards if cards is not None else build_cards()
    doc = {"tier": tier_name, "tier_config": tier, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "cards": {}}
    for c in cards:
        doc["cards"][c.id] = {}
        for lane in c.lanes:
            e = expected_cells(c, tier, lane)
            e["units_required"] = e["units"]
            doc["cards"][c.id][lane] = e
    CELLS.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def load_cells() -> dict | None:
    return json.loads(CELLS.read_text(encoding="utf-8")) if CELLS.exists() else None


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return write_manifest()
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def save_manifest(doc: dict) -> None:
    doc["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    MANIFEST.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def get_card(doc: dict, cid: str) -> Card:
    for d in doc["cards"]:
        if d["id"] == cid:
            return card_from_dict(d)
    raise KeyError(cid)


def update_card(doc: dict, cid: str, **fields) -> None:
    for d in doc["cards"]:
        if d["id"] == cid:
            if "status" in fields:
                assert fields["status"] in STATES and fields["status"] != "DONE", fields["status"]
            d.update(fields)
            return
    raise KeyError(cid)


def coverage(doc: dict) -> dict:
    by_state, by_trunk = {}, {}
    for d in doc["cards"]:
        by_state[d["status"]] = by_state.get(d["status"], 0) + 1
        t = by_trunk.setdefault(d["trunk"], {"total": 0, "resolved": 0})
        t["total"] += 1
        t["resolved"] += int(d["status"] in RESOLVED)
    mandatory = [d for d in doc["cards"] if d["trunk"] != "X"]
    return {"cards_total": len(doc["cards"]), "mandatory_total": len(mandatory),
            "mandatory_resolved": sum(d["status"] in RESOLVED for d in mandatory),
            "by_state": by_state, "by_trunk": by_trunk,
            "unresolved_mandatory": [d["id"] for d in mandatory if d["status"] not in RESOLVED],
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")}


def write_coverage(doc: dict) -> dict:
    cov = coverage(doc)
    COVERAGE.write_text(json.dumps(cov, indent=2), encoding="utf-8")
    return cov


def add_amendment(card: str, original: dict, replacement: dict, reason: str) -> None:
    doc = json.loads(AMENDMENTS.read_text(encoding="utf-8")) if AMENDMENTS.exists() else {"amendments": []}
    doc["amendments"].append({"card": card, "original": original, "replacement": replacement, "reason": reason,
                              "when": time.strftime("%Y-%m-%dT%H:%M:%S")})
    AMENDMENTS.write_text(json.dumps(doc, indent=2), encoding="utf-8")
