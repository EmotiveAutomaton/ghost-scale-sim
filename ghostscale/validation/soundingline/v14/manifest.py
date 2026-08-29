"""The V14 queue manifest: every mandatory card, the attack matrix, tiers, lineages, expected
cells and the machine-readable records (spec §5, §6, §8, §10). Written to
results/v14/QUEUE_MANIFEST.json; the validator compares it to the literal card ids in the spec
and refuses a program with a card missing, a factor absent, or a floor lowered without an
amendment.

A card's ``factors`` are its cell axes: every row a card emits carries these keys, and the
expected-cell receipt counts the realized crossings per independent unit. Levels here are DESIGN
levels every unit realizes by construction (V13's lesson: never a data-driven bin).
"""
from __future__ import annotations

import json
import time

from . import v14_dir
from .atomicio import write_json_atomic
from .schemas import Card, EXPANSIONS, RESOLVED, STATES, TIERS, card_from_dict, expected_cells

MANIFEST = v14_dir() / "QUEUE_MANIFEST.json"
COVERAGE = v14_dir() / "COVERAGE.json"
RUNTIME = v14_dir() / "RUNTIME.json"
CELLS_TEMPLATE = v14_dir() / "EXPECTED_CELLS_TEMPLATE.json"
CELLS = v14_dir() / "EXPECTED_CELLS.json"
WORKLOAD = v14_dir() / "WORKLOAD_LOCK.json"
AMENDMENTS = v14_dir() / "AMENDMENTS.json"
CLAIM_LEDGER = v14_dir() / "CLAIM_LEDGER.json"
CONFIRMATION_REGISTRY = v14_dir() / "CONFIRMATION_REGISTRY.json"
DEADLINE = v14_dir() / "DEADLINE.json"
PKG = "ghostscale.validation.soundingline.v14.cards"

ESTIMATORS = ["independent", "goal_process_preference", "process_goal_preference", "preference_goal_process", "joint"]
LATENTS = ["process", "goal", "preference"]
ROUTES = ["action", "semantic", "context", "forensic"]
REGIMES = ["full", "no_forensic", "artifact_only"]
DOSES = [1, 2, 4, 8]


def _c(cid, trunk, wave, question, construction, target, estimand, null, alt, rival, closure="",
       ceiling="CONSTRUCTED_MECHANISM", causal=True, factors=None, lanes=("discovery",), unit="world",
       unit_kind="world", weight=1.0, pilot=False, deps=(), paths=("exact",), min_rows=1, domains=2, families=2):
    gates = ["live", "placebo", "positive", "surface", "oracle", "prediction", "calibration"] if causal else ["identity", "positive", "placebo"]
    return Card(id=cid, trunk=trunk, wave=wave, question=question, construction=construction, target=target,
                estimand=estimand, null_expectation=null, alternative_expectation=alt, strongest_rival=rival,
                claim_ceiling=ceiling, solver_paths=list(paths), depends_on=list(deps), closure=closure, causal=causal,
                gates_required=gates, factors=dict(factors or {}), domains=domains, world_families=families,
                lanes=list(lanes), independent_unit=unit, unit_kind=unit_kind, min_rows_per_unit=min_rows, work_weight=weight, pilot=pilot,
                module=f"{PKG}.trunk_{trunk.lower()}:{cid}",
                output=f"results/validation/soundingline/v14/{cid}.json",
                checkpoint_key=f"results/v14/checkpoints/<lane>/{cid}/", completion_key=f"<lane>:{cid}")


def build_cards() -> list:
    C = []
    # ---- I: integrity, identities, V13 repairs ------------------------------------------------ #
    C += [
        _c("I01", "I", 0, "Do the V13 numeric anchors used by V14 reproduce from committed inputs?",
           "the committed V13 verdicts V14's design imports (C04, A03, G01, H03, C15, X10) are re-read and hashed; the cited fields compared",
           "anchor receipt", "max absolute deviation of cited fields from the committed files", "identity", "a mismatch blocks inheritance only",
           "provenance drift", causal=False, ceiling="METHOD", factors={"anchor": ["C04", "A03", "G01", "H03", "C15", "X10"]}, unit="anchor", unit_kind="list", weight=0.1),
        _c("I02", "I", 0, "Does the manifest enumerate all 64 cards, factors, attacks, lanes, and independent-unit floors?",
           "recursive validation of the manifest against the literal spec inventory", "manifest validity",
           "counts and presence of every required field", "complete", "an omission is a program defect", "a silent not-applicable",
           causal=False, ceiling="METHOD", factors={"check": ["cards_64", "attacks_12", "factors", "lanes", "floors"]}, unit_kind="single", weight=0.1),
        _c("I03", "I", 0, "Does the joint enumerator recover exact posteriors in tiny exhaustive worlds and remain invariant to labels/order?",
           "brute-force enumeration against the grid posterior; relabelled actions, tokens and menu options; permuted evidence order",
           "enumerator identities", "max deviation from brute force; max change under relabelling and reordering", "zero", "a deviation is a solver bug",
           "coordinate dependence", causal=False, ceiling="METHOD", factors={"check": ["normalization", "brute_force", "label_invariance", "order_invariance"]}, weight=0.5, pilot=True),
        _c("I04", "I", 0, "Are action, semantic, context, and forensic routes independently live and materially different in information?",
           "route-only posteriors on the same episodes; pairwise divergence; a shuffled null route", "route liveness",
           "pairwise Jensen-Shannon and information per route and latent; null-route information", "divergent and informative; null at zero",
           "a dead or duplicate route voids R", "two routes carrying one signal", causal=False, ceiling="METHOD",
           factors={"route": ROUTES, "latent": LATENTS}, weight=0.8),
        _c("I05", "I", 0, "Are surface collisions, equifinal histories, factor orthogonality, and lineage identities valid?",
           "matched-surface pairs hash equal; equivalent (plan, goal) pairs give identical non-forensic likelihoods; one-factor manipulations leak at floor; lanes disjoint",
           "construction identities", "collision hash equality; likelihood identity; max leak; lineage overlap", "all hold", "a leak or overlap voids dependent cards",
           "a leaking construction", causal=False, ceiling="METHOD", factors={"check": ["surface_collision", "equifinal_history", "factor_orthogonality", "lineage"]}, weight=0.5),
        _c("I06", "I", 0, "Repair V13 C03 once: within-common versus all-family prior, preserving its target and failed positive gate.",
           "V13's world and priors re-read (read-only); the positive floor derived from the construction's expected gain", "repaired C03",
           "log-score gain of within-common over all-family by dose against the construction floor", "no gain", "gain within the substrate",
           "a family label", ceiling="METHOD", factors={"route": ["within_common", "all_family"], "dose": [1, 2, 4]}, unit="v13_world", weight=1.5),
        _c("I07", "I", 0, "Repair V13 C05 once: self versus within-common prior, preserving reader-type interaction and convergence placebo.",
           "V13's readers by type; the convergence dose derived from the construction", "repaired C05",
           "self minus within-common by reader type and dose; convergence at the derived dose", "flat", "typical readers gain; routes converge",
           "typicality", ceiling="METHOD", factors={"reader_type": ["near", "typical", "atypical", "anti"], "dose": [1, 4, 16]}, unit="v13_world", weight=1.5),
        _c("I08", "I", 0, "Repair V13 P01 once: matched-local correction curve, preserving similarity bins, prospective endpoint, and failed calibration record.",
           "V13's matched routes and bins; correction RATE per bin; unit-level calibration at the prospective endpoint", "repaired P01",
           "half-life, residual, correction rate and endpoint calibration by route and bin", "no correction", "correction by route with rate ordering",
           "the unmatched generic", ceiling="METHOD", factors={"route": ["self", "equal_local", "generic_local"], "sim_bin": ["near", "mid", "far", "anti"]}, unit="v13_world", weight=1.5),
    ]
    # ---- J: joint partial identifiability ---------------------------------------------------- #
    C += [
        _c("J01", "J", 1, "Is process identifiable when goal and preference are supplied?", "oracle goal and preference; distinct and process-equivalent histories",
           "process identifiability", "class mass and single-state mass by history kind", "unidentifiable", "identifiable up to the class", "forced uniqueness",
           factors={"history": ["distinct", "equivalent"], "dose": [1, 4]}, weight=1.0),
        _c("J02", "J", 1, "Is episode goal identifiable when process and preference are supplied?", "oracle process and preference; the hidden next action within the episode",
           "goal identifiability", "next-action log score over the prior by dose", "no gain", "gain", "a retrospective label",
           factors={"dose": [1, 2, 4]}, weight=1.0),
        _c("J03", "J", 1, "Is standing preference identifiable when process and episode goal are supplied across episodes?", "oracle process and goals; the next episode's first action after the local goal changed",
           "preference identifiability", "next-episode action log score over the prior by episodes seen", "no gain", "gain", "a goal label",
           factors={"episodes": [2, 4, 8]}, weight=1.0),
        _c("J04", "J", 1, "Does recurrent joint inference beat independent marginals under matched evidence/compute?", "five estimators on one likelihood table; held-out next action; calibration; the independent estimator is the cross-latent ablation",
           "joint advantage", "held-out log score by estimator and dose; ECE", "tie", "joint wins", "the independent shortcut",
           factors={"estimator": ESTIMATORS, "dose": [2, 4]}, weight=1.5, pilot=True, closure="close the joint advantage if no estimator beats independent on a prospective target"),
        _c("J05", "J", 1, "Which staged order is best, and is any order uniformly best across expertise/access regimes?", "three orders x three access regimes x two reader competences",
           "order x regime", "held-out log score by order, regime and competence", "one order universal", "interaction", "purpose-first universality",
           factors={"order": ESTIMATORS[1:4], "regime": REGIMES}, weight=1.5),
        _c("J06", "J", 1, "At what evidence dose does each latent first improve prospective prediction?", "trajectories saved after every episode; provisional-confidence curve",
           "dose trajectories", "first improving dose per latent; confidence by dose", "never", "each by the maximum dose", "premature confidence",
           factors={"latent": LATENTS, "dose": [1, 2, 4, 6]}, weight=1.0),
        _c("J07", "J", 1, "Does diagnostic contradiction revise all affected latents rather than only the surface label?", "a planted contradiction after four episodes; a consistent continuation as placebo",
           "revision", "Jensen-Shannon movement per latent after contradiction and after continuation", "surface only", "all affected latents",
           "label-only revision", factors={"phase": ["contradiction", "continuation"], "latent": LATENTS}, weight=1.0),
        _c("J08", "J", 1, "Does the joint reader abstain on exact equifinality and contract uncertainty after resolving evidence?", "process-equivalent histories, then a forensic observation",
           "abstention", "single-state and class mass before and after resolution; risk-coverage", "confident uniqueness", "spread then contract",
           "overclaiming", factors={"phase": ["equifinal", "resolved"]}, weight=1.0),
        _c("J09", "J", 1, "Can it distinguish a changed episode goal from a changed standing preference?", "crossed intervention: goal changed, preference changed, neither; held-out future choices",
           "change attribution", "detection and confusion rates", "confounded", "separable", "the count of changed choices",
           factors={"change": ["goal", "preference", "none"]}, weight=1.0),
        _c("J10", "J", 3, "Does the joint advantage transfer to fresh world factorization and action vocabulary?", "the frozen estimators on transfer families and vocabularies",
           "transfer", "joint minus independent on fresh worlds", "collapse", "advantage kept; domain-bound effects named", "overfitting",
           factors={"estimator": ["independent", "joint"], "dose": [2, 4]}, weight=1.5, pilot=True, lanes=("transfer",)),
    ]
    # ---- R: route reliability, fluency, conflict --------------------------------------------- #
    C += [
        _c("R01", "R", 1, "Which route contains the most target information in each access regime?", "exact conditional information per route and latent; the prediction ruler on the dominant route",
           "route information", "information by route, latent and regime", "one route dominates all", "route x latent map", "a universal route",
           ceiling="METHOD", factors={"route": ROUTES, "latent": LATENTS}, weight=0.8),
        _c("R02", "R", 1, "Can a reader learn route reliability from feedback without receiving target labels at test?", "training makers with feedback; test makers without labels; learned, equal, random and fixed weights",
           "learned reliability", "held-out log score by weighting", "no gain", "learned wins", "equal weighting",
           factors={"weights": ["learned", "equal", "random", "fixed_action"]}, weight=1.2, pilot=True, closure="close reliable routing if learned never beats equal"),
        _c("R03", "R", 1, "Does planted ease bias route use when accuracy is equal?", "the wrong route made cheap and the right route dear, then reversed; ease-driven and learned readers",
           "ease bias", "route weight and score by condition and reader", "no bias", "ease-driven reader biased; learned reader not", "fluency as reliability",
           factors={"condition": ["easy_wrong", "easy_right"], "reader": ["ease_driven", "learned"]}, weight=1.0),
        _c("R04", "R", 1, "Does true accuracy control route use when ease is equal?", "equal ease; the accurate route hard or easy; ease-driven and learned readers",
           "accuracy control", "score and weight by condition and reader", "ease decides", "accuracy decides for the learned reader", "ease",
           factors={"condition": ["accurate_hard", "accurate_easy"], "reader": ["ease_driven", "learned"]}, weight=1.0),
        _c("R05", "R", 1, "When routes conflict, does expanding the latent set to missing goal, constraint, or strategic source improve prediction?", "consistent and strategic-source worlds; fixed and expanded readers; search cost",
           "expansion", "gain minus search cost; false positives in consistent worlds", "no gain", "gain in strategic worlds only", "always expanding",
           factors={"world": ["consistent", "strategic"], "reader": ["fixed", "expanded"]}, weight=1.2),
        _c("R06", "R", 1, "When is costly forensic access worth purchasing?", "exact information-per-cost, random, always-buy, never-buy policies",
           "forensic purchase", "realized gain per cost by policy", "never worth it", "exact rule best", "always buying",
           factors={"policy": ["exact", "random", "always", "never"]}, weight=1.0),
        _c("R07", "R", 1, "Can routes be fused without treating correlated or duplicated evidence as independent?", "independent, duplicated and paraphrased semantic evidence; naive and shared-cause fusion",
           "fusion", "confidence rise and calibration by evidence kind and fusion", "duplicates count twice", "shared-cause fusion is flat", "naive independence",
           factors={"evidence": ["independent", "duplicate", "paraphrase"], "fusion": ["naive", "shared_cause"]}, weight=1.0),
        _c("R08", "R", 3, "Do learned route reliabilities transfer, or should the reader reset under a new domain?", "a fresh vocabulary with changed reliabilities; reset, partial, full transfer",
           "transfer of reliability", "held-out score by transfer kind", "full transfer best", "reset or partial best under change", "full transfer",
           factors={"transfer": ["reset", "partial", "full"]}, weight=1.0, lanes=("transfer",)),
    ]
    # ---- E: competence versus attention history ---------------------------------------------- #
    C += [
        _c("E01", "E", 1, "Are demonstrated competence and past attention/value history independently live?", "full K x H manipulation; process accuracy and early relevance as the two measures",
           "K x H liveness", "each measure by each factor; cross-leak", "confounded", "independent", "history defined as competence",
           ceiling="METHOD", factors={"competence": ["low", "high"], "history": ["none", "strong"]}, weight=1.0, pilot=True),
        _c("E02", "E", 1, "With competence matched, does different attention history change initial route choice or prior?", "readers with none or strong stale route history; initial and corrected phases",
           "initial bias", "route-weight divergence and score gap by phase", "no bias", "bias then correction", "permanent bias",
           factors={"history": ["none", "strong"], "phase": ["initial", "corrected"]}, weight=1.0),
        _c("E03", "E", 1, "With history matched, does different competence change process reconstruction?", "readers at three competences with the same history",
           "competence effect", "process prediction and calibration by competence", "no effect", "monotone gain", "history",
           factors={"competence": ["low", "mid", "high"]}, weight=1.0),
        _c("E04", "E", 1, "Does a learned attention bias persist after its reward/value reverses?", "reward reversal at a planted episode; bias measured at increasing distances",
           "decay curve", "bias by episodes after reversal; current-utility cost", "instant reset", "geometric decay", "no decay",
           factors={"episodes_after": [0, 2, 4, 8]}, weight=1.0),
        _c("E05", "E", 1, "Can target evidence correct stale attention history without erasing genuine skill?", "stale-history readers before and after feedback correction; process accuracy retained",
           "correction", "bias reduction and skill retention by phase", "correction erases skill", "bias falls, skill stays", "reset",
           factors={"reader_history": ["none", "stale"], "phase": ["before", "after"]}, weight=1.0),
        _c("E06", "E", 1, "Does competence selectively improve early relevance detection?", "low and high competence makers; relevance detected by dose",
           "early relevance", "detection gain by competence and dose", "generic gain", "early-dose interaction", "generic intelligence",
           factors={"competence": ["low", "high"], "dose": [1, 2, 4]}, weight=1.0),
        _c("E07", "E", 3, "Does attention history transfer more broadly or narrowly than competence across domains?", "history and competence measured in the native and a fresh domain",
           "transfer breadth", "cross-domain conditional matrix", "equal breadth", "history narrower", "one breadth",
           factors={"object": ["history", "competence"], "domain": ["same", "cross"]}, weight=1.0, lanes=("discovery", "transfer")),
        _c("E08", "E", 1, "When skill was acquired through different attention paths, do equal competencies yield different maker signatures?", "two acquisition paths at equal competence; held-out early-transition signatures",
           "path signatures", "signature classifier accuracy; process accuracy gap", "no signature", "signature at equal skill", "skill itself",
           factors={"path": ["A", "B"]}, weight=1.0),
        _c("E09", "E", 1, "Can diverse calibrated readers shrink the compatible maker set without naive averaging?", "readers with different competence models; average, likelihood intersection, feasible set, best member",
           "reader combination", "log score and calibration by method", "average best", "intersection matches best member", "naive averaging",
           factors={"method": ["average", "likelihood_product", "feasible_set", "best_member"]}, weight=1.0),
        _c("E10", "E", 1, "Which object - competence, attention history, or current preference - best predicts the maker's next novel choice?", "factor interventions; hidden next novel choice; last-choice baseline",
           "next-choice tournament", "log score by object over baseline", "none", "one or more beat the baseline", "last choice",
           factors={"object": ["competence", "history", "preference"]}, weight=1.0),
    ]
    # ---- A: affect ownership and strategic communication ------------------------------------- #
    C += [
        _c("A01", "A", 2, "Are reader response, intended audience appraisal, maker appraisal, content support, communicative goal, reliability, and uptake independently live?",
           "each owner varied alone; every posterior measured", "owner identities", "own movement and leak per owner", "an overwritten owner", "each owner moves only itself",
           "reader appraisal as maker appraisal", ceiling="METHOD",
           factors={"owner": ["reader_response", "maker_appraisal", "intended_effect", "content", "goal", "reliability", "uptake"]}, weight=1.0),
        _c("A02", "A", 2, "When is the reader's own induced response a useful prior for intended audience effect?", "reader similar or dissimilar to the audience; projecting and neutral readers",
           "response as prior", "intended-effect log score by similarity and reader", "always useful", "useful when similar", "projection",
           factors={"similarity": ["similar", "dissimilar"], "reader": ["projecting", "neutral"]}, weight=1.0),
        _c("A03", "A", 2, "Can intended effect be recovered while maker appraisal remains uncertain?", "partial identifiability: appraisal cues withheld; held-out presentation choice",
           "partial identifiability", "intended-effect accuracy and appraisal entropy", "both or neither", "effect without appraisal", "one owner",
           factors={"maker_appraisal": ["known", "uncertain"]}, weight=1.0),
        _c("A04", "A", 2, "Can maker appraisal be recovered while intended effect differs?", "owner-swap construction; private/off-audience prediction",
           "appraisal recovery", "appraisal and private-action accuracy by swap", "effect overwrites appraisal", "recovered", "intended effect",
           factors={"owner_swap": ["same", "swapped"]}, weight=1.0),
        _c("A05", "A", 2, "Can derived honest warning, sincere fanatic, strategic propagandist, and neutral reporting be separated without labels?", "regions of the factorial; artifact-only against counterfactual evidence",
           "region separation", "region accuracy with and without counterfactual evidence", "templates", "counterfactuals separate", "a template classifier",
           factors={"region": ["honest_warning", "sincere_fanatic", "strategic_propagandist", "neutral_report"], "evidence": ["artifact_only", "counterfactual"]}, weight=1.2, pilot=True),
        _c("A06", "A", 2, "What minimally distinguishes sincere fanatic from strategic propagandist under matched surface and audience effect?", "matched artifacts; belief, private action and correction probes added one at a time; abstention",
           "fanatic/propagandist boundary", "pairwise accuracy by evidence; abstention mass", "telltale template", "probes separate, abstain otherwise", "surface",
           factors={"evidence": ["artifact_only", "plus_correction", "plus_private", "both"]}, weight=1.2,
           closure="close the boundary if no counterfactual discriminator exists"),
        _c("A07", "A", 2, "Does an inverse-inverse reader help only when the maker actually models the audience?", "plain and audience-modelling makers x plain and audience-aware readers",
           "maker x reader interaction", "log score by cell", "always helps", "helps only against audience-modelling makers", "always assuming manipulation",
           factors={"maker": ["plain", "audience_modelling"], "reader": ["plain", "audience_aware"]}, weight=1.0),
        _c("A08", "A", 2, "Does influence/source awareness improve attribution or merely suppress all uptake?", "aware, unaware and suppressing readers on true and false content",
           "awareness", "discrimination, criterion, calibration, selective uptake", "suppression", "selective", "blanket distrust",
           factors={"reader": ["aware", "unaware", "suppress"], "truth": ["true", "false"]}, weight=1.0),
        _c("A09", "A", 2, "Can acute response habituate while cumulative belief or policy uptake grows across exposure?", "repeated exposure; fast response and slow posterior",
           "two trajectories", "acute response and cumulative uptake by exposure", "one trajectory", "response falls, uptake rises", "a single channel",
           factors={"exposure": [1, 2, 4, 8]}, weight=0.8),
        _c("A10", "A", 2, "Does factored trust gate uptake after reconstruction without changing content evidence or inferred goal?", "factored, scalar and suppressing gates; reliable and unreliable sources",
           "uptake gate", "policy uptake, belief and goal by gate and reliability", "gate moves everything", "gate moves policy only", "a scalar",
           factors={"gate": ["factored", "scalar", "suppress"], "reliability": ["reliable", "unreliable"]}, weight=1.0),
    ]
    # ---- H: hierarchy, habit, value residue -------------------------------------------------- #
    C += [
        _c("H01", "H", 2, "Can repeated transition structure recover a subtask hierarchy when it exists?", "hierarchical and flat generators; boundary ruler; next-subtask prediction",
           "hierarchy recovery", "boundary recovery and next-subtask gain by structure", "no recovery", "recovered where present", "flat sequence model",
           factors={"structure": ["hierarchical", "flat"]}, weight=1.0),
        _c("H02", "H", 2, "Do identical local actions under different higher-order goals remain correctly ambiguous?", "shared and distinct opening windows",
           "ambiguity", "top-goal mass by window", "forced attribution", "ambiguous where shared", "unique attribution",
           factors={"window": ["shared", "distinct"]}, weight=1.0),
        _c("H03", "H", 2, "Can policy-equivalent reward transformations be distinguished without extra evidence?", "shaped and unshaped rewards; observational and intervened phases",
           "reward equivalence", "P(shaped) by phase", "distinguished observationally", "not without intervention", "an omitted alternative reward",
           factors={"phase": ["observational", "intervened"]}, weight=1.0),
        _c("H04", "H", 2, "Can a stable preference be distinguished from a practiced habit across changed incentives?", "preference-driven and habitual makers under original and changed incentives",
           "preference vs habit", "model log score by incentive and model", "confounded", "separable under change", "retrospective fit",
           factors={"incentive": ["original", "changed"], "model": ["preference", "habit"]}, weight=1.0),
        _c("H05", "H", 2, "Can stale attention/expertise residue be distinguished from a current preference?", "history reversal; hidden future choice",
           "residue vs preference", "prediction by model after reversal", "residue wins", "current preference wins", "stale value",
           factors={"phase": ["before_reversal", "after_reversal"]}, weight=1.0),
        _c("H06", "H", 2, "Does multi-episode evidence support progressively higher coordinating goals without requiring a terminal horizon?", "three levels; predictive compression; level uncertainty",
           "levels", "compression and level calibration", "one level", "higher levels compress", "a terminal horizon",
           factors={"level": ["action", "subgoal", "top"]}, weight=1.2, pilot=True),
        _c("H07", "H", 2, "When full interaction records are available, can role-relative control be recovered beyond coherence and shared brief?", "central and exact shared-brief twins; interaction and coherence readers; hidden next intervention",
           "role-relative control", "accuracy by team and reader; next-intervention prediction", "coherence suffices", "records only", "the shared brief",
           factors={"team": ["central", "shared_brief"], "reader": ["interaction", "coherence"]}, weight=1.0, pilot=True),
        _c("H08", "H", 2, "Which inferred hierarchy level best predicts the next changed-context action?", "top, subgoal, flat value and last-goal models",
           "level selection", "prospective log score by model", "flat value", "a level wins", "last goal",
           factors={"model": ["top", "subgoal", "flat_value", "last_goal"]}, weight=1.0),
    ]
    # ---- F: interest and epistemic foraging -------------------------------------------------- #
    C += [
        _c("F01", "F", 2, "Are novelty, complexity, prediction error, reducibility, learning progress, relevance, competence, and cost independently live?", "each factor varied alone; correlation audit",
           "factor liveness", "own movement and correlations", "collinear", "independent", "one scalar interest",
           ceiling="METHOD", factors={"factor": ["novelty", "complexity", "error", "reducibility", "progress", "relevance", "cost"]}, weight=0.8),
        _c("F02", "F", 2, "Is a novel but immediately explained item still selected over a familiar unresolved structure?", "novel-explained against familiar-unresolved items; three policies",
           "novelty vs residual", "share of picks by policy", "novelty wins", "progress prefers the residual", "novelty",
           factors={"policy": ["novelty", "learning_progress", "eig_per_cost"]}, weight=0.8),
        _c("F03", "F", 2, "Is a complex but compressible item preferred over a simpler unresolved one?", "complex-compressible against simple-unresolved; three policies",
           "complexity vs compressibility", "share of picks by policy", "complexity wins", "compressibility preferred", "complexity",
           factors={"policy": ["complexity", "learning_progress", "eig_per_cost"]}, weight=0.8),
        _c("F04", "F", 2, "Does random unlearnable noise lose to structured learnable error despite higher surprise?", "the noise trap; surprise, learning progress, gain per cost, random",
           "noise trap", "noise share and realized gain by policy", "noise wins by surprise", "progress ignores noise", "surprise",
           factors={"policy": ["surprise", "learning_progress", "eig_per_cost", "random"]}, weight=1.0, pilot=True,
           closure="close any curiosity ruler that noise wins"),
        _c("F05", "F", 2, "Does expected learning progress outperform raw current error in a nonstationary curriculum?", "item generators change mid-run; surprise against progress",
           "nonstationary", "realized gain by policy", "surprise suffices", "progress wins", "current error",
           factors={"policy": ["surprise", "learning_progress"]}, weight=0.8),
        _c("F06", "F", 2, "Does expected information gain per cost outperform novelty, surprise, and always-forensic policies?", "costed items; five policies",
           "gain per cost", "realized held-out gain per cost by policy", "novelty suffices", "gain per cost wins", "always forensic",
           factors={"policy": ["eig_per_cost", "novelty", "surprise", "always_forensic", "random"]}, weight=1.0),
        _c("F07", "F", 2, "Can pursuit value stay high while warrant stays low for a hoped-for explanation?", "an attractive weakly supported hypothesis against a dull diagnostic one; query allocation and posterior",
           "pursuit vs warrant", "query share and posterior mass by hypothesis", "one number", "pursuit high, warrant low", "hope as belief",
           factors={"hypothesis": ["attractive_weak", "dull_diagnostic"]}, weight=0.8),
        _c("F08", "F", 3, "Does an active selector transfer to new foraging ecologies and abstain when no probe is discriminative?", "fresh ecologies; discriminative and null probe sets",
           "transfer and abstention", "regret and no-action rate", "collapse", "transfers and abstains", "always acting",
           factors={"probe": ["discriminative", "null"]}, weight=1.0, pilot=True, lanes=("transfer",)),
    ]
    # ---- B: bridge and closure --------------------------------------------------------------- #
    C += [
        _c("B01", "B", 5, "Which validated rulers license an implementation in Sounding Line Stage 5?", "one row per candidate ruler from landed cards",
           "bridge ledger", "access, construction gate, cheap rival, endpoint, shape, ceiling per row", "nothing licensed", "licensed rows", "an unlicensed export",
           ceiling="METHOD", causal=False, factors={"item": ["deliverable"]}, unit_kind="single", weight=0.1),
        _c("B02", "B", 5, "Which V14 questions should be promoted, closed, or left as context after confirmation?", "final pursuit/warrant ledger, runtime audit, recommendation",
           "closure ledger", "promoted, closed, context per card", "no ledger", "a ledger", "an automatic V15",
           ceiling="METHOD", causal=False, factors={"item": ["deliverable"]}, unit_kind="single", weight=0.1),
    ]
    # ---- X: adversarial matrix ---------------------------------------------------------------- #
    X = [("X01", "surface", "Preserve the latent and alter cheap features; preserve surface and swap the latent."),
         ("X02", "route ease", "Make the wrong route easier and the correct route harder, then reverse; accuracy and ease must not alias."),
         ("X03", "equifinal history", "Give identical artifacts validly produced by different histories; require equivalence-class uncertainty."),
         ("X04", "duplicate evidence", "Duplicate or paraphrase one cause; confidence must not rise as though evidence were independent."),
         ("X05", "wrong generative model", "Change cost, competence, noise or source-selection process while retaining familiar surfaces."),
         ("X06", "attention/skill swap", "Hold competence fixed and swap history; hold history fixed and swap competence."),
         ("X07", "affect owner swap", "Swap reader response, maker appraisal, intended effect and content truth while matching intensity."),
         ("X08", "fanatic/propagandist collision", "Match artifact and intended audience effect; vary only counterfactual belief/private/correction behavior."),
         ("X09", "hierarchy equivalence", "Exact shared brief, reward shaping or locally equivalent higher goal must defeat unjustified unique attribution."),
         ("X10", "hope and salience", "Make an attractive hypothesis salient but weakly supported; make a dull hypothesis diagnostic."),
         ("X11", "aggregation", "Verify that global means cannot hide planned sign reversals across similarity, access or competence."),
         ("X12", "solver/lineage", "Seed, order, solver, process count and fresh-lane audit; exact/approximate disagreement is reported as such.")]
    for xid, attack, q in X:
        C.append(_c(xid, "X", 4, q, f"attack: {attack}, applied to every relevant promotion candidate, one method positive and one valid null; applicability recorded",
                    "survival", "effect under attack against unattacked; surviving region", "survives", "dies or narrows", attack,
                    ceiling="CONSTRUCTED_MECHANISM", causal=False, factors={"target": ["candidate", "positive", "null"]}, lanes=("attack",), weight=0.8))
    for c in C:
        c.status = "PLANNED"
    return C


MANDATORY_IDS = [c.id for c in build_cards() if c.trunk != "X"]
ATTACK_IDS = [c.id for c in build_cards() if c.trunk == "X"]
PILOT_IDS = [c.id for c in build_cards() if c.pilot]
assert len(MANDATORY_IDS) == 64, len(MANDATORY_IDS)
assert len(ATTACK_IDS) == 12
assert len(PILOT_IDS) == 10, PILOT_IDS


def lineages(tier: dict | None = None) -> dict:
    from .common import lane_ids
    t = tier or TIERS["T3"]
    return {"discovery": lane_ids("discovery", t), "transfer": lane_ids("transfer", t),
            "confirmation": lane_ids("confirmation", t), "pilot": lane_ids("pilot", {**t, "pilot_worlds": 4})}


def write_manifest(cards: list | None = None, note: str = "") -> dict:
    cards = cards if cards is not None else build_cards()
    doc = {"program": "V14 — The Routed Reader", "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "spec": "V14_SPEC.md", "allowed_states": list(STATES), "resolved_states": list(RESOLVED),
           "tiers": TIERS, "expansions": EXPANSIONS, "lineages": lineages(), "selected_tier": None,
           "mandatory": MANDATORY_IDS, "attacks": ATTACK_IDS, "pilot_cards": PILOT_IDS, "note": note,
           "cards": [c.to_dict() for c in cards]}
    write_json_atomic(MANIFEST, doc, newline=None)
    return doc


def write_cells_template(cards: list | None = None) -> dict:
    cards = cards if cards is not None else build_cards()
    doc = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "by_tier": {}}
    for tname, t in TIERS.items():
        doc["by_tier"][tname] = {c.id: {lane: expected_cells(c, t, lane) for lane in c.lanes} for c in cards}
    write_json_atomic(CELLS_TEMPLATE, doc, newline=None)
    return doc


def write_source_lineages() -> dict:
    """Spec §10: root seeds by lane, card, world family, maker, reader and replicate are derived from
    named strings; this record states the naming so ancestry can be audited without the code."""
    from .common import LANE_BASE, LANE_CAP, seed
    doc = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed_rule": "SEED_OFFSET_V14 + crc32('v14|' + name) % 1e6; never hash()",
           "world_seed": "world|<lane>|<wid>", "unit_seed": "<lane>|<card>|w<wid>|r<rep>|<tag>", "lane_base": LANE_BASE, "lane_cap": LANE_CAP,
           "examples": {"world|discovery|0": seed("world|discovery|0"), "world|transfer|2000": seed("world|transfer|2000"),
                        "world|confirmation|1000": seed("world|confirmation|1000"), "world|pilot|9000": seed("world|pilot|9000")},
           "lineages": lineages()}
    write_json_atomic(v14_dir() / "SOURCE_LINEAGES.json", doc)
    return doc


def write_construction_identities() -> dict:
    """Spec §10: the constructions the cards rely on, stated once: the equivalence class, the four
    routes, the owner variables, the region factorial, and the foraging kinds."""
    from .communication import BELIEFS, CORRECT, POLICIES, PRIVATE, REGIONS, SUPPORT, APPRAISALS
    from .foraging import KINDS
    from .world import N_ACT, N_FEAT, N_GOAL, N_PLAN, N_PREF, PLAN_EQUIVALENCE, ROUTE_COST, ROUTES, T_STEPS
    doc = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "grid": {"plans": N_PLAN, "goals": N_GOAL, "preferences": N_PREF, "states": N_PLAN * N_GOAL * N_PREF},
           "actions": N_ACT, "features": N_FEAT, "steps": T_STEPS, "routes": list(ROUTES), "route_cost": ROUTE_COST,
           "process_equivalence": {f"{k[0]},{k[1]}": f"{v[0]},{v[1]}" for k, v in PLAN_EQUIVALENCE.items()},
           "source_factorial": {"belief": BELIEFS, "support": SUPPORT, "appraisal": APPRAISALS, "policy": POLICIES, "correct": CORRECT, "private": PRIVATE},
           "regions": REGIONS, "foraging_kinds": list(KINDS)}
    write_json_atomic(v14_dir() / "CONSTRUCTION_IDENTITIES.json", doc)
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
    write_json_atomic(CELLS, doc, newline=None)
    return doc


def load_cells() -> dict | None:
    return json.loads(CELLS.read_text(encoding="utf-8")) if CELLS.exists() else None


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return write_manifest()
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def save_manifest(doc: dict) -> None:
    doc["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json_atomic(MANIFEST, doc, newline=None)


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
    write_json_atomic(COVERAGE, cov, newline=None)
    return cov


def add_amendment(card: str, original: dict, replacement: dict, reason: str) -> None:
    doc = json.loads(AMENDMENTS.read_text(encoding="utf-8")) if AMENDMENTS.exists() else {"amendments": []}
    doc["amendments"].append({"card": card, "original": original, "replacement": replacement, "reason": reason,
                              "when": time.strftime("%Y-%m-%dT%H:%M:%S")})
    write_json_atomic(AMENDMENTS, doc)


def claim(card: str, ceiling: str, sentence: str, state: str) -> None:
    """The claim ledger: one public sentence per resolved card with its ceiling (spec §10)."""
    doc = json.loads(CLAIM_LEDGER.read_text(encoding="utf-8")) if CLAIM_LEDGER.exists() else {"claims": {}}
    doc["claims"][card] = {"ceiling": ceiling, "sentence": sentence, "state": state, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json_atomic(CLAIM_LEDGER, doc)
