"""The V12 queue manifest: every mandatory card, its metadata, and its state (spec section 5,
section 17.4). Written to results/v12/QUEUE_MANIFEST.json and updated after every card.
"""
from __future__ import annotations

import json
import time

from . import v12_dir
from .common import CONFIRMATION_IDS, DISCOVERY_IDS
from .schemas import Card, RESOLVED, STATES, card_from_dict

MANIFEST = v12_dir() / "QUEUE_MANIFEST.json"
COVERAGE = v12_dir() / "COVERAGE.json"
RUNTIME = v12_dir() / "RUNTIME.json"

PKG = "ghostscale.validation.soundingline.v12.cards"


def _c(cid, trunk, wave, question, construction, target, estimand, null, alt, rival,
       unit="world", paths=("exact",), gates=("live", "positive", "placebo"), deps=(), cpu=5.0):
    return Card(id=cid, trunk=trunk, wave=wave, question=question, construction=construction,
                target=target, independent_unit=unit, primary_estimand=estimand,
                null_expectation=null, alternative_expectation=alt, strongest_rival=rival,
                solver_paths=list(paths), gates_required=list(gates), depends_on=list(deps),
                discovery_worlds=list(DISCOVERY_IDS), confirmation_worlds=list(CONFIRMATION_IDS),
                module=f"{PKG}.trunk_{trunk.lower()}:run_{cid}",
                output=f"results/validation/soundingline/v12/{cid}.json",
                completion_marker=f"results/validation/soundingline/v12/{cid}.produced",
                estimated_cpu_minutes=cpu)


def build_cards() -> list:
    C = []
    # ---- I: integrity ------------------------------------------------------------------------
    C += [
        _c("I01", "I", 0, "Do reconstructed inner loops reproduce V11's anchors?",
           "V12 world at default parameters", "S-15 and S-14 anchors", "max abs deviation from committed",
           "deviation within frozen tolerance", "a different world", "seed", cpu=2),
        _c("I02", "I", 1, "Which parameters carry S-14 and S-15 across randomized architectures?",
           "50+ random world draws", "reproduction surfaces", "survival rate and Morris/Sobol sensitivities",
           "results survive across draws", "results depend on narrow parameter regions", "particular parameter values",
           unit="architecture draw", cpu=40),
        _c("I03", "I", 0, "Where does mean-field inference become confidently wrong?",
           "small coupled worlds", "exact vs PyMDP posterior", "max posterior deviation and policy disagreement by coupling",
           "agreement in factor-independent worlds", "divergence under coupling", "solver approximation",
           paths=("exact", "pymdp"), cpu=10),
        _c("I04", "I", 0, "Do no-information, shuffled, and false-correspondence nulls return calibrated uncertainty?",
           "maker-independent emissions; shuffles; permuted correspondence", "posterior calibration",
           "posterior mass on truth vs chance", "chance", "above chance would be a leak", "residual structure", cpu=4),
        _c("I05", "I", 0, "Does every maker generator realize its assigned latents?",
           "profile, regime, expertise, habit manipulations", "emission change under manipulation",
           "JS distance between manipulated and unmanipulated emissions", "manipulation reaches the emitter",
           "no change would be VOID", "a dead knob", cpu=2),
        _c("I06", "I", 0, "Are seeds stable and lineages disjoint?", "manifest and seed derivations",
           "reproducibility invariants", "invariant checks", "all hold", "a violation stops the program",
           "hash randomisation", gates=("identity",), cpu=1),
    ]
    # ---- S: self and similarity ---------------------------------------------------------------
    C += [
        _c("S01", "S", 1, "Can a reader measure its own production model?", "reader as maker in two domains",
           "self-profile estimate", "held-out self log score minus frequency baseline", "no better than frequency",
           "self-model predicts held-out self choices", "frequency baseline", unit="reader", cpu=3),
        _c("S02", "S", 1, "Does the similarity ruler recover planted orderings one axis at a time?",
           "pairs varying one axis", "distance matrix", "rank correlation with planted ordering per axis",
           "no ordering recovered", "each axis recovered independently", "source-label identification", cpu=3),
        _c("S03", "S", 1, "What does reader expertise cost at off-ceiling difficulty?",
           "expertise corruption x length x transmission x separation", "goal and profile recovery",
           "log-score gap expert minus corrupted where neither is at ceiling", "zero gap", "expertise reduces evidence needed",
           "ceiling", cpu=8),
        _c("S04", "S", 2, "Does self-first beat an information-matched generic prior?",
           "six prior routes at fixed compute", "profile posterior and hidden next goal",
           "self-first log score minus matched-generic, by self-maker distance", "no gain",
           "selective gain near self", "coordinate privilege", cpu=10),
        _c("S05", "S", 2, "How does the self-first gain vary with distance and evidence?",
           "five distances x evidence doses", "phase diagram", "gain surface and self-directed error",
           "flat surface", "helpful near, harmful far, correction with evidence", "closeness", cpu=10),
        _c("S06", "S", 2, "Does diagnostic conflict correct projection?", "self-compatible then conflicting evidence, both orders",
           "correction dynamics", "anchoring, correction slope, residual self bias", "no order effect and full correction",
           "anchoring with slow correction", "order effect", cpu=8),
        _c("S07", "S", 3, "Does the self route predict a hidden continuation?", "history revealed, next choice hidden",
           "hidden continuation log score", "self-first minus generic on the hidden step", "no gain", "gain",
           "direct frequency", cpu=6),
        _c("S08", "S", 3, "Does the frozen self route transfer to fresh makers and a fresh domain?",
           "frozen settings on untouched worlds", "transfer", "gain on fresh domain", "collapse", "transfer",
           "overfitting to discovery worlds", cpu=8),
        _c("S09", "S", 3, "Does self-first improve calibration and selective refusal?", "abstention allowed",
           "risk-coverage", "risk at matched coverage", "no improvement", "better risk at coverage", "overconfidence", cpu=6),
        _c("S10", "S", 3, "Which similarity components retain independent predictive value?",
           "full maker x reader x axis matrix", "hierarchical decomposition", "partial R2 per axis",
           "none", "policy and process carry it", "surface similarity", cpu=6),
    ]
    # ---- Q: active interrogation --------------------------------------------------------------
    C += [
        _c("Q01", "Q", 1, "Does the PyMDP reader prefer the most informative probe?", "probes with known EIG",
           "probe choice", "agreement with exact EIG ranking", "random", "exact ranking", "utility-only agent",
           paths=("exact", "pymdp", "baseline"), cpu=6),
        _c("Q02", "Q", 1, "Does the reader choose the diagnostic commission?", "commissions targeting drives",
           "commission choice", "information captured vs random and uncertainty sampling", "no better than random",
           "diagnostic commissions preferred", "probe cost", paths=("exact", "pymdp", "baseline"), cpu=6),
        _c("Q03", "Q", 2, "Does self-first active episode selection beat uncertainty sampling?",
           "six episodes, inspect two, predict the seventh", "hidden-episode prediction",
           "final log score by selection policy", "no difference", "self-first selects better", "oracle pair",
           paths=("exact", "baseline"), cpu=8),
        _c("Q04", "Q", 2, "Does the reader buy context when it discriminates makers?",
           "artifact, biography, prior work, tool records, reputation at cost", "context purchase policy",
           "expected discrimination bought per unit cost", "buys polished context", "buys discriminating context",
           "polish", paths=("exact", "pymdp"), cpu=6),
        _c("Q05", "Q", 2, "What is the cost and stopping frontier?", "inspection and integration cost sweeps",
           "regret vs exact optimum", "regret, unnecessary and missed probes", "large regret", "near-optimal stopping",
           "premature stopping", paths=("exact", "pymdp"), cpu=8),
        _c("Q06", "Q", 3, "Does active selection survive an adversarial probe environment?",
           "concealer anticipates probes", "information gain under adversary", "gain vs uncertainty sampling",
           "walks into planted evidence", "still helps", "planted evidence", paths=("exact", "pymdp"), cpu=8),
    ]
    # ---- B: bard and concealer ----------------------------------------------------------------
    C += [
        _c("B01", "B", 1, "Do the three regimes differ only in inferential correspondence?",
           "bard, neutral, concealer at matched surface", "surface matching", "entropy and pair-mass equality",
           "matched", "unmatched cheap feature", "a cheap feature", cpu=3),
        _c("B02", "B", 1, "Does the cooperative assumption help on bards and cost on concealers?",
           "regime x reader assumption", "sample efficiency and confident error", "log score by cell",
           "universal benefit", "selective benefit and concealer cost", "an easier likelihood", cpu=8),
        _c("B03", "B", 2, "Can a reader learn a source's regime and recover after it changes?",
           "source histories with regime switches", "regime posterior", "regime recovery time", "no learning",
           "recovery", "source identity", cpu=8),
        _c("B04", "B", 2, "Does an active challenge separate scaffolding from strategic shaping?",
           "commission challenges", "regime discrimination", "log score by selection policy", "no gain",
           "regime-aware selection wins", "random challenge", paths=("exact", "pymdp"), cpu=8),
        _c("B05", "B", 2, "Where is partial concealment unreadable, and where does effort become the signal?",
           "deflection, omission, mimicry, mixed cues, bounded budget", "readability by concealment type",
           "log score and confident error", "unreadable everywhere", "signatures by type", "budget", cpu=10),
        _c("B06", "B", 3, "Does accurate reconstruction of an adversary improve exploitation rather than cooperation?",
           "accurate model of a concealer", "prediction, trust, adoption, payoffs", "own payoff vs maker payoff",
           "understanding equals cooperation", "understanding improves self-protection", "conflation", cpu=6),
    ]
    # ---- U: uptake bridge ---------------------------------------------------------------------
    C += [
        _c("U01", "U", 1, "Do the bridge identities hold?", "posterior -> C_AIF -> policy",
           "identities", "bit-identity at zero weight; direction at oracle; no move at uniform", "hold",
           "a violation is a construction bug", "direct injection", gates=("identity", "placebo", "positive"), cpu=2),
        _c("U02", "U", 1, "How do reconstruction accuracy and uptake weight interact?", "accuracy x weight",
           "policy behaviour", "regret and wrong-direction movement by cell", "weight alone drives it",
           "accuracy gates useful movement", "injection", cpu=6),
        _c("U03", "U", 2, "Which posterior representation gives the best downstream policy?",
           "MAP, mean, mixture, lower-confidence, gated, oracle, none", "downstream regret",
           "regret and catastrophic movement", "MAP best", "uncertainty-aware best under equifinality",
           "MAP", cpu=6),
        _c("U04", "U", 2, "Are competence, reliability, relevance, and value similarity separable?",
           "full cross", "policy movement by factor", "factor effects", "confounded", "separable",
           "substitution", cpu=8),
        _c("U05", "U", 2, "Which channel moves: process, belief, preference, or imitation?",
           "separate probes after exposure", "channel-specific movement", "per-channel deltas",
           "one aggregate", "process moves, preference does not", "aggregate score", cpu=6),
        _c("U06", "U", 3, "When does understanding worsen the reader's own task?", "divergent competent makers, concealers",
           "negative transfer", "own-task regret", "understanding helps", "exploitation", "conflation", cpu=6),
        _c("U07", "U", 3, "Is an update reversible under reliable counterevidence?", "counterevidence after update",
           "reversal", "harmful movement reversed, process retained", "irreversible", "reversible", "anchoring", cpu=6),
        _c("U08", "U", 3, "How do repeated small updates accumulate?", "longitudinal streams", "cumulative movement",
           "cumulative and reversal", "no accumulation", "accumulation", "single update", cpu=8),
    ]
    # ---- R: residue and preference ------------------------------------------------------------
    C += [
        _c("R01", "R", 1, "Is every planted latent separately recoverable from records?", "orthogonal factorial world",
           "latent separability", "conditional entropy of each latent given others", "coarsening", "separable",
           "collinearity", cpu=4),
        _c("R02", "R", 1, "Which estimator recovers the standing profile and predicts held-out choices?",
           "six estimators", "profile recovery and held-out prediction", "held-out log score by estimator",
           "all equal", "constrained inversion beats partialling", "partialling", cpu=8),
        _c("R03", "R", 2, "Do goal-dependent habits break subtraction?", "habits that store prior values",
           "estimator robustness", "recovery with interacting habits", "robust", "partialling deletes signal",
           "habit shadows", cpu=8),
        _c("R04", "R", 2, "Does the recovered profile predict structurally equivalent choices in a second domain?",
           "infer in domain one, predict in domain two", "cross-domain prediction", "log score in domain two",
           "identity only", "standing tradeoff transfers", "habit", cpu=6),
        _c("R05", "R", 2, "Does the posterior reflect opportunity strength rather than frequency?",
           "same choice under near tie vs large cost", "opportunity ruler", "posterior shift by opportunity strength",
           "frequency", "strength-weighted", "count reader", cpu=4),
        _c("R06", "R", 2, "Are current goal and standing profile jointly recoverable?", "full cross with aligned and opposed",
           "joint recovery", "interaction score", "one only", "both", "confound", cpu=6),
        _c("R07", "R", 3, "Does the profile predict prospective and counterfactual choices?", "changed cost, new domain, commission, new goal",
           "prospective prediction", "log score on prospective targets", "retrospective only", "prospective gain",
           "fit", cpu=6),
        _c("R08", "R", 3, "Do readers abstain where profiles are observationally equivalent?", "equifinal profile pairs",
           "identifiability boundary", "abstention and records separation", "false confidence", "abstain then separate",
           "overclaiming", cpu=4),
    ]
    # ---- T: supply ----------------------------------------------------------------------------
    C += [
        _c("T01", "T", 1, "What is the exact conditional-information ledger?", "generative construction",
           "entropies and conditional MI", "MI matrix", "deterministic edges", "measured edges", "construction", cpu=4),
        _c("T02", "T", 1, "What does supplying each latent buy for each other latent?", "no, true, shuffled, wrong, uncertain supply",
           "supply matrix", "recovery gain by supply type", "no gain", "gains", "hard labels", cpu=10),
        _c("T03", "T", 2, "Does supplying the standing profile improve process recovery?", "profile supplied",
           "drives to process edge", "gain on one common scale vs process to goal", "zero", "nonzero", "ceiling", cpu=6),
        _c("T04", "T", 2, "Does supplying the mechanic unlock goal, process, and profile?", "correct, related, wrong, generic mechanic",
           "expertise as entry point", "recovery gain by mechanic type", "no gain", "gain", "process trace supplied", cpu=6),
        _c("T05", "T", 2, "Does apparent directionality change with difficulty?", "supply battery across difficulty",
           "entry-point map", "gains in cells where both can move", "ceiling artifact", "stable shape", "ceiling", cpu=8),
        _c("T06", "T", 3, "Which topology predicts the supply matrix?", "chain, river, triangle, common cause, factor graph",
           "topology comparison", "held-out intervention prediction", "equivalence class", "one wins", "factorization", cpu=6),
    ]
    # ---- D: director topology -----------------------------------------------------------------
    C += [
        _c("D01", "D", 1, "Can matched multi-agent artifacts be produced under six ecologies?", "six ecologies",
           "realization", "match on quality, counts, style", "unmatched", "matched", "cheap feature", cpu=3),
        _c("D02", "D", 1, "How far does an upstream intervention reach?", "interventions at five levels",
           "causal reach", "downstream change fraction by level", "equal reach", "director reaches furthest",
           "count of decisions", cpu=6),
        _c("D03", "D", 2, "Can artifacts distinguish a central director from a shared brief?", "matched coherence",
           "artifact-only identification", "accuracy vs coherence baseline", "floor", "above baseline", "coherence", cpu=8),
        _c("D04", "D", 2, "Who controlled each decision level?", "contribution-level attribution", "attribution",
           "per-level accuracy", "token share", "level-specific", "token share", cpu=6),
        _c("D05", "D", 2, "Which contributor signals survive rewriting?", "rewrite ladder", "signal survival",
           "attribution by rewrite strength", "all die", "upstream reach survives", "style", cpu=6),
        _c("D06", "D", 3, "Can the director's next intervention be predicted?", "hidden next intervention",
           "prospective prediction", "log score vs baseline", "no gain", "gain", "coherence", cpu=6),
        _c("D07", "D", 3, "Are ratification and failure to notice distinguishable?", "identical artifacts, different histories",
           "non-identifiability", "abstention then later separation", "false confidence", "abstain", "overclaiming", cpu=4),
    ]
    # ---- F: flattened intent ------------------------------------------------------------------
    C += [
        _c("F01", "F", 1, "Does the layered/flattened manipulation reach the emitter?", "layered, flattened, non-invertible",
           "live gate", "bit difference under manipulation", "no change", "change", "dead knob", cpu=2),
        _c("F02", "F", 1, "Are the worlds matched on density and difficulty?", "marginal matching", "matching",
           "marginal and MI equality", "unmatched", "matched", "more signal", cpu=3),
        _c("F03", "F", 2, "Which ruler distinguishes layered from flat?", "ruler tournament", "topology recovery",
           "AUC by ruler", "none", "dependency recovery", "sequence baseline", cpu=8),
        _c("F04", "F", 2, "Can fewer goals, erased dependency, and low effort be told apart?", "adaptive flattening",
           "three-way discrimination", "accuracy", "confounded", "separable", "effort", cpu=6),
        _c("F05", "F", 3, "Does a high-level hand survive local flattening and rewriting?", "F x D cross",
           "survival", "attribution under rewrite", "erased", "survives", "template", cpu=8),
    ]
    # ---- X: adversarial matrix, applied to trunk headlines ------------------------------------
    for xid, q in [("X01", "surface/source match"), ("X02", "policy match, source change"),
                   ("X03", "false context"), ("X04", "equifinal history"), ("X05", "prior permutation"),
                   ("X06", "evidence-order reversal"), ("X07", "architecture randomization"),
                   ("X08", "solver substitution"), ("X09", "cheap baseline"), ("X10", "mixed control"),
                   ("X11", "adaptive adversary"), ("X12", "fresh confirmation")]:
        C.append(_c(xid, "X", 4, f"Does the top valid result in each trunk survive: {q}?",
                    "attack applied to promoted results", "survival", "effect under attack vs unattacked",
                    "survives", "dies", q, cpu=6))
    for c in C:
        if c.trunk == "X":
            c.module = f"{PKG}.trunk_x:run_{c.id}"
    return C


def write_manifest(cards: list | None = None, note: str = "") -> dict:
    cards = cards if cards is not None else build_cards()
    doc = {"program": "V12 — The Other Model", "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "spec": "docs/versions/v12-the-other-model/V12_SPEC.md", "allowed_states": list(STATES), "resolved_states": list(RESOLVED),
           "lineages": {"discovery": DISCOVERY_IDS, "confirmation": CONFIRMATION_IDS},
           "amendments": [], "note": note,
           "cards": [c.to_dict() for c in cards]}
    MANIFEST.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


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
