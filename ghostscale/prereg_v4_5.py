"""V4.5 acceptance criteria, as executable, hash-locked code.

Same discipline as ``prereg_v3`` and ``prereg_v4``: the criteria live here as functions the
experiments and the nulls both call, and the JSON file records their parameters plus a content
hash, so the written criterion and the applied criterion cannot drift.

V4.5 pre-registers three things, and each one is capable of returning the unwelcome answer:

  E21   whether the active-inference machinery is load-bearing at all. V4.5 §7: "E21 must be
        able to return 'the machinery is unnecessary'", and that outcome goes in the first
        line of its section.
  E28   whether beta does separable work, or collapses into kappa_p or theta.
  E29   whether the three gates dissociate, or whether the decomposition is a relabeling that
        is not earning its parameters.

-----------------------------------------------------------------------------------------
WHY THE THRESHOLDS ARE WHERE THEY ARE.

Every threshold below is anchored to a quantity that was measured BEFORE V4.5 existed, and
none is anchored to a value V4.5 produces. Where a criterion needs a scale, it is expressed
as a FRACTION OF A REFERENCE CELL measured in the same run rather than as a bare number —
the device V4 used for the N18 floor, for the same reason: a bare number can be slid until it
is cleared, and a ratio states the claim instead.

The confidence and disagreement thresholds come from V1's E2 (results/e2_cell_stats.csv):

    GHOST / SIG_CREATOR    within 0.0896   between 1.3793      the dissociation
    GHOST / SIG_GHOST      within 1.2926   between 1.3784      uncertainty, correctly held
    ceiling                                       1.3863       ln(4)

The dissociation is the JOINT of the two columns, and the second row is why: high
disagreement alone is not the signature, because truthfully-labelled synthetic content
produces it too while being honest about its uncertainty. What distinguishes fabrication is
that the disagreement is accompanied by CONFIDENCE. A heuristic can be confidently wrong; the
pre-registered claim is that it should not produce the joint pattern.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import foreign as FN
from .config import Config

# --------------------------------------------------------------------------- #
# E21 — the dissociation, and what counts as reproducing it.
#
# Anchored to E2 with deliberate slack, so that an arm which produces the phenomenon in a
# recognisably weaker form still counts as reproducing it. The slack runs AGAINST the
# framework: it makes it easier for a baseline to be credited, which is the direction the
# spec wants the benefit of the doubt to run ("if arm C or D reproduces either signature, the
# active-inference apparatus is scaffolding rather than mechanism").
# --------------------------------------------------------------------------- #
DISSOC_CONFIDENT_ENTROPY = 0.35   # within-observer entropy at or below this counts as confident.
                                  # E2's dissociation cell measured 0.0896; the truthful-GHOST
                                  # control, which must NOT qualify, measured 1.2926. The
                                  # threshold sits between them, nearer the control.
DISSOC_DISAGREE_ENTROPY = 1.10    # between-observer entropy at or above this counts as
                                  # disagreement. 79% of the ln(4) = 1.386 ceiling; E2 measured
                                  # 1.3793.

# The second discriminator, from E19. Sustained expensive attention that never resolves is a
# prediction about an agent that keeps EXPECTING to learn. A pure effort heuristic has no
# expectation to be wrong about.
FOREIGN_ENGAGED_FLOOR = 0.50      # E19 measured 0.746 of free steps DEEP on foreign content
FOREIGN_UNRESOLVED_ENTROPY = 0.50 # same threshold prereg_v4.crash_signature uses for "unresolved"

E21_ARMS = ("A_active_inference", "B_bayesian_always_deep", "C_label_truster",
            "D_effort_heuristic", "E_no_tom_classifier")
E21_BASELINE_ARMS = ("B_bayesian_always_deep", "C_label_truster", "D_effort_heuristic",
                     "E_no_tom_classifier")
# The two arms the spec singles out. C and D are the ones whose success would demote the
# apparatus to scaffolding: both are heuristics with no generative model of another agent.
E21_SCAFFOLDING_ARMS = ("C_label_truster", "D_effort_heuristic")

# --------------------------------------------------------------------------- #
# E28 — beta as inferred rationality.
# --------------------------------------------------------------------------- #
E28_BETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)      # true beta used to generate human content
E28_RECOVERY_RHO = 0.80          # Spearman(true beta, recovered E[beta]) at or above this
                                 # means beta is identifiable from content at all
E28_UPDATE_RHO = 0.80            # Spearman(true beta, psi_analogue). The PREDICTION: update
                                 # magnitude falls as beta falls.
E28_ACCURACY_FLOOR = 0.75        # goal accuracy must hold at or above this over the upper half
                                 # of the beta grid for "legible but unmoving" to be the claim.
                                 # Below it the observer is not reading the intent correctly and
                                 # the aesthetic category being described does not exist.
E28_BETA0_ENTROPY_TOL = 0.25     # nats. At beta = 0, E28 must land within this of E19's
                                 # exploratory-human cell. V4.5 §3.3 makes this a REQUIRED
                                 # consistency check: if a continuous beta near zero does not
                                 # recover the discrete EXPLORE result, the two are not the same
                                 # axis and the identification in §3.3 is wrong.
E28_BETA1_TOL = 0.05             # the N-series check. At beta = 1 across all conditions, V4.5
                                 # must reproduce V4 within tolerance, or beta has been wired
                                 # into the wrong pipeline position.

# --------------------------------------------------------------------------- #
# E29 — do the three gates dissociate?
#
# The update thresholds are FRACTIONS OF THE REFERENCE CELL (high kappa_p, beta = 1, theta
# open) measured in the same run. A bare nats threshold would be meaningless across a design
# that deliberately varies how much there is to update about.
# --------------------------------------------------------------------------- #
E29_UPDATE_NONE = 0.05           # <= 5% of the reference update counts as "none"
E29_UPDATE_LOW = 0.50            # (5%, 50%] counts as "low"; above 50% counts as "high"
E29_ENGAGED_HIGH = 0.50          # same threshold as the crash signature, deliberately
E29_RESOLVED_ENTROPY = 0.50      # same threshold as E19's convergence clause, deliberately
E29_DIVERGENCE_SPIKE = 2.0       # closed theta must show at least this multiple of the
                                 # reference cell's value divergence. It is the ONLY measure on
                                 # which closed theta is predicted to differ from low beta, so
                                 # it is the discriminator the decisive contrast rests on.


# --------------------------------------------------------------------------- #
# E20 — the omega sweep, with V4.5 §6's required addition.
#
# WRITTEN TO ITS OWN FILE, and that is not tidiness. E21, E28 and E29 had already run against
# ``v4_5_preregistration.json`` by the time E20 was built, and that file is hash-locked
# precisely so it cannot acquire new content after its experiments have reported. E20's
# criteria therefore live in ``v4_5_e20_preregistration.json``, written before E20 runs and
# leaving the earlier lock untouched. Adding to the locked file with force=True would have
# been the exact failure V4 spec §7 names.
# --------------------------------------------------------------------------- #
E20_OMEGA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)

# The misspecification signature. V4 §E20 predicts confident fabrication peaks at LOW BUT
# NONZERO omega: enough in-family structure to make an explanation seem available, not enough
# to make it correct. Scored as a single index so "peaks at" is a statement about a curve
# rather than about whichever cell a reader looks at first.
E20_CONFIDENT_ENTROPY = DISSOC_CONFIDENT_ENTROPY   # deliberately E21's, so the two are
E20_DISAGREE_ENTROPY = DISSOC_DISAGREE_ENTROPY     # commensurable and neither is bespoke


def fabrication_index(within: float, between: float, n_goals: int = 4) -> float:
    """Confident AND disagreeing, on one axis, in [0, 1].

        (1 - H_within/ln G) * (H_between/ln G)

    The product is the point: it is high only when both hold at once, which is the E2
    signature, and it is near zero for an observer that is honestly uncertain (first term
    small) or for one that is confident and agreeing (second term small). Reported alongside
    its two components, never instead of them.
    """
    ceiling = float(np.log(n_goals))
    conf = max(0.0, 1.0 - float(within) / ceiling)
    dis = min(1.0, max(0.0, float(between) / ceiling))
    return float(conf * dis)


def fabrication_index_strict(within: float, accuracy: float, n_goals: int = 4) -> float:
    """Confident AND WRONG, in [0, 1]:  (1 - H_within/ln G) * (1 - accuracy).

    ADDED AFTER E20 RAN, and logged as a deviation. ``fabrication_index`` above is the
    pre-registered measure and it has a confound that the sweep made visible: between-observer
    disagreement is not the same thing as being wrong. At omega = 0.10 the observer disagrees
    with its peers (0.702 nats) while being RIGHT 78% of the time, so part of what the
    pre-registered index scores as fabrication is ordinary partial learning with a minority
    getting it wrong.

    V4 §E20 asks whether the observer "converges confidently on a WRONG in-family goal", so
    wrongness belongs in the measure. This is that measure. Both are reported; the
    pre-registered one still decides the outcome string.

    THIS CANNOT MANUFACTURE THE WELCOME ANSWER AND IT WAS CHECKED BEFORE BEING ADDED: both
    indices peak at the same omega. The strict one changes what the peak MEANS, not where it is.
    """
    ceiling = float(np.log(n_goals))
    conf = max(0.0, 1.0 - float(within) / ceiling)
    return float(conf * max(0.0, 1.0 - float(accuracy)))


def e20_verdict(omegas, engaged, crashed, fabrication, accuracy,
                fabrication_strict=None, engaged_sem=None) -> dict:
    """Where along omega does the metabolic prediction flip, and where does fabrication peak?

    V4.5 §6 makes engagement and ``crash_signature`` PRIMARY outcomes rather than secondary
    columns, because E19's unpredicted finding — observers do not disengage from goal-foreign
    content — inverts V1-V3's metabolic prediction, and H1 and H3 in the preprint both assume
    goal-empty content. Goal-empty predicts an autonomic drop within 2-4 seconds; goal-foreign
    predicts sustained arousal with no resolution. A single pupillometry measurement
    discriminates them. E20's crossing point is what tells a human study where to look.

    Two quantities, both pre-registered:

      * the ENGAGEMENT CROSSING — the omega at which sustained attention falls through 0.50.
        Below it, foreign content holds attention and never resolves; above it, the observer
        reads the content and stops. This is the number a pupillometry design needs.
      * the FABRICATION PEAK — where the confident-and-disagreeing index is maximised. V4 §E20
        predicts an interior maximum. A peak at omega = 0 would mean the unidentifiability
        model's prediction was right after all and the reframe buys nothing here.
    """
    om = np.asarray(omegas, dtype=float)
    eng = np.asarray(engaged, dtype=float)
    fab = np.asarray(fabrication, dtype=float)

    crossing = None
    for i in range(len(om) - 1):
        a, b = eng[i], eng[i + 1]
        if (a - 0.5) * (b - 0.5) <= 0 and a != b:
            t = (0.5 - a) / (b - a)
            crossing = float(om[i] + t * (om[i + 1] - om[i]))
            break

    peak_i = int(np.argmax(fab))
    interior = bool(0 < peak_i < len(om) - 1)

    if interior:
        statement = (
            f"Confident fabrication peaks at omega = {om[peak_i]:.2f}, in the interior of the "
            f"range. That is V4 §E20's prediction and the unidentifiability model could not "
            f"generate it: partial overlap supplies enough in-family structure to make an "
            f"explanation seem available without making it correct, so the observer commits. "
            f"It is the strongest single argument for the reframe.")
        outcome = "INTERIOR_PEAK"
    elif peak_i == 0:
        statement = (
            "Confident fabrication is maximal at omega = 0 and falls monotonically. The "
            "predicted interior peak is absent, so on this measure the misspecification model "
            "behaves like the unidentifiability model it replaced and the reframe buys nothing "
            "here. Reported as a failed prediction rather than as a curve with a shoulder.")
        outcome = "PEAK_AT_ZERO"
    else:
        statement = (
            "Confident fabrication is maximal at full overlap, which is the human condition "
            "and should be the case where the observer is confident and CORRECT. Check the "
            "accuracy column before reading this as fabrication at all.")
        outcome = "PEAK_AT_ONE"

    strict_block = None
    if fabrication_strict is not None:
        fs = np.asarray(fabrication_strict, dtype=float)
        si = int(np.argmax(fs))
        strict_block = {
            "peak_omega": float(om[si]),
            "peak_value": float(fs[si]),
            "peak_is_interior": bool(0 < si < len(om) - 1),
            "agrees_with_preregistered_index": bool(si == peak_i),
            "why": ("confident AND WRONG, rather than confident and disagreeing. Between-"
                    "observer disagreement is not the same thing as being wrong, and the "
                    "sweep made the difference visible; V4 §E20 asks about converging "
                    "confidently on a WRONG in-family goal."),
        }

    # The crossing is the number a pupillometry design would target, so its RELIABILITY is
    # part of the result rather than a footnote. Engagement in this model is bimodal —
    # observers either sustain attention throughout or drop it almost at once — so a cell
    # mean sitting near 0.50 places the crossing under wide uncertainty even at large
    # observer counts, because the effective sample size is the number of SEEDS.
    crossing_robust = None
    if crossing is not None and engaged_sem is not None:
        sem = np.asarray(engaged_sem, dtype=float)
        near = [i for i in range(len(om)) if abs(eng[i] - 0.5) < 2.0 * sem[i]]
        crossing_robust = bool(not near) or bool(len(near) <= 1)

    return {
        "outcome": outcome,
        "statement": statement,
        "fabrication_peak_omega": float(om[peak_i]),
        "fabrication_peak_value": float(fab[peak_i]),
        "fabrication_peak_is_interior": interior,
        "fabrication_strict": strict_block,
        "engagement_crossing_omega": crossing,
        "engagement_crossing_is_well_determined": crossing_robust,
        "engagement_crossing_note": (
            "the omega at which sustained attention falls through 0.50. Below it, foreign "
            "content holds attention without resolving (E19's finding, which inverts V1-V3's "
            "metabolic prediction); above it, the observer reads the content and stops. A "
            "pupillometry study discriminating H1/H3 should target this value."
            if crossing is not None else
            "engagement never crosses 0.50 across the sweep, so no crossing exists and the "
            "metabolic prediction does not flip anywhere in the range"),
        "any_cell_crashes": bool(np.any(np.asarray(crashed, dtype=bool))),
        "crashing_omegas": [float(o) for o, c in zip(om, crashed) if bool(c)],
    }


def build_preregistration_e20(cfg: Config) -> dict:
    payload = {
        "version": "V4.5 / E20",
        "written_before": "E20 is run",
        "separate_file_because": (
            "v4_5_preregistration.json was already locked and its experiments had already "
            "reported. A hash-locked pre-registration that acquires new content after its "
            "experiments run is not a pre-registration."),
        "omega_grid": list(E20_OMEGA_GRID),
        "primary_outcomes": [
            "engaged_fraction", "crash_signature", "fabrication_index"],
        "why_engagement_is_primary": (
            "V4.5 §6. E19's unpredicted finding is that observers do NOT disengage from "
            "goal-foreign content (0.746 across free steps, no resolution, crash_signature "
            "false in every cell). That inverts V1-V3's metabolic prediction. H1 and H3 in "
            "the preprint both assume goal-empty content and both invert under goal-foreign: "
            "goal-empty predicts an autonomic drop within 2-4 seconds, goal-foreign predicts "
            "sustained arousal with no resolution. A single pupillometry measurement "
            "discriminates them, which makes this the sharpest and cheapest empirical "
            "prediction the framework has produced."),
        "confident_entropy_max": E20_CONFIDENT_ENTROPY,
        "disagreement_entropy_min": E20_DISAGREE_ENTROPY,
        "thresholds_shared_with_e21_because": (
            "the same phenomenon measured in two experiments should not be measured with two "
            "bespoke thresholds"),
        "predicted": (
            "confident fabrication peaks at LOW BUT NONZERO omega — enough in-family "
            "structure to make an explanation seem available, not enough to make it correct "
            "(V4 §E20)"),
        "falsification": (
            "a peak at omega = 0 means the misspecification model behaves like the "
            "unidentifiability model it replaced on this measure, and the reframe buys "
            "nothing here"),
        "outcomes": ["INTERIOR_PEAK", "PEAK_AT_ZERO", "PEAK_AT_ONE"],
        "observer": (
            "four real goals, EXPLORE disabled. E19 established that EXPLORE is inert on "
            "foreign content (0.2036 mass against a 0.2000 flat baseline, chosen slightly "
            "LESS often than chance), so including it would add an arm that E19 has already "
            "shown does nothing while doubling the sweep's cost. Stated so the choice is "
            "visible rather than assumed."),
    }
    payload["content_hash"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return payload


def write_preregistration_e20(cfg: Config, path: Path, force: bool = False) -> dict:
    payload = build_preregistration_e20(cfg)
    path = Path(path)
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("content_hash") != payload["content_hash"]:
            raise RuntimeError(
                f"{path.name} exists with a different content hash; E20's criteria were "
                f"pre-registered and must not change after the fact.")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _canonical(payload: dict) -> str:
    scrubbed = {k: v for k, v in payload.items() if k != "content_hash"}
    return json.dumps(scrubbed, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# E21 criteria.
# --------------------------------------------------------------------------- #
def reproduces_dissociation(within: float, between: float,
                            control_within: float = float("nan")) -> dict:
    """The E2 signature: CONFIDENT and DISAGREEING at the same time.

    Conjunctive, and the conjunction is the whole point. Either clause alone is cheap:
    a heuristic that always outputs a peaked distribution is confident, and one that
    outputs noise disagrees. Producing both simultaneously, on content that contains no
    recoverable goal, is the thing that is claimed to require a generative model of another
    agent.

    ``control_within`` is the SAME CONTENT under a truthful SIG_GHOST label, and it drives a
    SECONDARY measure rather than the primary criterion. The primary criterion above is the
    one V4.5 §2 names and it is left exactly as it was pre-registered. But E2's finding is
    sharper than "confidently wrong": the same content under a truthful label produced within
    1.2926 against the mislabeled cell's 0.0896, so the confidence is INDUCED BY THE LABEL.
    An arm that is confidently wrong regardless of what the label says has reproduced
    confident fabrication without reproducing the label induction, and A2's calibration
    result is a claim about the induction specifically. Both are reported; only the first
    decides the verdict.
    """
    w, b = float(within), float(between)
    conf = bool(np.isfinite(w) and w <= DISSOC_CONFIDENT_ENTROPY)
    dis = bool(np.isfinite(b) and b >= DISSOC_DISAGREE_ENTROPY)
    cw = float(control_within)
    control_uncertain = bool(np.isfinite(cw) and cw > DISSOC_CONFIDENT_ENTROPY)
    return {
        "within_observer": w, "between_observer": b,
        "control_within_observer": cw,
        "confident": conf, "disagreeing": dis,
        "reproduces": bool(conf and dis),
        "control_correctly_uncertain": control_uncertain,
        # SECONDARY. Not part of the verdict; reported alongside it.
        "label_induced": bool(conf and dis and control_uncertain),
        "criterion": (f"within <= {DISSOC_CONFIDENT_ENTROPY} (confident) AND between >= "
                      f"{DISSOC_DISAGREE_ENTROPY} (disagreeing), simultaneously"),
        "secondary_criterion": (
            f"label_induced additionally requires the truthfully-labelled control on the SAME "
            f"content to stay uncertain (within > {DISSOC_CONFIDENT_ENTROPY})"),
    }


def reproduces_foreign_engagement(engaged: float, final_entropy: float,
                                  control_engaged: float = float("nan"),
                                  control_entropy: float = float("nan")) -> dict:
    """E19's second discriminator: sustained expensive attention that never resolves.

    This is a prediction about an agent that keeps EXPECTING to learn and keeps being wrong.
    An effort-allocation heuristic has no expectation to be wrong about, so arm D is
    predicted to fail it. If arm D reproduces it anyway, that is informative about how little
    machinery the phenomenon requires, and it is reported as such rather than explained.

    THE SPECIFICITY CLAUSE, and it is not optional. E19's finding is a CONTRAST, not a level:
    the same observer that keeps paying 75% of free steps on foreign content resolves directed
    human content and stops paying almost entirely (0.00002). Scored as a level instead, two
    arms pass while having no behaviour at all:

      * arm C never updates a goal belief, so it is "unresolved" everywhere for free;
      * arm B has no engagement policy and is DEEP by construction, so it is "sustained"
        everywhere for free.

    So the control cell — ordinary directed human content, same arm — must both RESOLVE and
    DISENGAGE. An arm whose engagement does not respond to content has not reproduced a
    result about engagement responding to content.

    Both clauses were added before any E21 cell ran, from inspection of the arm definitions
    rather than of any result.
    """
    e, h = float(engaged), float(final_entropy)
    sustained = bool(np.isfinite(e) and e >= FOREIGN_ENGAGED_FLOOR)
    unresolved = bool(np.isfinite(h) and h > FOREIGN_UNRESOLVED_ENTROPY)
    ce, ch = float(control_engaged), float(control_entropy)
    control_resolves = bool(np.isfinite(ch) and ch <= FOREIGN_UNRESOLVED_ENTROPY)
    control_disengages = bool(np.isfinite(ce) and ce < FOREIGN_ENGAGED_FLOOR)
    return {
        "engaged_fraction": e, "final_entropy": h,
        "control_engaged_fraction": ce, "control_final_entropy": ch,
        "sustained": sustained, "unresolved": unresolved,
        "control_resolves_human_content": control_resolves,
        "control_disengages_from_human_content": control_disengages,
        "reproduces": bool(sustained and unresolved and control_resolves
                           and control_disengages),
        "criterion": (f"engaged >= {FOREIGN_ENGAGED_FLOOR} AND final entropy > "
                      f"{FOREIGN_UNRESOLVED_ENTROPY} on foreign content, AND on directed "
                      f"human content the same arm RESOLVES (entropy <= "
                      f"{FOREIGN_UNRESOLVED_ENTROPY}) and DISENGAGES (engaged < "
                      f"{FOREIGN_ENGAGED_FLOOR}) — otherwise an arm that never resolves "
                      f"anything, or one that is always engaged by construction, passes for "
                      f"free"),
    }


def e21_verdict(dissociation: dict, foreign_engagement: dict) -> dict:
    """Is the active-inference machinery load-bearing?

    ``dissociation`` and ``foreign_engagement`` each map arm name -> the dict returned by the
    corresponding criterion above.

    ARM A IS THE POSITIVE CONTROL, and it works exactly the way E19's did. If the full
    observer does not itself reproduce a signature in this build, then no baseline's failure
    to reproduce it carries information: the comparison would be measuring the harness rather
    than the arms. That case returns INCONCLUSIVE for that signature rather than crediting
    the framework with a win by default.

    Both outcomes are written here, before the run.
    """
    a_diss = dissociation.get("A_active_inference", {}).get("reproduces")
    a_eng = foreign_engagement.get("A_active_inference", {}).get("reproduces")

    scaffolding_diss = [a for a in E21_SCAFFOLDING_ARMS
                        if dissociation.get(a, {}).get("reproduces")]
    scaffolding_eng = [a for a in E21_SCAFFOLDING_ARMS
                       if foreign_engagement.get(a, {}).get("reproduces")]
    any_baseline_diss = [a for a in E21_BASELINE_ARMS
                         if dissociation.get(a, {}).get("reproduces")]
    any_baseline_eng = [a for a in E21_BASELINE_ARMS
                        if foreign_engagement.get(a, {}).get("reproduces")]

    if not a_diss and not a_eng:
        return {
            "verdict": "INCONCLUSIVE",
            "statement": (
                "The full active-inference observer did not reproduce either signature in "
                "this build, so it is not a usable reference. No baseline's failure carries "
                "information against a control that also fails. Fix the control before "
                "reading the baselines."),
            "positive_control_dissociation": bool(a_diss),
            "positive_control_foreign_engagement": bool(a_eng),
            "dissociation": dissociation,
            "foreign_engagement": foreign_engagement,
        }

    if scaffolding_diss or scaffolding_eng:
        verdict = "MACHINERY_UNNECESSARY"
        statement = (
            "A HEURISTIC REPRODUCES THE SIGNATURE. "
            + (f"Arms {', '.join(scaffolding_diss)} reproduce the simultaneous "
               f"confidence/disagreement dissociation. " if scaffolding_diss else "")
            + (f"Arms {', '.join(scaffolding_eng)} reproduce the sustained-unresolved-"
               f"attention result on foreign content. " if scaffolding_eng else "")
            + "Neither arm holds a generative model of another agent, so the active-inference "
              "apparatus is scaffolding rather than mechanism for the effect it reproduces. "
              "This does not delete the result — the phenomenon is still there — but the "
              "framework's claim to need a theory of mind to produce it does not survive, and "
              "the README must say so where that claim is currently made.")
    elif any_baseline_diss or any_baseline_eng:
        verdict = "MACHINERY_PARTLY_NECESSARY"
        statement = (
            "No pure heuristic (arm C or D) reproduces either signature, but "
            f"{', '.join(sorted(set(any_baseline_diss + any_baseline_eng)))} does. Those arms "
            "retain some of the apparatus, so this locates which component is load-bearing "
            "rather than vindicating the whole of it. The section reports which component and "
            "which signature.")
    else:
        verdict = "MACHINERY_NECESSARY"
        statement = (
            "No baseline reproduces either signature. Disengagement is cost-benefit "
            "arithmetic and several arms reproduce that, as predicted; the simultaneous "
            "confidence/disagreement dissociation and the sustained-unresolved-attention "
            "result are produced only by an observer carrying a generative model of another "
            "agent. The apparatus is load-bearing for the specific effects the framework "
            "claims it for.")

    return {
        "verdict": verdict,
        "statement": statement,
        "positive_control_dissociation": bool(a_diss),
        "positive_control_foreign_engagement": bool(a_eng),
        "scaffolding_arms_reproducing_dissociation": scaffolding_diss,
        "scaffolding_arms_reproducing_foreign_engagement": scaffolding_eng,
        "any_baseline_reproducing_dissociation": any_baseline_diss,
        "any_baseline_reproducing_foreign_engagement": any_baseline_eng,
        "dissociation": dissociation,
        "foreign_engagement": foreign_engagement,
    }


# --------------------------------------------------------------------------- #
# E28 criteria.
# --------------------------------------------------------------------------- #
def spearman(x, y) -> float:
    """Spearman rank correlation, computed here so the criterion has no scipy dependency
    hiding inside it and so ties are handled the one way this module documents."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(v.size, dtype=float)
        # average ranks within ties
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    rx, ry = rank(x), rank(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def e28_verdict(beta_grid, recovered_beta, accuracy, update_magnitude,
                beta0_entropy: float, e19_exploratory_entropy: float) -> dict:
    """Does beta do separable work?

    PREDICTED (V4.5 §4): goal accuracy stays high across the beta range while update
    magnitude falls with beta. That is the missing aesthetic category — *legible, competent,
    and unmoving*. The observer reads the intent correctly and correctly treats it as weak
    evidence.

    FALSIFIED (V4.5 §4): if update magnitude does not fall with beta while accuracy holds,
    beta is not doing separable work and collapses into either kappa_p or theta.

    A third outcome has to exist and it is worth naming before the run, because it is the one
    a careless read would report as the prediction. If accuracy falls WITH the update
    magnitude, beta is doing something — but it is not the predicted thing. Content generated
    at low beta genuinely carries less goal information, so an observer failing to read it is
    the ordinary consequence of a weaker stimulus rather than a distinct gate. The predicted
    result requires the DISSOCIATION: accuracy held while the update collapses.
    """
    rho_recovery = spearman(beta_grid, recovered_beta)
    rho_update = spearman(beta_grid, update_magnitude)

    beta_arr = np.asarray(beta_grid, dtype=float)
    acc_arr = np.asarray(accuracy, dtype=float)
    upper = acc_arr[beta_arr >= float(np.median(beta_arr))]
    accuracy_holds = bool(upper.size and float(np.min(upper)) >= E28_ACCURACY_FLOOR)
    accuracy_holds_throughout = bool(float(np.min(acc_arr)) >= E28_ACCURACY_FLOOR)

    identifiable = bool(np.isfinite(rho_recovery) and rho_recovery >= E28_RECOVERY_RHO)
    update_falls = bool(np.isfinite(rho_update) and rho_update >= E28_UPDATE_RHO)

    beta0_gap = abs(float(beta0_entropy) - float(e19_exploratory_entropy))
    beta0_consistent = bool(beta0_gap <= E28_BETA0_ENTROPY_TOL)

    if update_falls and accuracy_holds_throughout:
        verdict = "BETA_IS_SEPARABLE"
        statement = (
            "Update magnitude falls with beta while goal accuracy holds across the whole "
            "range. The observer reads the intent correctly and correctly treats it as weak "
            "evidence: legible, competent, and unmoving. That dissociation is not expressible "
            "by kappa_p, which would take the legibility with it, or by theta, which would "
            "leave the update intact until the recovered goal became unacceptable. beta is "
            "doing separable work.")
    elif update_falls and accuracy_holds:
        verdict = "BETA_IS_SEPARABLE_OVER_PART_OF_THE_RANGE"
        statement = (
            "Update magnitude falls with beta and accuracy holds over the upper half of the "
            "range but not the whole of it. beta is separable where the stimulus still "
            "carries enough goal information to read; below that point low beta and low "
            "kappa_p become empirically the same condition, which is a real limit on the "
            "decomposition and is reported as one rather than trimmed out of the grid.")
    elif update_falls:
        verdict = "CONFOUNDED_WITH_LEGIBILITY"
        statement = (
            "Update magnitude falls with beta, but accuracy falls with it too. That is the "
            "ordinary consequence of a weaker stimulus rather than a separate gate: content "
            "generated at low beta carries less goal information, so an observer that fails "
            "to read it has not demonstrated anything kappa_p could not explain. The "
            "predicted result required the dissociation and this is not it.")
    else:
        verdict = "BETA_COLLAPSES"
        statement = (
            "Update magnitude does not fall with beta. By V4.5 §4's own falsification "
            "criterion beta is not doing separable work and collapses into either kappa_p or "
            "theta. The three-gate decomposition loses its middle gate and should be reported "
            "as a two-gate model, or abandoned.")

    return {
        "verdict": verdict,
        "statement": statement,
        "spearman_true_vs_recovered_beta": rho_recovery,
        "beta_is_identifiable": identifiable,
        "spearman_true_beta_vs_update": rho_update,
        "update_magnitude_falls_with_beta": update_falls,
        "accuracy_holds_upper_half": accuracy_holds,
        "accuracy_holds_throughout": accuracy_holds_throughout,
        "min_accuracy": float(np.min(acc_arr)) if acc_arr.size else float("nan"),
        "beta0_final_entropy": float(beta0_entropy),
        "e19_exploratory_final_entropy": float(e19_exploratory_entropy),
        "beta0_gap": beta0_gap,
        "beta0_recovers_e19_explore_cell": beta0_consistent,
        "beta0_note": (
            "V4.5 §3.3 requires that at beta = 0 E28 reproduce E19's exploratory-human cell. "
            "If a continuous beta near zero does not recover the discrete EXPLORE result, the "
            "two are not the same axis and the identification of EXPLORE with beta = 0 is "
            "wrong. This is a REQUIRED check and its failure invalidates §3.3's shortcut "
            "regardless of what the rest of E28 shows."),
        "criteria": {
            "recovery_rho": E28_RECOVERY_RHO, "update_rho": E28_UPDATE_RHO,
            "accuracy_floor": E28_ACCURACY_FLOOR, "beta0_entropy_tol": E28_BETA0_ENTROPY_TOL,
        },
    }


# --------------------------------------------------------------------------- #
# E29 criteria.
# --------------------------------------------------------------------------- #
def classify_update(update: float, reference_update: float) -> str:
    """"none", "low" or "high", as a fraction of the reference cell measured in the same run.

    Expressed as a ratio for the N18 reason: a bare nats threshold could be slid until the
    predicted table came out, and it would not survive a change to the update scale. A ratio
    states the claim.
    """
    ref = float(reference_update)
    if not np.isfinite(ref) or ref <= 0:
        return "undefined"
    frac = float(update) / ref
    if frac <= E29_UPDATE_NONE:
        return "none"
    if frac <= E29_UPDATE_LOW:
        return "low"
    return "high"


def e29_cell_signature(engaged: float, final_entropy: float, update: float,
                       reference_update: float, value_divergence: float,
                       reference_divergence: float) -> dict:
    """The three-measure behavioural signature of one E29 cell."""
    return {
        "engaged_fraction": float(engaged),
        "engagement": "high" if float(engaged) >= E29_ENGAGED_HIGH else "low",
        "final_entropy": float(final_entropy),
        "resolution": "yes" if float(final_entropy) <= E29_RESOLVED_ENTROPY else "no",
        "update_magnitude": float(update),
        "update": classify_update(update, reference_update),
        "value_divergence": float(value_divergence),
        "divergence_ratio": (float(value_divergence) / float(reference_divergence)
                             if reference_divergence else float("nan")),
        "divergence_spike": bool(reference_divergence
                                 and float(value_divergence) / float(reference_divergence)
                                 >= E29_DIVERGENCE_SPIKE),
    }


# The table V4.5 §5 pre-registers, transcribed. Keys are the cell names the experiment uses.
E29_PREDICTED = {
    "low_kappa_p_goal_empty": {"engagement": "low", "resolution": "no", "update": "none"},
    "low_kappa_p_goal_foreign": {"engagement": "high", "resolution": "no", "update": "none"},
    "low_beta": {"engagement": "high", "resolution": "yes", "update": "low"},
    "closed_theta": {"engagement": "high", "resolution": "yes", "update": "none"},
}


def e29_verdict(signatures: dict) -> dict:
    """Is the three-gate decomposition real, or a relabeling?

    THE DECISIVE CONTRAST is low beta against closed theta (V4.5 §5). Both produce full
    extraction with little integration. They differ in WHY: low beta says the trajectory is
    weak evidence, closed theta says the recovered goal is unacceptable. If those two cells
    are behaviourally indistinguishable on every measure, the decomposition is not earning its
    parameters and V4.5 §5 says to report it as such rather than keeping it on theoretical
    grounds.

    Note honestly what "distinguishable" can mean here. Within a single artifact, low beta and
    closed theta are predicted to differ on update MAGNITUDE ("low" versus "none") and on the
    divergence spike, and to agree on engagement and resolution. The divergence spike is the
    sharper of the two because it is a positive prediction rather than a difference of degree,
    and it is the one this verdict weights.
    """
    low_beta = signatures.get("low_beta")
    closed_theta = signatures.get("closed_theta")
    if low_beta is None or closed_theta is None:
        return {"verdict": "INCOMPLETE",
                "statement": "E29 did not produce both cells the decisive contrast needs"}

    differs_on_update = bool(low_beta["update"] != closed_theta["update"])
    differs_on_divergence = bool(closed_theta["divergence_spike"]
                                 and not low_beta["divergence_spike"])
    differs_on_engagement = bool(low_beta["engagement"] != closed_theta["engagement"])
    differs_on_resolution = bool(low_beta["resolution"] != closed_theta["resolution"])
    distinguishable = bool(differs_on_update or differs_on_divergence
                           or differs_on_engagement or differs_on_resolution)

    matches = {}
    for name, predicted in E29_PREDICTED.items():
        sig = signatures.get(name)
        matches[name] = ({k: bool(sig.get(k) == v) for k, v in predicted.items()}
                         if sig else None)
    n_matched = sum(1 for name, m in matches.items()
                    if m is not None and all(m.values()))

    if not distinguishable:
        verdict = "DECOMPOSITION_NOT_EARNED"
        statement = (
            "Low beta and closed theta are behaviourally indistinguishable on every measure, "
            "including the divergence spike that was pre-registered as their discriminator. "
            "Three gates with different action points is a harder object to identify than one "
            "scalar — more parameters, more freedom, less falsifiability per experiment — and "
            "E29 was the mitigation. It failed. The decomposition is not earning its "
            "parameters and is reported as such rather than kept on theoretical grounds.")
    elif n_matched == len(E29_PREDICTED):
        verdict = "GATES_DISSOCIATE"
        statement = (
            "All four pre-registered signatures were produced, and the decisive contrast holds: "
            "low beta and closed theta are distinguishable. The three gates act at different "
            "points and produce different behaviour, so the decomposition is separable rather "
            "than a relabeling of one scalar.")
    else:
        verdict = "GATES_PARTLY_DISSOCIATE"
        statement = (
            f"The decisive contrast holds — low beta and closed theta are distinguishable — but "
            f"only {n_matched} of {len(E29_PREDICTED)} pre-registered cell signatures came out "
            f"as written. The decomposition survives its central test and fails some of its "
            f"detail predictions; both are reported, and the cells that missed are reported "
            f"before the ones that hit.")

    return {
        "verdict": verdict,
        "statement": statement,
        "decisive_contrast": {
            "low_beta": low_beta, "closed_theta": closed_theta,
            "differs_on_update": differs_on_update,
            "differs_on_divergence_spike": differs_on_divergence,
            "differs_on_engagement": differs_on_engagement,
            "differs_on_resolution": differs_on_resolution,
            "distinguishable": distinguishable,
        },
        "predicted_vs_measured": matches,
        "n_cells_fully_matching": n_matched,
        "signatures": signatures,
    }


# --------------------------------------------------------------------------- #
# The pre-registration file.
# --------------------------------------------------------------------------- #
def build_preregistration_v4_5(cfg: Config) -> dict:
    payload = {
        "version": "V4.5",
        "written_before": "any V4.5 experiment is run",
        "depends_on": "results/v4_preregistration.json, which is unchanged and still binding",
        "E21_criteria": {
            "arms": list(E21_ARMS),
            "scaffolding_arms": list(E21_SCAFFOLDING_ARMS),
            "dissociation": {
                "confident_entropy_max": DISSOC_CONFIDENT_ENTROPY,
                "disagreement_entropy_min": DISSOC_DISAGREE_ENTROPY,
                "rule": "conjunctive and simultaneous",
                "anchored_to": {
                    "e2_dissociation_cell_within": 0.0896,
                    "e2_dissociation_cell_between": 1.3793,
                    "e2_truthful_ghost_within": 1.2926,
                    "ln4_ceiling": float(np.log(4)),
                },
                "why_conjunctive": (
                    "high disagreement alone is not the signature: truthfully-labelled "
                    "synthetic content produces it too, while being honest about its "
                    "uncertainty. What distinguishes fabrication is that the disagreement "
                    "comes with confidence. A heuristic can be confidently wrong; the "
                    "pre-registered claim is that it should not produce the JOINT pattern."),
            },
            "foreign_engagement": {
                "engaged_floor": FOREIGN_ENGAGED_FLOOR,
                "unresolved_entropy_min": FOREIGN_UNRESOLVED_ENTROPY,
                "anchored_to": {"e19_foreign_engaged": 0.746, "e19_foreign_entropy": 1.4854},
                "why": (
                    "sustained expensive attention that never resolves is a prediction about "
                    "an agent that keeps EXPECTING to learn. A pure effort heuristic has no "
                    "expectation to be wrong about, so arm D should fail it; if arm D "
                    "reproduces it anyway that is informative about how little machinery the "
                    "phenomenon requires."),
                "specificity_clause": (
                    "on directed human content the same arm must both RESOLVE and DISENGAGE. "
                    "E19's finding is a contrast, not a level: the observer that pays 75% of "
                    "free steps on foreign content resolves human content and stops paying "
                    "(0.00002). Scored as a level, arm C passes for free because it never "
                    "resolves anything, and arm B passes for free because it has no "
                    "engagement policy and is DEEP by construction. An arm whose engagement "
                    "does not respond to content has not reproduced a result about engagement "
                    "responding to content. Added before any E21 cell ran, from inspection of "
                    "the arm definitions."),
            },
            "secondary_measure_label_induction": (
                "reported alongside the dissociation but NOT part of the verdict. E2's "
                "finding is that the SAME content under a truthful label produces within "
                "1.2926 against the mislabeled cell's 0.0896, so the confidence is induced by "
                "the label. An arm that is confidently wrong regardless of the label has "
                "reproduced confident fabrication without reproducing the induction, and A2's "
                "calibration result is a claim about the induction specifically."),
            "arm_D_is_given_its_best_shot": (
                "arm D's disengagement threshold is its one free parameter and there is no "
                "principled value for it. Rather than fitting it, D is run across a grid of "
                "thresholds and is credited with reproducing a signature if ANY threshold in "
                "the grid does. That is maximally generous to the baseline, which is the "
                "direction the benefit of the doubt should run in an experiment whose "
                "unwelcome outcome is that the apparatus is unnecessary. The full grid is "
                "reported, not just the crediting value."),
            "cells": (
                "content {human_directed, goal_empty, goal_foreign} x declared signal "
                "{SIG_CREATOR, SIG_GHOST, UNSIGNED}. goal_empty is V1-V3's noise_free_synth "
                "stimulus, which is what E2 actually ran on; goal_foreign is V4's. Both are "
                "included because the dissociation is a V1 result about goal-empty content "
                "and V4.5 is a delta on a model that replaced that stimulus — asking the "
                "necessity question about only one of them would answer it for only one model "
                "of synthetic content. The UNSIGNED cells carry the E19 engagement "
                "discriminator, which was measured unsigned."),
            "positive_control": (
                "arm A must itself reproduce a signature for any baseline's failure on it to "
                "carry information; otherwise INCONCLUSIVE"),
            "outcomes": ["MACHINERY_NECESSARY", "MACHINERY_PARTLY_NECESSARY",
                         "MACHINERY_UNNECESSARY", "INCONCLUSIVE"],
            "unwelcome_outcome_goes_first": (
                "V4.5 §2: if arm C or D reproduces either signature, that goes in the FIRST "
                "LINE of the section"),
            "baseline_fairness": (
                "every arm receives the same observation stream and the same D-prior draw as "
                "the full observer, used as its prior or regulariser. Without matched "
                "heterogeneity a deterministic heuristic scores zero between-observer "
                "disagreement by construction and arm A wins the comparison for a reason that "
                "has nothing to do with theory of mind. Arm E additionally carries a "
                "discriminative feature-to-goal head, because the dissociation is measured on "
                "goals and a provenance-only classifier could not be scored on it at all."),
        },
        "E28_criteria": {
            "beta_grid": list(E28_BETA_GRID),
            "recovery_rho_min": E28_RECOVERY_RHO,
            "update_rho_min": E28_UPDATE_RHO,
            "accuracy_floor": E28_ACCURACY_FLOOR,
            "beta0_entropy_tolerance_nats": E28_BETA0_ENTROPY_TOL,
            "beta1_reproduces_v4_tolerance": E28_BETA1_TOL,
            "predicted": (
                "goal accuracy stays high across the beta range while update magnitude falls "
                "with beta — legible, competent, and unmoving"),
            "falsification": (
                "if update magnitude does not fall with beta while accuracy holds, beta is "
                "not doing separable work and collapses into either kappa_p or theta"),
            "third_outcome_named_before_the_run": (
                "if accuracy falls WITH the update magnitude, beta is doing something but not "
                "the predicted thing: content generated at low beta genuinely carries less "
                "goal information, so failing to read it is the ordinary consequence of a "
                "weaker stimulus rather than a distinct gate. The predicted result requires "
                "the DISSOCIATION."),
            "required_consistency_check": (
                "V4.5 §3.3: at beta = 0, E28 must reproduce E19's exploratory-human cell. If "
                "a continuous beta near zero does not recover the discrete EXPLORE result, "
                "the two are not the same axis and §3.3's identification is wrong."),
            "outcomes": ["BETA_IS_SEPARABLE", "BETA_IS_SEPARABLE_OVER_PART_OF_THE_RANGE",
                         "CONFOUNDED_WITH_LEGIBILITY", "BETA_COLLAPSES"],
        },
        "E29_criteria": {
            "predicted_signatures": E29_PREDICTED,
            "update_none_max_fraction_of_reference": E29_UPDATE_NONE,
            "update_low_max_fraction_of_reference": E29_UPDATE_LOW,
            "engaged_high_min": E29_ENGAGED_HIGH,
            "resolved_entropy_max": E29_RESOLVED_ENTROPY,
            "divergence_spike_min_ratio": E29_DIVERGENCE_SPIKE,
            "decisive_contrast": "low_beta versus closed_theta",
            "why_thresholds_are_ratios": (
                "update magnitude is classified as a fraction of the reference cell measured "
                "in the same run, not as a bare number in nats. A bare threshold could be "
                "slid until the predicted table came out and would not survive a change to "
                "the update scale; a ratio states the claim."),
            "failure_is_reportable": (
                "V4.5 §5: three gates with different action points is a harder object to "
                "identify than one scalar. E29 is the mitigation. If it fails, say so plainly "
                "rather than keeping the decomposition on theoretical grounds."),
            "outcomes": ["GATES_DISSOCIATE", "GATES_PARTLY_DISSOCIATE",
                         "DECOMPOSITION_NOT_EARNED", "INCOMPLETE"],
        },
        "architecture_constraints": {
            "beta_position": (
                "beta acts on the DEMONSTRATOR MODEL — the observer's likelihood over what a "
                "creator at that rationality would emit — not on the update. At beta = 1 the "
                "construction must reduce to V4's A[0] exactly, which is the N-series check "
                "V4.5 §7 requires: 'if it does not, beta has been wired into the wrong "
                "pipeline position'."),
            "social_influence": (
                "V4.5 §3.2 and §7: social influence shifts P_0(R) and the prior over "
                "provenance BEFORE any gate runs. It is never a multiplier on integration. "
                "Asserted at construction — no social term may appear in the update-magnitude "
                "path."),
            "integrity_is_not_a_separate_construct": (
                "V4.5 §3.2: under revealed preference, values are defined by what behaviour "
                "reveals, so integrity cannot diverge from values by construction. What can "
                "diverge is STATED from REVEALED values, and that gap is already V4's C4 "
                "(declared_tier versus omega_true). No separate construct is added."),
        },
        "unchanged_from_v4": {
            "E8": "stays withheld with its xfail(strict) marker",
            "E27": "the V3 residual stays open and is not touched",
            "sig_EXPLORE": "still locked by v4_preregistration.json and N17",
        },
    }
    payload["content_hash"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return payload


def write_preregistration_v4_5(cfg: Config, path: Path, force: bool = False) -> dict:
    payload = build_preregistration_v4_5(cfg)
    path = Path(path)
    if path.exists() and not force:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if existing is not None and existing.get("content_hash") != payload["content_hash"]:
            raise RuntimeError(
                f"{path.name} already exists with a DIFFERENT content hash.\n"
                f"  on disk: {existing.get('content_hash')}\n"
                f"  now:     {payload['content_hash']}\n"
                "V4.5's criteria were pre-registered and must not change after the fact. If "
                "any V4.5 experiment has already run, changing this file is exactly the "
                "failure V4 spec §7 names. Investigate, or pass force=True and record it as a "
                "deviation.")
        if existing is not None:
            return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def assert_prereg_locked_v4_5(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"{path} not found. No V4.5 experiment may run before its criteria are "
            "pre-registered.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("content_hash")
    recomputed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if stated != recomputed:
        raise RuntimeError(
            f"{path.name} has been modified since it was written (hash {stated} != recomputed "
            f"{recomputed}). The pre-registered criteria are not trustworthy; the V4.5 "
            f"programme will not run.")
    return payload
