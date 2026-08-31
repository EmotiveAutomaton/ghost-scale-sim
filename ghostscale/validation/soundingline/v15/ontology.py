"""The shared constructed ontology (spec §3.1-§3.3) and the knobs the boundary atlas varies.

What is shared and what is not
------------------------------
Spec §5 permits a *common ontology* and forbids a family that merely relabels another family's
transition table. This module therefore holds the vocabulary -- what a knob setting means, what an
episode record contains, how realized coupling is measured -- and no generative code at all. Each
of ``world_chain``, ``world_composition`` and ``world_communication`` writes its own latent prior,
its own transition and its own emission, and card I06 audits that their code paths are genuinely
distinct while their *target semantics* (realized coupling, realized route information) agree.

The five distinctions the spec insists are kept apart (§3.2), stated once so no card can quietly
merge them:

``attention``
    a limited allocation of observation, computation, rehearsal or control precision.
``foreground goal``
    a currently selected target state that organizes policy this episode.
``expertise``
    a learned transformation and prediction structure produced by attention, feedback,
    instruction, constraint and practice.
``history residue``
    any lasting bias from that learning history.
``standing preference``
    a cross-context choice tendency, not reducible by definition to any one residue.

Attention may help create expertise. V15 must not *define* expertise as previous attention and
then report their relationship as a finding, so the generator carries them as separate variables
and the readers are never handed the mapping.

Labels are pointers (§3.3)
--------------------------
A label such as ``warn`` or ``explore`` is a lossy pointer. The ``ContextRealized`` record is what
a label is compared against: context and opportunity set, predicted action distribution, process
constraints, a conditional stopping rule and expected change under an intervention. A correct
label that does not improve a hidden event is not scored as understanding.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# --------------------------------------------------------------------------- #
# Knob levels. Every level is a declared factor level and therefore a cell axis.
# --------------------------------------------------------------------------- #
COUPLING_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
OVERLAP_LEVELS = (0.0, 0.33, 0.66, 1.0)
DOSE_LEVELS = (1, 2, 4, 8, 16)
TEMPERATURE_LEVELS = (0.25, 0.6, 1.0, 2.0)
COMPETENCE_LEVELS = (0.55, 0.75, 0.95)
SIMILARITY_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)

MISSINGNESS = ("none", "route", "context", "opportunity")
EQUIFINALITY = ("none", "exact", "approximate")
MODEL_SPACE = ("correct", "missing_latent", "extra_latent", "wrong_family")
DRIFT = ("stationary", "abrupt", "gradual")
DEPENDENCE = ("independent", "redundant", "synergistic")

#: The three latent components every family carries, in canonical order.
COMPONENTS = ("process", "goal", "tendency")


@dataclass(frozen=True)
class Knobs:
    """One point in the boundary atlas. Frozen: a card varies knobs by making a new one."""

    kappa: float = 0.0                 # latent coupling, 0 independent .. 1 strongly coupled
    overlap: float = 0.0               # route overlap, 0 disjoint .. 1 every route sees everything
    dose: int = 8                      # observations available to the reader
    dependence: str = "independent"    # independent | redundant | synergistic
    missing: str = "none"              # none | route | context | opportunity
    temperature: float = 0.6           # policy stochasticity
    competence: float = 0.85           # how reliably intention becomes action
    equifinality: str = "none"         # none | exact | approximate
    model_space: str = "correct"       # correct | missing_latent | extra_latent | wrong_family
    drift: str = "stationary"          # stationary | abrupt | gradual
    similarity: float = 0.0            # maker-reader pairwise similarity, -1 .. 1
    typicality: float = 0.0            # family location, varied SEPARATELY from similarity (C12)
    n_process: int = 4
    n_goal: int = 4
    n_tendency: int = 4
    seed_tag: str = ""

    def with_(self, **kw) -> "Knobs":
        return replace(self, **kw)

    def key(self) -> str:
        return (f"k{self.kappa:g}|o{self.overlap:g}|d{self.dose}|{self.dependence}|"
                f"m{self.missing}|t{self.temperature:g}|c{self.competence:g}|"
                f"e{self.equifinality}|s{self.model_space}|f{self.drift}|"
                f"sim{self.similarity:g}|typ{self.typicality:g}")

    def to_dict(self) -> dict:
        return {"kappa": self.kappa, "overlap": self.overlap, "dose": self.dose,
                "dependence": self.dependence, "missing": self.missing,
                "temperature": self.temperature, "competence": self.competence,
                "equifinality": self.equifinality, "model_space": self.model_space,
                "drift": self.drift, "similarity": self.similarity,
                "typicality": self.typicality,
                "n_process": self.n_process, "n_goal": self.n_goal,
                "n_tendency": self.n_tendency}


DEFAULT = Knobs()
#: V14's regime, reproduced as an anchor by C01/F01/S01/H01: disjoint routes, generous evidence.
V14_REGIME = Knobs(kappa=0.0, overlap=0.0, dose=8, dependence="independent", temperature=0.6)


# --------------------------------------------------------------------------- #
# Records.
# --------------------------------------------------------------------------- #
@dataclass
class Latent:
    """The triple a reader is asked about, plus whatever else a family carries."""

    process: int = 0
    goal: int = 0
    tendency: int = 0
    extra: dict = field(default_factory=dict)

    def triple(self) -> tuple:
        return (self.process, self.goal, self.tendency)

    def as_dict(self) -> dict:
        return {"process": self.process, "goal": self.goal, "tendency": self.tendency,
                **{k: v for k, v in self.extra.items()}}


@dataclass
class Episode:
    """One episode's observable record and the hidden events scored against it.

    ``routes`` maps route name -> list of observed tokens (ints), one per step. ``hidden`` maps
    endpoint name -> the value that was hidden during inference. ``context`` and ``opportunity``
    are observable unless a missingness knob removes them, which is why they are separate fields
    rather than folded into ``routes``.
    """

    routes: dict = field(default_factory=dict)
    context: int = 0
    opportunity: tuple = ()
    hidden: dict = field(default_factory=dict)
    latent: Latent = field(default_factory=Latent)
    meta: dict = field(default_factory=dict)

    def prefix(self, upto: int) -> "Episode":
        return Episode(routes={k: v[:upto] for k, v in self.routes.items()},
                       context=self.context, opportunity=self.opportunity,
                       hidden={}, latent=Latent(), meta=dict(self.meta))

    def n_steps(self) -> int:
        return max((len(v) for v in self.routes.values()), default=0)


@dataclass
class ContextRealized:
    """What a label is compared against (spec §3.3). A label is a lossy pointer to this."""

    context: int
    opportunity: tuple
    action_distribution: np.ndarray
    process_constraints: dict
    stopping_rule: dict
    intervention_delta: dict

    def to_dict(self) -> dict:
        return {"context": int(self.context), "opportunity": list(self.opportunity),
                "action_distribution": np.asarray(self.action_distribution, float).tolist(),
                "process_constraints": self.process_constraints,
                "stopping_rule": self.stopping_rule,
                "intervention_delta": self.intervention_delta}


# --------------------------------------------------------------------------- #
# Coupling semantics. Every family must hit these targets with its own code.
# --------------------------------------------------------------------------- #
def target_coupling_nats(kappa: float, n_a: int, n_b: int) -> float:
    """What ``kappa`` *means*: the mutual information, in nats, that a family's latent prior must
    realize between any two latent components, as a fraction of the smaller component's entropy.

    Stating the target here and letting each family reach it independently is what makes the
    three families comparable without sharing a table.
    """
    hmin = np.log(min(int(n_a), int(n_b)))
    return float(np.clip(kappa, 0.0, 1.0) * 0.75 * hmin)


def realized_coupling(joint: np.ndarray) -> float:
    """Mutual information in nats of a two-way marginal of the latent prior."""
    p = np.asarray(joint, float)
    p = p / p.sum()
    pa, pb = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    m = p > 0
    return float((p[m] * np.log(p[m] / (pa @ pb)[m])).sum())


def pairwise_coupling(prior: np.ndarray) -> dict:
    """All three pairwise couplings of a 3-d latent prior, in nats."""
    p = np.asarray(prior, float)
    p = p / p.sum()
    return {"process_goal": realized_coupling(p.sum(axis=2)),
            "process_tendency": realized_coupling(p.sum(axis=1)),
            "goal_tendency": realized_coupling(p.sum(axis=0)),
            "total_correlation": float(
                sum(_h(p.sum(axis=tuple(a for a in range(3) if a != i))) for i in range(3)) - _h(p))}


def _h(p) -> float:
    p = np.asarray(p, float).ravel()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def realized_route_information(route_tables: dict, prior: np.ndarray) -> dict:
    """How much each route says about each latent component, in nats, under the given prior.

    ``route_tables`` maps route name -> array ``[n_process, n_goal, n_tendency, n_tokens]`` of
    emission probabilities. This is the receipt every construction card writes: it is how a card
    shows that a route is *live* and how ``overlap`` is demonstrated rather than asserted.
    """
    prior = np.asarray(prior, float)
    prior = prior / prior.sum()
    out = {}
    for name, tab in route_tables.items():
        tab = np.asarray(tab, float)
        joint = prior[..., None] * tab                      # p(process, goal, tendency, token)
        pt = joint.sum(axis=(0, 1, 2))
        per = {}
        for i, comp in enumerate(COMPONENTS):
            axes = tuple(a for a in range(3) if a != i)
            pj = joint.sum(axis=axes)                        # p(component, token)
            pc = pj.sum(axis=1)
            m = pj > 0
            mi = float((pj[m] * np.log(pj[m] / np.outer(pc, pt)[m])).sum())
            per[comp] = mi
        tot = sum(per.values())
        per["dominant"] = max(COMPONENTS, key=lambda c: per[c])
        per["concentration"] = float(max(per[c] for c in COMPONENTS) / tot) if tot > 0 else float("nan")
        out[name] = per
    return out


def overlap_index(route_info: dict) -> float:
    """0 when every route is about exactly one latent, 1 when every route is equally about all
    three. This is the number ``overlap`` is validated against."""
    vals = []
    for per in route_info.values():
        v = np.array([max(per[c], 0.0) for c in COMPONENTS], float)
        s = v.sum()
        if s <= 0:
            continue
        q = v / s
        vals.append(float(_h(q) / np.log(len(COMPONENTS))))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# Model-space perturbations (spec §5, X05-X07). Applied to the *reader*, never the generator.
# --------------------------------------------------------------------------- #
def reader_components(model_space: str) -> tuple:
    """Which latent components a reader's model space contains under a misspecification."""
    if model_space == "missing_latent":
        return ("process", "goal")
    if model_space == "extra_latent":
        return COMPONENTS + ("distractor",)
    return COMPONENTS


def likelihood_family_error(model_space: str) -> float:
    """How wrong the reader's likelihood family is. ``wrong_family`` is superficially calibrated
    in training and wrong at test -- attack X05."""
    return 0.45 if model_space == "wrong_family" else 0.0
