"""V5 C1 — mu, the creator's estimated MODEL DEPTH, replacing V4.5's rationality beta.

WHY BETA WAS MEASURING THE WRONG THING (V5 C1). V4.5 introduced beta as the MaxEnt
rationality coefficient: how hard was the creator optimising. That is not what gates uptake.
What gates it is the observer's estimate of HOW COMPLEX A MODEL THE CREATOR HAD — the depth of
the hierarchy of decisions behind the artifact. The two dissociate, and the framework takes
the opposite side from beta:

                                    rationality beta   model depth mu
    fully committed, trivial goal        high               low
    offhand sketch by a master           low                high

The preprint says the second is more artful. A rationality parameter says the opposite.

-----------------------------------------------------------------------------------------
THE CONSTRUCTION, AND WHY THE OBVIOUS ONE WAS REJECTED.

Three routes were considered. Route C — mu as a scalar mixing coefficient over a depth-indexed
signature family — WAS REJECTED, and the reason is the whole point of C1. beta is
``normalize(b * sig[g] + (1 - b) * sig_EXPLORE)``, a one-dimensional mixture weight on a flat
categorical. Any scalar mu built the same way collapses onto the same axis: E30 would
reproduce E28's numbers and the correction would be a rename. Depth is a claim about
STRUCTURE AT MULTIPLE SCALES, and a single flat categorical cannot carry one, because any
mixture of categoricals is just another categorical.

Route B — composite observations, one observation being a window of k features over an F^k
alphabet — is kept as a documented fallback and is not built. It was needed only if route A
proved unaffordable. It did not: measured on this machine, a 24-step rollout costs 166 ms in
the V4.5 model and 657 ms here, so E30 runs in about ten minutes at 22 workers.

Route A, built here: mu indexes HOW MANY LEVELS of the creator's decision hierarchy reach the
artifact's surface.

    mu = 1   EXECUTION ONLY.   Features depend on the goal and nothing else. The sub-goal is
             invisible. This is V4 exactly, elementwise, and it is asserted at construction.
    mu = 2   + SUB-GOAL LEVEL. Features depend on which execution mode the creator is in.
             Modes persist, so the artifact carries BLOCKS. The modes are goal-generic: you
             can see that the maker was doing one thing and then another, but which things
             they were tells you nothing extra about the goal.
    mu = 3   + GOAL LEVEL.     The modes are goal-SPECIFIC, so the SEQUENCE of modes — which
             is governed by that goal's own plan in B — becomes diagnostic of the goal.
             Structure at execution, sub-goal, and goal level: the full hierarchy.

-----------------------------------------------------------------------------------------
DEPTH IS CORRELATIONAL, NOT MARGINAL, AND THAT IS ENFORCED RATHER THAN HOPED FOR.

The sub-goal signatures average EXACTLY to V4's ``sig[g]``, and the sub-goal chains are
doubly stochastic with a uniform stationary distribution. Both facts are asserted. Together
they mean the time-averaged feature histogram of a mu = 3 artifact is IDENTICAL to that of a
mu = 1 artifact, to machine precision. A reader who counts features and ignores their order
cannot tell a deep artifact from a shallow one at all.

That is the design commitment, and it is what makes mu something other than a redescription
of legibility: depth lives in the ORDER, and reading it requires a model of a creator who was
doing one thing and then another. It also means the E28 trap is closed by construction rather
than discovered afterwards. E28's failure was that ``sig_EXPLORE`` IS ``mean_g(sig[g])``, so
beta carried no information in the goal marginal and every bit of evidence about it came from
finite-sample luck in the joint (beta, goal) coupling. ``depth_marginal_invariance`` measures
the mu analogue of that quantity BEFORE E30 runs, which is V5 decision 7, and it is five lines
because that is all the check ever needed to be.

-----------------------------------------------------------------------------------------
WHY mu, goal AND SUB-GOAL SHARE ONE FACTOR, WHICH WAS DECIDED BY MEASUREMENT.

The factors are ``[provenance (P), (mu, goal, sub-goal) (M*G*S), attention]``. Attention keeps
index 2, so ``control_fac_idx=[K.F_ATTENTION]`` stays correct and every V1-V4 call site that
names it is untouched — the same reasoning V4.5 used when it appended F_BETA at index 3 rather
than inserting it.

Two things force the merge, and the second was not anticipated.

* B is per-factor in pymdp and cannot condition on another factor's state, so a GOAL-DEPENDENT
  sub-goal chain is not expressible with goal and sub-goal separate. Merged, B[1] is block
  diagonal: the goal never changes within an episode (V1-V4's identity B on goal, preserved
  exactly, since no mass ever crosses between blocks) and the sub-goal walks that goal's own
  plan inside its block.

* mu HAD TO JOIN THEM, and the reason is a property of the solver rather than of the theory.
  With mu in a separate factor from the sub-goal, pymdp legacy's MEAN-FIELD variational
  inference updates the mu posterior using the expected LOG likelihood under the current
  sub-goal marginal. By Jensen that systematically penalises any hypothesis carrying latent
  structure — and mu = 1 carries none, because at mu = 1 every sub-goal emits the same thing.
  So the shallow hypothesis wins for a reason that has nothing to do with the artifact.

  Measured on identical feature sequences, before the merge:

      true mu      exact filtering      mean-field agent
        1              1.52                  1.04
        2              2.20                  1.04
        3              2.27                  1.06

  Exact inference over the full joint recovers depth monotonically; the factorised agent
  returns the shallow answer for every artifact, CONFIDENTLY (posterior entropy 0.047). The
  depth is in the artifact. The solver could not see it. Merging mu into the same factor as
  the sub-goal makes the coupling that matters exact and removes the pathology, at no cost:
  the state space is 4 x 48 x 2 = 384 either way.

  This is why N21 runs BEFORE E30 rather than after. Left undetected, the defect would have
  surfaced as a small, plausible, entirely uninformative depth effect, and it would have been
  read as evidence against C1 rather than against the inference engine.

WHERE C2's EVIDENCE PATH WENT (V5 decision 10, revised).

Decision 10 originally merged provenance with mu so that a joint prior P(mu | provenance)
could be written, since pymdp factorises D per factor. That merge is incompatible with the one
above: putting all four into a single factor needs a cardinality-1 placeholder to keep
attention at index 2, and pymdp raises on it.

So provenance stays its own factor and the evidence path becomes a mu prior CONDITIONED ON THE
OBSERVED LABEL, supplied to ``build_v5_D`` by the caller. The mechanism is unchanged in
substance — a provenance signal shifts the observer's expectation of how much reasoning sits
behind the work, before any gate runs — and it is arguably more inspectable this way, because
the coupling is one argument that can be printed rather than a pattern inside a joint prior.
What it gives up is that the observer no longer infers provenance and depth jointly, so a
reader who doubts the label does not automatically doubt the depth estimate it induced. E31
manipulates that path and should report the limitation.

The observer INFERS the sub-goal rather than marginalising it away (decision 3). It is free
given the factor exists, and it is what "I can see what they were doing at each stage" means.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pymdp.legacy import utils
from pymdp.legacy.agent import Agent

from . import constants as K
from . import foreign as FN
from .config import Config
from .environment import Environment
from .generative_model import (GenerativeModel, alpha_by_provenance, assert_preferences_zero,
                               build_A2, build_C, cards)
from .metrics import normalize
from .v4_model import build_v4_synth, load_v4_config

# Factor indices. Attention stays at 2; see the module docstring.
F_PROVENANCE, F_MGS, F_ATTENTION = 0, 1, 2

# The depth levels. 1 = execution only (V4), 2 = + sub-goal, 3 = + goal plan.
MU_LEVELS = (1, 2, 3)
MU_SHALLOW = 1

_EPS = 1e-12


@dataclass
class V5World:
    gm: GenerativeModel
    sigs: FN.V4Signatures
    cfg: Config
    mu_levels: tuple
    n_subgoals: int
    subsig: np.ndarray          # (mu, goal, subgoal, features)
    chains: np.ndarray          # (goal, subgoal_next, subgoal_now)
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Index helpers for the two merged factors.
# --------------------------------------------------------------------------- #
def mgs_index(mu_i: int, g: int, s: int, ng: int, n_sub: int) -> int:
    """Flat index of (mu-level, goal, sub-goal) in factor 1."""
    return (int(mu_i) * int(ng) + int(g)) * int(n_sub) + int(s)


def as_mgs(q: np.ndarray, n_mu: int, ng: int, n_sub: int) -> np.ndarray:
    """Factor 1's posterior as a (mu, goal, sub-goal) array."""
    return np.asarray(q, dtype=float).reshape(int(n_mu), int(ng), int(n_sub))


def marginal_mu(q: np.ndarray, n_mu: int, ng: int, n_sub: int) -> np.ndarray:
    return as_mgs(q, n_mu, ng, n_sub).sum(axis=(1, 2))


def marginal_goal(q: np.ndarray, n_mu: int, ng: int, n_sub: int) -> np.ndarray:
    return as_mgs(q, n_mu, ng, n_sub).sum(axis=(0, 2))


def marginal_subgoal(q: np.ndarray, n_mu: int, ng: int, n_sub: int) -> np.ndarray:
    return as_mgs(q, n_mu, ng, n_sub).sum(axis=(0, 1))


def recovered_mu(q: np.ndarray, n_mu: int, ng: int, n_sub: int,
                 mu_levels=MU_LEVELS) -> float:
    """E[mu] under the observer's posterior. The quantity E30 regresses against true mu."""
    m = marginal_mu(q, n_mu, ng, n_sub)
    m = m / m.sum()
    return float(np.dot(m, np.asarray(mu_levels, dtype=float)))


# --------------------------------------------------------------------------- #
# The sub-goal signature family.
# --------------------------------------------------------------------------- #
def build_execution_modes(sig_g: np.ndarray, n_sub: int,
                          support: np.ndarray | None = None) -> np.ndarray:
    """``q[s]`` — S distributions over features whose PLAIN AVERAGE is exactly ``sig_g``.

    The exact-average property is the load-bearing one and it is why these are constructed
    rather than drawn. If the modes did not average back to the goal signature, a deep
    artifact's feature histogram would differ from a shallow one's and mu would be readable
    off a bare feature count — which is legibility, the thing kappa_p already measures, not
    depth. Depth has to be invisible to a counting reader or it is not depth.

    -------------------------------------------------------------------------------------
    THE FIRST VERSION OF THIS FUNCTION WAS BUILT, MEASURED, AND REPLACED. What follows is the
    replacement, and the reason is recorded here because the failure is instructive and would
    otherwise be invisible in the final code.

    The first family was (lead-with-a, lead-with-b, TIGHT, LOOSE), with the last mode solved
    for so the average came out exact. It satisfied every assertion. But ``sig_g`` carries 87%
    of its mass on two features, so TIGHT and LOOSE could only differ in how much of the
    remaining 13% sat in the tail: measured, they came out at (0.483, 0.483) and (0.384,
    0.384) on the pair — a separation of 0.116 against 0.651 for the first two modes. Only two
    of four modes were distinguishable, so the artifact carried about half the depth the
    design claimed, and N21's mu effect measured +0.18 on a 1-to-3 scale.

    THE FIX IS TO USE AN AXIS THE GOAL SIGNATURE LEAVES FREE. Modes differ in WHICH TAIL
    FEATURES they use, not only in how they split the dominant pair. The tail is 14 features
    that ``sig_g`` treats identically, so concentrating a mode's tail mass on its own subset
    separates the modes strongly while leaving the average untouched — the subsets partition
    the tail, so averaging over modes spreads that mass back out evenly.

    Read as a claim about creators: the modes are stages of work that all serve one goal and
    are told apart by which secondary material each one draws on. A reader who pools the whole
    artifact sees the goal's signature and nothing else. A reader who tracks it in order sees
    four stages.

    -------------------------------------------------------------------------------------
    EXACTNESS IS OBTAINED BY PROJECTION rather than by solving for the last mode. Given any
    proposal family, ``modes += sig_g - mean(modes)`` makes the average exactly ``sig_g`` and
    preserves every row sum, because both the mean and ``sig_g`` sum to one. That is general,
    it holds to machine precision, and it does not privilege one mode by making it absorb the
    others' error — which is what produced the degenerate LOOSE mode above.
    """
    sig_g = np.asarray(sig_g, dtype=float)
    nf = sig_g.size
    assert n_sub >= 2, "the mode family needs at least two modes"

    pair = np.sort(np.argsort(sig_g)[-2:])
    a, b = int(pair[0]), int(pair[1])
    # THE TAIL IS RESTRICTED TO THE OBSERVER'S OWN BLOCK, and this is not cosmetic. With the
    # tail taken as "every feature that is not the goal's pair", ``np.array_split`` hands the
    # later modes subsets that live in the FOREIGN block, so a deep human artifact would carry
    # mass on features V4 reserves for out-of-family content. Every C1 property rests on that
    # partition being disjoint — ``assert_c1_properties`` checks it for ``sig_true`` and would
    # not have seen it here, because it never inspects the mode family. Measured before the
    # restriction: modes 2 and 3 put all their tail mass on features 8-15.
    allowed = np.arange(nf) if support is None else np.asarray(support, dtype=int)
    tail = np.setdiff1d(allowed, pair)
    assert tail.size >= 1, "no tail features left to separate the execution modes with"
    tail_mass = float(sig_g[tail].sum())
    pair_mass = float(sig_g[a] + sig_g[b])

    # Partition the tail into S subsets, as evenly as the count allows.
    subsets = np.array_split(tail, n_sub)

    modes = np.zeros((n_sub, nf))
    for s in range(n_sub):
        sub = subsets[s]
        # Tail mass for this mode is scaled by its subset's SIZE, so that after averaging
        # every tail feature receives exactly the mass sig_g gives it, whatever the partition
        # sizes are. Without this an uneven split (4/4/3/3 over 14) would tilt the average.
        m_s = tail_mass * n_sub * len(sub) / len(tail)
        m_s = min(m_s, 0.9)                       # leave the pair something to work with
        modes[s, sub] = m_s / len(sub)
        # SECOND AXIS: how the goal's own pair is split. Every mode gets its own split rather
        # than the modes alternating between two, so all S differ on this axis as well as on
        # which tail features they use. Alternating gave modes 0 and 2 the same split and
        # modes 1 and 3 the same, which cost real separation for nothing.
        amps = np.linspace(0.4, 0.15, (n_sub + 1) // 2)
        amp = float(amps[s // 2])
        frac = 0.5 + amp if s % 2 == 0 else 0.5 - amp
        rest = 1.0 - m_s
        modes[s, a] = rest * frac
        modes[s, b] = rest * (1.0 - frac)

    # PROJECTION: make the average exactly sig_g without privileging any mode.
    modes = modes + (sig_g - modes.mean(axis=0))[None, :]

    if modes.min() < 0.0:
        # Shrink the whole family toward sig_g until it is feasible. Reported through the
        # separation diagnostic rather than silently: a family that had to be shrunk a long
        # way is a family that cannot carry much depth, and that is worth seeing.
        centred = modes - sig_g[None, :]
        neg = centred < 0
        # V6 FIX. ``sig_g[None, :]`` has shape (1, F) and ``neg`` has shape (S, F), so indexing
        # the first by the second raises rather than broadcasting. The path was unreachable with
        # the human goal family, whose mass is concentrated enough that the projection never
        # goes negative, and it crashed the moment V6 built a world over a FLATTER family.
        # Broadcasting explicitly is what the surrounding arithmetic already assumes.
        sig_b = np.broadcast_to(sig_g[None, :], centred.shape)
        scale = float(np.min(sig_b[neg] / -centred[neg])) if neg.any() else 1.0
        modes = sig_g[None, :] + centred * (scale * (1.0 - 1e-9))

    modes = np.clip(modes, 0.0, None)
    assert np.allclose(modes.sum(axis=1), 1.0, atol=1e-10), "execution modes must be normalised"
    assert np.allclose(modes.mean(axis=0), sig_g, atol=1e-10), (
        "execution modes must average EXACTLY to the goal signature; depth that shows up in "
        "the feature marginal is legibility wearing a new name")
    _ = pair_mass
    return modes


def mode_separation(modes: np.ndarray) -> float:
    """Mean pairwise Jensen-Shannon divergence between execution modes, in nats.

    The depth analogue of ``artifact_model.js_threshold``, which V1 applies to the goal
    signatures for exactly the same reason: hypotheses that are not distinguishable cannot be
    distinguished, and an experiment run on an indistinguishable family measures its own
    construction. Asserted against a floor at build time so a mode family that cannot carry
    depth fails loudly here rather than quietly in E30's residuals.
    """
    from .metrics import js_divergence
    n = modes.shape[0]
    vals = [js_divergence(modes[i], modes[j])
            for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals)) if vals else 0.0


def goal_mode_permutations(n_goals: int, n_sub: int) -> np.ndarray:
    """Which execution style each mode INDEX names, per goal, at mu = 3.

    Every permutation returned is a DERANGEMENT — no mode index keeps its generic meaning —
    and they are pairwise distinct. Both properties are asserted, and both were put here
    after a cyclic-shift version was built and measured.

    THE FAILURE THAT MOTIVATED IT, since it is the kind that does not announce itself. Rolling
    the mode family by ``g`` gives goal 0 a shift of zero, so goal 0's mu = 2 and mu = 3
    emissions are IDENTICAL and that goal's artifacts cannot discriminate the top two depths at
    all. With four goals and four modes there is no set of four distinct cyclic shifts that
    avoids this: one goal always draws the identity. The design would have survived — mode
    identity stays goal-diagnostic because the permutations still differ pairwise — but E30
    would have carried a per-goal heterogeneity with no theoretical content, and someone would
    eventually have found it in the residuals and had to explain it.

    Derangements remove it outright: at mu = 3, no mode means what it means generically, for
    any goal. There are 9 derangements of 4 elements, so the four needed here are available
    with room to spare.
    """
    from itertools import permutations

    base = tuple(range(n_sub))
    derangements = [p for p in permutations(base)
                    if all(p[i] != i for i in range(n_sub))]
    assert len(derangements) >= n_goals, (
        f"only {len(derangements)} derangements of {n_sub} modes exist, fewer than the "
        f"{n_goals} goals that need one. Raise the number of execution modes.")
    perms = np.array(derangements[:n_goals], dtype=int)
    assert len({tuple(p) for p in perms}) == n_goals, "goal mode permutations must be distinct"
    return perms


def build_subgoal_signatures(cfg: Config, sig_true: np.ndarray, n_sub: int,
                             delta: float, mu_levels=MU_LEVELS) -> np.ndarray:
    """``subsig[mu, g, s]`` — what a creator at depth ``mu`` emits in mode ``s`` of goal ``g``.

        subsig = (1 - delta) * sig[g] + delta * modes[g][s]

    ``delta`` is MODE DISTINCTNESS and it is the one free knob in the depth construction, so
    it is pre-registered rather than tuned. At delta = 0 every mode is the goal signature and
    depth is unreadable at any mu; at delta = 1 the modes are as distinct as the exact-average
    constraint permits. Both endpoints are reachable and the sweep between them is what says
    how much structure the claim needs.

    No renormalisation is applied and none is needed: both terms are distributions, so the
    convex combination is one, and the plain average over s stays EXACTLY sig[g] because
    ``mean_s(modes[g][s])`` is exactly sig[g]. A ``normalize`` call here would destroy that
    property in the fifth decimal and the whole design rests on it holding at 1e-12.

    THE THREE DEPTH LEVELS.

    * mu = 1 emits ``sig[g]`` in every mode. The sub-goal is invisible, the artifact is V4's,
      and ``assert_mu1_reproduces_v4`` checks that elementwise.
    * mu = 2 uses GOAL-GENERIC modes: the same execution styles, indexed the same way, for
      every goal. Blocks are visible and readable as blocks, but knowing which mode is active
      tells you nothing about the goal you did not already have.
    * mu = 3 uses GOAL-SPECIFIC modes, obtained by permuting which style each mode index names
      with that goal's derangement (see ``goal_mode_permutations``). The sub-goal chain in B is
      already goal-specific, so at mu = 3 the SEQUENCE of modes becomes goal-diagnostic and the
      third level of the hierarchy reaches the surface.
    """
    sig_true = np.asarray(sig_true, dtype=float)
    ng, nf = sig_true.shape
    d = float(delta)
    assert 0.0 <= d <= 1.0, f"mode distinctness delta must be in [0, 1], got {d}"

    # Modes vary only over the HUMAN block, so a deep human artifact never puts mass where V4
    # reserves the foreign family. See build_execution_modes for what happened without this.
    human = np.asarray(FN.HUMAN_FEATURES, dtype=int)
    generic = np.stack([build_execution_modes(sig_true[g], n_sub, support=human)
                        for g in range(ng)])
    perms = goal_mode_permutations(ng, n_sub)

    out = np.zeros((len(mu_levels), ng, n_sub, nf))
    for mi, mu in enumerate(mu_levels):
        for g in range(ng):
            if mu <= 1:
                out[mi, g, :, :] = sig_true[g][None, :]
            elif mu == 2:
                out[mi, g] = (1.0 - d) * sig_true[g][None, :] + d * generic[g]
            else:
                # GOAL-SPECIFIC modes: this goal's derangement decides which style each mode
                # index names, so no index means generically what it means here.
                out[mi, g] = (1.0 - d) * sig_true[g][None, :] + d * generic[g][perms[g]]
    return out


# --------------------------------------------------------------------------- #
# The sub-goal chain.
# --------------------------------------------------------------------------- #
def build_subgoal_chains(n_goals: int, n_sub: int, dwell: float,
                         next_share: float = 0.7) -> np.ndarray:
    """``chains[g]`` — goal g's plan over execution modes, as a column-stochastic matrix.

    Each chain is DOUBLY STOCHASTIC with a uniform stationary distribution, and each goal gets
    a DIFFERENT cyclic order. Those two properties together are the design:

      * doubly stochastic with uniform stationary means every goal spends the same average
        time in every mode, so the sub-goal marginal is identical across goals and across mu.
        Nothing about the plan leaks into a time-averaged histogram.
      * a different cyclic order per goal means the plans genuinely differ, and they differ
        only in what FOLLOWS what.

    So goal identity is carried by the ORDER of modes and by nothing else, which is what the
    goal level of the hierarchy is supposed to mean. Construction: hold ``dwell`` on the
    diagonal, send most of the leaving mass to the next mode in that goal's own cycle, and
    spread the rest. Column sums are 1 by construction; row sums are 1 because the cycle is a
    bijection, so each mode is the successor of exactly one other.

    ``dwell`` is the mean number of consecutive steps in a mode, set so that three to four
    blocks fit in a 24-step artifact (V5 decision 2). Too short and no block structure forms;
    too long and one artifact shows one mode, and either way depth is unreadable for a reason
    that has nothing to do with the theory.
    """
    p_stay = 1.0 - 1.0 / float(dwell)
    assert 0.0 < p_stay < 1.0, f"dwell {dwell} gives a degenerate self-transition {p_stay}"
    leave = 1.0 - p_stay
    q_next = leave * float(next_share)
    q_rest = (leave - q_next) / max(n_sub - 2, 1) if n_sub > 2 else 0.0

    chains = np.zeros((n_goals, n_sub, n_sub))
    for g in range(n_goals):
        # Goal g's cycle: step by (g + 1), which is a bijection on S whenever gcd(g+1, S) = 1
        # and otherwise still a permutation once composed with the shift below.
        step = (g % (n_sub - 1)) + 1
        for s in range(n_sub):
            nxt = (s + step) % n_sub
            for s2 in range(n_sub):
                if s2 == s:
                    chains[g, s2, s] = p_stay
                elif s2 == nxt:
                    chains[g, s2, s] = q_next
                else:
                    chains[g, s2, s] = q_rest
    assert np.allclose(chains.sum(axis=1), 1.0, atol=1e-12), "chains must be column-stochastic"
    return chains


def stationary(chain: np.ndarray, iters: int = 4000) -> np.ndarray:
    v = np.full(chain.shape[0], 1.0 / chain.shape[0])
    for _ in range(iters):
        v = chain @ v
    return v / v.sum()


def build_subgoal_chains_v5b(n_goals: int, n_sub: int, dwell: float,
                             next_share: float = 0.7) -> np.ndarray:
    """The repaired chain builder — V11 repair, deviation V5-5. NEW WORK ONLY.

    ``build_subgoal_chains`` above assigns goal g the cyclic step ``(g % (n_sub - 1)) + 1``,
    which with four goals and four modes maps goals 0 and 3 to the SAME step and hands them
    identical chains — contradicting the docstring's "a different cyclic order per goal" for that
    pair, and with it the claim that goal identity is carried by order alone. With S modes there
    are only S - 1 non-identity cyclic steps, so four goals cannot all get distinct cycles: the
    collision is a pigeonhole fact, not a typo, and the fix needs a richer successor family.

    Here each goal's successor map is a DERANGEMENT, drawn from the derangements NOT used by
    ``goal_mode_permutations`` so the order channel and the mu = 3 emission channel stay
    uncoupled. A derangement is a permutation, so each mode is the successor of exactly one other
    (rows sum to 1, the chain stays doubly stochastic, the stationary distribution stays
    uniform), and no mode ever succeeds itself, which keeps ``dwell`` the only source of
    self-transition. Pairwise distinctness of the chains is asserted — the assertion the original
    lacked, and the one that would have caught V5-5 at build time.

    Closed versions keep the original builder untouched: their committed worlds ARE the record.
    """
    from itertools import permutations

    base = tuple(range(n_sub))
    derangements = [p for p in permutations(base) if all(p[i] != i for i in range(n_sub))]
    emission_perms = {tuple(p) for p in goal_mode_permutations(n_goals, n_sub)}
    available = [p for p in derangements if tuple(p) not in emission_perms]
    assert len(available) >= n_goals, (
        f"only {len(available)} derangements of {n_sub} modes remain after excluding the "
        f"emission permutations; raise the number of execution modes.")

    p_stay = 1.0 - 1.0 / float(dwell)
    assert 0.0 < p_stay < 1.0, f"dwell {dwell} gives a degenerate self-transition {p_stay}"
    leave = 1.0 - p_stay
    q_next = leave * float(next_share)
    q_rest = (leave - q_next) / max(n_sub - 2, 1) if n_sub > 2 else 0.0

    chains = np.zeros((n_goals, n_sub, n_sub))
    for g in range(n_goals):
        succ = available[g]
        for s in range(n_sub):
            nxt = succ[s]
            for s2 in range(n_sub):
                if s2 == s:
                    chains[g, s2, s] = p_stay
                elif s2 == nxt:
                    chains[g, s2, s] = q_next
                else:
                    chains[g, s2, s] = q_rest
    assert np.allclose(chains.sum(axis=1), 1.0, atol=1e-12), "chains must be column-stochastic"
    assert np.allclose(chains.sum(axis=2), 1.0, atol=1e-12), (
        "chains must be row-stochastic too: each mode the successor of exactly one other")
    for g1 in range(n_goals):
        for g2 in range(g1 + 1, n_goals):
            assert not np.allclose(chains[g1], chains[g2], atol=1e-12), (
                f"goals {g1} and {g2} share a chain; the whole point of v5b is that they must "
                f"not (V5-5)")
    return chains


# --------------------------------------------------------------------------- #
# A, B, D.
# --------------------------------------------------------------------------- #
def build_v5_A0(cfg: Config, subsig: np.ndarray, noise_free_synth: np.ndarray,
                alpha: np.ndarray) -> np.ndarray:
    """A[0], shape (features, provenance, mu*goal*subgoal, attention).

    mu acts on the CREATOR (which structure exists), alpha on the CHANNEL (how much of it
    survives), composed in the only order that makes sense and the same order V4.5 used for
    beta:

        A0[:, p, (mu,g,s), DEEP] = alpha[p] * subsig[mu, g, s] + (1 - alpha[p]) * synth

    At mu = 1, ``subsig[0, g, s]`` is ``sig[g]`` for every s, so this is V4's A[0] element for
    element. That is asserted, not assumed.
    """
    n_mu, ng, n_sub, nf = subsig.shape
    npv = int(cfg.cardinalities.num_provenance)
    na = int(cfg.cardinalities.num_attention)
    A0 = np.zeros((nf, npv, n_mu * ng * n_sub, na))
    for p in range(npv):
        a = float(alpha[p])
        for mi in range(n_mu):
            for g in range(ng):
                for s in range(n_sub):
                    j = mgs_index(mi, g, s, ng, n_sub)
                    A0[:, p, j, K.DEEP] = a * subsig[mi, g, s] + (1.0 - a) * noise_free_synth
                    A0[:, p, j, K.SKIM] = noise_free_synth
    return A0


def build_v5_A1(cfg: Config, kappa: float, n_mgs: int) -> np.ndarray:
    """A[1] ghost_signal — V1's construction, constant over the merged factor.

    The signal row depends on PROVENANCE only, exactly as in V1-V4.5. The label is about where
    a thing came from, not about how much thought went into it, and making the channel itself
    mu-dependent would smuggle C2's conclusion into the likelihood: the observer would then be
    told that machine work is shallow rather than inferring it. C2's evidence path is a prior,
    and it lives in ``build_v5_D``.
    """
    cd = cards(cfg)
    uniform = np.full(cd.signals, 1.0 / cd.signals)
    A1 = np.zeros((cd.signals, cd.provenance, int(n_mgs), cd.attention))
    for p in range(cd.provenance):
        truthful = np.zeros(cd.signals)
        truthful[K.TRUTHFUL_SIGNAL[p]] = 1.0
        row = kappa * truthful + (1.0 - kappa) * uniform
        A1[:, p, :, :] = row[:, None, None]
    return A1


def build_v5_A2(cfg: Config, n_mu: int, ng: int, n_sub: int) -> np.ndarray:
    """A[2] effort, deterministic, tiled over the merged factor."""
    cd = cards(cfg)
    base = np.asarray(build_A2(cfg))            # (effort, prov, goal, att)
    out = np.zeros((cd.effort, cd.provenance, n_mu * ng * n_sub, cd.attention))
    for p in range(cd.provenance):
        for mi in range(n_mu):
            for g in range(ng):
                for s in range(n_sub):
                    out[:, p, mgs_index(mi, g, s, ng, n_sub), :] = base[:, p, g, :]
    return out


def build_v5_B(cfg: Config, chains: np.ndarray, n_mu: int) -> object:
    """B over the three factors.

    B[0] — provenance, the identity, exactly as in V1-V4.

    B[1] — (mu, goal, sub-goal) is BLOCK DIAGONAL with one block per (mu, goal), each block
    that goal's own chain. The zero off-diagonal blocks are what preserve V1-V4's identity B
    on the goal and add the same treatment for mu: no probability mass ever moves between
    goals or between depths, so both stay fixed within an episode and remain things to be
    INFERRED rather than things that change. Inside a block the sub-goal walks that goal's
    plan. The chain does not depend on mu — the creator's plan is the same object at every
    depth; what mu decides is how much of it reaches the surface.

    B[2] — attention, fully controllable, unchanged from V1.
    """
    cd = cards(cfg)
    ng, n_sub, _ = chains.shape
    B = utils.obj_array(3)
    B[0] = np.eye(cd.provenance)[:, :, None]

    n = n_mu * ng * n_sub
    big = np.zeros((n, n))
    for mi in range(n_mu):
        for g in range(ng):
            lo = mgs_index(mi, g, 0, ng, n_sub)
            big[lo:lo + n_sub, lo:lo + n_sub] = chains[g]
    B[1] = big[:, :, None]

    B2 = np.zeros((cd.attention, cd.attention, cd.attention))
    for a in range(cd.attention):
        B2[a, :, a] = 1.0
    B[2] = B2
    return B


def build_v5_D(cfg: Config, rng: np.random.Generator, n_mu: int, ng: int, n_sub: int,
               mu_prior: np.ndarray | None = None) -> object:
    """D priors over the three factors, with C2's provenance-to-depth evidence path.

    ``mu_prior`` is the observer's expectation about how much reasoning sits behind THIS
    artifact, and it is where V5 C2 lives after decision 10 was revised (see the module
    docstring). A caller that has seen a provenance label passes the mu prior that label
    induces; "learning that something is machine-generated lowers your estimate of how much
    reasoning produced it" is then a statement about a prior, which is what C2 says it is,
    rather than a term in a likelihood or a gate on integration.

    Default is UNIFORM, which switches the coupling off entirely. E30 runs that way ON
    PURPOSE: it asks what CONTENT ALONE reveals about depth, and an informative
    provenance-to-depth prior would be putting part of the answer in by hand. E31 is where the
    coupling is turned on and manipulated.

    The goal marginal keeps V1's heterogeneous Dirichlet perturbation — the H2 result is a
    between-observer claim and null N3 requires observers to differ. The sub-goal marginal is
    uniform, which is also its stationary distribution: which mode a creator happened to start
    in is not something a reader has a prior about, and any other choice would leak the plan.
    """
    cd = cards(cfg)
    eps = float(cfg.priors.eps)
    D = utils.obj_array(3)

    uni_p = np.full(cd.provenance, 1.0 / cd.provenance)
    D[0] = (normalize(uni_p + eps * rng.dirichlet(np.ones(cd.provenance)))
            if bool(cfg.priors.perturb_provenance) else uni_p)

    if mu_prior is None:
        mu_p = np.full(n_mu, 1.0 / n_mu)
    else:
        mu_p = np.asarray(mu_prior, dtype=float)
        mu_p = mu_p / mu_p.sum()

    uni_g = np.full(ng, 1.0 / ng)
    goal = normalize(uni_g + eps * rng.dirichlet(np.ones(ng)))
    joint = (mu_p[:, None, None] * goal[None, :, None]
             * np.full((1, 1, n_sub), 1.0 / n_sub))
    D[1] = joint.reshape(-1)
    D[1] = D[1] / D[1].sum()

    D[2] = np.full(cd.attention, 1.0 / cd.attention)
    return D


# --------------------------------------------------------------------------- #
# Assembly.
# --------------------------------------------------------------------------- #
def build_v5_world(cfg: Config, omega: float = 0.0, kappa: float | None = None,
                   delta: float | None = None, dwell: float | None = None,
                   n_subgoals: int | None = None, foreign_seed: int | None = None,
                   run_assertions: bool = True,
                   enforce_mode_separation: bool = True) -> V5World:
    """The V5 world model, with every C1 assertion run at construction."""
    kappa = float(cfg.signal_model.kappa) if kappa is None else float(kappa)
    n_sub = int(cfg.get("v5.subgoals.n", 4) if n_subgoals is None else n_subgoals)
    delta = float(cfg.get("v5.depth.delta", 0.75) if delta is None else delta)
    dwell = float(cfg.get("v5.subgoals.dwell", 6.5) if dwell is None else dwell)
    next_share = float(cfg.get("v5.subgoals.next_share", 0.7))

    sigs = FN.build_v4_signatures(cfg, omega=omega, include_explore=False,
                                  foreign_seed=foreign_seed)
    synth = build_v4_synth(cfg)
    alpha = alpha_by_provenance(cfg)

    subsig = build_subgoal_signatures(cfg, sigs.sig_true, n_sub, delta)
    chains = build_subgoal_chains(FN.NUM_REAL_GOALS, n_sub, dwell, next_share)
    n_mu = len(MU_LEVELS)

    A = utils.obj_array(3)
    A[0] = build_v5_A0(cfg, subsig, synth, alpha)
    A[1] = build_v5_A1(cfg, kappa, n_mu * FN.NUM_REAL_GOALS * n_sub)
    A[2] = build_v5_A2(cfg, n_mu, FN.NUM_REAL_GOALS, n_sub)

    gm = GenerativeModel(A=A, B=build_v5_B(cfg, chains, n_mu), C=build_C(cfg),
                         sig=sigs.sig_true, noise_free_synth=synth, alpha=alpha,
                         kappa=kappa, cfg=cfg, d=0.0, sig_true=sigs.sig_true,
                         synth_k=-1, synth_lean=0.0, goal_symmetric=True)

    world = V5World(gm=gm, sigs=sigs, cfg=cfg, mu_levels=MU_LEVELS, n_subgoals=n_sub,
                    subsig=subsig, chains=chains)
    if run_assertions:
        assert_preferences_zero(gm.C)
        diag = {}
        diag["mu1_vs_v4"] = assert_mu1_reproduces_v4(cfg, sigs, synth, alpha, subsig)
        # E30's CONTRAST sweep deliberately drives delta below the separation floor — that end
        # of the axis is the negative control, where there is no depth to find and the
        # observer must fail to find any. So the floor is skippable, EXPLICITLY and per call,
        # and the skip is recorded in the diagnostics rather than inferred from a threshold.
        # An earlier version tried to infer it by comparing delta against the floor, which is
        # a category error: delta is a mixture weight and the floor is in nats.
        if enforce_mode_separation:
            diag["mode_separation"] = assert_modes_are_separated(cfg, subsig)
        else:
            deepest = subsig[-1]
            diag["mode_separation"] = {
                "enforced": False,
                "min_pairwise_js_across_goals": float(np.min(
                    [mode_separation(deepest[g]) for g in range(deepest.shape[0])])),
                "note": "separation floor deliberately not enforced (negative-control cell)",
            }
        diag["beta_vs_v4_5"] = assert_beta_reduces_to_v4_5(subsig, sigs.sig_explore)
        diag["chains"] = assert_chains_uniform_stationary(chains)
        diag["dwell"] = assert_block_count_in_range(cfg, chains, dwell)
        diag["depth_marginal"] = depth_marginal_invariance(world)
        world.diagnostics = diag
    return world


def make_v5_observer(world: V5World, rng: np.random.Generator,
                     mu_prior: np.ndarray | None = None,
                     extra_modalities: list | None = None) -> Agent:
    """One observer over the V5 world.

    ``extra_modalities`` appends observation channels at construction, which is the only time
    they can be appended: pymdp caches the modality count and the Markov-blanket map when the
    Agent is built, so an A array grown afterwards raises inside the inference loop rather
    than being picked up. E33 uses this for the creator's self-report.

    EVERY EXTRA MODALITY GETS A ZERO PREFERENCE, and that is load-bearing rather than tidy.
    Null N7 requires the observer to have no preference over provenance or signal, so that
    disengagement can only come from the epistemic term collapsing and never from a distaste.
    A self-report channel is exactly the kind of thing a preference could be smuggled into —
    an observer that WANTED to be told a particular goal would produce C4's result by wishing
    — so C is extended with zeros and the existing ``assert_preferences_zero`` is extended to
    cover it.
    """
    D = build_v5_D(world.cfg, rng, len(world.mu_levels), FN.NUM_REAL_GOALS,
                   world.n_subgoals, mu_prior=mu_prior)
    A, C_ = world.gm.A, world.gm.C
    if extra_modalities:
        A = utils.obj_array(len(world.gm.A) + len(extra_modalities))
        for m in range(len(world.gm.A)):
            A[m] = world.gm.A[m]
        C_ = utils.obj_array(len(world.gm.C) + len(extra_modalities))
        for m in range(len(world.gm.C)):
            C_[m] = world.gm.C[m]
        for j, extra in enumerate(extra_modalities):
            A[len(world.gm.A) + j] = np.asarray(extra, dtype=float)
            C_[len(world.gm.C) + j] = np.zeros(np.asarray(extra).shape[0])
    # The solver switch, exactly as in ``generative_model.make_agent``; see validation item V-1.
    if bool(world.cfg.get("inference.exact", False)):
        from .exact import ExactAgent
        a = world.cfg.agent
        return ExactAgent(
            A=A, B=world.gm.B, C=C_, D=D,
            control_fac_idx=[F_ATTENTION],
            policy_len=int(a.policy_len),
            gamma=float(a.gamma),
            action_selection=str(a.action_selection),
            use_utility=bool(a.use_utility),
            use_states_info_gain=bool(a.use_states_info_gain),
            rng=rng)
    a = world.cfg.agent
    return Agent(
        A=A, B=world.gm.B, C=C_, D=D,
        control_fac_idx=[F_ATTENTION],
        policy_len=int(a.policy_len),
        inference_horizon=int(a.inference_horizon),
        use_utility=bool(a.use_utility),
        use_states_info_gain=bool(a.use_states_info_gain),
        action_selection=str(a.action_selection),
        gamma=float(a.gamma),
    )


# --------------------------------------------------------------------------- #
# Assertions and diagnostics.
# --------------------------------------------------------------------------- #
def assert_mu1_reproduces_v4(cfg: Config, sigs: FN.V4Signatures, synth: np.ndarray,
                             alpha: np.ndarray, subsig: np.ndarray) -> dict:
    """V5 decision 4 — at mu = 1 the V5 likelihood must equal V4's ELEMENTWISE.

    V4.5's own note applies and is the reason this is checked at construction rather than left
    to a test: a mis-wired depth parameter would not fail loudly. It would produce a plausible
    E30 curve for the wrong reason, and E31's whole design assumes mu acts on the demonstrator
    model rather than on the update.

    Tolerance is 1e-12, not "approximately": the construction reduces to V4 algebraically,
    because at mu = 1 every execution mode IS sig[g], so any nonzero difference is a wiring
    error and not a numerical one.
    """
    from .v4_model import build_v4_A0

    n_mu, ng, n_sub, _ = subsig.shape
    mi = list(MU_LEVELS).index(MU_SHALLOW)
    a5 = build_v5_A0(cfg, subsig, synth, alpha)
    a4 = build_v4_A0(cfg, sigs.sig_true, synth, alpha)
    npv = int(cfg.cardinalities.num_provenance)

    worst = 0.0
    for p in range(npv):
        for g in range(ng):
            for s in range(n_sub):
                got = a5[:, p, mgs_index(mi, g, s, ng, n_sub), :]
                worst = max(worst, float(np.max(np.abs(got - a4[:, p, g, :]))))
    tol = float(cfg.get("v5.depth.v4_boundary_tol", 1e-12))
    assert worst <= tol, (
        f"V5 decision 4: at mu = 1 the V5 likelihood differs from V4's by {worst:.3e} "
        f"(tolerance {tol}). mu has been wired into the wrong pipeline position. mu selects "
        f"WHICH EXECUTION MODES EXIST, inside the alpha channel mixture; it is not a term "
        f"beside it. Check build_subgoal_signatures: at mu = 1 every mode must be sig[g] "
        f"exactly, for every sub-goal index.")
    return {"max_abs_delta_from_v4_A0": worst, "tolerance": tol,
            "checked_subgoal_slices": int(npv * ng * n_sub)}


def assert_modes_are_separated(cfg: Config, subsig: np.ndarray) -> dict:
    """The execution modes must be distinguishable from one another at the deepest mu.

    V1 asserts a pairwise Jensen-Shannon floor on the goal signatures for exactly this reason
    and V5 owes depth the same guarantee. A mode family whose members are hard to tell apart
    produces an artifact that carries less depth than the design claims, and E30 would then
    measure the mode geometry rather than the theory — while returning a small, plausible,
    entirely uninformative effect.

    That is not hypothetical. The first mode family built here cleared every other assertion
    and had a separation of 0.019 nats; N21's mu effect measured +0.18 on a 1-to-3 scale
    against a prior mean of 2.0, which is indistinguishable from the observer ignoring the
    content. This assertion is what stops that from recurring silently.
    """
    floor = float(cfg.get("v5.depth.mode_js_floor", 0.10))
    deepest = subsig[-1]                    # (goal, subgoal, features) at the largest mu
    per_goal = [mode_separation(deepest[g]) for g in range(deepest.shape[0])]
    worst = float(np.min(per_goal))

    # The mode family must respect V4's partition too. assert_c1_properties checks this for
    # sig_true and never sees the modes, so without this the depth construction could put
    # human mass on the foreign block and every C1 property would still report clean.
    foreign_mass = float(np.max(subsig[:, :, :, FN.FOREIGN_FEATURES].sum(axis=-1)))
    assert foreign_mass < 0.10, (
        f"execution modes carry {foreign_mass:.3f} mass in the FOREIGN block. The depth "
        f"construction has broken the human/foreign partition that every V4 C1 property "
        f"rests on, and assert_c1_properties cannot see it because it only inspects "
        f"sig_true. Restrict the mode tail to the human block.")
    assert worst >= floor, (
        f"execution modes at the deepest mu are separated by only {worst:.4f} nats (floor "
        f"{floor}). Depth built on modes this similar is not depth: the artifact cannot carry "
        f"the structure the design claims, and E30 would measure the mode geometry rather "
        f"than the theory. Widen the mode family in build_execution_modes or raise "
        f"v5.depth.delta.")
    return {"min_pairwise_js_across_goals": worst,
            "per_goal": [float(x) for x in per_goal], "floor": floor,
            "max_mode_mass_in_foreign_block": foreign_mass}


def assert_beta_reduces_to_v4_5(subsig: np.ndarray, sig_explore: np.ndarray) -> dict:
    """At mu = 1, V5's beta target must BE V4.5's ``sig_EXPLORE``, elementwise.

    V5's beta mixes toward the goal-marginal of the mode family at that depth. At mu = 1 every
    mode is ``sig[g]``, so that marginal is ``mean_g(sig[g])``, and C2's construction makes
    that exactly ``sig_EXPLORE``. So V5's beta is a GENERALISATION of V4.5's and not a
    replacement, and every V4.5 quantity reached through a shallow creator is still reachable.

    Checked rather than argued, for the reason V4.5 gave about its own boundary check: a beta
    that had quietly stopped reducing to the previous version would not fail loudly. It would
    produce a plausible N21 table for the wrong reason.
    """
    shallow_marginal = subsig[0].mean(axis=0)      # (subgoal, features) at mu = 1
    ex = np.asarray(sig_explore, dtype=float)
    worst = float(np.max(np.abs(shallow_marginal - ex[None, :])))
    assert worst <= 1e-12, (
        f"at mu = 1, V5's beta target differs from V4.5's sig_EXPLORE by {worst:.3e}. The two "
        f"betas have come apart, so V4.5's results are no longer reachable through a shallow "
        f"V5 creator. mean_g(sig[g]) IS sig_EXPLORE by C2's construction; if this fails, the "
        f"mode family no longer averages to sig[g].")
    return {"max_abs_delta_from_sig_explore": worst}


def assert_chains_uniform_stationary(chains: np.ndarray, tol: float = 1e-9) -> dict:
    """Every goal's plan must spend equal average time in every mode.

    If it did not, the goals would differ in their time-averaged feature histograms at mu >= 2
    and not at mu = 1, so a reader could detect depth by counting features. That is legibility,
    which kappa_p already measures. Depth has to be invisible to a counting reader.
    """
    worst = 0.0
    for g in range(chains.shape[0]):
        pi = stationary(chains[g])
        worst = max(worst, float(np.max(np.abs(pi - 1.0 / chains.shape[1]))))
    assert worst <= tol, (
        f"sub-goal chain stationary distribution deviates from uniform by {worst:.3e}. The "
        f"chains must be doubly stochastic, or depth leaks into the feature marginal and mu "
        f"stops being a claim about structure and becomes one about legibility.")
    return {"max_abs_deviation_from_uniform_stationary": worst,
            "doubly_stochastic": True}


def assert_block_count_in_range(cfg: Config, chains: np.ndarray, dwell: float) -> dict:
    """V5 decision 2 — three to four mode blocks must fit in a 24-step artifact.

    Too short and no block structure forms in a single artifact; too long and every artifact
    shows one mode. Either way depth is unreadable for a reason that has nothing to do with
    the theory, and E30 would return a null that means nothing.
    """
    n_timesteps = int(cfg.get("experiments.e30.n_timesteps", 24))
    expected_blocks = 1.0 + (n_timesteps - 1) / float(dwell)
    lo = float(cfg.get("v5.subgoals.min_blocks", 3.0))
    hi = float(cfg.get("v5.subgoals.max_blocks", 4.5))
    assert lo <= expected_blocks <= hi, (
        f"a dwell of {dwell} gives {expected_blocks:.2f} expected mode blocks in "
        f"{n_timesteps} steps, outside [{lo}, {hi}]. V5 decision 2 fixes three to four; "
        f"outside that window a null E30 measures the block geometry, not the theory.")
    return {"dwell": float(dwell), "expected_blocks": float(expected_blocks),
            "n_timesteps": n_timesteps, "window": [lo, hi]}


def depth_marginal_invariance(world: V5World) -> dict:
    """V5 decision 7 — is depth visible in an averaged likelihood, BEFORE E30 runs?

    THE E28 LESSON, MADE INTO A PRE-RUN CHECK. E28 discovered only after running that
    ``sig_EXPLORE`` IS ``mean_g(sig[g])`` by construction, so beta carried no information at
    all in the goal marginal: every bit of evidence about it came from the joint (beta, goal)
    coupling, which is exactly what a finite trajectory biases. The estimate was compressed at
    both ends and beta = 0 lost to chance. Five lines run in advance would have predicted that
    from the construction alone. These are those five lines, for mu.

    THREE MARGINALS, AND THEY ARE EXPECTED TO COME OUT DIFFERENTLY. All three are needed to
    say where mu's evidence actually comes from; any two of them would support an overclaim.

    * ``subgoal_marginal_deviation`` — the mode-averaged emission, per goal. Expected ZERO to
      machine precision, and its being zero is the DESIGN, not a warning: a reader who counts
      features and ignores their order cannot tell a deep artifact from a shallow one.
    * ``full_marginal_deviation`` — averaged over BOTH goal and mode. Also expected zero, and
      it follows from the first. This is the honest statement of what mu does NOT have: there
      is no depth evidence in the flat likelihood, exactly as there was none for beta.
    * ``goal_marginal_deviation`` — averaged over goals but held at a mode. This is where the
      two constructs part company. For beta the corresponding quantity was zero BY
      CONSTRUCTION, because ``sig_EXPLORE`` is ``mean_g(sig[g])``, so all of beta's evidence
      came from the joint (beta, goal) coupling and therefore from finite-sample luck about
      which goal happened to look favoured. If this is nonzero, mu's evidence flows through
      the (mu, sub-goal) coupling instead — and the sub-goal is TEMPORALLY PERSISTENT, so the
      observer accumulates evidence about it across a block rather than betting on a lucky
      draw. That is a materially better identification channel than beta ever had, and it is
      the concrete reason to expect E30 not to reproduce E28's compression at the ends.

    Reported, not asserted. The first two are design invariants; the third is a measurement,
    and turning a measurement into an assertion is how a result gets built into a
    construction.
    """
    subsig = world.subsig
    n_mu, ng, n_sub, _ = subsig.shape

    sub_dev = 0.0
    for mi in range(n_mu):
        for g in range(ng):
            sub_dev = max(sub_dev, float(np.max(np.abs(
                subsig[mi, g].mean(axis=0) - world.sigs.sig_true[g]))))

    base = full_base = None
    goal_dev = full_dev = 0.0
    for mi in range(n_mu):
        marg = subsig[mi].mean(axis=0)          # (subgoal, features), averaged over goals
        full = subsig[mi].mean(axis=(0, 1))     # (features,), averaged over goals AND modes
        if base is None:
            base, full_base = marg, full
        goal_dev = max(goal_dev, float(np.max(np.abs(marg - base))))
        full_dev = max(full_dev, float(np.max(np.abs(full - full_base))))

    return {
        "subgoal_marginal_deviation": sub_dev,
        "full_marginal_deviation": full_dev,
        "goal_marginal_deviation": goal_dev,
        "reading": (
            "subgoal_marginal_deviation and full_marginal_deviation are DESIGN INVARIANTS and "
            "should be ~0: the mode family averages exactly to sig[g], so there is no depth "
            "evidence in the flat likelihood and a reader who counts features without regard "
            "to order cannot tell a deep artifact from a shallow one. goal_marginal_deviation "
            "is the E28 analogue and is a MEASUREMENT. For beta it was zero BY CONSTRUCTION, "
            "which left finite-sample luck in the (beta, goal) coupling as the only evidence "
            "channel and produced the compression at both ends. Nonzero here means mu's "
            "evidence flows through the (mu, sub-goal) coupling, and the sub-goal persists in "
            "time, so evidence accumulates across a block instead of resting on a lucky draw."),
    }


# --------------------------------------------------------------------------- #
# The world side: a creator that actually walks the hierarchy.
# --------------------------------------------------------------------------- #
class HierarchicalCreator:
    """A creator at depth ``mu`` and rationality ``beta``, producing one artifact.

    V5 decision 5: the world generates from the SAME ``subsig`` family the observer's
    likelihood is built from. An observer whose demonstrator model and whose world disagreed
    about what depth means would make the recovery measurement meaningless, and the
    disagreement would be invisible in the output. This is E28's precedent, kept deliberately.

    beta is retained ALONGSIDE mu, and only as a world-side generator setting, because N21
    needs to generate content that is committed-but-shallow and offhand-but-deep.

    WHAT BETA MIXES TOWARD, AND WHY IT IS NOT V4.5's TARGET. V4.5 mixes toward
    ``sig_EXPLORE``, which is mode-INDEPENDENT, so a low beta washes the execution modes out
    along with everything else. Under that construction "offhand but deep" is not a thing that
    can exist: the mixing destroys the hierarchy it is supposed to leave standing, and C1's
    own 2 x 2 is unrealisable. Measured before the change, at beta = 0.25 the mu effect was
    +0.025 on a 1-to-3 scale — not contamination, but the artifact genuinely no longer
    carrying any depth to recover.

    So beta mixes toward the GOAL-MARGINAL OF THE MODE FAMILY at that depth:

        emit[s] = beta * subsig[mu, g, s] + (1 - beta) * mean_g( subsig[mu, g, s] )

    The mode structure survives at any beta; what beta attenuates is which GOAL the modes are
    in service of. That is the aesthetic category C1 is pointing at and it is worth naming:
    you can see the craft — the stages, the practised order — and not what it was for.

    THIS GENERALISES V4.5's BETA RATHER THAN REPLACING IT, and the reduction is exact. At
    mu = 1 every mode is ``sig[g]``, so ``mean_g(subsig[1, g, s])`` is ``mean_g(sig[g])``,
    which IS ``sig_EXPLORE`` by C2's construction. At the shallow limit this is V4.5's beta
    element for element, and ``assert_beta_reduces_to_v4_5`` checks it.

    THE SUB-GOAL TRAJECTORY IS DRAWN ONCE PER ARTIFACT, not stepped as the reader reads. An
    artifact is a fixed object: its structure exists before anyone looks at it, and two
    readers who look at it differently must be looking at the same thing. That is E21's
    ``ObservationTape`` lesson applied at construction rather than retrofitted — there, arms
    that generated content as they read it stopped reading the same artifact after their first
    differing attention choice, and the between-observer measures silently stopped meaning
    what their names said.
    """

    def __init__(self, world: V5World, goal: int, mu: int, beta: float,
                 n_positions: int, rng: np.random.Generator):
        self.world = world
        self.goal = int(goal)
        self.mu = int(mu)
        self.beta = float(beta)
        assert 0.0 <= self.beta <= 1.0, f"beta must be in [0, 1], got {self.beta}"
        assert self.mu in world.mu_levels, f"mu must be one of {world.mu_levels}, got {self.mu}"

        chain = world.chains[self.goal]
        n_sub = world.n_subgoals
        traj = np.empty(int(n_positions) + 1, dtype=int)
        traj[0] = int(rng.integers(n_sub))       # uniform start: the stationary distribution
        for t in range(1, traj.size):
            traj[t] = int(rng.choice(n_sub, p=chain[:, traj[t - 1]]))
        self.trajectory = traj

        mi = list(world.mu_levels).index(self.mu)
        base = world.subsig[mi, self.goal]                      # (subgoal, features)
        # The goal-marginal of the mode family AT THIS DEPTH. Mode structure survives; goal
        # identity is what beta attenuates. At mu = 1 this is sig_EXPLORE exactly.
        goal_marginal = world.subsig[mi].mean(axis=0)           # (subgoal, features)
        mix = self.beta * base + (1.0 - self.beta) * goal_marginal
        mix = np.clip(mix, _EPS, None)
        self.emission = mix / mix.sum(axis=1, keepdims=True)     # (subgoal, features)
        self._pos = 0

    def next_feature(self, rng: np.random.Generator) -> int:
        s = int(self.trajectory[min(self._pos, self.trajectory.size - 1)])
        self._pos += 1
        return int(rng.choice(self.emission.shape[1], p=self.emission[s]))

    def advance(self) -> None:
        """Consume a position without emitting — a SKIM step still happens at a position."""
        self._pos += 1

    def reset(self) -> None:
        """Rewind to the start of the SAME artifact, for the next observer to read it.

        The trajectory is not redrawn. Every observer in a cell reads one artifact, which is
        what makes a between-observer disagreement measure mean anything — E19 found this the
        hard way when a fresh draw per observer turned that column into a measurement of the
        sampler, reporting 1.379 nats of 'disagreement' on content every observer read
        correctly.
        """
        self._pos = 0

    def realised_blocks(self) -> int:
        t = self.trajectory
        return int(1 + np.sum(t[1:] != t[:-1]))


class V5Environment(Environment):
    """Environment whose DEEP features come from a ``HierarchicalCreator``.

    SKIM still returns ``noise_free_synth``, unchanged since V1: skimming never resolves a
    goal, and it must not resolve a sub-goal either, or the effort cost would stop being the
    thing that gates access to structure.
    """

    def __init__(self, *args, creator: HierarchicalCreator | None = None, **kwargs):
        # The V1 creator bank is never consulted under V5 and CANNOT be built here: the config
        # a V5 rollout carries has the MERGED goal cardinality (mu x goal x sub-goal), while
        # ``gm.sig`` still has one row per real goal, so ``build_creator_bank`` would index
        # off the end of it. Passing an empty bank says that explicitly rather than letting a
        # stale default decide. Unconditional, including when ``creator`` is None: the foreign
        # branch of ``sample_feature`` needs no bank either, and making it conditional meant
        # E31's foreign cells tripped the same IndexError the hierarchical cells had avoided.
        kwargs.setdefault("creator_bank", {})
        super().__init__(*args, **kwargs)
        self.creator = creator

    def sample_feature(self, artifact, attention: int, rng: np.random.Generator) -> int:
        if self.creator is None:
            return super().sample_feature(artifact, attention, rng)
        if attention == K.SKIM:
            self.creator.advance()
            return int(rng.choice(self.cd.features, p=self.synth))
        a = float(self.alpha[artifact.provenance])
        if rng.random() < a:
            return self.creator.next_feature(rng)
        self.creator.advance()
        return int(rng.choice(self.cd.features, p=self.synth))


def load_v5_config(quick: bool = False, overrides: dict | None = None) -> Config:
    """A config at V4 cardinality, with the two merged factors sized for V5.

    ``num_provenance`` and ``num_goals`` are left at their V4 values here. The MERGED
    cardinalities are what the pymdp agent sees, and they are applied by the experiment
    workers, because every construction function in this module needs the unmerged values to
    build from and would have to un-merge them again otherwise.
    """
    base = load_v4_config(quick=quick, include_explore=False)
    if overrides:
        for k, v in overrides.items():
            base.set(k, v)
    return base
