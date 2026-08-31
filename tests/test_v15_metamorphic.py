"""V15 metamorphic relations: things that must be true of the construction whatever it produces.

Each of these is a known answer. Several of them found a real defect while the program was being
built, and the comment on those says which -- a metamorphic test that never caught anything is a
test whose relation was chosen to be easy.
"""
from __future__ import annotations

import numpy as np
import pytest

from ghostscale.validation.soundingline.v15 import common as C
from ghostscale.validation.soundingline.v15 import coverage as CV
from ghostscale.validation.soundingline.v15 import exact as EX
from ghostscale.validation.soundingline.v15 import world_chain, world_communication, world_composition
from ghostscale.validation.soundingline.v15.ontology import (Knobs, fit_uniform_marginals,
                                                             overlap_index, pairwise_coupling)

FAMILIES = (world_chain, world_composition, world_communication)


def _w(mod, **kw):
    return mod.sample_world(Knobs(**kw), np.random.default_rng(7))


# --------------------------------------------------------------------------- #
# Coupling and overlap: the two axes of the atlas.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", FAMILIES, ids=lambda m: m.FAMILY)
def test_zero_coupling_is_exactly_a_product(mod):
    w = _w(mod, kappa=0.0)
    pc = pairwise_coupling(w.prior)
    for k in ("process_goal", "process_tendency", "goal_tendency"):
        assert abs(pc[k]) < 1e-9, f"{mod.FAMILY} {k} = {pc[k]}"


@pytest.mark.parametrize("mod", FAMILIES, ids=lambda m: m.FAMILY)
def test_coupling_rises_with_the_knob_and_marginals_stay_uniform(mod):
    """A log-linear tilt fails this: mutual information is not monotone in the tilt strength, so
    a bisection walks past the peak and returns degenerate worlds."""
    got = [_w(mod, kappa=k).meta["realized_coupling"] for k in (0.0, 0.25, 0.5)]
    assert got[0] < got[1] < got[2], got
    for k in (0.0, 0.5, 1.0):
        w = _w(mod, kappa=k)
        assert w.meta["marginal_uniformity"] < 1e-6, (mod.FAMILY, k,
                                                      w.meta["marginal_uniformity"])


def test_overlap_index_does_not_move_with_coupling():
    """Measured under the world's own prior it reads 0.89 at zero overlap as soon as coupling is
    on, which would make the two axes of the phase surface impossible to separate."""
    for k in (0.0, 0.5, 1.0):
        w = _w(world_chain, kappa=k, overlap=0.0)
        assert w.meta["overlap_index"] < 1e-6, (k, w.meta["overlap_index"])


def test_overlap_index_rises_with_the_overlap_knob():
    got = [_w(world_chain, kappa=0.5, overlap=o).meta["overlap_index"]
           for o in (0.0, 0.33, 0.66, 1.0)]
    assert all(got[i] < got[i + 1] for i in range(len(got) - 1)), got


def test_fit_uniform_marginals_is_marginal_preserving_and_dependence_preserving():
    rng = np.random.default_rng(3)
    t = rng.random((4, 4, 4)) ** 3
    f = fit_uniform_marginals(t)
    assert abs(f.sum() - 1.0) < 1e-12
    for ax in range(3):
        m = f.sum(axis=tuple(a for a in range(3) if a != ax))
        assert np.abs(m - 0.25).max() < 1e-6
    assert pairwise_coupling(f)["process_goal"] > 1e-6


# --------------------------------------------------------------------------- #
# Exactness.
# --------------------------------------------------------------------------- #
def test_exact_matches_brute_force_on_a_tiny_chain_world():
    w = _w(world_chain, kappa=0.5, overlap=0.33, n_process=3, n_goal=3, n_tendency=3)
    r = np.random.default_rng(11)
    lat = world_chain.sample_latent(w, r)
    ep = world_chain.rollout(w, lat, r, 6)
    fast = EX.joint_posterior(world_chain, w, ep, 3, channels=("routes",))
    brute = EX.brute_force_posterior(world_chain, w, ep, 3, channels=("routes",))
    assert np.abs(fast - brute).max() < 1e-9


def test_relabelling_a_latents_values_permutes_the_posterior_and_nothing_else():
    w = _w(world_chain, kappa=0.5)
    r = np.random.default_rng(5)
    lat = world_chain.sample_latent(w, r)
    ep = world_chain.rollout(w, lat, r, 8)
    perm = r.permutation(w.n_p)
    assert EX.relabel_invariance(world_chain, w, ep, 4, perm, axis=0) < 1e-9


@pytest.mark.parametrize("mod", FAMILIES, ids=lambda m: m.FAMILY)
def test_posteriors_are_normalized(mod):
    w = _w(mod, kappa=0.5, overlap=0.33)
    r = np.random.default_rng(2)
    lat = mod.sample_latent(w, r)
    ep = mod.rollout(w, lat, r, 8)
    for post in (EX.joint_posterior(mod, w, ep, 4),
                 EX.independent_posterior(mod, w, ep, 4),
                 EX.staged_posterior(mod, w, ep, 4)[0]):
        assert abs(float(post.sum()) - 1.0) < 1e-9


def test_the_independent_rival_never_beats_exact_on_average():
    """The tell that the rival was double-counting: an approximation cannot beat the exact Bayes
    posterior in expectation under a correctly specified model."""
    from ghostscale.validation.soundingline.v15 import architectures as A
    diffs = []
    for wi in range(8):
        r = np.random.default_rng(100 + wi)
        w = world_chain.sample_world(Knobs(kappa=0.5, overlap=0.33, dose=4), r)
        for _ in range(12):
            lat = world_chain.sample_latent(w, r)
            ep = world_chain.rollout(w, lat, r, 10)
            y = ep.hidden["next_action"]
            j = C.log_score(A.read("joint_exact", world_chain, w, ep, 4, "next_action",
                                   rng=r).dist, y)
            i = C.log_score(A.read("independent", world_chain, w, ep, 4, "next_action",
                                   rng=r).dist, y)
            diffs.append(j - i)
    assert float(np.mean(diffs)) > -0.02, float(np.mean(diffs))


# --------------------------------------------------------------------------- #
# The source collision, which V14 got for free and V15 must not.
# --------------------------------------------------------------------------- #
def test_the_artifact_recovers_the_collision_class_and_not_the_motive():
    w = _w(world_communication, kappa=0.0, dose=8)
    r = np.random.default_rng(3)
    inside, klass = [], []
    for _ in range(80):
        lat = world_communication.sample_latent(w, r)
        ep = world_communication.rollout(w, lat, r, 10)
        post = EX.joint_posterior(world_communication, w, ep, 8)
        mp = world_communication.motive_posterior(w, post)
        kl = world_communication.collision_class(lat.tendency)
        truth = world_communication.motive_of(lat.tendency)
        klass.append(sum(mp[m] for m in kl))
        pair = {m: mp[m] for m in kl}
        inside.append(float(max(pair, key=pair.get) == truth))
    assert float(np.mean(klass)) > 0.9, float(np.mean(klass))
    assert abs(float(np.mean(inside)) - 0.5) < 0.15, float(np.mean(inside))


# --------------------------------------------------------------------------- #
# Families, lineages and the coverage stream.
# --------------------------------------------------------------------------- #
def test_the_three_families_agree_on_the_declared_coupling_semantics():
    got = [_w(m, kappa=0.5).meta["realized_coupling"] for m in FAMILIES]
    assert max(got) - min(got) < 0.35, got


def test_the_three_families_do_not_share_a_generative_symbol():
    import ast
    from pathlib import Path
    privates = {}
    for m in FAMILIES:
        tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
        from_ontology = {a.name for node in ast.walk(tree)
                         if isinstance(node, ast.ImportFrom)
                         and (node.module or "").endswith("ontology") for a in node.names}
        privates[m.FAMILY] = {n.name for n in ast.walk(tree)
                              if isinstance(n, ast.FunctionDef) and n.name.startswith("_")
                              and n.name not in from_ontology}
    assert not set.intersection(*privates.values()), privates


def test_lane_seeds_differ_even_for_the_same_world_id():
    assert C.seed("world|discovery|5") != C.seed("world|confirmation|5")
    with pytest.raises(AssertionError):
        C.world_seed("confirmation", 5)


def test_lane_id_ranges_are_disjoint():
    from ghostscale.validation.soundingline.v15.schemas import TIERS
    ids = {ln: C.lane_ids(ln, TIERS["T3"])
           for ln in ("discovery", "transfer", "confirmation", "coverage")}
    assert C.lineage_disjoint(ids)


def test_the_coverage_stream_regenerates_and_verifies():
    d = CV.sequence_definition(64)
    v = CV.verify_prefix(64, d)
    assert v["block_0_digest_matches"] and v["chain_matches"] and v["definition_matches"]
    b = CV.block(3)
    assert b["n_cells"] == CV.BLOCK_CELLS
    assert CV.block_digest(3) == CV.block_digest(3)


def test_every_coverage_block_crosses_the_primary_axes_completely():
    """Truncating at a whole block must leave a balanced design; that is the whole reason the
    executed prefix is interpretable however far the clock got."""
    b = CV.block(11)
    seen = {(c["family"], c["kappa"], c["dose"]) for c in b["cells"]}
    want = {(f, k, d) for f in CV.PRIMARY["family"] for k in CV.PRIMARY["kappa"]
            for d in CV.PRIMARY["dose"]}
    assert seen == want


# --------------------------------------------------------------------------- #
# Scores and rulers.
# --------------------------------------------------------------------------- #
def test_exact_shapley_sums_to_the_total():
    vals = {}
    rng = np.random.default_rng(1)
    comps = ["process", "goal", "tendency"]
    base = {frozenset(): 0.0}
    for mask in range(8):
        s = frozenset(c for i, c in enumerate(comps) if mask & (1 << i))
        vals[s] = float(rng.normal()) if s else 0.0
    out = C.shapley_decomposition(vals)
    assert out["sums_to_total"]


def test_pid_atoms_are_consistent_with_the_mutual_informations():
    rng = np.random.default_rng(4)
    joint = rng.random((4, 4, 3))
    joint = joint / joint.sum()
    p = C.pid_two_source(joint)
    assert abs((p["redundancy"] + p["unique_1"]) - p["mi_1"]) < 1e-9
    assert abs((p["redundancy"] + p["unique_2"]) - p["mi_2"]) < 1e-9
    assert abs((p["redundancy"] + p["unique_1"] + p["unique_2"] + p["synergy"])
               - p["mi_joint"]) < 1e-9


def test_criterion_separates_held_from_failed_without_touching_a_gate():
    c = C.criterion("x", 0.03, 0.02, "greater", "basis")
    assert c["held"] and c["bar"] == 0.02
    c2 = C.criterion("x", 0.01, 0.02, "greater", "basis")
    assert not c2["held"]
    assert C.criterion_status([c, c2]) == "FAILED"
    assert C.criterion_status([c]) == "HELD"
    assert C.criterion_status([]) == "NOT_APPLICABLE"


def test_equivalence_reports_inconclusive_rather_than_claiming_a_null():
    rng = np.random.default_rng(0)
    a = {i: rng.normal(0, 1, 5) for i in range(6)}
    b = {i: rng.normal(0, 1, 5) for i in range(6)}
    out = C.equivalence(a, b, rng, margin=0.01)
    assert out["verdict"] in ("equivalent", "different", "inconclusive")
    assert not (out["verdict"] == "equivalent" and not out["equivalent"])


def test_the_sparsest_cell_rule_holds_for_list_cards():
    """V14's I01 blocked twice on this: a list card's cells each live in exactly one unit, so
    requiring every cell in every unit is arithmetically impossible."""
    from ghostscale.validation.soundingline.v15 import manifest as M
    from ghostscale.validation.soundingline.v15.schemas import TIERS, expected_cells
    card = next(c for c in M.build_cards() if c.unit_kind == "list")
    e = expected_cells(card, TIERS["T0"], "discovery")
    assert e["units"] == 1
