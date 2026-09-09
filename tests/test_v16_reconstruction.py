import math
from itertools import product
import pytest
from ghostscale.validation.soundingline.v16.reconstruction import (
    library_prior, prepare, reader, observation_mass, LIBRARIES, RELIABILITIES,
    TOPIC_PROBABILITIES, MOTIFS)
from ghostscale.validation.soundingline.v16.reference import artifact_distribution
from ghostscale.validation.soundingline.v16.records import canonical
from ghostscale.validation.soundingline.v16.learning import learn


def test_acquisition_prior_matches_independent_executed_training_enumeration():
    expected = [0.0]*4
    for r, q, direction in product(RELIABILITIES, TOPIC_PROBABILITIES, range(2)):
        p = [r*(q if direction == 0 else 1-q), r*(1-q if direction == 0 else q), 1-r]
        for outcomes in product(range(3), repeat=4):
            traces = [MOTIFS[item] if item < 2 else (0,) for item in outcomes]
            targets = [3 if item != 1 else 12 for item in outcomes]
            library = learn(traces, targets).library
            index = int(MOTIFS[0] in library) + 2*int(MOTIFS[1] in library)
            expected[index] += math.prod(p[item] for item in outcomes)/18
    assert max(abs(a-b) for a,b in zip(expected, library_prior(4))) < 1e-10
    assert abs(sum(library_prior(16))-1) < 1e-10


def test_all_artifact_likelihoods_match_independent_scalar_anchor():
    for library_index, library in enumerate(LIBRARIES):
        for target in range(16):
            expected = artifact_distribution(library, target)
            assert max(abs(observation_mass(library_index,target,artifact)-expected[artifact])
                       for artifact in range(16)) < 1e-10


def test_data_use_correlated_history_and_direct_table_known_boundary():
    public, private, _ = prepare("reader-fixture", "P03", {"id":"dose-8","prior_works":8}, 0)
    public["task_id"] = "opaque-id"
    for item in public["permitted_prior_artifacts"]:
        item["artifact"] = 3
    public["final_artifact"] = 3
    public["declared_context"]["current_observation"]["artifact"] = 3
    prediction = reader(canonical(public))
    table = reader(canonical(public), "direct-table")
    ignored = reader(canonical(public), "generic")
    assert max(abs(a-b) for a,b in zip(prediction["future_probabilities"],table["future_probabilities"])) < 1e-10
    assert max(abs(a-b) for a,b in zip(prediction["future_probabilities"],ignored["future_probabilities"])) > 1e-8
    # Deliberately erase the dependence between repeated works from one maker.
    assert prediction["history_probabilities"] != list(library_prior(16))
    private["controller_state"] = 999
    assert prediction == reader(canonical(public))
    public["task_id"] = "unrelated-filename-and-seed"
    assert prediction == reader(canonical(public))


def test_impossible_observation_is_explicit_mismatch_and_private_fields_rejected():
    public, _, _ = prepare("impossible-fixture", "P01", {"id":"artifact-only"}, 0)
    public["declared_context"]["current_observation"]["artifact"] = 15
    assert reader(canonical(public))["model_mismatch"]
    public["true_library"] = [0,1]
    with pytest.raises(ValueError, match="schema"):
        reader(canonical(public))
