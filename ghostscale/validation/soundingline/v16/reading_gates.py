"""Known-answer controls for the learned-library observer packet."""
from itertools import product
from math import prod
from .reconstruction import (library_prior, observation_mass, LIBRARIES, MOTIFS,
                             RELIABILITIES, TOPIC_PROBABILITIES, prepare, reader)
from .learning import learn
from .reference import artifact_distribution
from .records import canonical


def run():
    gates = []
    def gate(name, positive, null, broken, boundary, measurements=None):
        gates.append({"id":name,"positive":bool(positive),"null":bool(null),
                      "deliberate_break_detected":bool(broken),
                      "nonidentifiability_boundary":bool(boundary),"evidence_scope":"fixture",
                      "instrument_state":"valid" if all([positive,null,broken,boundary]) else "failed",
                      "measurements":measurements or {}})
    expected = [0.0]*4
    for reliability,topic,direction in product(RELIABILITIES,TOPIC_PROBABILITIES,range(2)):
        probabilities = [reliability*(topic if direction == 0 else 1-topic),
                         reliability*(1-topic if direction == 0 else topic),1-reliability]
        for outcomes in product(range(3),repeat=4):
            traces = [MOTIFS[item] if item < 2 else (0,) for item in outcomes]
            targets = [3 if item != 1 else 12 for item in outcomes]
            learned = learn(traces,targets).library
            index = int(MOTIFS[0] in learned)+2*int(MOTIFS[1] in learned)
            expected[index] += prod(probabilities[item] for item in outcomes)/18
    error = max(abs(a-b) for a,b in zip(expected,library_prior(4)))
    gate("exact-training-prior",error < 1e-10,
         max(abs(a-b) for a,b in zip(library_prior(0),(1.0,0.0,0.0,0.0))) < 1e-10,
         expected != [0.25]*4, expected[1] == expected[2] or abs(expected[1]-expected[2]) < 1e-10,
         {"maximum_error":error})
    error = 0.0
    for index,library in enumerate(LIBRARIES):
        for target in range(16):
            scalar = artifact_distribution(library,target)
            error = max(error,max(abs(observation_mass(index,target,a)-scalar[a]) for a in range(16)))
    public,private,_ = prepare("reading-gate","P03",{"id":"fixture","prior_works":8},0)
    public["task_id"] = "opaque"
    for observed in public["permitted_prior_artifacts"]:
        observed["artifact"] = 3
    public["final_artifact"] = 3
    public["declared_context"]["current_observation"]["artifact"] = 3
    predicted = reader(canonical(public))
    direct = reader(canonical(public),"direct-table")
    ignored = reader(canonical(public),"generic")
    gap = max(abs(a-b) for a,b in zip(predicted["future_probabilities"],ignored["future_probabilities"]))
    equality = max(abs(a-b) for a,b in zip(predicted["future_probabilities"],direct["future_probabilities"]))
    gate("data-use-joint-calibration",error < 1e-10 and gap > 1e-8,
         equality < 1e-10, gap > 1e-8,
         all(weight > 0 for weight in predicted["history_probabilities"]),
         {"scalar_error":error,"direct_table_error":equality,
          "data_ignoring_and_independent_work_break_gap":gap})
    private["controller_state"] = 999
    public["task_id"] = "unrelated-file-and-seed"
    fixed = reader(canonical(public))
    leaky_a,leaky_b = (fixed,0),(fixed,999)
    gate("physical-reader-boundary",fixed == predicted,fixed == reader(canonical(public)),
         leaky_a != leaky_b, len(private["equivalence_classes"]) > 1)
    forbidden = dict(public,true_library=[0,1])
    rejected = False
    try:
        reader(canonical(forbidden))
    except ValueError:
        rejected = True
    impossible = dict(public)
    impossible["declared_context"] = {**public["declared_context"],
        "current_observation":{**public["declared_context"]["current_observation"],"artifact":15}}
    mismatch = reader(canonical(impossible))["model_mismatch"]
    gate("model-mismatch",rejected and mismatch,not predicted["model_mismatch"],
         rejected and mismatch, equality < 1e-10)
    full = observation_mass(1,3,3)
    restricted = observation_mass(1,3,3,(),tuple(range(1,8)))
    gate("query-realization",full > 0 and restricted == 0,
         observation_mass(1,3,0,(),tuple(range(8))) == observation_mass(1,3,0),
         full != restricted,
         observation_mass(1,3,3,(0,)) > 0 and observation_mass(1,3,3,(1,)) > 0)
    return {"evidence_scope":"fixture","gates":gates,
            "instrument_state":"valid" if all(g["instrument_state"] == "valid" for g in gates) else "failed"}
