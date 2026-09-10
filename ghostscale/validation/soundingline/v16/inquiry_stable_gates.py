"""Bounded R-family tie repair; original inquiry_gates.py remains frozen.
"""
"""Known-answer learning, noise and finite expected-value admission controls."""
from itertools import product
from .inquiry_stable import PRIOR,GOALS,update,construction,competence,expected_after,prediction,choose
from .inquiry import choose as frozen_choose
from .inquiry_stable_reference import chosen as reference_choice
from .records import canonical
from .recoding import MAPPINGS, inquiry_observation, inquiry_result_equivalent, close


def run():
    mapping=(2,0,3,1)
    def score(belief):
        return sum(sum(1<<mapping[command] for command in construction(belief,goal))==goal for goal in GOALS)/6
    belief=PRIOR
    for command in range(3):
        belief=update(belief,command,mapping[command])
    noise=update(update(PRIOR,0,0),0,1)
    known=sum(((1<<a)|(1<<b))==GOALS[0] for a,b in product(range(4),repeat=2))/16
    expected=expected_after(PRIOR,(0,))[0]
    scalar=sum(0.25*competence(update(PRIOR,0,cell)) for cell in range(4))
    cases={"inquiry-learning":{"live_execution_gain":score(belief)==1.0,
                               "erasing_learning_breaks_execution":score(PRIOR)==1/6,
                               "data_ignoring_reader_detected":score(belief)!=score(PRIOR)},
           "inquiry-noise":{"irreducible_noise_is_admitted":noise[-1]==1,
                             "noise_no_skill_gain":expected_after(noise,(0,1))[0]==competence(noise),
                             "independent_random_execution":known==0.125==competence(noise)},
           "inquiry-value":{"exact_one_step_matches_scalar":abs(expected-scalar)<1e-10,
                             "learning_can_change_future_competence":expected>competence(PRIOR),
                             "certainty_has_no_information":expected_after(noise,(0,))[1]==0}}
    partial=[1/6 if item[0]==2 else 0.0 for item in MAPPINGS]+[0.0]
    public={"schema_version":"v16.inquiry.1","task_id":"known-answer",
        "beliefs":[partial,PRIOR],"offers":[0,1],"histories":[[],[]],"familiarity":[4,0],
        "pending_commands":[[],[]],"feedback_batches":[2,2],"future_weights":[0.5,0.5],
        "opportunity_cost":0.01,"remaining_interactions":8,"tie_draw":0.25,"committed_domain":None,
        "offer_probabilities":[[0.1,0.2,0.6,0.1],[0.4,0.3,0.2,0.1]]}
    renamed=inquiry_observation(public,(2,0,3,1))
    original=frozen_choose(canonical(public),"value-learning")
    changed=frozen_choose(canonical(renamed),"value-learning")
    corrected=choose(canonical(public),"value-learning")
    corrected_changed=choose(canonical(renamed),"value-learning")
    cases["inquiry-numeric-recoding"]={
        "original_failure_detected":not inquiry_result_equivalent("inquiry",original,changed,public,renamed,(2,0,3,1)),
        "amended_recoding":inquiry_result_equivalent("inquiry",corrected,corrected_changed,public,renamed,(2,0,3,1)),
        "independent_scalar_agreement":close(corrected,reference_choice(public,"value-learning"))}
    mixed=[0.99]+[0.0]*23+[0.01]
    public.update(beliefs=[mixed,mixed],opportunity_cost=0.0,feedback_batches=[1,1])
    cases["inquiry-numeric-abstention"]={
        "original_spurious_query_detected":frozen_choose(canonical(public),"value-learning")["domain"] is not None,
        "amended_zero_gain_abstention":choose(canonical(public),"value-learning")["domain"] is None,
        "independent_zero_gain_abstention":reference_choice(public,"value-learning")["domain"] is None}
    gates=[{"id":name,"checks":checks,"evidence_scope":"fixture",
            "instrument_state":"valid" if all(checks.values()) else "failed"} for name,checks in cases.items()]
    return {"gates":gates,"instrument_state":"valid" if all(row["instrument_state"]=="valid" for row in gates) else "failed"}
