"""Known-answer learning, noise and finite expected-value admission controls."""
from itertools import product
from .inquiry import PRIOR,GOALS,update,construction,competence,expected_after,prediction


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
    gates=[{"id":name,"checks":checks,"evidence_scope":"fixture",
            "instrument_state":"valid" if all(checks.values()) else "failed"} for name,checks in cases.items()]
    return {"gates":gates,"instrument_state":"valid" if all(row["instrument_state"]=="valid" for row in gates) else "failed"}
