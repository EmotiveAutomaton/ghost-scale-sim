"""Fixed purpose-relation and real inhibition controls."""
from .purpose_craft import inhibition
from .purpose_study import alignment


def run():
    aligned=alignment(3,7)
    partial=alignment(3,5)
    opposed=alignment(3,12)
    retained,checks=inhibition([(0,1)],7)
    rejected,opposed_checks=inhibition([(0,1)],12)
    cases={"purpose-realization":{"aligned_positive_reference":aligned["reward_correlation"]>0,
            "partial_reference_is_orthogonal":abs(partial["reward_correlation"])<1e-10,
            "opposed_reference_is_negative":abs(opposed["reward_correlation"]+1)<1e-10,
            "all_reference_programs":aligned["reference_programs"]==585},
           "purpose-inhibition":{"aligned_habit_retained":retained==[(0,1)],"opposed_steps_inhibited":rejected==[],
            "null_no_unsupported_checking":inhibition([],12)==([],[]),
            "checking_has_actual_primitive_cost":sum(row["primitive_cost"] for row in opposed_checks)==2}}
    gates=[{"id":name,"checks":checks,"instrument_state":"valid" if all(checks.values()) else "failed",
            "evidence_scope":"fixture"} for name,checks in cases.items()]
    return {"gates":gates,"instrument_state":"valid" if all(row["instrument_state"]=="valid" for row in gates) else "failed"}
