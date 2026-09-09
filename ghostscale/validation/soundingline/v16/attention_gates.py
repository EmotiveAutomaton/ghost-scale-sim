"""Known repetition, withheld-feedback and enacted-allocation controls."""
from .attention_craft import learn_processed,prepare,perform


def run():
    condition={"instruction":True,"feedback":False,"offers":32}
    design=prepare("attention-fixture",condition,0,8)
    public,private=perform(design,condition,0,"attention-fixture")
    repeat=[{"program":[0,1],"feedback":True}]*4
    rejected=[{"program":[0,1],"feedback":False}]*4
    unknown=[{"program":[0,1],"feedback":None}]*4
    cases={"attention-realization":{"actual_high_allocation":[sum(item["topic"]==topic for item in public["processed"]["focal-effort"]) for topic in range(2)]==[12,4],
                "actual_low_allocation":[sum(item["topic"]==topic for item in public["processed"]["other-effort"]) for topic in range(2)]==[4,12],
                "equal_attempt_count":all(len(records)==16 for records in public["processed"].values()),
                "real_repetition_acquires":learn_processed(repeat)==[[0,1]],
                "empty_training_null":learn_processed([])==[]},
           "attention-information":{"failed_feedback_is_not_eligible":learn_processed(rejected)==[],
                "absent_feedback_is_not_secret_failure_filter":learn_processed(unknown)==[[0,1]],
                "withheld_feedback_not_serialized":all(item["feedback"] is None for records in public["processed"].values() for item in records),
                "training_targets_stay_private":all("intended_target" not in item for records in public["processed"].values() for item in records)}}
    gates=[{"id":name,"checks":checks,"instrument_state":"valid" if all(checks.values()) else "failed","evidence_scope":"fixture"}
           for name,checks in cases.items()]
    return {"gates":gates,"instrument_state":"valid" if all(row["instrument_state"]=="valid" for row in gates) else "failed"}
