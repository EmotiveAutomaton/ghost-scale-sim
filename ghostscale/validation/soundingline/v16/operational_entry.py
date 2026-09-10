"""The seven campaign stages share a finite queue and preserve existing packets."""
import importlib
from .records import read, write, now, file_digest
from .runtime import REPO, campaign
from .runtime_status_retry import supervisor
from .record_integrity import source_locks
from .campaign_ownership import campaign_owner
from .operational_queue import freeze_plan, queue_state, record_failure, dispatchable, verify_closeout

PREFIX = "ghostscale.validation.soundingline.v16."


def job(name, stage, cap, admission, completion, dependencies, contribution, handler=None, forecast=None):
    return {"job_id": name, "stage": stage, "unit_cap": cap, "admission": admission, "completion": completion,
        "dependencies": dependencies, "contribution": contribution, "handler": handler, "forecast": forecast}


def remaining_plan():
    return {"schema_version": "v16.remaining-queue.1", "scope": "remaining work after all thirty native scouts and B02 search",
        "cap_scope": "upper bounds; exact scientific allocations remain in separately frozen implementation packets",
        "jobs": [
        job("constructor-expansion-1", "discovery", 76800, "expansion-setup/ADMISSION.json", "constructor-expansion-1/COMPLETION.json", [],
            "Fresh constructor tests of the twenty-six named scout boundaries", "expansion_runner"),
        job("final-boundary-disposition", "discovery", 307200, "final-expansion-setup/ADMISSION.json", "expansion-closure/COMPLETION.json", ["constructor-expansion-1"],
            "At most one further 1024-maker allocation per retained condition, only for named informative imprecision; otherwise explicit exhaustion", "boundary_runner", "final-expansion-setup/FORECAST.json"),
        job("explanatory-catalogue", "transfer", 64, "catalogue-setup/ADMISSION.json", "explanatory-catalogue/COMPLETION.json", [],
            "Index at most sixty-four inspectable cases across the eight commissioned lenses; distinguish descriptive examples from validated search discoveries", "catalogue_runner"),
        job("confirmation-selection", "confirmation", 3, "confirmation-setup/SELECTION_ADMISSION.json", "confirmation-selection/COMPLETION.json", ["constructor-expansion-1", "final-boundary-disposition", "explanatory-catalogue"],
            "Freeze at most three valid claims by target, serious rival, transfer relevance and cost", "confirmation_plan"),
        job("confirmations", "confirmation", 12288, "confirmation-setup/ADMISSION.json", "confirmation/COMPLETION.json", ["confirmation-selection"],
            "At most three unchanged claim packets with at most 4096 fresh independent makers each; preserve every failure", "confirmation_runner", "confirmation-setup/FORECAST.json"),
        job("closeout", "close", 1, "closeout/ADMISSION.json", "CLOSEOUT.json", ["final-boundary-disposition", "explanatory-catalogue", "confirmations"],
            "Verify every commissioned card and all four scientific/documentary closeout proofs") ]}


def plan(root):
    path = root/"operations/QUEUE_PLAN.json"
    return read(path)["identity"]["plan"] if path.exists() else remaining_plan()


def preflight(root):
    accepted = campaign(root)
    if not (root/"RUNNER_STATUS.json").exists():
        # A second directory with the same identity must first prove that it is
        # not taking over an owned campaign. This probe never writes live status.
        with campaign_owner(root):
            return preflight_record(root, accepted)
    return preflight_record(root, accepted)


def preflight_record(root, accepted):
    sources = source_locks(root, REPO)
    queue = queue_state(root, plan(root))
    return {"execution_state": "completed", "instrument_state": "valid", "stage": "preflight",
        "campaign_complete": False, "accepted_at": accepted["accepted_at"], "deadline": accepted["deadline"],
        "source_checks": sources, "queue": queue, "writes_live_status": False,
        "live_status": read(root/"RUNNER_STATUS.json") if (root/"RUNNER_STATUS.json").exists() else None}


def run(root, stage):
    if stage == "preflight":
        return preflight(root)
    accepted = campaign(root)
    with supervisor(root, stage) as heartbeat:
        source_locks(root, REPO)
        definition = plan(root)
        frozen = freeze_plan(root, definition)
        queue = queue_state(root, definition)
        eligible = [row for row in queue["jobs"] if row["state"] == "eligible" and (stage == "resume" or row["stage"] == stage)]
        active_path = root/"operations/ACTIVE_JOB.json"
        if stage == "resume" and active_path.exists():
            active = read(active_path)
            if active["plan_sha256"] != frozen["sha256"] or active["accepted_at"] != accepted["accepted_at"]:
                raise ValueError("active job identity or accepted clock changed")
            eligible.sort(key=lambda row: row["job_id"] != active["job_id"])
        if not eligible:
            result = {"execution_state": "blocked", "instrument_state": "valid", "campaign_complete": False,
                "reason": "no admitted eligible job for this stage; unresolved dependencies and closeout remain explicit", "queue": queue}
            heartbeat(execution_state="blocked", result=result)
            return result
        selected = next(row for row in definition["jobs"] if row["job_id"] == eligible[0]["job_id"])
        if selected["forecast"]:
            forecast = dispatchable(root, selected, read(root/selected["forecast"]))
            if not forecast["eligible_within_deadline"]:
                result = {"execution_state": "checkpointed", "instrument_state": "valid", "campaign_complete": False,
                    "reason": "measured forecast exceeds remaining immutable horizon", "forecast": forecast}
                heartbeat(execution_state="checkpointed", result=result)
                return result
        write(active_path, {"job_id": selected["job_id"], "stage": selected["stage"], "plan_sha256": frozen["sha256"],
            "accepted_at": accepted["accepted_at"], "deadline": accepted["deadline"]}, immutable=False)
        try:
            if selected["job_id"] == "closeout":
                result = verify_closeout(root, read(root/"closeout/INPUTS.json"))
                if result["campaign_closed"]:
                    write(root/"CLOSEOUT.json", result)
            else:
                module = importlib.import_module(PREFIX+selected["handler"])
                packet_name = getattr(module, "PACKET", None)
                existing_packet = packet_name is None or (root/"packets"/(packet_name+".json")).exists()
                result = module.execute(root, heartbeat, resume=stage == "resume" and existing_packet)
        except Exception as error:
            failure = record_failure(root, selected["job_id"], error,
                root_cause=type(error).__name__+": "+str(error), family=selected["job_id"])
            heartbeat(execution_state=failure["execution_state"], failure=failure)
            raise
        heartbeat(execution_state=result["execution_state"], result=result)
        return result
