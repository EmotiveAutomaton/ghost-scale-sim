"""Amended R-family audit: original physical audit plus independent reader math."""
from pathlib import Path
from .audit_inquiry import audit_unit, close, checked
from .audit_statistics import verify
from .inquiry_stable_reference import readout
from .records import read
from runners.replay_v16_readers import recursive_requests


def audit(root, summary):
    rows = [read(path) for path in sorted((root/"units").glob("*_points.json"))]
    requests = 0
    for row in rows:
        audit_unit(root, row)
        for path in sorted((root/"predictions").glob(row["unit_id"]+"*.json")):
            for frame in recursive_requests(read(path)):
                if not frame["kind"].startswith("ghostscale.validation.soundingline.v16.inquiry_stable:"):
                    raise ValueError("amended reader frame invokes another implementation")
                close(frame["result"], readout(frame["kind"], frame["public"], frame["options"]))
                requests += 1
        resources = checked(root/"private"/f'{row["unit_id"]}-resources.json',
                            row["reader_resources_sha256"])
        if not resources["requests"]:
            raise ValueError("amended unit lacks actual reader resource observations")
    count = verify(rows, summary)
    return {"execution_state": "completed", "instrument_state": "valid", "n_raw_units": len(rows),
            "reproduced_contrasts": count, "all_reported_aggregates_reproduced": True,
            "independent_scalar_reader_requests": requests, "actual_physics_and_evidence_flow": "verified",
            "reader_predictions_reexecuted": "independent scalar model, not merely saved-score reuse",
            "full_rollout_replay": False}
