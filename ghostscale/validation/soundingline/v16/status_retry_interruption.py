"""Actual last-expansion CLI restart control on independent known campaigns."""
import os
import subprocess
import sys
import time
from .runtime import REPO
from .records import read, write, file_digest, now
from .expansion_interruption import fixture_campaign


def control(directory, card):
    root=directory/"campaign"
    if root.exists():
        raise ValueError("prior boundary interruption evidence retained")
    fixture_campaign(root)
    command=[sys.executable,"-B","-m","runners.run_v16_boundary_resilient","--root",str(root),"--fixture",card]
    options={"cwd":REPO,"env":dict(os.environ,PYTHONPATH=str(REPO),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1"),
             "creationflags":subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0}
    base=root/("boundary-fixture-"+card)/card
    with (directory/"interrupt.log").open("wb") as log:
        child=subprocess.Popen(command,stdout=log,stderr=log,**options)
        try:
            until=time.monotonic()+60
            while time.monotonic()<until:
                if list((base/"units").glob("*_points.json")):
                    break
                if child.poll() is not None:
                    raise ValueError("boundary CLI exited before known unit; see interrupt.log")
                time.sleep(.01)
            else:
                raise ValueError("boundary CLI made no bounded known progress")
            child.terminate()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    before=read(root/"RUNNER_STATUS.json")
    write(directory/"INTERRUPTED_STATUS.json",before)
    saved={path.relative_to(root).as_posix():path.read_bytes() for path in root.rglob("*.json") if path.name!="RUNNER_STATUS.json"}
    with (directory/"resume.log").open("wb") as log:
        resumed=subprocess.run(command+["--resume"],stdout=log,stderr=log,timeout=150,**options)
    if resumed.returncode:
        raise ValueError("actual last-expansion resume failed; see resume.log")
    completion=read(base.parent/"COMPLETION.json")
    checks={"actual_cli_interrupted":child.returncode!=0 and before["execution_state"]!="completed",
            "all_saved_inputs_outputs_and_clocks_preserved":all((root/name).read_bytes()==content for name,content in saved.items()),
            "complete_unit_bound_and_independent_audit":completion["execution_state"]=="completed" and completion["instrument_state"]=="valid",
            "job_does_not_close_campaign":completion["campaign_complete"] is False}
    result={"execution_state":"completed","instrument_state":"valid" if all(checks.values()) else "failed",
            "checks":checks,"card_id":card,"packet_hash":completion["packet_hash"],"completed_at":now(),
            "scope":"Known final-expansion CLI fixture; no scientific source outcomes reused",
            "files":{path.relative_to(directory).as_posix():file_digest(path) for path in directory.rglob("*.json")}}
    write(directory/"RECEIPT.json",result)
    return result
