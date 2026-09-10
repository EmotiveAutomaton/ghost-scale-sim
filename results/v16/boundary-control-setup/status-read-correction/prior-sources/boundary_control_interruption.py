"""Actual final-control interruption using a separately identified known campaign."""
import os
import subprocess
import sys
import time
from .runtime import REPO
from .records import read,write,file_digest,now
from .expansion_interruption import fixture_campaign


def control(directory, source, source_name):
    root=directory/"campaign"
    if root.exists():
        raise ValueError("prior final-control interruption retained; choose a new attempt")
    fixture_campaign(root)
    command=[sys.executable,"-B","-m","runners.control_v16_boundary","--root",str(root),
             "--fixture-source",str(source),"--fixture-name",source_name]
    options={"cwd":REPO,"env":dict(os.environ,PYTHONPATH=str(REPO),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1"),
             "creationflags":subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0}
    with (directory/"interrupt.log").open("wb") as log:
        child=subprocess.Popen(command,stdout=log,stderr=log,**options)
        try:
            until=time.monotonic()+90
            while time.monotonic()<until:
                if list((root/"boundary-controls-1").glob("*/conditions/*/RECEIPT.json")):
                    break
                if child.poll() is not None:
                    raise ValueError("final-control CLI exited before a condition; see interrupt.log")
                time.sleep(.005)
            else:
                raise ValueError("final-control CLI made no bounded known progress")
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
        resumed=subprocess.run(command+["--resume"],stdout=log,stderr=log,timeout=300,**options)
    if resumed.returncode:
        raise ValueError("actual final-control resume failed; see resume.log")
    completed=read(root/"boundary-controls-1/COMPLETION.json")
    plan=read(root/"packets/boundary-controls-1.json")["identity"]["design"]
    checks={"actual_cli_terminated":child.returncode!=0 and before["execution_state"]!="completed",
            "all_saved_partial_outputs_and_clocks_preserved":all((root/name).read_bytes()==content for name,content in saved.items()),
            "every_condition_completed":completed["condition_controls"]==plan["planned_controls"],
            "actual_controls_valid":completed["instrument_state"]=="valid",
            "job_does_not_close_campaign":completed["campaign_complete"] is False}
    result={"execution_state":"completed","instrument_state":"valid" if all(checks.values()) else "failed",
        "checks":checks,"completed_at":now(),"source_packet_hash":completed["source_packet_hash"],
        "packet_hash":completed["packet_hash"],"condition_controls":completed["condition_controls"],
        "scope":"Actual final-control CLI on known fixtures; no final scientific makers generated",
        "files":{path.relative_to(directory).as_posix():file_digest(path) for path in directory.rglob("*.json")}}
    write(directory/"RECEIPT.json",result)
    return result
