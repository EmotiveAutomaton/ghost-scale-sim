"""Execute the strongest public-law compiler on retained G5 construction tasks."""
import argparse
import gzip
import json
from pathlib import Path
import time
from ghostscale.validation.soundingline.v18_2 import direct_control as d,model as m
from ghostscale.validation.soundingline.v18_2.runtime import load
from ghostscale.validation.soundingline.v18_2.verify import replay
from ghostscale.validation.soundingline.v16.records import read,write,file_digest,now
from runners.verify_v18_2 import assembly_execution


def report(campaign,output):
    start=time.process_time();wall=time.monotonic();rows=[];inputs={}
    for name in ('g5-final','g5-dependency-final','g5-assembly'):
        root=campaign/name;complete=read(root/'COMPLETE.json');inputs[name]=file_digest(root/'PLAN.json')
        for block in complete['blocks']:
            for unit in load(root,block):
                if name=='g5-assembly':
                    seen=set()
                    for row in unit['rows']:
                        w=row['world'];target=row['outputs'][0]['target'];key=m.canonical([w,target])
                        if key in seen:continue
                        seen.add(key);program=d.assembly(w,target);actual=assembly_execution(w,program)
                        success=actual['legal'] and actual['successfully_stopped'] and actual['state']==target
                        rows.append(dict(batch=name,case_id=unit['case_id'],world=w,target=target,program=program,
                            success=success,compile_checks=6,selection_units=3,emission_units=len(program),execution_units=len(program)))
                else:
                    case=unit['case'];w=case['world'];target=sum(1<<i for i in w['groups'][0])
                    program=d.board(w,target);success=replay(program)==target
                    rows.append(dict(batch=name,case_id=case['case_id'],world=w,target=target,program=program,
                        success=success,compile_checks=4,selection_units=0,emission_units=len(program),execution_units=len(program)))
    assert all(row['success'] for row in rows)
    output.mkdir(parents=True,exist_ok=True);raw=output/'DIRECT_points.json.gz';raw.write_bytes(gzip.compress(m.canonical(rows),mtime=0))
    record=dict(passed=True,at=now(),rows=len(rows),lineages=len({r['case_id'] for r in rows}),success_fraction=1.,
        sampling='retained exposed tasks; no new independent makers',information='public world law and learner own target only; no maker latent state or future result',
        comparator='direct compiler may omit acquisition that its known-law construction task does not require',
        scope='construction control only; not a prediction of the other maker or a learned goal-uptake model',
        inputs=inputs,raw_sha256=file_digest(raw),compiler_sha256=file_digest(Path(d.__file__)),
        report_source_sha256=file_digest(Path(__file__)),cpu_seconds=time.process_time()-start,wall_seconds=time.monotonic()-wall)
    write(output/'SUMMARY.json',record)
    write(campaign/'audit/attempts/direct-controls.json',dict(cpu_seconds=record['cpu_seconds'],wall_seconds=record['wall_seconds'],child_cpu_seconds=0))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(report(a.campaign,a.output)))
