"""Outcome-independent selected replay from an extracted packet source checkout."""
import argparse
import json
import math
import time
from pathlib import Path
from ghostscale.validation.soundingline.v16.records import read,write,canonical,file_digest
from ghostscale.validation.soundingline.v16.runtime import local_owner
from ghostscale.validation.soundingline.v18_3.runtime import load,dispatch,REPO
from ghostscale.validation.soundingline.v18_3.verify import metrics,check_unit


def replay(root,output,campaign):
    started=time.process_time()
    with local_owner(campaign/'scientific-worker-owner'):
        plan=read(root/'PLAN.json');complete=read(root/'COMPLETE.json')
        assert complete['plan_sha256']==file_digest(root/'PLAN.json')
        assert complete['summary_sha256']==file_digest(root/'SUMMARY.json')
        assert all(file_digest(REPO/name)==value for name,value in plan['sources'].items())
        units=[];cells={};checks=0
        for block in complete['blocks']:
            for unit in load(root,block):
                check_unit(unit);checks+=1;units.append(unit)
                for tags,values in metrics(unit):
                    key=canonical(tags).decode()
                    for metric,value in values.items():cells.setdefault((key,metric),[]).append(value)
        summary=read(root/'SUMMARY.json');checked=0
        for (key,metric),values in cells.items():
            report=summary['cells'][key][metric]
            assert report['n']==len(values)
            if all(value is not None and math.isfinite(value) for value in values):
                assert abs(math.fsum(map(float,values))/len(values)-report['mean'])<1e-10
            else:assert report['mean'] is None
            checked+=1
        # Fixed evenly spaced index rule, independent of outcomes and effect sizes.
        indices=sorted({int(i*(len(units)-1)/7) for i in range(8)})
        for i in indices:
            fresh=json.loads(canonical(dispatch(units[i]['request'])))
            assert fresh==units[i],f'source replay differs at unit {i}'
        write(output,dict(passed=True,plan_sha256=file_digest(root/'PLAN.json'),summary_sha256=file_digest(root/'SUMMARY.json'),
              units_checked=checks,aggregate_means_checked=checked,replayed_indices=indices,source_replays=len(indices),
              verifier_sha256=file_digest(Path(__file__)),cpu_seconds=time.process_time()-started,
              scope='all retained unit validity and means; eight fixed source-extracted whole-unit replays, not full rerun'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();replay(a.root,a.output,a.campaign)
