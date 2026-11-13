"""Finite variant adapter for the separately frozen V29 continuation."""
import importlib,shutil,json
from pathlib import Path
import numpy as np
from .common import read,write,digest,audit_controls
from . import ledger,b3_uncertainty

class Adapter:
    def __init__(self,plan):
        self.plan=plan;self.module=importlib.import_module('ghostscale.validation.soundingline.v29.'+plan['module'])
    def settings(self):return self.plan['roster']
    def pilot_exclusion(self,s,z):
        return self.module.pilot_exclusion(s,z) if hasattr(self.module,'pilot_exclusion') else None
    def run(self,destination,**kw):
        result=self.module.run(destination,**kw)
        if self.plan['module']=='b3':
            out=Path(destination);source=Path(kw['original']) if kw.get('mode')=='reconstruct' else out
            repair=out.parent/(out.name+'-refit-intervals')
            b3_uncertainty.run(source,repair,reference=kw.get('mode')=='reconstruct',summary_override=out/'SUMMARY.json',cpu_limit=self.plan['cpu_cap_seconds'])
            shutil.copyfile(repair/'CORRECTED_SUMMARY.json',out/'SUMMARY.json');result=read(out/'SUMMARY.json')
        return result
    def controls(self,destination,pulse=lambda **kw:None):
        out=Path(destination);out.mkdir(parents=True,exist_ok=True);checks=[];seen=set()
        for base in self.settings():
            key=json.dumps({k:base.get(k) for k in ('part','truth','slope','tremble','hub_weights','order','mu','population','reminder_step','start','tail','loss')},sort_keys=True)
            if key in seen:continue
            seen.add(key);s=base|dict(worlds=8,blocks=8)
            if 'length' in s:s['length']=12
            else:s['steps']=64
            if 'life' in s:s['life']=max(s.get('start',0)+s['steps'],64)
            if base.get('extinction_start') is not None:s.update(steps=640)
            family=self.plan['module']
            raw=ledger.generate(s,'variant-control') if family in ('x1','x2') else self.module.generate(s,'variant-control')
            a=self.module.infer(raw,s)
            if family=='q2':
                from .q2_reference import infer
                b=infer(raw,s)
            else:b=self.module.infer(raw,s,True)
            passed=all(np.allclose(a[k],b[k],rtol=0,atol=1e-8,equal_nan=True) for k in a)
            checks.append(dict(name=key+' independent complete excluded path',passed=passed));pulse(control=len(checks))
        if self.plan['module']=='b3':checks.extend(b3_uncertainty.controls()['checks'])
        if self.plan['module']=='q2' and any('F' in s['order'] for s in self.settings()):
            s=self.settings()[0]|dict(length=12,worlds=8,blocks=8);raw=self.module.generate(s,'failure-control')
            checks.append(dict(name='watched failure contains no successes or owned actions',passed=bool(np.all(raw['successes'][:,:12]==0)) and bool(np.all(raw['actions'][:,:12]==0)) and bool(np.all(raw['watching'][:12]==1))))
        if self.plan['module']=='b2' and any(s.get('paired_reminder') for s in self.settings()):
            s=self.settings()[0]|dict(worlds=8,blocks=8,reminder_step=None);a=self.module.generate(s,'paired-control');b=self.module.generate(s|dict(reminder_step=512),'paired-control')
            checks.extend([dict(name='paired reminder arms share cue paths and pre-reminder actions',passed=bool(np.array_equal(a['cues'],b['cues'])) and bool(np.array_equal(a['alarms'][:,:513],b['alarms'][:,:513]))),dict(name='extinction and single forced harm reminder',passed=bool(np.all(a['harm'][:,256:]==0)) and bool(np.all(b['harm'][:,512]==1)) and bool(np.all(b['harm'][:,513:]==0)))])
        result=audit_controls(checks);write(out/'CONTROLS.json',result);return result
