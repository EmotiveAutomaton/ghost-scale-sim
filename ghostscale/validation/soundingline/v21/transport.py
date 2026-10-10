"""Blind constructed fixtures consumed by the canonical Sounding Line exporter."""
import importlib,types,sys,json
from pathlib import Path
import numpy as np
from ..v20 import transport as T
from ..v20.integration import load
from ..v18_3.io import write,file_digest
from . import world as W,oracle as O
from .studies import controls

def run(root,d,pulse):
    sl=Path(d['soundingline']);module=load('v21_canonical_process',sl/'soundingline/process_record.py')
    pkg=types.ModuleType('v21_sl_stage13');pkg.__path__=[str(sl/'runners/stage13')];sys.modules[pkg.__name__]=pkg
    exporter=importlib.import_module('v21_sl_stage13.export_tompathy');common=importlib.import_module('v21_sl_stage13.common')
    cases=[];rows=W.enumerate_world(71000400);meanlaw=dict(route=4,select=4,revise=4,goal=4,inspect=4,skill=6.5,shift='native')
    for i in range(24):
        tier=W.TIERS[i%4];row=rows[(137*i+19)%len(rows)];identity=('v21-fixture',i)
        packet=T.packet(row,tier,identity,True)
        case=module.ProcessCase(case_id=packet['case_id'],lineage_id=packet['source_lineage_id'],domain='constructed',
            medium='binary-linked-artifact',brief_id='synthetic',declared_context=packet['evidence'],
            participants={'human':'constructed','tool':'constructed'},route_family=packet['workflow_family'],
            events=[module.ProcessEvent(**e) for e in packet['observed_events']],artifact_final=packet['artifact'])
        case.validate();T.validate(packet)
        x=W.project(np.asarray([row]),tier,True)
        lp,_=O.posterior(x,tier,[meanlaw],observed=True)
        supplied,_=O.posterior(x,tier,[W.law(71000400)],observed=True)
        artifact,reading,outer=T.offline_input(packet,np.full(128,1/128))
        context=[]
        if tier!='artifact':
            excerpt=json.dumps(packet['evidence'],sort_keys=True,separators=(',',':'));source='Ghost V21 constructed observed context'
            context=[dict(id=common.digest([source,excerpt]),source=source,excerpt=excerpt,selected=True)]
        exported,sidecar=exporter.export(artifact,reading,context=context,outer=outer)
        base=root/'reader'/packet['case_id'];write(base/'observed.json',packet);write(base/'offline-packet.json',exported);write(base/'sidecar.json',sidecar)
        write(base/'candidates.json',dict(target='seven-bit constructed process and aims',labels=list(range(128))))
        write(root/'evaluator'/(packet['case_id']+'.json'),dict(truth=int(row[10]),law_marginal=np.exp(lp[0]).tolist(),
            supplied_law_diagnostic=np.exp(supplied[0]).tolist(),brier_normalization='full multiclass sum; half-Brier = full / 2',
            scope='H=0 exact generator-law prior; evaluator only; browser/live inference not tested'))
        cases.append(dict(case_id=packet['case_id'],tier=tier,canonical_process_case=True,offline_export=True))
        pulse(phase='canonical-export',case=i)
    sources={name:file_digest(sl/name) for name in ('soundingline/process_record.py','runners/stage13/export_tompathy.py','runners/stage13/common.py','runners/stage13/scoring.py')}
    write(root/'CONSUMER.json',dict(passed=True,sources=sources,cases=cases,browser_tested=False,
        coverage='opening evidence-tier fixtures; copying/correction/omission intervention fixtures follow their admitted family'))
    return dict(card='G21-F03',cases=cases,controls=controls(),scope='24 canonical offline fixtures; opening subset of F03; no real source text or evaluator truth in reader packet')
