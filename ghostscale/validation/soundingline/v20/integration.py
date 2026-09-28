"""Read-only canonical Sounding Line integration and privacy-bounded intake."""
from pathlib import Path
import importlib.util,sys,types,importlib,json,hashlib
import numpy as np
from . import transport as T,world as W
from ..v18_3.io import write,file_digest

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def consume(soundingline,out):
    soundingline=Path(soundingline);out=Path(out)
    module=load('v20_canonical_process',soundingline/'soundingline/process_record.py')
    pkg=types.ModuleType('v20_sl_stage13');pkg.__path__=[str(soundingline/'runners/stage13')];sys.modules[pkg.__name__]=pkg
    exporter=importlib.import_module('v20_sl_stage13.export_tompathy');common=importlib.import_module('v20_sl_stage13.common')
    rows=W.enumerate_world(0);receipts=[]
    for tier in W.TIERS:
        p=T.packet(rows[13579],tier,(0,13579),True)
        case=module.ProcessCase(case_id=p['case_id'],lineage_id=p['source_lineage_id'],domain='constructed',medium='binary-linked-artifact',brief_id='synthetic',declared_context=p['evidence'],participants={'human':'constructed','tool':'constructed'},route_family=p['workflow_family'],events=[module.ProcessEvent(**e) for e in p['observed_events']],artifact_final=p['artifact'])
        case.validate();artifact,reading,outer=T.offline_input(p,np.full(128,1/128))
        context=[]
        if tier!='artifact':
            excerpt=json.dumps(p['evidence'],sort_keys=True,separators=(',',':'));source='Ghost V20 constructed observed context'
            context=[dict(id=common.digest([source,excerpt]),source=source,excerpt=excerpt,selected=True)]
        packet,sidecar=exporter.export(artifact,reading,context=context,outer=outer)
        assert packet['scope']['kind']==('artifact' if tier=='artifact' else 'context')
        assert any(h['status']=='unknown' and h['kind']=='value' for h in packet['hypotheses'])
        write(out/tier/'observed.json',p);write(out/tier/'packet.json',packet);write(out/tier/'sidecar.json',sidecar)
        receipts.append(dict(tier=tier,canonical_process_case=True,actual_offline_export=True,visibility=reading['region-0']['visibility']))
    source=soundingline/'results/phase_2_4_stage_13/raw/sources/coauthor-v3-train.json'
    example=json.loads(source.read_bytes())[0]
    # Preserve only schema/topology and explicit unknown states in the public receipt.
    fields={k:sorted(v) for k,v in example['views'].items()}
    receipt=dict(passed=True,consumer='Sounding Line canonical ProcessCase validator and Stage 13 offline exporter',
        parser_scope='Exporter executed; ToMpathy browser import was not exercised',tiers=receipts,
        sources={n:file_digest(soundingline/n) for n in ['soundingline/process_record.py','runners/stage13/export_tompathy.py','runners/stage13/common.py','runners/stage13/scoring.py']},
        known_source=dict(sha256=file_digest(source),example_sha256=hashlib.sha256(json.dumps(example,sort_keys=True).encode()).hexdigest(),top_level_fields=sorted(example),view_fields=fields,private_text_exported=False,
        mapping='Witnessed proposal, selection, revision and exact-span handling map to separate event roles. Endorsement, understanding and review remain unknown unless witnessed. No author-share scalar.'),
        downstream_role='Synthetic calibration fixtures for retrospective contribution and selected-context offline export')
    write(out/'CONSUMER.json',receipt);return receipt
