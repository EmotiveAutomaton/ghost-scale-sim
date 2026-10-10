"""Immutable source capsules and validated packet admission (no scientific dispatch)."""
import zipfile
from pathlib import Path
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest,now
from ghostscale.validation.soundingline.v21.runtime import REPO
from ghostscale.validation.soundingline.v20.runtime import fingerprint

def source_files():
    files=list((REPO/'ghostscale').rglob('*.py'))+list((REPO/'runners').glob('*.py'))+list((REPO/'tests').glob('test_v21*.py'))
    return {p.relative_to(REPO).as_posix():file_digest(p) for p in files}

def freeze(campaign,name,validation):
    if not validation['passed']:raise ValueError('controls required')
    mapping=source_files()
    if validation.get('sources')!=mapping:raise ValueError('controls must bind exact current source')
    capsule=campaign/'capsules'/name;source=capsule/'source';source.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(capsule/'SOURCE.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(mapping):
            out=source/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((REPO/name).read_bytes());z.write(out,name)
    record=dict(sources=mapping,archive_sha256=file_digest(capsule/'SOURCE.zip'),environment=fingerprint(),validation=validation)
    write(capsule/'ADMISSION.json',record);return capsule

def admit(campaign,capsule,identity,design,**queue_fields):
    from ghostscale.validation.soundingline.v21.checker import validate_design
    validate_design(design)
    root=campaign/'packets'/identity;root.mkdir(parents=True,exist_ok=False);admission=read(capsule/'ADMISSION.json')
    (root/'SOURCE.zip').write_bytes((capsule/'SOURCE.zip').read_bytes())
    ph=write(root/'PLAN.json',dict(design=design,acceptance_sha256=file_digest(campaign/'ACCEPTANCE.json'),
        sources=admission['sources'],archive_sha256=admission['archive_sha256'],environment=admission['environment']))
    queue=read(campaign/'QUEUE.json');job=dict(id=identity,root=str(root),source=str(capsule/'source'),plan_sha256=ph,
        not_before=now(),cpu_cap_seconds=design['cpu_cap_seconds'],wall_cap_seconds=design['cpu_cap_seconds']*2+300,
        requires=[],review_boundary=False);job.update(queue_fields);queue['jobs'].append(job);queue['revision']+=1
    write(campaign/'QUEUE.json',queue,immutable=False);return root
