"""Retain a failed attempt's scientific artifacts without local operating logs."""
import argparse
from pathlib import Path
import shutil
import zipfile
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest


def failed_E(root,destination):
    if (root/'COMPLETE.json').exists():raise ValueError('attempt is not incomplete')
    destination.mkdir(parents=True,exist_ok=False)
    paths=[root/name for name in ('PLAN.json','SOURCE.zip','TRAINING.json')]
    paths.extend(p for p in (root/'data').rglob('*') if p.is_file() and p.suffix in ('.npz','.gz','.json'))
    paths.extend(p for p in (root/'neural').rglob('*') if p.is_file() and p.name in ('BEST.pt','COMPLETE.json','IDENTITY.json','CONTROLS.json'))
    for pointer in (root/'neural').glob('*/CURRENT.json'):
        receipt=read(pointer);checkpoint=(pointer.parent/receipt['file']).resolve()
        if not checkpoint.is_relative_to(pointer.parent.resolve()) or file_digest(checkpoint)!=receipt['sha256']:raise ValueError('resume checkpoint changed')
        paths.extend((pointer,checkpoint))
    manifest={p.relative_to(root).as_posix():file_digest(p) for p in sorted(set(paths))}
    for name in ('PLAN.json','SOURCE.zip'):shutil.copyfile(root/name,destination/name)
    archive=destination/'RETAINED_SCIENCE.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in manifest:z.write(root/name,name)
    fits=[p.parent.name for p in (root/'neural').glob('*/COMPLETE.json')]
    write(destination/'DISPOSITION.json',dict(state='retained_failed_attempt',failure='Windows atomic status replacement denied while parent reader shared the file',
        superseded_by='E-discovery-r1',completed_fit_names=sorted(fits),scientific_result_complete=False,
        same_data_repair_not_replication=True,files=manifest,archive_sha256=file_digest(archive),
        privacy='machine paths, process status and operating logs retained only locally; scientific checkpoints retained here'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();failed_E(a.root,a.output)
