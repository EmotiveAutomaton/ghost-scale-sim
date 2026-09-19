"""Complete scientific neural inputs/outputs/weights, excluding local operations."""
import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest

PART_BYTES=40_000_000


def selected(root):
    files=[root/name for name in ('PLAN.json','SOURCE.zip','SUMMARY.json','TRAINING.json','neural_points.json.gz','neural/COMPLETE.json','neural/IDENTITY.json')]
    files.extend(p for p in (root/'data').rglob('*') if p.is_file() and p.suffix in ('.npz','.gz','.json','.pt'))
    files.extend(p for p in (root/'neural').rglob('*') if p.is_file() and
        (p.name in ('BEST.pt','READOUT.npz','FIT.json','COMPLETE.json','IDENTITY.json','CONTROLS.json') or p.name.endswith('-PREDICTIONS.npz')))
    return sorted(set(files))


def export(root,destination):
    proof=read(root/'INDEPENDENT_REPLAY.json')
    if not proof['passed'] or proof['plan_sha256']!=file_digest(root/'PLAN.json') or proof['summary_sha256']!=file_digest(root/'SUMMARY.json'):
        raise ValueError('unverified neural packet')
    driver=Path(__file__).with_name('replay_v18_3_neural.py')
    if file_digest(driver)!=proof['verifier_sha256']:raise ValueError('neural replay driver changed')
    destination.mkdir(parents=True,exist_ok=False);files=selected(root)
    scientific=dict(files={p.relative_to(root).as_posix():file_digest(p) for p in files},
        original_complete_sha256=file_digest(root/'COMPLETE.json'),
        retained='all scientific input capsules, evaluator truth, case records, predictions, selected weights, fit curves and source',
        excluded='machine-local status/ownership/CPU process receipts and intermediate optimizer snapshots; retained locally')
    write(destination/'SCIENTIFIC_MANIFEST.json',scientific)
    for name in ('PLAN.json','SOURCE.zip','SUMMARY.json','INDEPENDENT_REPLAY.json'):
        shutil.copyfile(root/name,destination/name)
    shutil.copyfile(root/'COMPLETE.json',destination/'ORIGINAL_COMPLETE.json')
    shutil.copyfile(driver,destination/'REPLAY_DRIVER.py')
    archive=destination/'SCIENTIFIC.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_STORED) as z:
        for path in files:z.write(path,path.relative_to(root).as_posix())
    archive_hash=file_digest(archive);parts=[]
    with archive.open('rb') as stream:
        while chunk:=stream.read(PART_BYTES):
            path=destination/f'SCIENTIFIC.zip.part{len(parts):03d}';path.write_bytes(chunk)
            parts.append(dict(file=path.name,sha256=file_digest(path),bytes=len(chunk)))
    if not archive.resolve().is_relative_to(destination.resolve()):raise ValueError('unexpected archive target')
    archive.unlink()
    write(destination/'BUNDLE.json',dict(parts=parts,archive_sha256=archive_hash,format='concatenate binary parts in listed order to obtain one ordinary ZIP archive'))
    write(destination/'EXPORT.json',dict(files={p.name:file_digest(p) for p in destination.iterdir() if p.is_file()},
        scope='portable scientific bundle; operating telemetry and intermediate optimizer checkpoints remain private'))


def unpack(packet,output):
    manifest=read(packet/'BUNDLE.json');output.mkdir(parents=True,exist_ok=False);archive=output/'bundle.zip';digest=hashlib.sha256()
    with archive.open('xb') as stream:
        for part in manifest['parts']:
            path=packet/part['file']
            if file_digest(path)!=part['sha256']:raise ValueError('bundle part corruption')
            data=path.read_bytes();digest.update(data);stream.write(data)
    if digest.hexdigest()!=manifest['archive_sha256']:raise ValueError('bundle assembly corruption')
    with zipfile.ZipFile(archive) as z:
        if any(not (output/name).resolve().is_relative_to(output.resolve()) for name in z.namelist()):raise ValueError('archive escape')
        z.extractall(output)
    shutil.copyfile(packet/'SCIENTIFIC_MANIFEST.json',output/'SCIENTIFIC_MANIFEST.json')
    for name,value in read(output/'SCIENTIFIC_MANIFEST.json')['files'].items():
        if file_digest(output/name)!=value:raise ValueError('extracted scientific artifact differs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['export','unpack']);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();(export if a.action=='export' else unpack)(a.root,a.output)
