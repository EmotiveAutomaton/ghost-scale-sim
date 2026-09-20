"""Verified portable scientific bundles; local operating telemetry stays private."""
import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest

PART_BYTES=40_000_000


def export(root,destination,driver):
    proof=read(root/'INDEPENDENT_REPLAY.json')
    if not proof['passed'] or proof['plan_sha256']!=file_digest(root/'PLAN.json') or proof['summary_sha256']!=file_digest(root/'SUMMARY.json'):raise ValueError('packet lacks matching replay')
    if file_digest(driver)!=proof['verifier_sha256']:raise ValueError('executed replay driver required')
    destination.mkdir(parents=True,exist_ok=False)
    names=['PLAN.json','SOURCE.zip','SUMMARY.json','COMPLETE.json','INDEPENDENT_REPLAY.json']
    if read(root/'PLAN.json')['design']['engine']=='neural':
        names+=['TRAINING.json','neural_points.json.gz']
        names += [p.relative_to(root).as_posix() for p in (root/'data').rglob('*') if p.is_file() and p.suffix in ('.npz','.gz','.json','.pt')]
        names += [p.relative_to(root).as_posix() for p in (root/'neural').rglob('*') if p.is_file() and
            (p.name in ('BEST.pt','READOUT.npz','COMPLETE.json','IDENTITY.json','CONTROLS.json') or p.name.endswith('-PREDICTIONS.npz'))]
    else:
        names += [p.relative_to(root).as_posix() for part in ('raw','blocks') for p in (root/part).glob('*') if p.is_file()]
    names=sorted(set(names));manifest=dict(files={n:file_digest(root/n) for n in names},
        retained='all raw scientific cases, forecasts, selected weights, fit curves, plans and source',
        excluded='local process telemetry, logs, ownership and resumable intermediate optimizer states')
    write(destination/'SCIENTIFIC_MANIFEST.json',manifest)
    for name in ('PLAN.json','SOURCE.zip','SUMMARY.json','INDEPENDENT_REPLAY.json'):shutil.copyfile(root/name,destination/name)
    shutil.copyfile(driver,destination/'REPLAY_DRIVER.py')
    archive=destination/'SCIENTIFIC.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_STORED) as z:
        for name in names:z.write(root/name,name)
        z.write(destination/'SCIENTIFIC_MANIFEST.json','SCIENTIFIC_MANIFEST.json')
    archive_sha=file_digest(archive);parts=[]
    with archive.open('rb') as stream:
        while data:=stream.read(PART_BYTES):
            path=destination/f'SCIENTIFIC.zip.part{len(parts):03d}';path.write_bytes(data)
            parts.append(dict(file=path.name,sha256=file_digest(path),bytes=len(data)))
    if not archive.resolve().is_relative_to(destination.resolve()):raise ValueError('archive path escapes')
    archive.unlink()
    write(destination/'BUNDLE.json',dict(parts=parts,archive_sha256=archive_sha,format='concatenate parts in order to recover the ordinary ZIP'))
    write(destination/'EXPORT.json',dict(files={p.name:file_digest(p) for p in destination.iterdir() if p.is_file()},scientific_files=len(names)))


def unpack(packet,output):
    manifest=read(packet/'BUNDLE.json');output.mkdir(parents=True,exist_ok=False);archive=output/'bundle.zip';digest=hashlib.sha256()
    with archive.open('xb') as stream:
        for part in manifest['parts']:
            p=packet/part['file']
            if file_digest(p)!=part['sha256']:raise ValueError('part corruption')
            data=p.read_bytes();digest.update(data);stream.write(data)
    if digest.hexdigest()!=manifest['archive_sha256']:raise ValueError('assembled hash differs')
    with zipfile.ZipFile(archive) as z:
        if any(not (output/name).resolve().is_relative_to(output.resolve()) for name in z.namelist()):raise ValueError('archive path escapes')
        z.extractall(output)
    for name,sha in read(output/'SCIENTIFIC_MANIFEST.json')['files'].items():
        if file_digest(output/name)!=sha:raise ValueError('scientific artifact differs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['export','unpack']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--driver',type=Path);a=p.parse_args()
    if a.action=='export':export(a.root,a.output,a.driver)
    else:unpack(a.root,a.output)
