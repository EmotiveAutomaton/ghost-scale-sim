"""Freeze committed certified-fit-1 engine bytes without reading dirty sibling files."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile
from ghostscale.validation.soundingline.v23.common import digest,write
from ghostscale.validation.soundingline.v23.profile_successor import (
    TAG,TAG_OBJECT,ENGINE_COMMIT,CERTIFICATION_SHA256)


def git(repo,*args):
    return subprocess.run(['git','-c',f'safe.directory={repo}','-C',str(repo),*args],
        check=True,capture_output=True,text=True).stdout.strip()


def prepare(campaign,repo,evidence):
    campaign=campaign.resolve();repo=repo.resolve();evidence=evidence.resolve()
    if git(repo,'rev-parse',f'{TAG}^{{tag}}')!=TAG_OBJECT:raise ValueError('tag object differs')
    if git(repo,'rev-parse',f'{TAG}^{{commit}}')!=ENGINE_COMMIT:raise ValueError('tag target differs')
    verification=evidence/'VERIFICATION.json';publication=evidence/'PUBLICATION.json'
    if digest(verification)!=CERTIFICATION_SHA256:raise ValueError('certification receipt differs')
    verified=json.loads(verification.read_text(encoding='utf-8'))
    published=json.loads(publication.read_text(encoding='utf-8'))
    if verified['state']!='PASS' or verified['counts']!={'tests':27,'errors':0,'failures':0,'skipped':0}:
        raise ValueError('certification tests did not pass exactly')
    if not published['remote_tag_and_peeled_commit_verified'] or published['target_commit']!=ENGINE_COMMIT:
        raise ValueError('published tag is not verified')
    target=campaign/'canonical'/(ENGINE_COMMIT+'-'+TAG)
    if target.exists():
        pin=json.loads((target/'ENGINE_PIN.json').read_text(encoding='utf-8'))
        for name,value in pin['files'].items():
            if digest(target/'engine-snapshot'/name)!=value:raise ValueError('existing certified snapshot changed')
        return target
    target.mkdir(parents=True,exist_ok=False);snapshot=target/'engine-snapshot';snapshot.mkdir()
    with tempfile.TemporaryDirectory(dir=target) as temporary:
        archive=Path(temporary)/'engine.zip'
        subprocess.run(['git','-c',f'safe.directory={repo}','-C',str(repo),'archive',
            '--format=zip',f'--output={archive}',ENGINE_COMMIT,'engine/tompathy_engine','engine/pyproject.toml'],check=True)
        with zipfile.ZipFile(archive) as source:source.extractall(snapshot)
    files={p.relative_to(snapshot).as_posix():digest(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}
    write(target/'ENGINE_PIN.json',dict(tag=TAG,tag_object=TAG_OBJECT,commit=ENGINE_COMMIT,
        files=files,verification_sha256=digest(verification),publication_sha256=digest(publication),
        dirty_sibling_bytes_used=False))
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--repository',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args();print(prepare(a.campaign,a.repository,a.evidence))
