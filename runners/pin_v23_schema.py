"""Snapshot canonical ledger blobs from a named ToMpathy commit, without edits."""
import argparse
from pathlib import Path
import subprocess
from ghostscale.validation.soundingline.v23.common import write, digest
from ghostscale.validation.soundingline.v23.ledger import canonical_schema


def pin(campaign, repository, commit):
    campaign, repository = Path(campaign).resolve(), Path(repository).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repository), *args])
    resolved = git('rev-parse', commit+'^{commit}').decode().strip()
    if len(resolved) != 40: raise ValueError('commit must resolve exactly')
    target = campaign/'canonical'/resolved/'ledger'
    target.mkdir(parents=True, exist_ok=False)
    blobs = {}
    for name in ('schema.py', 'ids.py'):
        relative = 'engine/tompathy_engine/ledger/'+name
        raw = git('show', resolved+':'+relative)
        with (target/name).open('xb') as f: f.write(raw)
        blobs[name] = git('rev-parse', resolved+':'+relative).decode().strip()
    schema, hashes = canonical_schema(target)
    record = dict(commit=resolved, directory=str(target), files=hashes,
        schema_version=schema.SCHEMA_VERSION, schema_hash=schema.schema_hash(),
        committed_files_verified=True, git_blobs=blobs,
        ownership='canonical ToMpathy classes; read-only committed dependency snapshot')
    write(campaign/'SCHEMA_PIN.json', record)
    return record


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--repository',required=True)
    p.add_argument('--commit',required=True);a=p.parse_args();print(pin(a.campaign,a.repository,a.commit))
