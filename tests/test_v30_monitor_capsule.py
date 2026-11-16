"""Monitor capsule includes its late-imported shutdown dependency exactly once."""
import zipfile
from runners import run_v27
from ghostscale.validation.soundingline.v30.common import read,digest

def test_monitor_capsule_carries_review_shutdown(tmp_path,monkeypatch):
    repo=tmp_path/'repo';(repo/'ghostscale').mkdir(parents=True);(repo/'runners').mkdir()
    (repo/'ghostscale/__init__.py').write_text('')
    for name in ('run_v27','watch_v23_events','review_shutdown','launch_background'):
        (repo/f'runners/{name}.py').write_text('# capsule fixture\n')
    monkeypatch.setattr(run_v27,'REPO',repo)
    source=run_v27.frozen_source(tmp_path/'campaign',['runners/review_shutdown.py'])
    manifest=read(source/'SOURCE_MANIFEST.json')
    assert manifest['runners/review_shutdown.py']==digest(source/'runners/review_shutdown.py')
    with zipfile.ZipFile(source/'SOURCE.zip') as archive:
        assert archive.namelist().count('runners/review_shutdown.py')==1
