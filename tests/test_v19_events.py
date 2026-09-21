from datetime import datetime,timezone
from runners.watch_v18_4_events import checkpoint_events
from ghostscale.validation.soundingline.v18_3.io import read,file_digest


def test_new_campaign_checkpoints_are_due_once(tmp_path):
    acceptance=dict(interim_at='2030-01-05T00:00:00+00:00',deadline='2030-01-08T00:00:00+00:00')
    checkpoint_events(tmp_path,acceptance,{},datetime(2030,1,4,tzinfo=timezone.utc))
    assert not (tmp_path/'events').exists()
    checkpoint_events(tmp_path,acceptance,{},datetime(2030,1,5,tzinfo=timezone.utc))
    path=tmp_path/'events/interim-report-due.json';sha=file_digest(path)
    checkpoint_events(tmp_path,acceptance,{},datetime(2030,1,9,tzinfo=timezone.utc))
    assert file_digest(path)==sha
    assert read(tmp_path/'events/final-report-due.json')['kind']=='final_report_due'


def test_legacy_checkpoint_retained(tmp_path):
    checkpoint_events(tmp_path,dict(minimum_exploration_until='2030-01-01T00:00:00+00:00'),{},datetime(2030,1,2,tzinfo=timezone.utc))
    assert (tmp_path/'events/minimum-window-reached.json').exists()
