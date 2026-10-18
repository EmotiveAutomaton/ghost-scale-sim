from datetime import datetime
from runners.watch_v24_handoff import eligible

def test_handoff_does_not_open_early_or_after_authorized_window():
    config=dict(due='2026-10-22T00:00:00-07:00',expires='2026-10-25T22:00:00-07:00')
    assert not eligible(datetime.fromisoformat('2026-10-21T23:59:59-07:00'),config)
    assert eligible(datetime.fromisoformat(config['due']),config)
    assert not eligible(datetime.fromisoformat(config['expires']),config)
