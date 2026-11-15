from datetime import datetime,timezone,timedelta
import pytest
from runners.run_v30 import admissible,equivalent
from ghostscale.validation.soundingline.v30.common import write
import numpy as np

def test_reservations_and_deadline_are_real_boundaries():
    now=datetime.now(timezone.utc)
    a=dict(cpu_limits={"P":100,"reserve":20},total_cpu_seconds=110,
           new_starts_close=(now+timedelta(hours=2)).isoformat(),
           drain=(now+timedelta(hours=3)).isoformat())
    p=dict(pool="P",cpu_cap_seconds=40,wall_cap_seconds=60)
    assert admissible(a,{"P":60,"reserve":0},p,now)
    assert not admissible(a,{"P":61,"reserve":0},p,now)
    assert not admissible(a,{"P":60,"reserve":11},p,now)
    assert not admissible(a,{"P":0,"reserve":0},p,now+timedelta(hours=2))

def test_replay_rejects_missing_or_changed_raw(tmp_path):
    a=tmp_path/"a";b=tmp_path/"b";a.mkdir();b.mkdir()
    (a/"raw").write_bytes(b"123");(b/"raw").write_bytes(b"123")
    equivalent(a,b,True)
    (b/"raw").write_bytes(b"124")
    with pytest.raises(ArithmeticError):equivalent(a,b,True)

def test_reconstruction_checks_world_metrics_not_just_means(tmp_path):
    a=tmp_path/"a";b=tmp_path/"b";a.mkdir();b.mkdir()
    write(a/"SUMMARY.json",dict(mean=2));write(b/"SUMMARY.json",dict(mean=2))
    np.savez(a/"cell_measures.npz",scores=np.array([1.,3.]))
    np.savez(b/"cell_measures.npz",scores=np.array([2.,2.]))
    with pytest.raises(ArithmeticError):equivalent(a,b)
