import pytest
from ghostscale.validation.soundingline.v19 import runtime as R
from ghostscale.validation.soundingline.v18_3.io import write


def test_source_closure_includes_portable_replay():
    assert 'runners/replay_v19.py' in R.source_files()
    assert all((R.REPO/p).is_file() for p in R.source_files())
    assert 'scipy' in R.fingerprint()


def test_native_and_process_cost_are_not_double_counted(tmp_path):
    write(tmp_path/'attempts/a.json',dict(state='complete',cpu_seconds=5,native_cpu_seconds=7,child_cpu_seconds=3))
    write(tmp_path/'attempts/b.json',dict(state='failed',cpu_seconds=2,uncertainty_cpu_seconds=4))
    assert R.consumed(tmp_path)==14


def test_unreconciled_attempt_blocks_second_worker(tmp_path):
    write(tmp_path/'attempts/a.json',dict(state='running',cpu_seconds=2))
    with pytest.raises(ValueError,match='unreconciled'):R.consumed(tmp_path)


def test_card_cap_includes_replay_and_setup(tmp_path):
    write(tmp_path/'attempts/a.json',dict(packet='science',accounting_card='G19-04',state='complete',cpu_seconds=5,native_cpu_seconds=7))
    write(tmp_path/'attempts/b.json',dict(packet='replay',accounting_card='G19-04',state='failed',cpu_seconds=3))
    write(tmp_path/'attempts/c.json',dict(packet='other',state='complete',cpu_seconds=99))
    assert R.card_consumed(tmp_path,dict(accounting_card='G19-04',setup_charge_seconds=2),'science')==12
