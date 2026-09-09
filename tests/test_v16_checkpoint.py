from pathlib import Path
import shutil
import pytest
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.acquisition_checkpoint import admission,execute


def test_scientific_admission_survives_an_extended_unlocked_entrypoint():
    # The saved scientific inputs are unchanged even though new packet routes were
    # added after this admission receipt. This must neither regenerate nor refuse them.
    assert admission(REPO/"results/v16")["X08"]["instrument_state"]=="valid"


def test_completed_acquisition_is_read_without_repreparing_or_new_units():
    root=REPO/"results/v16"
    paths=[root/"CAMPAIGN.json",root/"packets/k01-scout-1.json",root/"k01-scout-1/SUMMARY.json"]
    before={str(path):path.read_bytes() for path in paths}
    result=execute(root,lambda **updates:None,resume=True)
    assert result["already_completed"] and result["n_maker_packets"]==192
    assert {str(path):path.read_bytes() for path in paths}==before


def test_missing_scientific_control_is_never_inferred_passed(tmp_path):
    with pytest.raises(FileNotFoundError):
        admission(tmp_path)
