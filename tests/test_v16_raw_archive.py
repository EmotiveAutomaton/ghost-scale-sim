import zipfile
import pytest
from ghostscale.validation.soundingline.v16.raw_archive import manifest,create,verify


def test_raw_archive_is_accessible_complete_and_rejects_missing_or_extra_members(tmp_path):
    repo=tmp_path/"repo"
    repo.mkdir()
    source=repo/"unit_points.json"
    source.write_bytes(b'{"known":true}\n')
    plan=manifest(repo,[source],scope="known archive fixture")
    receipt=create(repo,tmp_path/"archive",plan)
    assert receipt["all_expected_member_bytes_verified"] and receipt["verified_members"]==1
    assert not receipt["complete_campaign_archive"]
    assert create(repo,tmp_path/"archive",plan)==receipt
    with zipfile.ZipFile(tmp_path/"archive/raw.zip","a") as archive:
        archive.writestr("unregistered.json",b"{}")
    with pytest.raises(ValueError,match="added members"):
        verify(tmp_path/"archive/raw.zip",plan)


def test_raw_archive_preserves_failed_attempt_and_rejects_changed_sources(tmp_path):
    repo=tmp_path/"repo"
    repo.mkdir()
    source=repo/"unit_points.json"
    source.write_bytes(b"original")
    plan=manifest(repo,[source],scope="known mutation fixture")
    source.write_bytes(b"changed")
    with pytest.raises(ValueError,match="source changed"):
        create(repo,tmp_path/"archive",plan)
    assert (tmp_path/"archive/FAILURE.json").exists()
    assert (tmp_path/"archive/raw.zip").exists()
    with pytest.raises(ValueError,match="incomplete raw archive attempt"):
        create(repo,tmp_path/"archive",plan)
    with pytest.raises(ValueError,match="duplicate"):
        manifest(repo,[source,source],scope="duplicate")
