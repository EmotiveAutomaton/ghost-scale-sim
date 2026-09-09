import pytest
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.gates import fixture_observation
from ghostscale.validation.soundingline.v16.inference import read_public
from ghostscale.validation.soundingline.v16.records import canonical


def test_separate_reader_matches_known_prediction_and_cannot_open_private_record(tmp_path):
    private=tmp_path/"private-truth.json"
    private.write_text('{"secret_fixture_answer":7}',encoding="utf-8")
    public=fixture_observation()
    with ReaderProcess(tmp_path/"public-workspace") as reader:
        actual=reader.request("native",public)
        assert actual==read_public(canonical(public))
        with pytest.raises(RuntimeError,match="PermissionError"):
            reader.request("_probe_forbidden_read",{"path":str(private)})
        private.write_text('{"secret_fixture_answer":999}',encoding="utf-8")
        public["task_id"]="unrelated-public-identifier"
        assert reader.request("native",public)==actual
