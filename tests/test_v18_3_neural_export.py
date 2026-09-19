from pathlib import Path
import pytest
from runners import export_v18_3_neural as E
from ghostscale.validation.soundingline.v18_3.io import write,read,file_digest


def test_scientific_bundle_roundtrip_and_private_exclusions(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();public=tmp_path/'public';out=tmp_path/'unpacked'
    names=['PLAN.json','SUMMARY.json','TRAINING.json','COMPLETE.json','neural/COMPLETE.json','neural/IDENTITY.json','data/reader/INPUTS.json','neural/m/FIT.json']
    for name in names:write(root/name,dict(fixture=True))
    for name in ('SOURCE.zip','neural_points.json.gz','data/reader/encoder.pt','data/test.npz','neural/m/READOUT.npz','neural/m/BEST.pt','neural/m-PREDICTIONS.npz'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'known scientific bytes')
    write(root/'neural/STATUS.json',dict(private_operator=True))
    (root/'neural/m/step-0001.pt').write_bytes(b'intermediate optimizer')
    driver=Path(E.__file__).with_name('replay_v18_3_neural.py')
    write(root/'INDEPENDENT_REPLAY.json',dict(passed=True,plan_sha256=file_digest(root/'PLAN.json'),summary_sha256=file_digest(root/'SUMMARY.json'),verifier_sha256=file_digest(driver)))
    monkeypatch.setattr(E,'PART_BYTES',128)
    E.export(root,public);E.unpack(public,out)
    assert not (out/'neural/STATUS.json').exists() and not (out/'neural/m/step-0001.pt').exists()
    assert (out/'data/reader/encoder.pt').read_bytes()==b'known scientific bytes'
    assert (out/'neural/m/READOUT.npz').read_bytes()==b'known scientific bytes'
    part=public/read(public/'BUNDLE.json')['parts'][0]['file'];part.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corruption'):E.unpack(public,tmp_path/'corrupt-output')
