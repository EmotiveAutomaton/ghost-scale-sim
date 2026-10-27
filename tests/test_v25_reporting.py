from ghostscale.validation.soundingline.v23.common import write,digest
from ghostscale.validation.soundingline.v25.reporting import verified_primary

def test_repaired_probe_cannot_displace_completed_primary(tmp_path):
    packets=tmp_path/'packets';packets.mkdir()
    for suffix,kind,scientific in [('study-1','study','study'),('healing-1','healing','probe')]:
        p=packets/('v25-w4-'+suffix);p.mkdir()
        write(p/'PLAN.json',dict(kind=kind,scientific_kind=scientific))
        write(p/'COMPLETE.json',dict(plan_sha256=digest(p/'PLAN.json')))
        write(p/'STATUS.json',dict(state='COMPLETE'))
    assert verified_primary(tmp_path,'W4').name=='v25-w4-study-1'

def test_completed_study_repair_supersedes_original(tmp_path):
    packets=tmp_path/'packets';packets.mkdir()
    for suffix in ('study-1','healing-2'):
        p=packets/('v25-w6-'+suffix);p.mkdir()
        write(p/'PLAN.json',dict(kind='study' if suffix=='study-1' else 'healing',scientific_kind='study'))
        write(p/'COMPLETE.json',dict(plan_sha256=digest(p/'PLAN.json')))
        write(p/'STATUS.json',dict(state='COMPLETE'))
    assert verified_primary(tmp_path,'W6').name=='v25-w6-healing-2'


def test_calibrated_study_keeps_native_primary_separate(tmp_path):
    for suffix in ('study-1','calibrated-study-1'):
        p=tmp_path/'packets'/('v25-w5-'+suffix);p.mkdir(parents=True)
        write(p/'PLAN.json',dict(kind='study',calibration_variant='wb1-calibrated' if suffix.startswith('calibrated') else None))
        write(p/'COMPLETE.json',dict(plan_sha256=digest(p/'PLAN.json')))
        write(p/'STATUS.json',dict(state='COMPLETE'))
    assert verified_primary(tmp_path,'W5').name=='v25-w5-study-1'
