"""Complete original calibration replay under equivalent arithmetic only."""
from pathlib import Path
from . import adaptive_numerics, calibration
from .common import digest, read, write


def run(root, plan, pulse):
    reference = Path(plan['calibration_reference'])
    if digest(reference/'COMPLETE.json') != plan['calibration_reference_complete_sha256']:
        raise ValueError('calibration reference binding changed')
    from .family import verify_packet
    _, summary = verify_packet(reference)
    if summary['state'] != 'CALIBRATION_PASSED':
        raise ValueError('original calibration unresolved')
    adaptive_numerics.install()
    calibration.run(Path(root), pulse, cpu_cap_seconds=plan['cpu_cap_seconds']/2)
    expected = read(reference/'REPLAY.json')['files']
    actual = {p.relative_to(root).as_posix(): digest(p) for p in Path(root).rglob('*')
              if p.is_file() and not p.name.endswith('-cost.json')}
    if actual != expected:
        raise ValueError('equivalent arithmetic changed calibration decisions: '+str(
            [name for name in set(actual)|set(expected) if actual.get(name) != expected.get(name)]))
    write(Path(root)/'EQUIVALENCE.json', dict(state='CALIBRATION_EQUIVALENT',
        original_complete_sha256=plan['calibration_reference_complete_sha256'],
        scientific_files=expected, full_tuning_and_validation_reexecuted=True,
        thresholds_priors_nodes_tolerance_unchanged=True,
        limitation='Same complete calibration outcomes; no universal claim outside its validated distribution.'))
