"""One owner-approved, source-frozen S.b validation and conditional admission."""
import argparse
import itertools
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v31.common import read, write, digest, run_cell
from runners.run_v27 import frozen_source
from runners.run_v31 import equivalent, accounting
from runners.v31_build import PINS

FAMILY = 'S_LEARNING_R2'
PROTOCOL = 'docs/versions/v31-four-settings/protocols/S_LEARNING_R2.md'
AUTHORIZATION = 'results/v31/S_LEARNING_R2_AUTHORIZATION.json'
MODULE = 'ghostscale.validation.soundingline.v31.s_learning_r2'


def validate(c):
    from ghostscale.validation.soundingline.v31 import s_learning_r2 as module
    runtime.below_normal()
    out = c / 'preparation' / FAMILY
    frozen = read(out / 'FREEZE.json')
    source = Path(frozen['source_root'])
    assert Path(module.__file__).resolve().is_relative_to(source)
    assert digest(source / 'SOURCE_MANIFEST.json') == frozen['source_manifest_sha256']
    for name, sha in read(source / 'SOURCE_MANIFEST.json').items():
        assert digest(source / name) == sha, name
    assert digest(c / 'ACCEPTANCE.json') == frozen['acceptance_sha256']
    started = time.process_time()
    opened = time.monotonic()
    last = [0.]

    def pulse(**progress):
        if (time.process_time() > 600 or time.monotonic() - opened > 1800
                or (c / 'STOP').exists()
                or datetime.now(timezone.utc) >= datetime.fromisoformat(read(c / 'ACCEPTANCE.json')['drain'])):
            raise TimeoutError('bounded S.b R2 validation limit')
        if time.monotonic() - last[0] > 5:
            write(out / 'PROGRESS.json', dict(pid=os.getpid(), cpu_seconds=time.process_time(),
                  at=datetime.now(timezone.utc).isoformat(), **progress), immutable=False)
            last[0] = time.monotonic()

    try:
        controls = module.controls(out / 'controls', pulse)
        assert controls['passed'], controls
        retained = []
        checked = {}
        for family in ('S_LEARNING', 'S_LEARNING_R1'):
            original = c / 'preparation' / family / 'original'
            setting = read(original / 'SUMMARY.json')['setting']
            key = (digest(original / 'cell_points.npz'), digest(original / 'SUMMARY.json'))
            if key not in checked:
                dest = out / ('retained-' + family)
                run_cell(module, setting, dest, mode='reconstruct', original=original,
                         lane='excluded-cost-profile', pulse=pulse)
                equivalent(original, dest)
                with np.load(dest / 'cell_measures.npz', allow_pickle=False) as values:
                    assert all(np.isfinite(values[k]).all() for k in values.files)
                checked[key] = dest.name
            retained.append(dict(family=family, raw_sha256=key[0], summary_sha256=key[1],
                                 reconstruction=checked[key], passed=True))
        probe = frozen['settings'][-1] | dict(worlds=8, blocks=8)
        profile_start = time.process_time()
        run_cell(module, probe, out / 'original', lane='excluded-cost-profile', pulse=pulse)
        run_cell(module, probe, out / 'replay', lane='excluded-cost-profile', pulse=pulse)
        equivalent(out / 'original', out / 'replay', True)
        run_cell(module, probe, out / 'reconstruct', mode='reconstruct', original=out / 'original',
                 lane='excluded-cost-profile', pulse=pulse)
        equivalent(out / 'original', out / 'reconstruct')
        # Production, seeds and outputs must also remain identical to failed R1's
        # valid original pass; only its independent reconstruction is repaired.
        equivalent(c / 'preparation/S_LEARNING_R1/original', out / 'original', True)
        elapsed = time.process_time() - profile_start
        write(out / 'VALIDATION.json', dict(passed=True, controls=controls,
              retained_fixtures=retained, production_unchanged=True, full_seed_replay=True,
              independent_reconstruction=True, profile_worlds=8,
              profile_cpu_seconds=elapsed, cpu_cap_seconds=max(120., elapsed / 8 * 240 * 1.8),
              freeze_sha256=digest(out / 'FREEZE.json')))
    except BaseException as exc:
        write(out / 'FAILURE.json', dict(error=str(exc), traceback=traceback.format_exc(),
              cpu_seconds=time.process_time(), freeze_sha256=digest(out / 'FREEZE.json')))
        raise


def prepare(repo, c):
    auth = read(repo / AUTHORIZATION)
    assert auth['approved'] is True and auth['family'] == FAMILY
    out = c / 'preparation' / FAMILY
    if out.exists():
        raise ValueError('R2 already prepared; no automatic rerun')
    acceptance = read(c / 'ACCEPTANCE.json')
    totals = accounting(c)
    assert totals['healing'] + 600 <= acceptance['cpu_limits']['healing']
    assert sum(totals.values()) + 600 <= acceptance['total_cpu_seconds']
    assert datetime.now(timezone.utc) < datetime.fromisoformat(acceptance['new_starts_close'])
    kinds = ['BASE', 'SHIFT', 'SHARPEN', 'SLOW']
    settings = [dict(truth=k, kinds=kinds, odds=p, cost=cost, temperature=temp,
                     steps=400, worlds=200, blocks=8)
                for p, cost, temp, k in itertools.product((.25, .5, .75, .9), (1., 2.), (.5, 1.), kinds)]
    source = frozen_source(c, PINS + ['runners/prepare_v31_s_learning_r2.py', PROTOCOL, AUTHORIZATION])
    write(out / 'FREEZE.json', dict(at=datetime.now(timezone.utc).isoformat(), settings=settings,
          source_root=str(source), source_manifest_sha256=digest(source / 'SOURCE_MANIFEST.json'),
          acceptance_sha256=digest(c / 'ACCEPTANCE.json'),
          protocol_sha256=digest(repo / PROTOCOL), authorization_sha256=digest(repo / AUTHORIZATION)))
    family = FAMILY + '_VALIDATION'
    gatefile = c / 'families' / family / 'ADMISSION.json'
    write(gatefile, dict(passed=True, validation_only=True, no_scientific_admission=True,
          authorization_sha256=digest(repo / AUTHORIZATION), freeze_sha256=digest(out / 'FREEZE.json')))
    write(c / 'jobs' / family / 'PLAN.json', dict(family=family, pool='healing', kind='validation',
          module='ghostscale.validation.soundingline.v31.s_learning_r2_gate', setting=dict(campaign=str(c)),
          order=-10, cpu_cap_seconds=600., wall_cap_seconds=1800., source_root=str(source),
          source_manifest_sha256=digest(source / 'SOURCE_MANIFEST.json'),
          acceptance_sha256=digest(c / 'ACCEPTANCE.json'), admission_sha256=digest(gatefile),
          interpreter=str(repo / '.venv/Scripts/python.exe')))
    print('Frozen R2 validation queued in the existing Gear 2 controller', flush=True)


def admit(repo, c):
    out = c / 'preparation' / FAMILY
    frozen = read(out / 'FREEZE.json')
    job = c / 'jobs' / (FAMILY + '_VALIDATION')
    complete = read(job / 'COMPLETE.json')
    worker = read(job / 'WORKER_COMPLETE.json')
    assert complete['state'] == worker['state'] == 'VERIFIED'
    assert complete['plan_sha256'] == worker['plan_sha256'] == digest(job / 'PLAN.json')
    assert complete['worker_receipt_sha256'] == digest(job / 'WORKER_COMPLETE.json')
    assert worker['validation_sha256'] == digest(job / 'output/VALIDATION.json')
    assert read(job / 'output/VALIDATION.json') == read(out / 'VALIDATION.json')
    assert frozen['protocol_sha256'] == digest(repo / PROTOCOL)
    assert frozen['authorization_sha256'] == digest(repo / AUTHORIZATION)
    source = Path(frozen['source_root'])
    assert digest(source / 'SOURCE_MANIFEST.json') == frozen['source_manifest_sha256']
    for name, sha in read(source / 'SOURCE_MANIFEST.json').items():
        assert digest(source / name) == sha, name
    settings = frozen['settings']
    proof = read(out / 'VALIDATION.json')
    assert proof['passed']
    gate = dict(passed=True, validation_sha256=digest(out / 'VALIDATION.json'),
                protocol_sha256=digest(repo / PROTOCOL), source_root=str(source),
                source_manifest_sha256=digest(source / 'SOURCE_MANIFEST.json'),
                authorization_sha256=digest(repo / AUTHORIZATION))
    gatefile = c / 'families' / FAMILY / 'ADMISSION.json'
    write(gatefile, gate)
    write(c / (FAMILY + '_ROSTER.json'), settings)
    cap = proof['cpu_cap_seconds']
    for i, setting in enumerate(settings):
        write(c / 'jobs' / f'{FAMILY}-{i:04d}' / 'PLAN.json', dict(family=FAMILY, pool='S',
              module=MODULE, setting=setting, order=12200+i, cpu_cap_seconds=cap,
              wall_cap_seconds=max(1800, cap*4), source_root=str(source),
              source_manifest_sha256=gate['source_manifest_sha256'],
              acceptance_sha256=digest(c / 'ACCEPTANCE.json'), admission_sha256=digest(gatefile),
              interpreter=str(repo / '.venv/Scripts/python.exe')))
    print(dict(admitted=len(settings), profile_cpu_seconds=proof['profile_cpu_seconds'], cap=cap), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--admit', action='store_true')
    args = parser.parse_args()
    if args.validate:
        validate(args.campaign.resolve())
    elif args.admit:
        admit(args.repo.resolve(), args.campaign.resolve())
    else:
        prepare(args.repo.resolve(), args.campaign.resolve())
