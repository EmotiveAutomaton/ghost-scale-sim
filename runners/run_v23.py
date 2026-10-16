"""Hidden module-form V23 worker; never synchronizes an environment."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v23.runtime import run, execute_pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--single-pass', choices=['original', 'replay'])
    args = parser.parse_args()
    if args.single_pass: execute_pass(args.root, args.campaign, args.single_pass)
    else: run(args.root, args.campaign)
