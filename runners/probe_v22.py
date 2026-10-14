"""Harmless native launch fixture; writes no scientific or live queue state."""
import argparse
import os
import time
from pathlib import Path
from ghostscale.validation.soundingline.v18_3.io import write, now
from ghostscale.validation.soundingline.v18_4.priority import below_normal
from ghostscale.validation.soundingline.v18_3.native import ProcessClock

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    below_normal()
    clock = ProcessClock(os.getpid())
    write(a.output, dict(at=now(), pid=os.getpid(), parent_pid=os.getppid(), created=clock.created,
                         cwd=str(Path.cwd()), module_form=True, fixture_only=True))
    time.sleep(15)
    clock.close()
