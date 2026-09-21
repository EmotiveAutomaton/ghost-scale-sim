"""Narrow installed-Torch regression child; no campaign fits or queue ownership."""
import argparse,os,time
from pathlib import Path
from ..v18_3.io import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    start=time.process_time();state='failed';result=None
    from .runtime import REPO
    from ..v18_4.priority import below_normal
    below_normal()
    write(args.output/'STATUS.json',dict(state='running',pid=os.getpid(),parent_pid=os.getppid(),cpu_seconds=0),immutable=False)
    try:
        import pytest
        # The scientific module configures Torch once on import. Calling the
        # interop setter here as well prevents pytest from collecting that module.
        from . import tiny_worker
        import torch
        if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:
            raise ValueError('tiny regression thread configuration differs')
        class StopAtBoundary:
            def pytest_runtest_setup(self,item):
                if (args.output/'STOP').exists():pytest.exit('owning validation stopped',returncode=2)
        result=int(pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(args.output/'pytest-temp'),str(REPO/'tests/test_v19_tiny_torch.py')],plugins=[StopAtBoundary()]))
        state='complete' if result==0 else 'failed'
    finally:
        write(args.output/'STATUS.json',dict(state=state,pid=os.getpid(),parent_pid=os.getppid(),cpu_seconds=time.process_time()-start,exit_code=result),immutable=False)
    raise SystemExit(result)

if __name__=='__main__':main()
