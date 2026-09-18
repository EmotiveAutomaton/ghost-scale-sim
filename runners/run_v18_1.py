"""Module-form, source-bound V18.1 branch worker."""
import argparse
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
if os.name=='nt':
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)

from ghostscale.validation.soundingline.v16.records import read
from ghostscale.validation.soundingline.v18_1.runtime import freeze,run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','run'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--acceptance',type=Path)
    parser.add_argument('--admission',type=Path)
    parser.add_argument('--constructors',type=int,default=64)
    parser.add_argument('--stop-after-blocks',type=int)
    args=parser.parse_args()
    if args.command=='freeze':
        if not args.acceptance or not args.admission:
            parser.error('freeze requires acceptance and admission')
        freeze(args.root,read(args.acceptance),args.archive,read(args.admission),constructors=args.constructors)
        print('frozen')
    else:
        result=run(args.root,args.archive,stop_after_blocks=args.stop_after_blocks)
        print(result.get('execution_state',result.get('state')))


if __name__=='__main__':
    main()
