"""Harmless receipt for the complete hidden Windows launch chain."""
import argparse,ctypes,os,time,json
from pathlib import Path
from ghostscale.validation.soundingline.v18_4.priority import below_normal

def main(out):
    below_normal();k=ctypes.windll.kernel32;k.GetCurrentProcess.restype=ctypes.c_void_p
    k.GetPriorityClass.argtypes=[ctypes.c_void_p]
    receipt=dict(pid=os.getpid(),parent_pid=os.getppid(),console_window=int(k.GetConsoleWindow()),priority=int(k.GetPriorityClass(k.GetCurrentProcess())),module_form=True)
    out.write_text(json.dumps(receipt),encoding='utf-8');time.sleep(3)
    if receipt['console_window']!=0 or receipt['priority']!=0x4000:raise RuntimeError('native hidden/below-normal fixture failed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
