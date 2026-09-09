"""Observed per-request process resources; separate from scientific cost estimands."""
import os
from pathlib import Path
import time
from .records import canonical

def process_snapshot(pid):
    try:
        if os.name=="nt":
            import ctypes
            from ctypes import wintypes
            class Memory(ctypes.Structure):
                _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[
                    (name,ctypes.c_size_t) for name in ["PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage",
                     "QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage",
                     "PagefileUsage","PeakPagefileUsage"]]
            kernel=ctypes.WinDLL("kernel32",use_last_error=True)
            kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
            kernel.OpenProcess.restype=wintypes.HANDLE
            kernel.CloseHandle.argtypes=[wintypes.HANDLE]
            kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
            kernel.K32GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]
            handle=kernel.OpenProcess(0x410,False,pid)
            if not handle:
                raise OSError(ctypes.get_last_error(),"OpenProcess query failed")
            try:
                created,exited,system,user=[wintypes.FILETIME() for _ in range(4)]
                if not kernel.GetProcessTimes(handle,*[ctypes.byref(x) for x in [created,exited,system,user]]):
                    raise OSError(ctypes.get_last_error(),"process time query failed")
                memory=Memory();memory.cb=ctypes.sizeof(memory)
                if not kernel.K32GetProcessMemoryInfo(handle,ctypes.byref(memory),memory.cb):
                    raise OSError(ctypes.get_last_error(),"process memory query failed")
                ticks=sum((entry.dwHighDateTime<<32)+entry.dwLowDateTime for entry in [system,user])
                return {"state":"measured","cpu_seconds":ticks/1e7,"resident_bytes":memory.WorkingSetSize,
                        "peak_resident_bytes":memory.PeakWorkingSetSize,"method":"OS process time and working-set counters"}
            finally:
                kernel.CloseHandle(handle)
        stat=(Path("/proc")/str(pid)/"stat").read_text().rsplit(")",1)[1].split()
        status=(Path("/proc")/str(pid)/"status").read_text().splitlines()
        peak=next(int(line.split()[1])*1024 for line in status if line.startswith("VmHWM:"))
        return {"state":"measured","cpu_seconds":(int(stat[11])+int(stat[12]))/os.sysconf("SC_CLK_TCK"),
                "resident_bytes":int(stat[21])*os.sysconf("SC_PAGE_SIZE"),"peak_resident_bytes":peak,
                "method":"proc process time and resident-memory counters"}
    except (OSError,AttributeError,StopIteration,ValueError) as error:
        return {"state":"unavailable","reason":f"{type(error).__name__}: {error}"}

class MeasuredReader:
    def __init__(self,reader):
        self.reader=reader
        self.samples=[]
    def request(self,kind,public,**options):
        before=process_snapshot(self.reader.child.pid)
        started=time.perf_counter()
        parent_started=time.process_time()
        result=self.reader.request(kind,public,**options)
        parent_cpu=time.process_time()-parent_started
        elapsed=time.perf_counter()-started
        after=process_snapshot(self.reader.child.pid)
        measured=before["state"]==after["state"]=="measured"
        self.samples.append({"reader":kind,"options":options,"request_bytes":len(canonical(public)),
            "wall_seconds":elapsed,"parent_cpu_seconds":parent_cpu,
            "reader_cpu_seconds":max(0.0,after["cpu_seconds"]-before["cpu_seconds"]) if measured else None,
            "resident_bytes":after.get("resident_bytes"),"peak_resident_bytes":after.get("peak_resident_bytes"),
            "resource_state":"measured" if measured else "unavailable",
            "measurement":after.get("method",after.get("reason")),
            "scope":"request including serialization/IPC; OS CPU has finite clock resolution; worker startup excluded"})
        return result

