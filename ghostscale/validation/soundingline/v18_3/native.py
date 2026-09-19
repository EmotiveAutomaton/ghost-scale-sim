"""Retained native process handles for CPU accounting across Windows exits."""
import os


class ProcessClock:
    def __init__(self,pid):
        self.pid=pid;self.handle=None
        if os.name=='nt':
            import ctypes
            self.ctypes=ctypes;self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            self.kernel.OpenProcess.restype=ctypes.c_void_p
            self.kernel.GetProcessTimes.argtypes=[ctypes.c_void_p]+[ctypes.c_void_p]*4
            self.kernel.CloseHandle.argtypes=[ctypes.c_void_p]
            self.handle=self.kernel.OpenProcess(0x1000,False,pid)
            if not self.handle:raise OSError(ctypes.get_last_error(),'cannot bind native worker clock')
            self.created=self.times()[0]

    def times(self):
        if self.handle is None:raise RuntimeError('native process clock unavailable')
        values=[self.ctypes.c_ulonglong() for _ in range(4)]
        if not self.kernel.GetProcessTimes(self.handle,*[self.ctypes.byref(v) for v in values]):
            raise OSError(self.ctypes.get_last_error(),'native process time query failed')
        return values[0].value,(values[2].value+values[3].value)/1e7

    def seconds(self):
        created,seconds=self.times()
        if created!=self.created:raise RuntimeError('native process identity changed')
        return seconds

    def close(self):
        if self.handle is not None:self.kernel.CloseHandle(self.handle);self.handle=None
