"""Bounded mutable-status reads that permit Windows atomic replacement."""
import errno
import json
import os
import time


def _bytes(path):
    if os.name != "nt":
        return path.read_bytes()
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path.resolve()), 0x80000000, 0x00000007, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def read_status(path, *, attempts=40, delay=.025):
    if path.name != "RUNNER_STATUS.json" or attempts < 1 or delay < 0:
        raise ValueError("bounded status reads apply only to the mutable heartbeat record")
    for attempt in range(attempts):
        try:
            return json.loads(_bytes(path))
        except OSError as error:
            transient = getattr(error, "winerror", None) in {5, 32, 33} or error.errno == errno.EACCES
            if not transient or attempt+1 == attempts:
                raise
            time.sleep(min(.25, delay*(attempt+1)))
