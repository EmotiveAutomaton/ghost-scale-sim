"""Verified Windows priority control; handle width must be explicit on Win64."""
import os


def below_normal():
    if os.name!='nt':return None
    import ctypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    kernel.SetPriorityClass.restype=ctypes.c_int
    kernel.GetPriorityClass.argtypes=[ctypes.c_void_p]
    kernel.GetPriorityClass.restype=ctypes.c_ulong
    handle=ctypes.c_void_p(-1)
    if not kernel.SetPriorityClass(handle,0x4000):raise ctypes.WinError(ctypes.get_last_error())
    actual=kernel.GetPriorityClass(handle)
    if actual!=0x4000:raise RuntimeError('requested process priority was not applied')
    return actual


if __name__=='__main__':print(below_normal())
