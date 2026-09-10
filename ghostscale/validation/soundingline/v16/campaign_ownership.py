"""Operational campaign ownership across checkouts; no scientific model state."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
import threading
from .records import read,write,file_digest,digest,now

_ACTIVE=set()
_REGISTRY=threading.Lock()

def acceptance(root,repo):
    accepted=read(root/"CAMPAIGN.json")
    if file_digest(repo/"docs/versions/v16-acquired-craft/CODING_PACKAGE.md")!=accepted["commission_sha256"]:
        raise ValueError("commission hash mismatch")
    lock_path=root/"ACCEPTANCE_LOCK.json"
    manifest=root/"COMMISSION_MANIFEST.json"
    identity={"campaign_sha256":file_digest(root/"CAMPAIGN.json"),
        "manifest_sha256":file_digest(manifest) if manifest.exists() else None}
    if lock_path.exists():
        if read(lock_path)["identity"]!=identity:
            raise ValueError("immutable acceptance or commission manifest changed")
    else:
        # Existing packet clocks anchor the migration; never bless a changed clock.
        for path in (root/"packets").glob("*.json"):
            packet=read(path)
            if packet["accepted_at"]!=accepted["accepted_at"] or packet["deadline"]!=accepted["deadline"] or packet["identity"]["commission_hash"]!=accepted["commission_sha256"]:
                raise ValueError("accepted campaign differs from an existing packet clock")
        write(lock_path,{"schema_version":"v16.acceptance-lock.1","identity":identity,"recorded_at":now()})
    return accepted

def key_for(root):
    path=root/"CAMPAIGN.json"
    if path.exists():
        accepted=read(path)
        return digest([accepted["campaign_id"],accepted["commission_sha256"],accepted["accepted_at"]])
    return digest(["scratch-fixture",str(root.resolve())])

@contextmanager
def campaign_owner(root):
    """Nonblocking ownership survives path changes and releases on process death."""
    key=key_for(root)
    with _REGISTRY:
        if key in _ACTIVE:
            raise RuntimeError("another supervisor owns this campaign")
        _ACTIVE.add(key)
    handle=None;locked=False;api=None
    try:
        if os.name=="nt":
            import ctypes
            from ctypes import wintypes
            api=ctypes.WinDLL("kernel32",use_last_error=True)
            api.CreateMutexW.argtypes=[ctypes.c_void_p,wintypes.BOOL,wintypes.LPCWSTR]
            api.CreateMutexW.restype=wintypes.HANDLE
            api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
            api.WaitForSingleObject.restype=wintypes.DWORD
            api.ReleaseMutex.argtypes=[wintypes.HANDLE]
            api.ReleaseMutex.restype=wintypes.BOOL
            api.CloseHandle.argtypes=[wintypes.HANDLE]
            api.CloseHandle.restype=wintypes.BOOL
            handle=api.CreateMutexW(None,False,"Global\\GhostScaleV16_"+key)
            if not handle:
                raise OSError(ctypes.get_last_error(),"cannot create campaign ownership mutex")
            result=api.WaitForSingleObject(handle,0)
            # WAIT_ABANDONED also grants ownership after an actual owner death.
            if result not in [0,0x80]:
                raise RuntimeError("another supervisor owns this campaign")
            locked=True
        else:
            import fcntl
            directory=Path(tempfile.gettempdir())/f"ghostscale-v16-ownership-{os.getuid()}"
            directory.mkdir(mode=0o700,exist_ok=True)
            handle=(directory/(key+".lock")).open("a+b")
            try:
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                locked=True
            except OSError as error:
                raise RuntimeError("another supervisor owns this campaign") from error
        yield key
    finally:
        if handle is not None:
            if os.name=="nt":
                if locked:
                    api.ReleaseMutex(handle)
                api.CloseHandle(handle)
            else:
                handle.close()
        with _REGISTRY:
            _ACTIVE.discard(key)
