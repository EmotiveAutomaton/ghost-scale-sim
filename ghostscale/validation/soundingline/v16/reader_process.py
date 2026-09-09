"""Owned persistent reader process; only serialized public observations cross it."""
from concurrent.futures import ThreadPoolExecutor,TimeoutError
import json
import os
from pathlib import Path
import subprocess
import sys
from .records import canonical
from .runtime import REPO


class ReaderProcess:
    def __init__(self,directory:Path,timeout=20,extensions=()):
        self.directory=directory
        self.timeout=timeout
        self.child=None
        self.pool=None
        self.log=None
        self.extensions=tuple(extensions)

    def __enter__(self):
        self.directory.mkdir(parents=True,exist_ok=True)
        environment=dict(os.environ,PYTHONPATH=str(REPO),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
        self.log=(self.directory/"reader-stderr.log").open("ab")
        command=[sys.executable,"-s","-B","-u","-m","runners.v16_reader_worker"]
        for extension in self.extensions:
            command.extend(["--extension",extension])
        self.child=subprocess.Popen(command,
                                    cwd=self.directory,env=environment,stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,stderr=self.log,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        self.pool=ThreadPoolExecutor(max_workers=1)
        try:
            ready=self._receive()
        except Exception:
            self.close()
            raise
        if not ready.get("ready") or Path(ready["code_root"]).resolve()!=REPO:
            self.close()
            raise ValueError("reader imported the wrong checkout")
        self.identity=ready
        return self

    def _receive(self):
        try:
            line=self.pool.submit(self.child.stdout.readline).result(timeout=self.timeout)
        except TimeoutError:
            self.child.terminate()
            self.child.wait(timeout=5)
            raise RuntimeError("owned reader exceeded its finite request timeout")
        if not line:
            raise RuntimeError("owned reader exited before replying; inspect reader-stderr.log")
        return json.loads(line)

    def request(self,kind,public,**options):
        self.child.stdin.write(canonical({"kind":kind,"public":public,"options":options})+b"\n")
        self.child.stdin.flush()
        response=self._receive()
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    def close(self):
        if self.child is not None and self.child.poll() is None:
            try:
                self.child.stdin.write(b'{"kind":"shutdown"}\n')
                self.child.stdin.flush()
                self.child.wait(timeout=5)
            except (OSError,subprocess.TimeoutExpired):
                self.child.terminate()
                self.child.wait(timeout=5)
        if self.pool is not None:
            self.pool.shutdown(wait=True,cancel_futures=True)
        if self.log is not None:
            self.log.close()

    def __exit__(self,*error):
        self.close()
