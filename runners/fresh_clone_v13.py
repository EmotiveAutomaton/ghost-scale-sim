"""Fresh-clone validation for V13 (spec §20.2): clone HEAD, run the validators there, compare the
scientific fields, and write a receipt.

    python runners/fresh_clone_v13.py [--install]

Without --install, the clone is validated with THIS interpreter (the pinned venv) pointed at the
clone; --install additionally creates a fresh venv in the clone and installs from the lockfile's
pins (pip; may need the network or a warm wheel cache). The receipt records which was done.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ghostscale.validation.soundingline.v13 as V                          # noqa: E402
from ghostscale.validation.soundingline.v13 import common as C              # noqa: E402

WORK = V.V13_RESULTS / "fresh_clone_work"
RECEIPT = V.V13_RESULTS / "FRESH_CLONE_RECEIPT.json"


def sh(args, cwd=None, env=None, timeout=3600):
    return subprocess.run(args, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout, env=env)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--install", action="store_true")
    args = ap.parse_args()
    head = sh(["git", "rev-parse", "HEAD"], cwd=REPO).stdout.strip()
    dirty = bool(sh(["git", "status", "--porcelain"], cwd=REPO).stdout.strip())
    dest = WORK / "clone"
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = sh(["git", "-c", "core.longpaths=true", "clone", "--quiet", str(REPO), str(dest)])
    if r.returncode != 0 or not (dest / "runners" / "validate_v13_program.py").exists():
        print(r.stderr or "clone checkout incomplete (deep legacy paths exceed the platform limit without core.longpaths)")
        return 1
    py = sys.executable
    install = "shared pinned interpreter (no install)"
    if args.install:
        r = sh([sys.executable, "-m", "venv", str(dest / ".venv_clone")], timeout=600)
        pyc = dest / ".venv_clone" / "Scripts" / "python.exe"
        r2 = sh([str(pyc), "-m", "pip", "install", "-q", "-e", ".[methods,dev]"], cwd=dest, timeout=3000)
        if r2.returncode == 0:
            py, install = str(pyc), "fresh venv, pip install -e .[methods,dev]"
        else:
            install = f"install failed ({r2.stderr[-300:]}); fell back to the shared interpreter"
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(dest)
    env["OMP_NUM_THREADS"] = "1"
    checks = {}
    for name, cmd in (("determinism", [py, "-m", "ghostscale.validation.soundingline.v13.determinism", "--order", "forward"]),
                      ("validator", [py, str(dest / "runners" / "validate_v13_program.py"), "--interim"]),
                      ("gates_test", [py, "-m", "pytest", "-q", str(dest / "tests" / "test_v13_gates.py")]),):
        rr = sh(cmd, cwd=dest, env=env, timeout=3600)
        checks[name] = {"returncode": rr.returncode, "head": (rr.stdout + rr.stderr)[:1200], "tail": (rr.stdout + rr.stderr)[-500:]}
    # compare determinism output with the local tree
    local = sh([sys.executable, "-m", "ghostscale.validation.soundingline.v13.determinism", "--order", "forward"], cwd=REPO)
    clone_det = checks["determinism"]
    fields_match = None
    if clone_det["returncode"] == 0 and local.returncode == 0:
        rr = sh([py, "-m", "ghostscale.validation.soundingline.v13.determinism", "--order", "forward"], cwd=dest, env=env)
        try:
            fields_match = bool(json.loads(rr.stdout) == json.loads(local.stdout))
        except json.JSONDecodeError:
            fields_match = False
    ok = all(c["returncode"] in (0, 2) for c in checks.values()) and fields_match is not False
    receipt = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "head": head, "git_dirty": dirty, "install": install,
               "checks": checks, "scientific_fields_match_local": fields_match,
               "note": "a dirty local tree makes field comparison meaningless for uncommitted generator changes", "ok": bool(ok)}
    RECEIPT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("head", "git_dirty", "install", "scientific_fields_match_local", "ok")}, indent=2))
    shutil.rmtree(dest, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
