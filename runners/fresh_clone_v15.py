"""Clean-clone reproduction receipt for V15 (spec §9.1, hours 166-168).

Clones the repository at HEAD into a scratch directory and re-derives, from that clone alone:

* the card set and its hashes;
* the structural lock's payload;
* the balanced coverage sequence's definition and its hash chain.

The point is not that the numbers match -- they are read from the same committed files -- but that
the *derivation* runs from a fresh checkout with nothing in the working tree. V14's receipt caught
a CRLF issue this way: two generator hashes differed between the working copy and a fresh clone,
which would have broken the lock in continuous integration and not on the machine that wrote it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ghostscale.validation.soundingline.v15 import v15_dir              # noqa: E402
from ghostscale.validation.soundingline.v15.atomicio import write_json_atomic  # noqa: E402

RECEIPT = v15_dir() / "FRESH_CLONE_RECEIPT.json"

PROBE = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from ghostscale.prereg_v15 import structural_payload
from ghostscale.validation.soundingline.v15 import manifest as M
from ghostscale.validation.soundingline.v15 import coverage as CV
cards = M.build_cards()
p = structural_payload()
print(json.dumps({
    "n_mandatory": len(M.mandatory(cards)),
    "n_attacks": len(M.attacks(cards)),
    "cards_sha256": p["cards_sha256"],
    "sesoi_sha256": p["sesoi_sha256"],
    "generators": p["generators"],
    "coverage_block_0": CV.block_digest(0),
    "coverage_chain_64": CV.hash_chain(64),
}))
'''


def run() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()
    tmp = Path(tempfile.mkdtemp(prefix="v15_clone_"))
    out = {"program": "v15", "head": head, "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "clone": str(tmp)}
    try:
        r = subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(REPO), str(tmp / "r")],
                           capture_output=True, text=True)
        if r.returncode:
            out["error"] = r.stderr[-500:]
            write_json_atomic(RECEIPT, out)
            return out
        subprocess.run(["git", "checkout", "--quiet", head], cwd=tmp / "r", capture_output=True)
        probe = tmp / "probe.py"
        probe.write_text(PROBE, encoding="utf-8")
        env = dict(os.environ)
        env.pop("GS_V15_SMOKE", None)
        p = subprocess.run([sys.executable, str(probe), str(tmp / "r")], capture_output=True,
                           text=True, env=env, cwd=str(tmp / "r"))
        if p.returncode:
            out["error"] = p.stderr[-1500:]
            write_json_atomic(RECEIPT, out)
            return out
        clone = json.loads(p.stdout.strip().splitlines()[-1])
        sys.path.insert(0, str(REPO))
        from ghostscale.prereg_v15 import structural_payload
        from ghostscale.validation.soundingline.v15 import coverage as CV
        from ghostscale.validation.soundingline.v15 import manifest as M
        local_payload = structural_payload()
        local = {"n_mandatory": len(M.mandatory(M.build_cards())),
                 "n_attacks": len(M.attacks(M.build_cards())),
                 "cards_sha256": local_payload["cards_sha256"],
                 "sesoi_sha256": local_payload["sesoi_sha256"],
                 "generators": local_payload["generators"],
                 "coverage_block_0": CV.block_digest(0),
                 "coverage_chain_64": CV.hash_chain(64)}
        diffs = {k: {"clone": clone.get(k), "local": local.get(k)}
                 for k in local if clone.get(k) != local.get(k)}
        out.update({"clone_result": clone, "local_result": local, "differences": diffs,
                    "ok": not diffs,
                    "note": ("generator hashes are computed on the bytes on disk; a CRLF working "
                             "copy and an LF clone differ, which is what this receipt is for")})
    finally:
        write_json_atomic(RECEIPT, out)
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "clone_result"}, indent=2)[:2000])
    sys.exit(0 if r.get("ok") else 1)
