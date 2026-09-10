"""Optional machine-local routing to the accepted isolated implementation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def resolve(repo):
    repo = repo.resolve()
    outer = repo.parent.parent.parent if repo.parent.name == "v16-acquired-craft" and repo.parent.parent.name == ".local" else repo.parent
    path = outer/".local/v16-acquired-craft/ACTIVE_IMPLEMENTATION.json"
    if not path.exists():
        return None
    binding = json.loads(path.read_bytes())
    source = Path(binding["source_root"]).resolve()
    root = Path(binding["result_root"]).resolve()
    if not source.is_relative_to(outer) or root != source/"results/v16":
        raise ValueError("active implementation binding escapes this workspace")
    accepted = json.loads((root/"CAMPAIGN.json").read_bytes())
    if accepted["campaign_id"] != binding["campaign_id"] or accepted["commission_sha256"] != binding["commission_sha256"]:
        raise ValueError("active implementation binding differs from accepted campaign")
    package = source/"docs/versions/v16-acquired-craft/CODING_PACKAGE.md"
    if hashlib.sha256(package.read_bytes()).hexdigest() != binding["commission_sha256"]:
        raise ValueError("bound source does not contain the accepted commission")
    return {"source": source, "root": root, "binding_path": path}


def forward(repo, arguments):
    # An explicit root selects a fixture or user-named campaign and prevents recursion.
    if any(argument == "--root" or argument.startswith("--root=") for argument in arguments):
        return None
    binding = resolve(repo)
    if binding is None or binding["source"] == repo.resolve():
        return None
    result = subprocess.run([sys.executable, "-B", "-m", "runners.run_v16", *arguments,
        "--root", str(binding["root"])], cwd=binding["source"],
        stdout=sys.stdout, stderr=sys.stderr,
        env=dict(os.environ, PYTHONPATH=str(binding["source"]), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return result.returncode
