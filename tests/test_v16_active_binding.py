import json
import sys
from pathlib import Path
import pytest
from ghostscale.validation.soundingline.v16.active_binding import resolve, forward
from ghostscale.validation.soundingline.v16.records import write, file_digest
from ghostscale.validation.soundingline.v16.operational_entry import remaining_plan
from ghostscale.validation.soundingline.v16.operational_queue import validate_plan


def arrange(tmp_path):
    repo = tmp_path/"ghost-scale-sim"
    source = tmp_path/".local/v16-acquired-craft/implementation"
    package = source/"docs/versions/v16-acquired-craft/CODING_PACKAGE.md"
    package.parent.mkdir(parents=True)
    package.write_text("known fixture commission", encoding="utf-8")
    accepted = {"campaign_id": "fixture-bound", "commission_sha256": file_digest(package)}
    write(source/"results/v16/CAMPAIGN.json", accepted)
    binding = {**accepted, "source_root": str(source), "result_root": str(source/"results/v16")}
    write(source.parent/"ACTIVE_IMPLEMENTATION.json", binding)
    return repo,source,binding


def test_default_binding_resolves_same_accepted_isolated_checkout(tmp_path):
    repo,source,binding = arrange(tmp_path)
    assert resolve(repo)["source"] == source.resolve()
    assert resolve(source)["root"] == (source/"results/v16").resolve()
    assert forward(source, ["--stage", "preflight"]) is None
    assert forward(repo, ["--stage", "pilot", "--root", "explicit-fixture"]) is None


def test_binding_rejects_changed_identity_and_source(tmp_path):
    repo,source,binding = arrange(tmp_path)
    package = source/"docs/versions/v16-acquired-craft/CODING_PACKAGE.md"
    package.write_text("changed commission", encoding="utf-8")
    with pytest.raises(ValueError, match="accepted commission"):
        resolve(repo)


def test_general_cli_forwards_module_form_with_explicit_bound_root(tmp_path, monkeypatch):
    repo,source,binding = arrange(tmp_path)
    observed = {}
    def subprocess_run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return type("Result", (), {"returncode": 0})()
    monkeypatch.setattr("ghostscale.validation.soundingline.v16.active_binding.subprocess.run", subprocess_run)
    assert forward(repo, ["--stage", "resume"]) == 0
    assert observed["command"][1:4] == ["-B", "-m", "runners.run_v16"]
    assert observed["command"][-2:] == ["--root", str(source/"results/v16")]
    assert observed["cwd"] == source.resolve()
    assert observed["stdout"] is sys.stdout and observed["stderr"] is sys.stderr


def test_remaining_queue_is_finite_and_confirmation_requires_boundary_disposition():
    jobs = validate_plan(remaining_plan())
    assert set(jobs) == {"constructor-expansion-1", "final-boundary-disposition", "explanatory-catalogue", "confirmation-selection", "confirmations", "closeout"}
    assert "final-boundary-disposition" in jobs["confirmation-selection"]["dependencies"]
    assert jobs["confirmations"]["unit_cap"] == 3*4096
