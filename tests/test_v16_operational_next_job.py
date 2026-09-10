from types import SimpleNamespace
from ghostscale.validation.soundingline.v16 import operational_entry as entry
from ghostscale.validation.soundingline.v16.operational_queue import freeze_plan
from ghostscale.validation.soundingline.v16.expansion_interruption import fixture_campaign
from ghostscale.validation.soundingline.v16.records import write
from ghostscale.validation.soundingline.v16.runtime import freeze, REPO


def test_resume_stage_starts_new_eligible_packet_and_resumes_existing_one(tmp_path, monkeypatch):
    fixture_campaign(tmp_path)
    sources = [REPO/"tests/test_v16_operational_next_job.py"]
    freeze(tmp_path, "known-predecessor", sources, {"scope": "known operational fixture"})
    definition = {"jobs": [entry.job("next", "discovery", 2, "ADMISSION.json", "NEXT_COMPLETION.json", [],
        "Known next-job resume semantics", "known_handler")]}
    freeze_plan(tmp_path, definition)
    write(tmp_path/"ADMISSION.json", {"instrument_state": "valid"})
    calls = []
    def execute(root, heartbeat, *, resume):
        calls.append(resume)
        return {"execution_state": "checkpointed", "campaign_complete": False}
    monkeypatch.setattr(entry.importlib, "import_module", lambda name: SimpleNamespace(PACKET="next-packet", execute=execute))
    entry.run(tmp_path, "resume")
    assert calls == [False]
    freeze(tmp_path, "next-packet", sources, {"scope": "known operational fixture"})
    entry.run(tmp_path, "resume")
    assert calls == [False, True]
