"""Known construction job exercised through the actual general CLI."""
from .runtime import campaign
from .expansion_runner import execute as execute_expansion


def execute(root, heartbeat, *, resume=False):
    if not campaign(root)["campaign_id"].startswith("fixture-"):
        raise ValueError("operational fixture cannot dispatch into the scientific campaign")
    return execute_expansion(root, heartbeat, resume=resume, fixture="K01")
