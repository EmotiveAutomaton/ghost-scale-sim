"""Read-only extraction of every declared public reader call in V16 scout units.

This extends coverage in a new module; the earlier reader replay stays frozen.
No evaluator truth is inserted into a request.
"""
from .records import read
from runners.replay_v16_readers import requests as original_requests

PREFIX = "ghostscale.validation.soundingline.v16."
SPECS = {
    "K04": ("options_study:public_read", None),
    "K03": ("purpose_craft:public_read", None),
    "K05": ("attention_craft:public_read", None),
    "P04": ("mechanism_reader:public_reader", "strategy"),
    "S03": ("dependency_monitor:public_reader", "policy"),
    "M01": ("recognition:public_reader", "method"),
    "M02": ("selection:public_reader", "method"),
    "M03": ("audience:public_reader", "policy"),
    "M04": ("multi_actor:public_reader", "policy"),
    "V01": ("tradeoffs:public_reader", "policy"),
    "V02": ("tradeoffs:public_reader", "policy"),
    "V03": ("preference_probe:public_reader", "policy")}
EXTENSIONS = sorted({PREFIX+item[0] for item in SPECS.values()} |
                    {PREFIX+"attention_craft:public_learn"})


def frame(kind, public, expected, **options):
    return {"kind": kind, "public": public, "options": options, "result": expected}


def requests(root, row):
    if row["card_id"] not in SPECS:
        yield from original_requests(root, row)
        return
    uid = row["unit_id"]
    public = read(root/"public"/f"{uid}.json")
    predicted = read(root/"predictions"/f"{uid}.json")
    extension, option = SPECS[row["card_id"]]
    kind = PREFIX+extension
    if row["card_id"] == "K05":
        learning = read(root/"predictions"/f"{uid}-acquisition.json")
        yield frame(PREFIX+"attention_craft:public_learn", learning["public_input"], learning["arms"])
        yield frame(kind, public["construction"], predicted["arms"])
    elif "public_inputs" in predicted:
        first = read(root/"predictions"/f"{uid}-phase-1.json")
        for name, expected in first["arms"].items():
            yield frame(kind, public, expected, **{option: name})
        for name, expected in predicted["arms"].items():
            yield frame(kind, predicted["public_inputs"][name], expected, **{option: name})
    elif option is None:
        yield frame(kind, public, predicted["arms"])
    else:
        for name, expected in predicted["arms"].items():
            yield frame(kind, public, expected, **{option: name})


def scout_units(root):
    """One predeclared seed-index-zero unit per condition, all admitted scouts."""
    for packet_file in sorted((root/"packets").glob("*scout-1.json")):
        packet_id = packet_file.stem
        for path in sorted((root/packet_id).glob("**/units/*_points.json")):
            row = read(path)
            if row["seed_components"]["index"] == 0:
                yield packet_id, path, row
