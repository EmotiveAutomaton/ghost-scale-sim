"""Current consumer identities, with explicit preserved inquiry supersession."""
import re
from .records import read, file_digest
from .runtime import REPO
from .admission import CONSUMERS
from .consumer_frames import requests


def packets(root):
    for path in sorted((root/"packets").glob("*.json")):
        if path.stem == "inquiry-scout-1" and (root/"packets/inquiry-scout-2.json").exists():
            continue
        if "-scout-" in path.stem:
            yield path, read(path)


def attack_consumers(root):
    consumers = {attack: set(cards) for attack, cards in CONSUMERS.items()}
    def collect(value, card=None):
        if isinstance(value, dict):
            card = value.get("card_id", card)
            if card:
                for attack in value.get("adversaries", [])+value.get("dependencies", []):
                    if attack in consumers:
                        consumers[attack].add(card)
            for key, item in value.items():
                collect(item, key if re.fullmatch(r"[KPOSRMVXB][0-9]{2}", key) else card)
        elif isinstance(value, list):
            for item in value:
                collect(item, card)
    for _, packet in packets(root):
        collect(packet["identity"]["design"])
    return {attack: sorted(cards) for attack, cards in consumers.items()}


def selected_units(root, cards=None):
    for packet_path, packet in packets(root):
        for relative, expected in packet["identity"]["files"].items():
            if file_digest(REPO/relative) != expected:
                raise ValueError("consumer source differs from its frozen identity")
        for path in sorted((root/packet_path.stem).glob("**/units/*_points.json")):
            row = read(path)
            if row["seed_components"]["index"] != 0 or cards is not None and row["card_id"] not in cards:
                continue
            completion = read(path.parent.parent/"COMPLETION.json")
            if completion["execution_state"] != "completed" or completion["instrument_state"] != "valid":
                raise ValueError("consumer lacks completed valid source")
            if row["packet_hash"] != packet["packet_hash"]:
                raise ValueError("consumer raw unit belongs to another source")
            yield packet_path.stem, path, row
