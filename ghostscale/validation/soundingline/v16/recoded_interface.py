"""Explicit macro syntax adapter before the unchanged primitive consumer APIs.

This validates a declared interface, not invariance to an unknown grammar or to
changing an algorithm's action-order policy. Physical recodings are separate.
"""
import importlib
import json
from .recoding import macro_record, compile_macros
from .records import canonical
from .consumer_frames import EXTENSIONS
from runners.v16_reader_worker import native, craft, reading, opportunity, self_monitor, trajectory, critic_reader

OPERATIONS = {"native": native, "craft": craft, "reading": reading, "opportunity": opportunity,
              "self": self_monitor, "trajectory": trajectory, "critic": critic_reader}
for specification in [*EXTENSIONS, *["ghostscale.validation.soundingline.v16.inquiry_stable:"+name
        for name in ("choose", "learn_public", "construct_public", "read_maker_public")]]:
    module, function = specification.split(":")
    OPERATIONS[specification] = getattr(importlib.import_module(module), function)

SINGLE = {"program", "prefix", "commands", "trace", "continuation_program"}
MULTIPLE = {"attempts", "instructions", "library", "pending_commands"}


def encode(public, spelling, split):
    def program(value):
        scalar = type(value) is int
        actions = [value] if scalar else value
        if not isinstance(actions, list) or any(type(item) is not int for item in actions):
            return walk(value)
        cuts = list(range(1, len(actions))) if split else []
        return {"_v16_macro": macro_record(actions, cuts, spelling, reverse_definitions=split),
                "shape": "scalar" if scalar else "program"}
    def walk(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key in SINGLE and (type(item) is int or isinstance(item, list)):
                    result[key] = program(item)
                elif key in MULTIPLE and isinstance(item, list):
                    result[key] = [program(part) for part in item]
                else:
                    result[key] = walk(item)
            return result
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value
    return walk(public)


def decode(encoded):
    counts = {"program_fields": 0, "nonempty_program_fields": 0, "expanded_primitive_tokens": 0, "macro_calls": 0}
    def walk(value):
        if isinstance(value, dict):
            if set(value) == {"_v16_macro", "shape"}:
                expanded = compile_macros(value["_v16_macro"])
                counts["program_fields"] += 1
                counts["nonempty_program_fields"] += bool(expanded)
                counts["expanded_primitive_tokens"] += len(expanded)
                counts["macro_calls"] += len(value["_v16_macro"]["calls"])
                if value["shape"] == "scalar":
                    if len(expanded) != 1:
                        raise ValueError("scalar command must expand to one primitive")
                    return expanded[0]
                if value["shape"] != "program":
                    raise ValueError("unknown recoded program shape")
                return expanded
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value
    return walk(encoded), counts


def public_reader(payload, operation, reader_options):
    observation, counts = decode(json.loads(payload))
    if operation not in OPERATIONS:
        raise ValueError("operation outside the declared recoding interface")
    return {"result": OPERATIONS[operation](canonical(observation), **reader_options),
            "representation": counts,
            "cost_scope": "expanded tokens represented in input; repeated observations and library definitions are not new executions"}
