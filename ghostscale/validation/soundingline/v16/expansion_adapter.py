"""Fresh-source orchestration with unchanged scientific unit bytecode.

The three early in-process studies receive copied function globals with only
public-reader call bindings redirected through the existing guarded worker.
Original modules, generators, scores and acquired recipes remain untouched.
"""
import importlib
import json
import time
from types import FunctionType, SimpleNamespace
from .records import write
from .resource_accounting import MeasuredReader

PREFIX = "ghostscale.validation.soundingline.v16."
READING = {"K02", "P01", "P02", "P03"}
BEHAVIOR = {"O01", "O02", "O03", "O04", "S01", "S02", "S04", "S05"}
INQUIRY = {f"R{i:02d}" for i in range(1, 6)}
SINGLE = {"K03": "purpose", "K04": "options", "K05": "attention", "S03": "dependency",
          "P04": "mechanism", "M01": "recognition", "M02": "selection", "M03": "audience",
          "M04": "multi_actor", "V01": "tradeoffs", "V02": "tradeoffs", "V03": "preference_probe"}
CARDS = sorted({"K01"}|READING|BEHAVIOR|INQUIRY|set(SINGLE))


def module(card):
    name = "study" if card == "K01" else "reading_study" if card in READING else "behavior_study" if card in BEHAVIOR else "inquiry_stable_study" if card in INQUIRY else SINGLE[card]+"_study"
    return importlib.import_module(PREFIX+name)


def design(card):
    owner = module(card)
    if card in READING|BEHAVIOR|INQUIRY:
        return owner.DESIGNS[card]
    if card in {"V01", "V02"}:
        return importlib.import_module(PREFIX+"tradeoffs").design(card)
    return owner.DESIGN


def summarizer(card, rows):
    owner = module(card)
    if card in READING|BEHAVIOR|INQUIRY:
        return owner.summarize(card, rows)
    if card in {"V01", "V02"}:
        return owner.summarize(rows, card=card)
    return owner.summarize(rows)


def auditor(card):
    name = "craft" if card == "K01" else "reading" if card in READING else "behavior" if card in BEHAVIOR else "inquiry_stable" if card in INQUIRY else SINGLE[card]
    return importlib.import_module(PREFIX+"audit_"+name).audit


class Calls:
    def __init__(self, reader, crosscheck=False):
        self.reader = MeasuredReader(reader)
        self.crosscheck = crosscheck
        self.reading_invocations = []
        self.calls = []

    def bind(self, kind, original, positional=None):
        def call(payload, *args, **kwargs):
            public = json.loads(payload)
            options = dict(kwargs)
            if args:
                if len(args) != 1 or positional is None or positional in options:
                    raise ValueError("unexpected public reader call signature")
                options[positional] = args[0]
            if kind == "reading":
                measured = self.reader.request(PREFIX+"reading_cost_meter:public_reader", public, **options)
                result = measured["result"]
                self.reading_invocations.append({"strategy": options.get("strategy", "maker"),
                    "invocations": measured["invocations"], "scope": measured["scope"]})
            else:
                result = self.reader.request(kind, public, **options)
            if self.crosscheck and result != original(payload, *args, **kwargs):
                raise ValueError("guarded dispatch changes original scientific output")
            self.calls.append({"operation": kind, "options": options})
            return result
        return call


def copy_function(function, bindings):
    copied = FunctionType(function.__code__, bindings, function.__name__, function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


def guarded_function(card, calls):
    owner = module(card)
    function = owner.unit if card == "K01" else owner.execute_unit
    bindings = dict(function.__globals__)
    if card == "K01":
        bindings["construct_public"] = calls.bind("craft", owner.construct_public)
    elif card in READING:
        bindings["reader"] = calls.bind("reading", owner.reader, "strategy")
    elif card in BEHAVIOR:
        for attribute, kind, function_name, positional in [
            ("opportunities", "opportunity", "public_reader", "model"),
            ("selves", "self", "public_reader", "strategy"),
            ("trajectories", "trajectory", "reader", "strategy")]:
            old = getattr(owner, attribute)
            changed = SimpleNamespace(**vars(old))
            setattr(changed, function_name, calls.bind(kind, getattr(old, function_name), positional))
            bindings[attribute] = changed
        bindings["critic_reader"] = calls.bind("critic", owner.critic_reader)
        bindings["execute_trajectory"] = copy_function(owner.execute_trajectory, bindings)
    else:
        raise ValueError("this study already uses a guarded reader")
    return copy_function(function, bindings)


def execute_unit(card, root, condition, index, *, namespace, packet, reader, constructors,
                 scope="discovery expansion", crosscheck=False):
    owner = module(card)
    calls = Calls(reader, crosscheck)
    start, cpu = time.perf_counter(), time.process_time()
    if card == "K01":
        row = guarded_function(card, calls)(root, condition, index, namespace=namespace,
            packet_hash=packet["packet_hash"], constructors=constructors, evidence_scope=scope)
    elif card in READING:
        row = guarded_function(card, calls)(root, card, condition, index, namespace=namespace,
            packet_hash=packet["packet_hash"], constructors=constructors, evidence_scope=scope)
    elif card in BEHAVIOR:
        row = guarded_function(card, calls)(root, card, condition, index, namespace=namespace,
            packet=packet, constructors=constructors, evidence_scope=scope)
    elif card in INQUIRY:
        row = owner.execute_unit(root, card, condition, index, namespace=namespace, packet=packet,
            reader=calls.reader, constructors=constructors, scope=scope)
    else:
        options = {"card": card} if card in {"V01", "V02"} else {}
        row = owner.execute_unit(root, condition, index, namespace=namespace, packet=packet,
            reader=calls.reader, constructors=constructors, scope=scope, **options)
    if calls.reader.samples:
        write(root/"private"/(row["unit_id"]+"-expansion-resources.json"), {"unit_id": row["unit_id"],
            "reader_calls": calls.calls, "requests": calls.reader.samples,
            "reading_invocation_corrections": calls.reading_invocations,
            "parent_cpu_seconds": time.process_time()-cpu, "wall_seconds": time.perf_counter()-start,
            "original_scientific_bytecode_preserved": True, "startup_included": False,
            "cost_scope": "OS measurements and actual invocation supplements do not replace frozen physical or primary outcome estimands"})
    return row
