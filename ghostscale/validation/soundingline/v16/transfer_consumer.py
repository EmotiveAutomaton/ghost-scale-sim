"""Standalone B01 consumer. Copied unchanged beside the v16_reference package.

Launch from the exported directory: python -s -B -u -m consumer
Requests and predictions use standard input/output. The trusted reference needs
NumPy; Ghost Scale and Sounding Line need not be installed. No model calls.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

FIELDS = {"task_id", "schema_version", "lineage_id", "access_tier",
          "final_artifact", "declared_context", "permitted_prior_artifacts",
          "permitted_query_descriptions", "query_costs", "target_request",
          "reader_action_budget"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def guard(event, args):
    if event == "open" or event in {"os.listdir", "os.scandir", "os.system",
            "subprocess.Popen", "socket.connect", "socket.bind", "ctypes.dlopen", "ctypes.dlsym"}:
        raise PermissionError("consumer file/process/network access closed after trusted imports")


def validate(observation):
    if set(observation) != FIELDS or observation["schema_version"] != "v16.transfer.1":
        raise ValueError("transfer public schema violation")
    if any(not isinstance(observation[key], str) or len(observation[key]) != 32
           for key in ("task_id", "lineage_id")):
        raise ValueError("opaque 128-bit public aliases required")
    context = observation["declared_context"]
    if set(context) != {"operation", "reader_options", "observation"}:
        raise ValueError("undeclared transfer context")
    return context


def main():
    from v16_reference.craft import construct_public
    from v16_reference.reconstruction import reader as reading
    from v16_reference.opportunity import public_reader as opportunity
    from v16_reference.self_monitor import public_reader as self_reader
    from v16_reference.self_trajectory import reader as trajectory
    from v16_reference.inquiry import choose, learn_public, construct_public as inquiry_construct
    from v16_reference.inquiry import read_maker_public
    import numpy
    readers = {"craft": construct_public, "reading": reading, "opportunity": opportunity,
               "self": self_reader, "trajectory": trajectory, "inquiry": choose,
               "inquiry-learning": learn_public, "inquiry-construction": inquiry_construct,
               "inquiry-reading": read_maker_public}
    directory = Path(__file__).resolve().parent
    sources = {str(path.relative_to(directory)).replace("\\", "/"):
               hashlib.sha256(path.read_bytes()).hexdigest()
               for path in [Path(__file__).resolve(), *sorted((directory/"v16_reference").glob("*.py"))]}
    forbidden_imports = [name for name in sys.modules if name == "ghostscale" or name.startswith("ghostscale.")]
    if forbidden_imports:
        raise RuntimeError("standalone consumer imported the scientific repository")
    sys.addaudithook(guard)
    print(json.dumps({"ready": True, "pid": os.getpid(), "sources": sources,
                      "numpy": numpy.__version__, "python": sys.version,
                      "ghostscale_imports": forbidden_imports,
                      "access": "public JSON; post-import CPython audit guard"}), flush=True)
    for line in sys.stdin.buffer:
        try:
            if len(line) > 4_000_000:
                raise ValueError("public frame exceeds finite transport limit")
            frame = json.loads(line)
            if frame.get("operation") == "shutdown":
                break
            if frame.get("operation") == "probe-private":
                Path(frame["path"]).read_bytes()
                raise AssertionError("forbidden read was permitted")
            context = validate(frame)
            function = readers[context["operation"]]
            started = time.perf_counter()
            cpu = time.process_time()
            result = function(canonical(context["observation"]), **context["reader_options"])
            response = {"ok": True, "task_id": frame["task_id"],
                        "observation_sha256": hashlib.sha256(canonical(frame)).hexdigest(),
                        "result": result,
                        "runtime": {"reader_cpu_seconds": time.process_time()-cpu,
                                    "reader_wall_seconds": time.perf_counter()-started}}
        except Exception as error:
            response = {"ok": False, "error": type(error).__name__ + ": " + str(error)}
        sys.stdout.buffer.write(canonical(response) + b"\n")
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
