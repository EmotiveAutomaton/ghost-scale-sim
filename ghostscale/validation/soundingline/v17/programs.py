"""Small typed cell programs, expanded through the unchanged V16 graphic executor."""
from ..v16 import graphic_world


def cell(value, arguments=()):
    if isinstance(value, dict) and set(value) == {"arg"}:
        index = value["arg"]
        if type(index) is not int or not 0 <= index < len(arguments):
            raise ValueError("unbound cell argument")
        value = arguments[index]
    if type(value) is not int or not 0 <= value < 16:
        raise ValueError("cell is not an integer in 0..15")
    return value


def expand(expression, library=(), arguments=(), *, depth=0):
    if depth > 12 or not isinstance(expression, dict):
        raise ValueError("invalid or too deeply nested expression")
    op = expression.get("op")
    if op in ("place", "remove") and set(expression) == {"op", "cell"}:
        return [cell(expression["cell"], arguments) + (16 if op == "remove" else 0)]
    if op == "seq" and set(expression) == {"op", "items"} and isinstance(expression["items"], list):
        return [action for item in expression["items"]
                for action in expand(item, library, arguments, depth=depth+1)]
    if op == "call" and set(expression) == {"op", "index", "args"}:
        index = expression["index"]
        if type(index) is not int or not 0 <= index < len(library):
            raise ValueError("unknown abstraction")
        definition = library[index]
        arity = definition["arity"]
        if type(arity) is not int or arity not in (0, 1, 2) or len(expression["args"]) != arity:
            raise ValueError("invalid abstraction arity")
        bound = tuple(cell(arg, arguments) for arg in expression["args"])
        return expand(definition["body"], library[:index], bound, depth=depth+1)
    raise ValueError("invalid typed expression")


def encode(program):
    if any(type(a) is not int or a not in range(32) for a in program):
        raise ValueError("invalid primitive for encoding")
    return {"op": "seq", "items": [{"op": "place" if action < 16 else "remove", "cell": action % 16}
                                  for action in program]}


def execute(program, *, initial=0, forbidden=(), max_steps=6):
    """Tool restrictions reject attempted actions; they do not change primitive physics."""
    if not isinstance(program, (tuple, list)):
        raise ValueError("program must be a list")
    if any(type(a) is not int or a not in range(32) for a in forbidden):
        raise ValueError("invalid forbidden primitive")
    # Validate the initial state/limit even for an empty program.
    graphic_world.execute([], initial=initial, max_steps=max_steps)
    state, trace = initial, []
    for i, action in enumerate(program):
        if i == max_steps:
            return {"artifact": state, "legal": False, "error": "timeout",
                    "primitive_cost": len(trace), "trace": trace}
        if type(action) is not int or action not in range(32) or action in forbidden:
            trace.append({"before": state, "action": action, "after": state, "legal": False})
            return {"artifact": state, "legal": False, "error": "invalid/tool-forbidden action",
                    "primitive_cost": len(trace), "trace": trace}
        result = graphic_world.execute([action], initial=state, max_steps=1)
        trace.extend(result["trace"])
        state = result["artifact"]
    return {"artifact": state, "legal": True, "error": None,
            "primitive_cost": len(trace), "trace": trace}
