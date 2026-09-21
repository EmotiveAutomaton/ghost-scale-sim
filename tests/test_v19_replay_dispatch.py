"""New scientific handlers must be reachable through the portable replay entry."""
import ast
from pathlib import Path


def test_runtime_science_handlers_have_portable_dispatch():
    repo=Path(__file__).resolve().parents[1]
    def choices(path,name):
        values=set()
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if not isinstance(node,ast.Compare):continue
            left=ast.unparse(node.left)
            if left!=name:continue
            for value in node.comparators:
                if isinstance(value,ast.Constant) and isinstance(value.value,str):values.add(value.value)
                if isinstance(value,(ast.Tuple,ast.List)):
                    values.update(x.value for x in value.elts if isinstance(x,ast.Constant))
        return values
    native=choices(repo/'ghostscale/validation/soundingline/v19/runtime.py',"plan['design']['handler']")
    portable=choices(repo/'runners/replay_v19.py','kind')
    assert native-{'validation-suite'}<=portable
