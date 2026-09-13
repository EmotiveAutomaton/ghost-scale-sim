"""Exercise the actual Windows GUI interpreter and its console-free child."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows native launch")
LAUNCHER = Path(__file__).resolve().parents[1] / "runners/launch_background.py"


def launch(tmp_path, command):
    log = tmp_path / "output with spaces.log"
    result = subprocess.run(
        [str(Path(sys.executable).with_name("pythonw.exe")), "-B", str(LAUNCHER),
         "--cwd", str(tmp_path), "--log", str(log), "--", *command],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW, timeout=20,
    )
    return result.returncode, log.read_text(encoding="utf-8")


def test_gui_entry_preserves_streams_cwd_arguments_and_exit(tmp_path):
    probe = tmp_path / "probe with spaces.py"
    probe.write_text(
        "import ctypes, json, os, sys\n"
        "k = ctypes.windll.kernel32\n"
        "k.GetConsoleWindow.restype = ctypes.c_void_p\n"
        "h = k.GetConsoleWindow()\n"
        "print(json.dumps(dict(cwd=os.getcwd(), arg=sys.argv[1], "
        "visible=bool(ctypes.windll.user32.IsWindowVisible(ctypes.c_void_p(h))))), flush=True)\n"
        "print('stderr retained', file=sys.stderr, flush=True)\n"
        "raise SystemExit(17)\n", encoding="utf-8")
    code, output = launch(tmp_path, [sys.executable, "-B", str(probe), "literal $() & spaces"])
    assert code == 17
    receipt = json.loads(output.splitlines()[0])
    assert Path(receipt["cwd"]) == tmp_path
    assert receipt["arg"] == "literal $() & spaces"
    assert receipt["visible"] is False
    assert "stderr retained" in output


def test_missing_executable_records_failure(tmp_path):
    code, output = launch(tmp_path, [str(tmp_path / "absent.exe")])
    assert code == 1
    assert "FileNotFoundError" in output
