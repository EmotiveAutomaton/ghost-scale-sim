"""Convert actual pytest JUnit results into source-bound admission evidence."""
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
from ghostscale.validation.soundingline.v16.runtime import REPO, PACKAGE
from ghostscale.validation.soundingline.v16.records import write, file_digest, now


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    args = parser.parse_args()
    tree = ET.parse(args.junit)
    tests = {}
    for case in tree.findall(".//testcase"):
        state = "failed" if case.find("failure") is not None or case.find("error") is not None else (
            "skipped" if case.find("skipped") is not None else "passed")
        if case.attrib["name"] in tests:
            raise ValueError("ambiguous duplicate test identity")
        tests[case.attrib["name"]] = state
    if not tests or any(state != "passed" for state in tests.values()):
        raise ValueError("admission suite is empty or not fully passed")
    sources = [*sorted(PACKAGE.glob("*.py")), *sorted((REPO / "tests").glob("test_v16*.py")),
               REPO / "runners/run_v16.py", Path(__file__).resolve()]
    write(REPO / "results/v16/ADMISSION_CHECKS.json", {
        "recorded_at": now(), "scope": "fixture", "tests": tests,
        "junit_sha256": file_digest(args.junit),
        "source_hashes": {str(path.relative_to(REPO)).replace("\\", "/"): file_digest(path) for path in sources}})
    print(f"Recorded {len(tests)} actual passed tests.")


if __name__ == "__main__":
    main()
