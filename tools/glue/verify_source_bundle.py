"""Compare the opt-in source result with the pinned, selected B runtime files.

This is a source check only. It does not infer wheel reproducibility or run a model.
"""

import argparse
import hashlib
import json
from pathlib import Path


PINS = Path(__file__).with_name("SOURCE_PINS.json")


def digest(path: Path, normalize_windows_line_endings: bool = False) -> str:
    data = path.read_bytes()
    if normalize_windows_line_endings:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def verify(sglang: Path, kt: Path, pins: dict,
           normalize_windows_line_endings: bool = False) -> dict:
    results = {}
    for group, root in (("sglang", sglang), ("kt", kt)):
        for relative, expected in pins[group].items():
            path = root / relative
            actual = digest(path, normalize_windows_line_endings) if path.is_file() else None
            results[f"{group}/{relative}"] = {
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "PASS" if actual == expected else "FAIL",
            }
    return {"status": "PASS" if all(x["status"] == "PASS" for x in results.values())
            else "FAIL", "files": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--kt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--normalize-windows-line-endings", action="store_true",
                        help="For a Windows git-apply replay only; never use for a runtime receipt")
    args = parser.parse_args()
    result = verify(args.sglang, args.kt, json.loads(PINS.read_text(encoding="utf-8")),
                    args.normalize_windows_line_endings)
    result["normalized_windows_line_endings"] = args.normalize_windows_line_endings
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
