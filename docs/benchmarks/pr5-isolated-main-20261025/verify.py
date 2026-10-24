"""Verify the published PR5 observations with Python's standard library.

Run: python docs/benchmarks/pr5-isolated-main-20261025/verify.py
This checks recorded evidence; it does not launch an inference server.
"""
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def finite_hex(value):
    require(isinstance(value, str) and len(value) == 8, "invalid float32 bits")
    require(math.isfinite(struct.unpack("!f", bytes.fromhex(value))[0]), "nonfinite score")


def main():
    packed = (ROOT / "observations.json.gz").read_bytes()
    require(hashlib.sha256(packed).hexdigest() ==
            (ROOT / "OBSERVATIONS_SHA256.txt").read_text().split()[0], "archive hash differs")
    data = json.loads(gzip.decompress(packed))
    receipts, hot, isolation = read("case_receipts.json"), read("fixed_hot_receipts.json"), read("ISOLATION.json")
    arms = ["00", "10", "01", "11"]
    require(list(data) == arms, "wrong switch matrix")
    names = [f"piece-{i:02d}" for i in range(24)] + ["batch-1", "batch-2", "batch-4", "repeat-piece-00"]
    placements = {row["placement_sha256"] for row in isolation["arms"]}
    require(len(placements) == 1, "placement differs across arms")
    for arm in arms:
        require(list(data[arm]) == names, "case coverage/order differs")
        rec, witness = receipts[arm], hot[arm]
        require(rec["status"] == "TOOLS_COMPLETE_PENDING_NUMERICAL_REVIEW", "arm incomplete")
        env = witness["environment"]
        require(env == witness["observed_process_environment"], "live environment differs")
        require(env["KT_SHARED_AFTER_SUBMIT"] == arm[0] and env["KT_HANDOFF_PACK"] == arm[1], "switch differs")
        require(env["KT_KV_HOT_DYNAMIC_SLOTS"] == env["KT_KV_HOT_DYNAMIC_RESIDENT"] == "0", "dynamic hot enabled")
        require(env["KT_KV_HOT_EXTRA"] == "2" and env["KT_KV_HOT_SLOTS_MAX"] == "80", "slot budget differs")
        require(witness["fixed_slot_witnesses"] == [["80", "2", "2", "2"]], "allocation differs")
        require(not witness["release_log"], "slots were released")
        for case, record in zip(names, rec["cases"]):
            values = data[arm][case]
            raw = (json.dumps(values, indent=2, allow_nan=False) + "\n").encode()
            require(record["name"] == case and hashlib.sha256(raw).hexdigest() == record["numeric_sha256"], "recorded bits differ")
            batch = int(case.split("-")[1]) if case.startswith("batch-") else 1
            length = 64 if case.startswith("batch-") else 480
            require(len(values) == batch, "batch coverage differs")
            for row in values:
                require(len(row["input"]) == length and len(row["output"]) == 8, "selected coverage differs")
                require(row["input"][0][0] is None, "first input score should be absent")
                for bits, token in row["input"][1:] + row["output"]:
                    finite_hex(bits)
                    require(type(token) is int and 0 <= token < 129280, "invalid token ID")
                require(row["output_ids"] == [v[1] for v in row["output"]], "output IDs differ")
                for key, count in [("input_top_logprobs", length), ("output_top_logprobs", 8)]:
                    require(len(row[key]) == count, "top-8 coverage differs")
                    for top in row[key][1:] if key.startswith("input") else row[key]:
                        require(len(top) == 8, "top-8 width differs")
                        for bits, token in top:
                            finite_hex(bits)
                            require(type(token) is int and 0 <= token < 129280, "invalid top-8 ID")
        require(data[arm]["piece-00"] == data[arm]["repeat-piece-00"], "repeat differs")
    for left, right in itertools.combinations(arms, 2):
        require(data[left] == data[right], f"numerical difference: {left}/{right}")
    print("PASS: 4 arms, 112 cases, 128 requests; all 6 pairs bit-identical; fixed 80 slots, dynamic hot off.")


if __name__ == "__main__":
    main()
