"""Per-token log-probabilities of the running server on a fixed text, and how two runs compare.

  python tools/nll.py score <model dir> <out.json>        the server on $URL (default http://127.0.0.1:8080)
  python tools/nll.py compare <a.json> <b.json> [...]     each against the first

The text is tools/nll_pieces.py's: 24 pieces of 480 tokens from two public-domain texts (Japanese and
English), scored teacher-forced through /generate with input_ids (no chat template). tools/ref-nll.py
scores the same pieces with the checkpoint's own reference implementation (inference/model.py), so

  python tools/nll.py compare ref.json served.json

says how far the server is from the reference: the mean NLL of each, the mean difference with its standard
error taken over the 24 pieces (tokens of one piece are not independent), and the mean absolute difference
per token. Two launches of the same build may differ by a few thousandths of a nat when the hot experts
differ (what runs on the GPU and what on the CPU rounds differently); the standard error says how much a
difference means.
"""

import json
import math
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nll_pieces  # noqa: E402

URL = os.environ.get("URL", "http://127.0.0.1:8080")
KEY = os.environ.get("SGLANG_API_KEY", "")


def score(model_dir, out):
    tok = nll_pieces.tokenizer(model_dir)
    result = []
    for ids in nll_pieces.pieces(tok):
        body = json.dumps({"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                           "return_logprob": True, "logprob_start_len": 0}).encode()
        headers = {"Content-Type": "application/json"}
        if KEY:
            headers["Authorization"] = f"Bearer {KEY}"
        r = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/generate", body, headers), timeout=1800))
        lp = r["meta_info"]["input_token_logprobs"]
        result.append([t[0] for t in lp[1:len(ids)]])
    json.dump(result, open(out, "w"))
    n = sum(len(x) for x in result)
    print("scored %d tokens in %d pieces, mean NLL %.4f" % (n, len(result), -sum(sum(x) for x in result) / n))


def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    names = [os.path.basename(p) for p in paths]
    for name, run in zip(names, runs):
        n = sum(len(x) for x in run)
        print("%-24s mean NLL %.4f over %d tokens" % (name, -sum(sum(x) for x in run) / n, n))
    base = runs[0]
    for name, run in zip(names[1:], runs[1:]):
        assert [len(x) for x in run] == [len(x) for x in base], f"{name}: not the same pieces"
        per_piece = [-(sum(b) - sum(a)) / len(a) for a, b in zip(base, run)]  # NLL(run) - NLL(base)
        k = len(per_piece)
        m = sum(per_piece) / k
        se = math.sqrt(sum((x - m) ** 2 for x in per_piece) / (k - 1) / k)
        absd = [abs(x - y) for a, b in zip(base, run) for x, y in zip(a, b)]
        same = sum(x == 0 for x in absd)
        z = "" if se == 0 else " (z %+.2f)" % (m / se)
        print("%-24s minus %s: %+.4f nats a token, SE %.4f over %d pieces%s; mean |diff| a token %.4f; "
              "%d of %d tokens identical" % (name, names[0], m, se, k, z, sum(absd) / len(absd), same, len(absd)))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        compare(sys.argv[2:])
    else:
        sys.exit(__doc__)
