#!/usr/bin/env python3
"""The README's numbers against a running server (./serve.sh): decode on four Japanese prose
prompts, then the prefill of a 35,000-token prompt, twice.

  python tools/bench.py [--url http://127.0.0.1:8080] [--api-key KEY]
  python tools/bench.py --prefill-only --lengths 35000,131072,524288,1000000   (about an hour)

- decode: each prompt asks for a story of about 2,000 characters; 1,024 tokens are generated
  at temperature 1.0, top_p 0.95. The tok/s of a request is its completion tokens over its
  time, so it includes the short prompt's prefill; the median of the four is printed.
- prefill: 35,000 token ids drawn at random (fixed seed) from the vocabulary, one token
  generated; then 35,000 other ids. The first long prompt of a launch can carry one-time
  setup, so both are printed. --lengths gives other prompt lengths, each its own random
  ids (no prefix shared with an earlier prompt), up to the 1M-token context.

Only the standard library is used. The server's own log has the decode rate per step too
("gen throughput").
"""
import argparse
import json
import random
import statistics
import time
import urllib.request

PROMPTS = [
    "夜のダンジョンの野営地を舞台に、2000字の場面を書いて。",
    "大正時代の東京、雨の夜に古書店を訪れた女学生と店主の会話を中心に、2000字の短編小説を書いてください。",
    "宇宙港の整備士の女性が、引退する老パイロットの最後の飛行を見送る場面を、一人称で2000字ほどで書いてください。",
    "山奥の温泉宿に一人で泊まりに来た作家が、宿の女将の秘密に気づいていく導入部分を、情景描写を多めに2000字で書いてください。",
]


def post(url, body, key):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    return json.load(urllib.request.urlopen(req, timeout=7200))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--prefill-tokens", type=int, default=35000)
    ap.add_argument("--lengths", default="", help="prompt lengths to prefill once each, comma-separated")
    ap.add_argument("--prefill-only", action="store_true")
    ap.add_argument("--vocab", type=int, default=129280, help="token ids are drawn from 1000..vocab")
    a = ap.parse_args()

    rates = []
    for i, p in enumerate([] if a.prefill_only else PROMPTS):
        body = {"model": "x", "max_tokens": 1024, "temperature": 1.0, "top_p": 0.95,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": p}]}
        t = time.time()
        r = post(a.url + "/v1/chat/completions", body, a.api_key)
        dt = time.time() - t
        n = r["usage"]["completion_tokens"]
        rates.append(n / dt)
        print(f"decode p{i}: {n} tokens in {dt:.1f} s = {n / dt:.1f} tok/s", flush=True)
    if rates:
        print(f"decode median: {statistics.median(rates):.1f} tok/s", flush=True)

    runs = [("first", 0, a.prefill_tokens), ("second", 1, a.prefill_tokens)]
    if a.lengths:
        runs = [(f"#{i + 1}", 100 + i, int(n)) for i, n in enumerate(a.lengths.split(","))]
    for name, seed, length in runs:
        rng = random.Random(seed)
        ids = [rng.randrange(1000, a.vocab) for _ in range(length)]
        t = time.time()
        post(a.url + "/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}},
             a.api_key)
        dt = time.time() - t
        print(f"prefill, {name} {length}-token prompt: {dt:.2f} s = {length / dt:.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
