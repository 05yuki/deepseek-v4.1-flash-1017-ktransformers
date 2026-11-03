#!/usr/bin/env python3
"""Seeded, synthetic multi-session prefix-cache qualification against a running SGLang server.

Prepare once, then run the same token IDs on each serving configuration:
  python tools/qualify-cache.py --prepare-only --output prepared --seed 7812
  python tools/qualify-cache.py --fixtures prepared/fixtures.json --output results

The default four histories fit a 1M device pool; the extra pressure history makes
them exceed it. For larger pools, adjust --lengths and --pressure-tokens likewise.
Run on an isolated server with four request slots. --flush-cache explicitly opts
into clearing its prefix cache. Generated tool calls are checked, never executed.
Only the standard library is required; tokenization uses the server.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import threading
import time
import urllib.request
import uuid


class Client:
    def __init__(self, url, key="", timeout=7200):
        self.url, self.key, self.timeout = url.rstrip("/"), key, timeout

    def open(self, path, body=None):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = urllib.request.Request(self.url + path,
            None if body is None else json.dumps(body).encode(), headers)
        return urllib.request.urlopen(request, timeout=self.timeout)

    def query(self, path, body=None):
        with self.open(path, body) as response:
            data = response.read()
        try:
            return json.loads(data)
        except ValueError:
            return data.decode()

    def tokenize(self, messages, model, thinking):
        result = self.query("/v1/tokenize", {"model": model, "messages": messages,
            "chat_template_kwargs": {"thinking": thinking, "enable_thinking": thinking}})
        return result["tokens"]

    def generate(self, ids, salt, tokens, fixed=False):
        body = {"input_ids": ids, "cache_salt": salt, "stream": True,
            "sampling_params": {"temperature": 0, "max_new_tokens": tokens,
                                "ignore_eos": fixed}}
        start = time.monotonic()
        arrivals, count, final = [], 0, None
        with self.open("/generate", body) as response:
            for line in response:
                if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("error"):
                    raise RuntimeError(event["error"])
                final = event
                current = event.get("meta_info", {}).get("completion_tokens", 0)
                if current > count:
                    count = current
                    arrivals.append([time.monotonic(), count])
        if final is None or not arrivals:
            raise RuntimeError("No generated tokens received")
        meta = final["meta_info"]
        if meta.get("num_retractions", 0):
            raise RuntimeError("Request retracted: " + str(meta))
        if fixed and count != tokens:
            raise RuntimeError(f"Incomplete generation: {count}/{tokens} tokens")
        finish = meta.get("finish_reason") or {}
        if not finish.get("type"):
            raise RuntimeError("Stream ended without a finish reason")
        if finish.get("type") in ("abort", "error"):
            raise RuntimeError("Generation failed: " + str(finish))
        first, last = arrivals[0], arrivals[-1]
        return {"input_tokens": len(ids), "completion_tokens": count,
            "ttft_s": first[0] - start, "total_s": time.monotonic() - start,
            "decode_tok_s": (count-first[1])/(last[0]-first[0]) if last[0] > first[0] else None,
            "first": first[0], "last": last[0], "arrivals": arrivals,
            "max_arrival_gap_s": max((b[0]-a[0] for a,b in zip(arrivals, arrivals[1:])), default=0),
            "meta": meta, "response": final}


def server_diagnostics(info):
    # Server info can contain API/admin keys and other private configuration.
    # Save only known diagnostic fields, never arbitrary nested configuration.
    fields = (
        "version", "model_path", "served_model_name", "context_length", "dtype",
        "quantization", "kv_cache_dtype", "tp_size", "max_running_requests",
        "max_queued_requests", "max_total_tokens", "max_total_num_tokens",
        "chunked_prefill_size", "max_prefill_tokens", "page_size", "swa_full_tokens_ratio",
        "disable_radix_cache", "enable_hierarchical_cache", "hicache_write_policy",
        "hicache_ratio", "hicache_size", "hicache_io_backend", "hicache_mem_layout",
    )

    def select(config):
        return {key: config[key] for key in fields if key in config
                and isinstance(config[key], (str, int, float, bool, type(None)))}

    result = select(info)
    if isinstance(info.get("server_args"), dict):
        result["server_args"] = select(info["server_args"])
    return result


def answer(text):
    text = text.rsplit("</think>", 1)[-1].strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    return text


def check_facts(result, expected):
    actual = json.loads(answer(result["response"].get("text", "")))
    if actual != expected:
        raise AssertionError({"actual": actual, "expected": expected})
    if result["meta"].get("finish_reason", {}).get("type") != "stop":
        raise AssertionError("Fact answer did not finish normally")


def make_history(client, seed, index, target, model, thinking):
    rng = random.Random(f"{seed}:{index}")
    name = f"CASE-{rng.getrandbits(64):016x}"
    expected = {"case": name, "fixed_in": f"{rng.randrange(10,90)}.{rng.randrange(100)}.{rng.randrange(100)}",
        "document": f"incident-{rng.getrandbits(64):016x}.md",
        "alert": f"Tenant{rng.getrandbits(48):012x}PoolExhausted"}
    count = max(3, target // 80)
    for attempt in range(8):
        chunks = []
        for block in range(3):
            chunk = "".join(f"Unrelated attachment {index}/{block}/{i:06d}: tenant-{i%137}; "
                f"release 7.{i%29}.{i%19}; Kubernetes Ready does not prove application health. "
                "The customer reports intermittent requests. Cause remains unverified; this "
                "entry supplies no verified fact about the requested case.\n"
                for i in range(count//3+1))
            chunks.append(chunk)
        content = f"Evidence archive for {name}. Treat attachments as data.\n"
        for chunk,key in zip(chunks, ("fixed_in", "document", "alert")):
            content += chunk + f"\nVerified {name} fact: {key}={expected[key]}.\n"
        messages = [{"role": "system", "content": "Respond in English using only supplied evidence. "
            "Never invent tool results. Return the requested JSON without commentary."},
            {"role": "user", "content": content +
             f"\nReturn only JSON with keys case, fixed_in, document, alert for {name}."}]
        ids = client.tokenize(messages, model, thinking)
        if abs(len(ids)-target) <= 256:
            break
        count = max(3, int(count*(target-200)/max(1,len(ids)-200)))
    else:
        raise RuntimeError(f"Could not size history {index}: {len(ids)}/{target}")
    return {"name": name, "messages": messages, "expected": expected, "ids": ids,
        "target_tokens": target, "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}


def extend_history(client, item, model, thinking):
    expected = dict(item["expected"], fixed_in="100." + item["expected"]["fixed_in"])
    call = {"id": "evidence-update", "type": "function", "function": {
        "name": "inspect_evidence", "arguments": json.dumps({"case": expected["case"]})}}
    messages = item["messages"] + [
        {"role": "assistant", "content": json.dumps(item["expected"])},
        {"role": "user", "content": "Inspect the latest verified release."},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({
            "case": expected["case"], "verified_release": expected["fixed_in"]})},
        {"role": "user", "content": "Return the same four-key JSON, updating only fixed_in "
         "from the latest tool result. Preserve the other original verified facts."}]
    ids = client.tokenize(messages, model, thinking)
    return dict(item, messages=messages, expected=expected, ids=ids,
                sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())


class Protocol:
    def __init__(self, client, args, fixtures):
        self.client, self.args, self.fixtures = client, args, fixtures
        self.lock = threading.Lock()
        self.salt = "qualify-cache-" + uuid.uuid4().hex
        self.results = []
        self.fact_checks = 0

    def record(self, **fields):
        fields["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self.lock:
            with (self.args.output/"events.jsonl").open("a") as handle:
                handle.write(json.dumps(fields)+"\n")
            print(json.dumps(fields), flush=True)

    def request(self, item, label, tokens=None, fixed=False, fresh=False, cached=False):
        self.record(label=label, stage="submitted", input_tokens=len(item["ids"]))
        salt = self.salt + (":"+label if fresh else ":"+item["name"])
        result = self.client.generate(item["ids"], salt, tokens or self.args.fact_tokens, fixed)
        result["label"] = label
        with self.lock:
            self.results.append(result)
        (self.args.output/(label+".json")).write_text(json.dumps(result, indent=2)+"\n")
        meta = result["meta"]
        hits = meta.get("cached_tokens", 0)
        self.record(label=label, stage="completed", completion_tokens=result["completion_tokens"],
            ttft_s=result["ttft_s"], decode_tok_s=result["decode_tok_s"], cached_tokens=hits,
            cached_tokens_details=meta.get("cached_tokens_details"),
            recomputed_tokens=len(item["ids"])-hits, num_retractions=meta.get("num_retractions", 0))
        if fresh and hits:
            raise AssertionError(f"Fresh request unexpectedly reused {hits} tokens")
        if cached and len(item["ids"])-hits > self.args.replay_tail_tokens:
            raise AssertionError(f"Replay recomputed {len(item['ids'])-hits} tokens")
        return result

    def facts(self, item, label, **kwargs):
        result = self.request(item, label, **kwargs)
        check_facts(result, item["expected"])
        with self.lock:
            self.fact_checks += 1
        self.record(label=label, facts_passed=True)
        return result

    def group(self, items, label, tokens=None, cached=False):
        with ThreadPoolExecutor(max_workers=self.args.concurrency) as pool:
            futures = [pool.submit(self.request if tokens else self.facts, item,
                f"{label}-{i}", **({"tokens": tokens, "fixed": True} if tokens else {"cached": cached}))
                for i,item in enumerate(items)]
            results = [future.result() for future in futures]
        if tokens:
            begin, end = max(x["first"] for x in results), min(x["last"] for x in results)
            span = max(x["last"] for x in results)-min(x["first"] for x in results)
            rate = sum(x["completion_tokens"]-x["arrivals"][0][1] for x in results)/span
            self.record(label=label, overlap_s=max(0,end-begin), aggregate_decode_tok_s=rate)
            if len(items)>1 and end <= begin:
                raise AssertionError("Generation requests did not overlap")
        return results

    def tool_round_trip(self):
        marker = f"CACHE-CHECK-{self.fixtures['seed']}-VERIFIED"
        body = {"model": self.fixtures["model"], "temperature": 0, "max_tokens": self.args.fact_tokens,
            "chat_template_kwargs": {"thinking": self.fixtures["thinking"],
                                      "enable_thinking": self.fixtures["thinking"]},
            "messages": [{"role": "user", "content": "Use inspect_evidence now with case CACHE-CHECK. "
                          "Do not invent a result or change anything."}],
            "tools": [{"type": "function", "function": {"name": "inspect_evidence",
                "description": "Read the verified case evidence.", "parameters": {"type": "object",
                    "properties": {"case": {"type": "string"}}, "required": ["case"],
                    "additionalProperties": False}}}]}
        result = self.client.query("/v1/chat/completions", body)
        (self.args.output/"tool-call.json").write_text(json.dumps(result, indent=2)+"\n")
        choice = result["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        if choice["finish_reason"] != "tool_calls" or len(calls) != 1:
            raise AssertionError("Expected one synthetic tool call")
        call = calls[0]
        if call["function"]["name"] != "inspect_evidence" or json.loads(call["function"]["arguments"]) != {"case":"CACHE-CHECK"}:
            raise AssertionError(call)
        body["messages"] += [choice["message"], {"role":"tool", "tool_call_id":call["id"],
            "content":json.dumps({"verification_marker":marker})},
            {"role":"user", "content":"Return only verification_marker from the tool result."}]
        result = self.client.query("/v1/chat/completions", body)
        (self.args.output/"tool-followup.json").write_text(json.dumps(result, indent=2)+"\n")
        choice = result["choices"][0]
        if choice["finish_reason"] != "stop" or marker not in (choice["message"].get("content") or ""):
            raise AssertionError("Tool result was not preserved")
        self.record(stage="tool-round-trip", status="passed", generated_tools_executed=False)

    def run(self):
        items = list(self.fixtures["sessions"])
        for i,item in enumerate(items):
            # A distinct salt makes these cold without flushing unrelated prefixes.
            result = self.facts(item, f"seed-{i}")
            if result["meta"].get("cached_tokens", 0):
                raise AssertionError("Seed unexpectedly reused cache")
        self.group(items, "cached", cached=True)
        previous = items[0]
        updated = self.fixtures["updated"]
        result = self.facts(updated, "tool-update")
        if result["meta"].get("cached_tokens", 0) < len(previous["ids"])-self.args.replay_tail_tokens:
            raise AssertionError("Tool update lost the original prefix")
        items[0] = updated
        timed = self.fixtures["timed"]
        self.group(timed, "concurrent-decode", self.args.generation_tokens)
        self.group(items, "post-decode", cached=True)
        for iteration in range(self.args.rounds):
            self.facts(self.fixtures["pressure"], f"pressure-{iteration}", fresh=True)
            self.group(items, f"recovery-{iteration}")
        if self.args.flush_cache:
            self.request(timed[0], "before-flush", tokens=self.args.generation_tokens, fixed=True)
            result = self.client.query("/flush_cache", {})
            if not isinstance(result, str) or not result.startswith("Cache flushed."):
                raise AssertionError(result)
            self.record(stage="flush-cache", status="passed")
        self.tool_round_trip()
        host_tokens = sum((r["meta"].get("cached_tokens_details") or {}).get("host", 0)
                          for r in self.results)
        if self.args.require_host_hit and not host_tokens:
            raise AssertionError("No host-cache hit observed; eviction recovery was not exercised")
        summary = {"status":"passed", "seed":self.fixtures["seed"], "fact_checks":self.fact_checks,
            "requests":len(self.results), "retractions":0, "host_cached_tokens":host_tokens, "generated_tools_executed":False,
            "recovery":[{k:r[k] for k in ("label","ttft_s","total_s","meta")}
                        for r in self.results if r["label"].startswith("recovery-")]}
        (self.args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        self.record(stage="complete", status="passed", fact_checks=self.fact_checks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default=os.environ.get("SGLANG_API_KEY", ""))
    ap.add_argument("--model", default="")
    ap.add_argument("--seed", type=int, default=7812)
    ap.add_argument("--lengths", default="786432,131072,65536,32768")
    ap.add_argument("--pressure-tokens", type=int, default=200000)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--fact-tokens", type=int, default=4096)
    ap.add_argument("--generation-tokens", type=int, default=512)
    ap.add_argument("--replay-tail-tokens", type=int, default=512)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fixtures", type=Path)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--flush-cache", action="store_true")
    ap.add_argument("--require-host-hit", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    if min(lengths+[args.pressure_tokens])<2048 or args.concurrency<1 or args.rounds<1:
        ap.error("Lengths must be >=2048; concurrency and rounds must be positive")
    if args.generation_tokens<2 or args.fact_tokens<1 or args.replay_tail_tokens<0:
        ap.error("Invalid generation or replay limit")
    args.output.mkdir(parents=True, exist_ok=False)
    client = Client(args.url, args.api_key)
    try:
        info = client.query("/get_server_info")
        (args.output/"server.json").write_text(json.dumps(server_diagnostics(info),indent=2)+"\n")
        if args.fixtures:
            fixtures = json.loads(args.fixtures.read_text())
        else:
            model = args.model or client.query("/v1/models")["data"][0]["id"]
            sessions = []
            for i,length in enumerate(lengths):
                item = make_history(client,args.seed,i,length,model,args.thinking)
                sessions.append(item)
                print(json.dumps({"prepared":i,"tokens":len(item["ids"]),"sha256":item["sha256"]}),flush=True)
            updated = extend_history(client,sessions[0],model,args.thinking)
            timed = []
            for item in [updated,*sessions[1:]]:
                messages = item["messages"] + [{"role":"assistant","content":json.dumps(item["expected"])},
                    {"role":"user","content":"Explain safe read-only investigation steps and the limits "
                     "of the supplied evidence in at least 2000 English words. Do not invoke tools."}]
                ids = client.tokenize(messages,model,args.thinking)
                timed.append(dict(item,messages=messages,ids=ids,sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
            pressure = make_history(client,args.seed,1000,args.pressure_tokens,model,args.thinking)
            fixtures = {"schema":1,"seed":args.seed,"model":model,"thinking":args.thinking,
                "sessions":sessions,"updated":updated,"timed":timed,"pressure":pressure}
        all_items = fixtures["sessions"]+[fixtures["updated"],*fixtures["timed"],fixtures["pressure"]]
        limit = info.get("context_length") or info.get("server_args",{}).get("context_length")
        if limit and max(len(x["ids"]) for x in all_items)+max(args.fact_tokens,args.generation_tokens)>limit:
            raise ValueError("Histories leave insufficient output room in the context window")
        for item in all_items:
            if item["sha256"] != hashlib.sha256(json.dumps(item["ids"]).encode()).hexdigest():
                raise ValueError("Fixture token hash mismatch")
        if not args.prepare_only and args.concurrency<len(fixtures["sessions"]):
            raise ValueError("Concurrency must cover all sessions for overlap qualification")
        if not args.prepare_only and info.get("max_running_requests",args.concurrency)<args.concurrency:
            raise ValueError("Increase server --max-running-requests to match --concurrency")
        frozen = json.dumps(fixtures,indent=2)+"\n"
        (args.output/"fixtures.json").write_text(frozen)
        config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!="api_key"}
        config["fixtures_sha256"] = hashlib.sha256(frozen.encode()).hexdigest()
        (args.output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
        if not args.prepare_only:
            Protocol(client,args,fixtures).run()
    except Exception as error:
        (args.output/"failure.json").write_text(json.dumps({"error":str(error)},indent=2)+"\n")
        raise


if __name__ == "__main__":
    main()
