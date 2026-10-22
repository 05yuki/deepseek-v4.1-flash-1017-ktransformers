"""Tool calls through a running server's OpenAI API, end to end.

  tools/tool-call-check.py [http://127.0.0.1:8080]

Checks, each with the model card's sampling (temperature 1.0, top_p 1.0):
  plain      a request with one tool comes back with finish_reason "tool_calls" and the
             call's name and JSON arguments in message.tool_calls
  stream     the same request streamed: the deltas join into the same call
  result     the call and its result sent back: the next turn answers in content
  namespace  a tool declared with a namespace: the call carries the bare name and the
             namespace (V4.1's reference encoding; tier 0's chat encoding patch)
  thinking   thinking on (chat_template_kwargs): reasoning_content and the call both arrive

Needs the server started with --tool-call-parser deepseekv41 and --reasoning-parser
deepseek-v41 (serve.sh's defaults). Exit status 1 when a check fails.
"""

import json
import os
import sys
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:" + os.environ.get("SGLANG_PORT", "8080"))
KEY = os.environ.get("SGLANG_API_KEY")

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}
ASK = [{"role": "user", "content": "What is the weather in Tokyo right now? Use the tool."}]


def post(body, stream=False):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = "Bearer " + KEY
    body = {"model": "default", "temperature": 1.0, "top_p": 1.0, "max_tokens": 512, "stream": stream, **body}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), headers)
    r = urllib.request.urlopen(req, timeout=900)
    if not stream:
        return json.load(r)
    calls, finish, reasoning = {}, None, ""
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        choice = json.loads(line[5:])["choices"][0]
        delta = choice.get("delta", {})
        reasoning += delta.get("reasoning_content") or ""
        for tc in delta.get("tool_calls") or []:
            c = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
            fn = tc.get("function") or {}
            c["name"] += fn.get("name") or ""
            c["arguments"] += fn.get("arguments") or ""
        finish = choice.get("finish_reason") or finish
    return {"calls": [calls[i] for i in sorted(calls)], "finish": finish, "reasoning": reasoning}


results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")


def city_of(arguments):
    try:
        return json.loads(arguments).get("city", "")
    except (TypeError, ValueError):
        return None


# plain
r = post({"messages": ASK, "tools": [WEATHER]})
msg, finish = r["choices"][0]["message"], r["choices"][0]["finish_reason"]
calls = msg.get("tool_calls") or []
ok = finish == "tool_calls" and len(calls) == 1 and calls[0]["function"]["name"] == "get_weather" \
    and "tokyo" in (city_of(calls[0]["function"]["arguments"]) or "").lower()
check("plain", ok, f"finish={finish} calls={[(c['function']['name'], c['function']['arguments']) for c in calls]}")
first_call = calls[0] if calls else None

# stream
s = post({"messages": ASK, "tools": [WEATHER]}, stream=True)
ok = s["finish"] == "tool_calls" and len(s["calls"]) == 1 and s["calls"][0]["name"] == "get_weather" \
    and "tokyo" in (city_of(s["calls"][0]["arguments"]) or "").lower()
check("stream", ok, f"finish={s['finish']} calls={s['calls']}")

# result
if first_call:
    history = ASK + [
        {"role": "assistant", "content": None, "tool_calls": [first_call]},
        {"role": "tool", "tool_call_id": first_call["id"], "content": json.dumps({"city": "Tokyo", "weather": "light rain", "celsius": 14})},
    ]
    r = post({"messages": history, "tools": [WEATHER]})
    m = r["choices"][0]["message"]
    text = (m.get("content") or "")
    ok = r["choices"][0]["finish_reason"] == "stop" and not m.get("tool_calls") and ("14" in text or "rain" in text.lower())
    check("result", ok, f"finish={r['choices'][0]['finish_reason']} content={text[:120]!r}")
else:
    check("result", False, "no call from the plain check to answer")

# namespace
ns_tool = json.loads(json.dumps(WEATHER))
ns_tool["function"]["namespace"] = "weather"
r = post({"messages": ASK, "tools": [ns_tool]})
calls = r["choices"][0]["message"].get("tool_calls") or []
ok = len(calls) == 1 and calls[0]["function"]["name"] == "get_weather" and calls[0].get("namespace") == "weather"
check("namespace", ok, f"calls={[(c['function']['name'], c.get('namespace')) for c in calls]}")

# thinking: a request that needs a step of reasoning before the call. A bare "use the tool" is answered
# with an empty reasoning block, by the checkpoint's own inference code too.
PLAN = [{"role": "user", "content": "I land at Haneda at 18:00 and must decide between an outdoor dinner and an "
         "indoor one. Work out which city's weather decides this, then check it with the tool."}]
r = post({"messages": PLAN, "tools": [WEATHER], "max_tokens": 4096,
          "chat_template_kwargs": {"thinking": True}})
m = r["choices"][0]["message"]
calls = m.get("tool_calls") or []
ok = bool((m.get("reasoning_content") or "").strip()) and len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
check("thinking", ok, f"reasoning {len(m.get('reasoning_content') or '')} chars, calls={[c['function']['name'] for c in calls]}")

print(f"{sum(results)} of {len(results)} passed")
sys.exit(0 if all(results) else 1)
