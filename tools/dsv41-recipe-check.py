"""SGLang's DeepSeek-V4.1 chat prompts (rendered through DeepSeek's `deepseek-recipe` package since
sgl-project/sglang #45315) against the checkpoint's reference encoding/encoding.py, token for token.

The reference renders a prompt string; SGLang's DeepSeekRecipe renders token ids. The string is tokenized
with the checkpoint's tokenizer (special tokens parsed from the text) and the id lists are compared; on a
difference both are decoded and the first differing token shown. Groups:

  golden      every tests/test_input_N.json next to the reference (and the reference against test_output_N.txt)
  namespace   tool namespaces: a namespaced tool in the system message, calls carrying a namespace, the
              qualified "ns::name" spelling, and the inputs both sides must reject
  thinking    drop_thinking across a tool-results-only user turn

Needs the SGLang tree on sys.path (--sglang <python dir>) with deepseek-recipe installed, and the checkpoint
(--model) for its tokenizer; no GPU.

Usage: python tools/dsv41-recipe-check.py --sglang <sglang python dir> --ref <encoding.py> --model <checkpoint dir>
           [--tests <dir with test_input_N.json>] [-v]
Exit status 1 when a case differs.
"""
import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--sglang", required=True, help="the python/ dir of the SGLang tree")
ap.add_argument("--ref", required=True, help="the reference encoding/encoding.py")
ap.add_argument("--model", required=True, help="the checkpoint dir (tokenizer)")
ap.add_argument("--tests", help="golden cases (default: tests/ next to --ref)")
ap.add_argument("-v", "--verbose", action="store_true")
args = ap.parse_args()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REF = load("dsv41_reference_encoding", args.ref)
TESTS = Path(args.tests) if args.tests else Path(args.ref).parent / "tests"
sys.path.insert(0, args.sglang)
from transformers import AutoTokenizer  # noqa: E402

from sglang.srt.parser.deepseek_recipe_renderer import DeepSeekRecipe  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
recipe = DeepSeekRecipe.from_hf_tokenizer("dsv41", tok)

results = []  # (group, name, ok, detail)


def ref_ids(prompt: str):
    return tok.encode(prompt, add_special_tokens=False)


def with_ids(messages):
    """recipe's request adapter requires an id on every history tool call (the reference does not); clients
    always send one, so both sides get the same ids."""
    messages = copy.deepcopy(messages)
    for m in messages:
        for k, call in enumerate(m.get("tool_calls") or []):
            call.setdefault("id", f"call_{k}")
    return messages


def sgl_ids(messages, tools=None, thinking=False, reasoning_effort=None):
    body = {"messages": with_ids(messages), "thinking": thinking}
    if tools:
        body["tools"] = copy.deepcopy(tools)
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    return recipe.encode(body)


def outcome(fn):
    """The ids, or the error text when the input is rejected."""
    try:
        return ("ok", fn())
    except (AssertionError, ValueError) as ex:
        return ("rejected", str(ex))


def record(group, name, a, b):
    ok = a == b
    detail = ""
    if not ok:
        if a[0] == "ok" and b[0] == "ok":
            x, y = a[1], b[1]
            i = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
            detail = (f"first difference at token {i} of {len(x)} vs {len(y)}: "
                      f"ref ...{tok.decode(x[max(0, i - 8):i + 8])!r} / sgl ...{tok.decode(y[max(0, i - 8):i + 8])!r}")
        else:
            detail = f"ref {a[0]} {str(a[1])[:160]!r} / sgl {b[0]} {str(b[1])[:160]!r}"
    results.append((group, name, ok, detail))
    if args.verbose or not ok:
        print(f"{'ok  ' if ok else 'DIFF'} [{group}] {name}" + (f": {detail}" if detail else ""))


# ---------------------------------------------------------------- golden
for inp in sorted(TESTS.glob("test_input_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
    n = inp.stem.split("_")[-1]
    case = REF.load_cases(str(inp))[0]
    mode = case.get("thinking_mode") or "chat"
    case["messages"] = with_ids(case["messages"])
    ref_prompt, _ = REF.encode_case(case, thinking_mode=mode)
    golden = TESTS / f"test_output_{n}.txt"
    if golden.exists():
        record("golden", f"case {n}: reference == test_output_{n}.txt", ("ok", golden.read_text()), ("ok", ref_prompt))
    # a case that ends with an assistant message is a whole conversation, not a request: the reference
    # renders it as history, recipe as a request to continue; reported apart
    group = "golden" if case["messages"][-1]["role"] != "assistant" else "other"
    record(group, f"case {n} ({mode})", ("ok", ref_ids(ref_prompt)),
           outcome(lambda: sgl_ids(case["messages"], case.get("tools"), mode == "thinking", case.get("reasoning_effort"))))


# ---------------------------------------------------------------- namespace
def make_tool():
    return {"type": "function", "function": {"name": "lookup", "description": "Look up a value",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}}}


def make_tool_call_messages():
    return [
        {"role": "user", "content": "question"},
        {"role": "assistant", "reasoning_content": "  reason  ", "content": "summary",
         "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": '{"query":"value","limit":2}'}}]},
    ]


def both(name, messages, tools=None, thinking=False):
    """The reference's prompt for a system message carrying the tools, SGLang's for a body carrying them."""
    sys_msgs = ([{"role": "system", "content": "system", "tools": tools}] if tools else []) + messages
    r = outcome(lambda: ref_ids(REF.encode_messages(with_ids(sys_msgs), thinking_mode="thinking" if thinking else "chat")))
    s = outcome(lambda: sgl_ids(([{"role": "system", "content": "system"}] if tools else []) + messages, tools, thinking))
    record("namespace", name, r, s)


ns_tool = make_tool()
ns_tool["namespace"] = {"name": "search", "description": "Search tools."}
both("namespaced tool in the schema", [{"role": "user", "content": "question"}], [ns_tool])
both("namespaced tool, thinking", [{"role": "user", "content": "question"}], [ns_tool], thinking=True)
for location in ("tool", "function", "qualified_name"):
    messages = make_tool_call_messages()
    call = messages[1]["tool_calls"][0]
    if location == "qualified_name":
        call["function"]["name"] = "search::lookup"
    else:
        (call if location == "tool" else call["function"])["namespace"] = "search"
    # recipe requires the call's result before the next turn (the reference does not)
    messages[1]["tool_calls"][0]["id"] = "call_0"
    both(f"history call with the namespace on the {location}",
         messages + [{"role": "tool", "tool_call_id": "call_0", "content": "value"}, {"role": "user", "content": "next"}],
         [ns_tool], thinking=True)
same = []
for namespace in (None, "search", "files"):
    t = make_tool()
    if namespace is not None:
        t["namespace"] = {"name": namespace}
    same.append(t)
both("three same-named tools in three namespaces", [{"role": "user", "content": "question"}], same)
for name, ns in (("search::lookup", "files"), ("search::nested::lookup", None), ("lookup", "search::nested")):
    t = make_tool()
    t["function"]["name"] = name
    if ns is not None:
        t["namespace"] = ns
    both(f"rejects tool {name!r} ns={ns!r}", [{"role": "user", "content": "question"}], [t])

# ---------------------------------------------------------------- thinking
DROP = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "reasoning_content": "r2",
     "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "value"},
]
record("thinking", "tool results keep the current turn's reasoning",
       outcome(lambda: ref_ids(REF.encode_messages(with_ids(DROP), thinking_mode="thinking", drop_thinking=True))),
       outcome(lambda: sgl_ids(DROP, None, True)))
record("thinking", "the next user turn drops it",
       outcome(lambda: ref_ids(REF.encode_messages(with_ids(DROP) + [{"role": "user", "content": "q3"}], thinking_mode="thinking"))),
       outcome(lambda: sgl_ids(DROP + [{"role": "user", "content": "q3"}], None, True)))

# ---------------------------------------------------------------- verdict
bad = [r for r in results if not r[2]]
by_group = {}
for g, _, ok, _ in results:
    by_group.setdefault(g, [0, 0])
    by_group[g][0 if ok else 1] += 1
print("; ".join(f"{g}: {v[0]} same, {v[1]} differ" for g, v in by_group.items()))
sys.exit(1 if any(r[0] != "other" for r in bad) else 0)
