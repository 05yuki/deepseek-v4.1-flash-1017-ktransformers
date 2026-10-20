"""SGLang's DeepSeek-V4.1 chat encoding against DeepSeek's reference encoding/encoding.py, byte for byte.

Imports both encoders by file path (SGLang's encoding_dsv41.py needs only the standard library) and,
for each case, compares what the two return: the prompt string, the image records, or the error
text when both reject the input (the reference asserts, SGLang raises ValueError; the message is
what is compared). Groups:

  golden      every tests/test_input_N.json against the reference (and the reference against
              test_output_N.txt)
  namespace   the tool-namespace scenarios of the reference's test_encoding.py (dba1be0a40aa)
  thinking    drop_thinking across a tool-results-only user turn (314113f8a7da)
  image-order tool-result image order, four input shapes (314113f8a7da)
  other       the rest of test_encoding.py's prompt checks; reported, not part of the verdict

The string effort tiers are a separate axis: SGLang trees before upstream #39929 map low/high to
25/50 where the reference has 50/75, so every thinking-mode prompt with the default effort differs
in its first line. golden and thinking therefore pass the reference's default as a number when a
case leaves the effort unset (the reference renders that byte-identically to its default, which
the golden check against test_output_N.txt confirms); "other" repeats those cases with the effort
unset and the tier mapping itself.

--detector also imports SGLang itself (pydantic, partial_json_parser; no torch) and checks the
request and response path: tools and history tool calls through the protocol models, the V4.1
detector's non-streaming and streaming parse of namespaced calls against the reference parser,
and that a ToolCall without a namespace serializes as before.

Usage: python tools/dsv41-encoding-check.py --sglang <sglang python dir> --ref <encoding.py>
           [--tests <dir with test_input_N.json>] [--detector] [-v]
Exit status 1 when a case outside "other" differs.
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
ap.add_argument("--tests", help="golden cases (default: tests/ next to --ref)")
ap.add_argument("--detector", action="store_true")
ap.add_argument("-v", "--verbose", action="store_true")
args = ap.parse_args()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SGL_DIR = Path(args.sglang)
REF = load("dsv41_reference_encoding", args.ref)
SGL = load("dsv41_sglang_encoding", SGL_DIR / "sglang/srt/entrypoints/openai/encoding_dsv41.py")
TESTS = Path(args.tests) if args.tests else Path(args.ref).parent / "tests"
IMG = REF.IMAGE_PLACEHOLDER
# the reference's default effort as a number (see the docstring)
EFF = REF.REASONING_EFFORT_MAPPINGS[REF.DEFAULT_REASONING_EFFORT]

results = []  # (group, name, same, detail)


def outcome(fn):
    try:
        return ("ok", fn())
    except (AssertionError, ValueError, KeyError, NotImplementedError) as e:
        return ("error", str(e))


def short(x, n=160):
    s = repr(x)
    return s if len(s) <= n else s[:n] + "..."


def images_by_url(x):
    """Image records compared by url and order: SGLang trees from upstream #44751 on record
    {"url": ...} where the reference has {"type": "image", "url": ...}; the prompt is unchanged."""
    if isinstance(x, dict) and isinstance(x.get("images"), list):
        return {**x, "images": [i.get("url") if isinstance(i, dict) else i for i in x["images"]]}
    if isinstance(x, (tuple, list)):
        return type(x)(images_by_url(v) for v in x)
    return x


def record(group, name, a, b):
    same = images_by_url(a) == images_by_url(b)
    detail = "" if same else f"ref={short(a)}\n        sgl={short(b)}"
    results.append((group, name, same, detail))


def compare(group, name, fn):
    """fn(module) runs against both encoders."""
    record(group, name, outcome(lambda: fn(REF)), outcome(lambda: fn(SGL)))


# ---------------------------------------------------------------- fixtures (test_encoding.py)
V41_TOOL_CALL_OUTPUT = (
    "  reason  </think>summary\n\n"
    "<｜DSML｜ calls>\n"
    '<｜DSML｜ invoke name="lookup">\n'
    '<｜DSML｜ parameter name="query" string="true">value</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="limit" string="false">2</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n"
    "</｜DSML｜ calls><｜end▁of▁sentence｜>"
)


def make_tool():
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            },
        },
    }


def make_tool_call_messages():
    return [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "reasoning_content": "  reason  ",
            "content": "summary",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query":"value","limit":2}'},
                }
            ],
        },
    ]


def namespaced_call_messages(location):
    messages = make_tool_call_messages()
    call = messages[1]["tool_calls"][0]
    if location == "qualified_name":
        call["function"]["name"] = "search::lookup"
    else:
        (call if location == "tool" else call["function"])["namespace"] = "search"
    return messages


def same_named_case():
    tools, calls = [], []
    for namespace in (None, "search", "files"):
        tool = make_tool()
        call = {"type": "function", "function": {"name": "lookup", "arguments": '{"query":"value"}'}}
        if namespace is not None:
            tool["namespace"] = {"name": namespace}
            call["namespace"] = namespace
        tools.append(tool)
        calls.append(call)
    return [
        {"role": "system", "content": "system", "tools": tools},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "summary", "tool_calls": calls},
    ]


def unchanged(fn, *inputs):
    """fn's result, plus whether it left its inputs as they were."""
    before = copy.deepcopy(inputs)
    out = fn(*inputs)
    return out, inputs == before


# ---------------------------------------------------------------- golden
for inp in sorted(TESTS.glob("test_input_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
    n = inp.stem.split("_")[-1]
    case = REF.load_cases(str(inp))[0]
    golden = (TESTS / f"test_output_{n}.txt").read_text()
    ref_prompt, _ = REF.encode_case(case, thinking_mode="chat")
    record("golden", f"case {n}: reference == test_output_{n}.txt", golden, ref_prompt)

    def enc(m, case=case, default=EFF):
        effort = case.get("reasoning_effort")
        return m.encode_messages(
            copy.deepcopy(case["messages"]),
            thinking_mode=case.get("thinking_mode") or "chat",
            context=copy.deepcopy(case.get("context")),
            reasoning_effort=default if effort is None else effort,
            return_multi_modal_data=True,
        )

    compare("golden", f"case {n}", enc)
    if case.get("reasoning_effort") is None:
        compare("other", f"golden case {n}, effort unset", lambda m, enc=enc: enc(m, default=None))

# ---------------------------------------------------------------- namespace
for location in ("tool", "function"):
    for ns in ("search", {"name": "search", "description": "Search tools."}):
        def schema(m, location=location, ns=ns):
            tool = make_tool()
            (tool if location == "tool" else tool["function"])["namespace"] = copy.deepcopy(ns)
            return unchanged(
                lambda t: m.encode_messages(
                    [{"role": "system", "content": "system", "tools": [t]}], thinking_mode="chat"
                ),
                tool,
            )

        compare("namespace", f"schema namespace on {location}, {type(ns).__name__}", schema)

for mode in ("chat", "thinking"):
    for location in ("tool", "function", "qualified_name"):
        compare(
            "namespace",
            f"history call render, {mode}, {location}",
            lambda m, mode=mode, location=location: unchanged(
                lambda msgs: m.render_message(1, msgs, thinking_mode=mode),
                namespaced_call_messages(location),
            ),
        )

        # the reference parses its own completion; both encoders re-encode the parsed message
        def reencode(m, mode=mode, location=location):
            messages = namespaced_call_messages(location)
            completion = REF.render_message(1, messages, thinking_mode=mode)
            parsed = REF.parse_message_from_completion_text(completion, thinking_mode=mode)
            return m.encode_messages([parsed], thinking_mode=mode, context=messages[:1])

        compare("namespace", f"parsed call re-encoded, {mode}, {location}", reencode)

compare("namespace", "same-named tools: prompt",
        lambda m: m.encode_messages(same_named_case(), thinking_mode="chat"))
compare("namespace", "same-named tools: assistant render",
        lambda m: m.render_message(2, same_named_case(), thinking_mode="chat"))


def qualified_tool(m):
    tool = make_tool()
    tool["function"]["name"] = "search::lookup"
    tool["namespace"] = {"name": "search", "description": "Search tools."}
    return m.tools_from_openai_format([tool])


compare("namespace", "qualified name + namespace not doubled: schema", qualified_tool)


def qualified_call(m):
    messages = make_tool_call_messages()
    call = messages[1]["tool_calls"][0]
    call["function"]["name"] = "search::lookup"
    call["namespace"] = "search"
    return m.render_message(1, messages, thinking_mode="thinking")


compare("namespace", "qualified name + namespace not doubled: call", qualified_call)

for name, ns in (("search::lookup", "files"), ("search::nested::lookup", None), ("lookup", "search::nested")):
    def bad_tool(m, name=name, ns=ns):
        tool = make_tool()
        tool["function"]["name"] = name
        tool["namespace"] = ns
        return m.tools_from_openai_format([tool])

    def bad_call(m, name=name, ns=ns):
        messages = make_tool_call_messages()
        call = messages[1]["tool_calls"][0]
        call["function"]["name"] = name
        call["namespace"] = ns
        return m.render_message(1, messages, thinking_mode="thinking")

    compare("namespace", f"rejects tool {name!r} ns={ns!r}", bad_tool)
    compare("namespace", f"rejects call {name!r} ns={ns!r}", bad_call)

# ---------------------------------------------------------------- thinking
DROP = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
    {"role": "user", "content": "q2"},
    {
        "role": "assistant",
        "reasoning_content": "r2",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "value"},
]
compare("thinking", "tool results keep the current turn's reasoning",
        lambda m: m.encode_messages(copy.deepcopy(DROP), thinking_mode="thinking", drop_thinking=True,
                                    reasoning_effort=EFF))
compare("thinking", "the next user turn drops it",
        lambda m: m.encode_messages(copy.deepcopy(DROP) + [{"role": "user", "content": "q3"}],
                                    thinking_mode="thinking", reasoning_effort=EFF))
compare("thinking", "same, results passed as a later call's context",
        lambda m: m.encode_messages(copy.deepcopy(DROP[4:]), thinking_mode="thinking",
                                    context=copy.deepcopy(DROP[:4]), reasoning_effort=EFF))
compare("thinking", "mixed text + tool result still starts a turn",
        lambda m: m.encode_messages(
            copy.deepcopy(DROP[:4]) + [{"role": "user", "content_blocks": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "value"},
                {"type": "text", "text": "and q3"}]}],
            thinking_mode="thinking", reasoning_effort=EFF))
compare("other", "drop_thinking across tool results, effort unset",
        lambda m: m.encode_messages(copy.deepcopy(DROP), thinking_mode="thinking"))


# ---------------------------------------------------------------- image-order
def image_order(m, input_format):
    history = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "history.png"}}]},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": n, "type": "function", "function": {"name": n, "arguments": "{}"}} for n in ("a", "b")
            ],
        },
    ]
    results_ = [
        {
            "type": "tool_result",
            "tool_use_id": n,
            "content": [
                {"type": "text", "text": f"result {n}"},
                {"type": "image_url", "image_url": {"url": f"{n}.png"}},
            ],
        }
        for n in ("b", "a")
    ]
    if input_format in ("tool_messages", "tool_content_blocks"):
        field = "content_blocks" if input_format == "tool_content_blocks" else "content"
        messages = history + [
            {"role": "tool", "tool_call_id": b["tool_use_id"], field: b["content"]} for b in results_
        ]
        context = []
    elif input_format == "context":
        messages, context = [{"role": "user", "content_blocks": results_}], history
    else:
        messages, context = history + [{"role": "user", "content_blocks": results_}], []
    return unchanged(
        lambda ms, cx: m.encode_messages(ms, thinking_mode="chat", context=cx, return_multi_modal_data=True),
        messages, context,
    )


for fmt in ("content_blocks", "tool_messages", "tool_content_blocks", "context"):
    compare("image-order", f"tool-result images, {fmt}", lambda m, fmt=fmt: image_order(m, fmt))

# ---------------------------------------------------------------- other (informational)
Q = [{"role": "user", "content": "question"}]
for effort in (None, "low", "high", "max", 1, 42, 100, -1, 0, 101, "medium", True, 1.5):
    compare("other", f"reasoning_effort={effort!r}",
            lambda m, e=effort: m.encode_messages(copy.deepcopy(Q), thinking_mode="thinking", reasoning_effort=e))
compare("other", "chat mode ignores effort",
        lambda m: m.encode_messages([{"role": "user", "content": "hello"}], thinking_mode="chat", reasoning_effort="max"))
compare("other", "leading system", lambda m: m.encode_messages(
    [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "hello"}],
    thinking_mode="chat"))
compare("other", "mid-conversation system", lambda m: m.encode_messages(
    [{"role": "system", "content": "sys"}, {"role": "user", "content": "q1"},
     {"role": "assistant", "content": "a1", "reasoning_content": "r1"}, {"role": "system", "content": "mid sys"}],
    thinking_mode="thinking", reasoning_effort=88))
compare("other", "tools prompt, chat", lambda m: m.encode_messages(
    [{"role": "system", "content": "system", "tools": [make_tool()]}, {"role": "user", "content": "question"}],
    thinking_mode="chat"))
compare("other", "assistant tool call render",
        lambda m: m.render_message(1, make_tool_call_messages(), thinking_mode="thinking"))
compare("other", "parsed completion re-encoded", lambda m: m.encode_messages(
    [REF.parse_message_from_completion_text(V41_TOOL_CALL_OUTPUT, thinking_mode="thinking")],
    thinking_mode="thinking", context=make_tool_call_messages()[:1]))
compare("other", "merge_tool_messages", lambda m: m.merge_tool_messages([
    {"role": "assistant", "content": "", "tool_calls": []},
    {"role": "tool", "tool_call_id": "a", "content": "r1"},
    {"role": "tool", "tool_call_id": "b", "content": "r2"}]))
compare("other", "task sp token", lambda m: m.encode_messages(
    [{"role": "user", "content": "classify me", "task": "query"}], thinking_mode="chat"))
compare("other", "image block", lambda m: m.encode_messages(
    [{"role": "user", "content": [{"type": "text", "text": "inspect"},
                                  {"type": "image_url", "image_url": {"url": "/unused/image.png"}}]}],
    thinking_mode="chat", return_multi_modal_data=True))
for role in ("user", "tool"):
    compare("other", f"rejects placeholder in {role} text", lambda m, r=role: m.encode_messages(
        [{"role": r, "content": f"hi {IMG}"}], thinking_mode="chat"))


# ---------------------------------------------------------------- detector / protocol (SGLang import)
def detector_checks():
    sys.path.insert(0, str(SGL_DIR))
    from sglang.srt.entrypoints.openai import chat_encoding, encoding_dsv41
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionMessageGenericParam, FunctionResponse, Tool, ToolCall)
    from sglang.srt.function_call.deepseekv41_detector import DeepSeekV41Detector

    def ref_calls(completion, mode):
        parsed = REF.parse_message_from_completion_text(completion, thinking_mode=mode)
        return [(c.get("namespace"), c["function"]["name"], json.loads(c["function"]["arguments"]))
                for c in parsed["tool_calls"]]

    def body(completion):
        """What the detector sees: the reasoning parser has taken the think part, EOS is a stop token."""
        return completion.split("</think>", 1)[-1].removesuffix(REF.eos_token)

    def sgl_calls(text, tools):
        res = DeepSeekV41Detector().detect_and_parse(text, tools)
        out = []
        for c in res.calls:
            tc = ToolCall(id="x", index=c.tool_index, namespace=getattr(c, "namespace", None),
                          function=FunctionResponse(name=c.name, arguments=c.parameters)).model_dump()
            out.append((tc.get("namespace"), tc["function"]["name"], json.loads(tc["function"]["arguments"])))
        return out

    def sgl_stream(text, tools, step=5):
        det, calls = DeepSeekV41Detector(), {}
        for i in range(0, len(text), step):
            for c in det.parse_streaming_increment(text[i:i + step], tools).calls:
                d = calls.setdefault(c.tool_index, {"ns": None, "name": None, "args": ""})
                if c.name:
                    d["name"], d["ns"] = c.name, getattr(c, "namespace", None)
                d["args"] += c.parameters or ""
        return [(d["ns"], d["name"], json.loads(d["args"])) for _, d in sorted(calls.items())]

    def tools_for(location):
        tool = make_tool()
        if location == "qualified_name":
            tool["function"]["name"] = "search::lookup"
        else:
            (tool if location == "tool" else tool["function"])["namespace"] = "search"
        return [Tool(**tool)]

    for mode in ("chat", "thinking"):
        for location in ("tool", "function", "qualified_name"):
            completion = REF.render_message(1, namespaced_call_messages(location), thinking_mode=mode)
            want = outcome(lambda: ref_calls(completion, mode))
            record("detector", f"parse, {mode}, tools ns on {location}", want,
                   outcome(lambda: sgl_calls(body(completion), tools_for(location))))
            record("detector", f"stream, {mode}, tools ns on {location}", want,
                   outcome(lambda: sgl_stream(body(completion), tools_for(location))))

    case = same_named_case()
    completion = REF.render_message(2, case, thinking_mode="chat")
    tools = [Tool(**t) for t in case[0]["tools"]]
    want = outcome(lambda: ref_calls(completion, "chat"))
    record("detector", "parse, same-named tools in 3 namespaces", want, outcome(lambda: sgl_calls(body(completion), tools)))
    record("detector", "stream, same-named tools in 3 namespaces", want, outcome(lambda: sgl_stream(body(completion), tools)))
    record("detector", "tool_index of the 3 same-named calls", [0, 1, 2],
           outcome(lambda: [c.tool_index for c in DeepSeekV41Detector().detect_and_parse(body(completion), tools).calls])[1])
    plain = REF.render_message(1, make_tool_call_messages(), thinking_mode="chat")
    record("detector", "parse, plain tool (no namespace)", outcome(lambda: ref_calls(plain, "chat")),
           outcome(lambda: sgl_calls(body(plain), [Tool(**make_tool())])))

    # request path: tools through the protocol model and dsv41_tool_payload, as serving_chat does
    for location in ("tool", "function"):
        for ns in ("search", {"name": "search", "description": "Search tools."}):
            tool = make_tool()
            (tool if location == "tool" else tool["function"])["namespace"] = ns
            ref = outcome(lambda: REF.encode_messages(
                [{"role": "system", "content": "system", "tools": [copy.deepcopy(tool)]}], thinking_mode="chat"))
            sgl = outcome(lambda: encoding_dsv41.encode_messages(
                [{"role": "system", "content": "system", "tools": [chat_encoding.dsv41_tool_payload(Tool(**tool))]}],
                thinking_mode="chat"))
            record("detector", f"request tool, ns on {location}, {type(ns).__name__}", ref, sgl)
    # history: an assistant message with a namespaced call through the protocol model
    for location in ("tool", "function", "qualified_name"):
        messages = namespaced_call_messages(location)
        ref = outcome(lambda: REF.render_message(1, messages, thinking_mode="thinking"))
        dumped = [messages[0], ChatCompletionMessageGenericParam(**messages[1]).model_dump()]
        record("detector", f"request history call, ns on {location}", ref,
               outcome(lambda: encoding_dsv41.render_message(1, dumped, thinking_mode="thinking")))
    # serialization stays OpenAI-shaped without a namespace
    record("detector", "ToolCall without namespace serializes as before",
           {"id": "x", "index": 0, "type": "function", "function": {"name": "f", "arguments": "{}"}},
           ToolCall(id="x", index=0, function=FunctionResponse(name="f", arguments="{}")).model_dump())
    record("detector", "Tool without namespace dumps as before", False,
           "namespace" in json.dumps(Tool(**make_tool()).model_dump()))


if args.detector:
    detector_checks()

# ---------------------------------------------------------------- report
order = ["golden", "namespace", "thinking", "image-order", "detector", "other"]
groups = [g for g in order if any(r[0] == g for r in results)]
print(f"sglang: {SGL_DIR}\nreference: {args.ref}")
bad = 0
for g in groups:
    rows = [r for r in results if r[0] == g]
    same = sum(r[2] for r in rows)
    print(f"{g:12s} identical {same:3d}  differ {len(rows) - same:3d}")
    for _, name, ok, detail in rows:
        if not ok or args.verbose:
            print(f"    {'same  ' if ok else 'DIFFER'} {name}" + (f"\n        {detail}" if detail else ""))
    if g != "other":
        bad += len(rows) - same
total = [r for r in results if r[0] != "other"]
print(f"verdict (all but other): identical {len(total) - bad}  differ {bad}")
sys.exit(1 if bad else 0)
