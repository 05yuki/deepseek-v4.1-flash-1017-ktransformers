# Serving

The server speaks SGLang's OpenAI-compatible API on `127.0.0.1:8080`, one request at a
time by default. `SGLANG_MAX_RUNNING_REQUESTS=4` runs up to four at once.

- Four streams of Japanese prose decode at 74 tok/s together (18.5 each), against 42.7 for one.
  Their routes gather on the same experts.
- Four streams of different kinds of text (Japanese, Chinese, JSON, Python) stay at 47 tok/s together.
  The CPU reads every expert any of them routes to.
- A long prompt holds the others while it prefills: the prefill lends the GPU's weights and free KV
  pages to itself. Three running answers waited 62 s for a 131K-token prompt.

The prefix cache is on, with a copy in host memory (SGLang's HiCache).
A prompt that starts like an earlier one reads that part from the cache.
The same 35K-token prompt again takes 2 s instead of 20; a 900K-token one 5-17 s instead of 855.
`DSV41_RADIX=0` turns the cache off ([tier1.md](tier1.md#21-hicache)).

To serve other machines, set `SGLANG_HOST=0.0.0.0` and `SGLANG_API_KEY=<key>`
(clients then send `Authorization: Bearer <key>`). The same setup also serves the earlier
DeepSeek-V4.1-Flash release: point `MODEL` at it.

## Tool calls and thinking

`serve.sh` starts the server with V4.1's tool-call parser and reasoning parser
(`--tool-call-parser deepseekv41 --reasoning-parser deepseek-v41`), so the OpenAI API returns
a tool call in `message.tool_calls` and the model's thinking in `reasoning_content`, streamed
or not. `SGLANG_TOOL_CALL_PARSER=` or `SGLANG_REASONING_PARSER=` (empty) leaves one out.

- Thinking is off unless a request turns it on:
  `"chat_template_kwargs": {"thinking": true}`.
- The prompt is rendered by SGLang main's `deepseek-recipe` package, which differs from the
  checkpoint's `encoding/encoding.py` in three places, tool namespaces among them
  ([patches.md](patches.md#chat-encoding-dropped)).
- `tools/tool-call-check.py [URL]` checks a running server: a call, the same call streamed,
  the answer after the tool's result, and a call with thinking on (4 of 4 pass); a namespaced
  tool is reported as the known gap above.
