# model_client.py notes

`model_client.py` is a Python 3 stdlib-only client for the historical SM89
portable-versus-dual validation at recipe `8719948`/`a9ce91c`. Those arms each
passed its 76 checks under the six-node, 64K serving profile; see
[SM89 validation](../../docs/sm89.md) for that configuration. Those results
do not describe the current `00fd12a` full tier-1 profile. Canonical A-r8
instead ran the exact upstream `check.sh`, tool-call, reference-NLL and bench
tools, plus three **sequential** retrieval checks through this client's
`check_retrieval` path. The [current evidence](../../docs/benchmarks/sm89-00fd12a-20261024.json)
records its one-GPU, eight-node profile and output. This older 76-check suite
was not rerun in A-r8.

## Usage

```
python3 model_client.py --arm portable --output OUT_PORTABLE   # defaults: --base-url http://127.0.0.1:18209 --model sm89-pr3
python3 model_client.py --arm dual     --output OUT_DUAL
python3 model_client.py --compare OUT_PORTABLE OUT_DUAL        # offline, no server needed
```

- `--output` must be new or empty. The client refuses to overwrite earlier evidence.
- `--timeout` is the per-HTTP-request timeout. It defaults to 600 s, and larger values are rejected.
- Exit status is 0 only if every recorded check passed. Any failed check, exception, timeout, non-200 reply or interrupt gives exit status 1.
- `report.json` is rewritten atomically after every check. It holds the check list, the counts, the `/server_info` snapshot and the throughput table.
- Every HTTP exchange is saved in `exchanges/NNNN-<label>.json`. The file holds the request object, status, wall time, any transport error, and the raw `response_text`. This includes `/tokenize` token lists and the full retrieval prompts.
- `logprob_probes.json` holds the teacher-forced probe data used for cross-arm comparison.

## Scope

The server under test is single-request-stateless: no sessions and no streaming. The server is expected to run with CUDA graphs on, speculation off, fixed expert placement, 64K context, 4 max active requests and language-model-only. The client asserts only the four settings that `/server_info` is known to expose:
- `disable_cuda_graph == False`
- `speculative_algorithm is None`
- `context_length == 65536`
- `max_running_requests == 4`

A key that is absent counts as a failed check. Expert placement, attention backend and language-only mode are recorded in `report.json`, not asserted. The coordinator should verify them from the launch recipe.

Checks, in run order. All chat checks use `/v1/chat/completions` with `chat_template_kwargs={"thinking": false}`, `temperature=0`, `max_tokens=96` and a fixed `seed`:

| Section | Checks | Description |
|---|---|---|
| server-info | 5 | One fetch check plus the four settings above. |
| basic | 11 | 3 facts, 2 JSON-structure, 3 arithmetic and 3 Unicode cases, each judged against an exact expected object. |
| concurrency | 21 | Cohorts of 1, 2 and 4 simultaneous conversations, 3 turns each, so (1+2+4)×3 checks. Each conversation has its own owner and codes. Turns are released together with a barrier. |
| tools | 14 | Cohorts of 1, 2 and 4. Each slot has a parsed tool-call check (`tool_choice=auto`, exactly one call of the right name and arguments) and a tool-result round-trip check. |
| logprob probes | 4 | 2 fixed probes × 2 repeats. See the alignment rules below. |
| retrieval | 3 | About 4K, 16K and 35K prompt tokens, one chat call each. |
| throughput | 18 | N=1, 2 and 4 with raw `text` prompts, 128 `ignore_eos` tokens; one warmup and five measured rounds. |

The total is **76 checks** when every section runs normally. A section-level crash adds extra failed entries named `section:<name>:crashed`, `...-start-barrier-...` or `client:aborted`, so the count can be higher. The client does not force any particular total.

Design choices that affect interpretation:

- **Canonical history in concurrency cohorts.** Conversations are stateless: each turn re-sends the whole history. Assistant turns in that history are the canonical expected JSON, not the model's output. Every turn is therefore an independent exact check, a single early miss does not cascade, and the prompts are byte-identical across arms. Cross-request contamination shows up as a wrong owner or code.
- **Tool round trip.**
  - The follow-up reuses the model's own parsed tool call if the first check passed.
  - Otherwise it uses a canonical call, and `tool_call_source` in the details says which. The follow-up check is still run and still counted, never skipped.
  - The follow-up uses `tool_choice=none` and asks for `{"available": <int>}`.
- **JSON judging.** The first JSON object in the answer is parsed, and a surrounding markdown fence is tolerated. The object must equal the expected value semantically: key order and whitespace are ignored, but key sets and JSON types are exact, so `"10081"` or `10081.0` is not `10081`, and `true` is not `1`. Extra prose outside the object does not fail a check, but it is recorded as `extra_text`.
- **Retrieval.**
  - The filler is deterministic digit-free prose, generated from a fixed LCG.
  - One unique code sits in a paragraph at the middle of the document, and the question comes after the document with an explicit exact-copy instruction.
  - The prompt is sized by estimating 4.8 chars/token, then calling `/tokenize` (messages + `chat_template_kwargs`) and rescaling the filler. This is at most 3 tokenizations per case, with no loops beyond that.
  - It is accepted within ±5%, and any final size more than ±10% off target or at or above 60K tokens is a failed check.
  - Each case makes exactly 1 generation call, so there are 3 long calls per arm.
  - The check also requires the server's `usage.prompt_tokens` to be within 1% of the tokenized count. This confirms the server saw the whole prompt.
  - The actual token list and payload are in the tokenize exchange files.
- **Throughput.**
  - Per-request latency, tokens/s (`completion_tokens / client wall time`) and cohort aggregate tokens/s are recorded in `report.json` under `throughput`.
  - There is no speed gate. The functional check requires each request to return exactly 128 tokens with `finish_reason=length`.
  - Throughput runs last, so graphs are already warm.
- **Not covered.** Streaming, sessions, speculation, other-model checks, quality benchmarks and statistical equivalence of the two arms are out of scope. No cross-arm numeric tolerance is applied anywhere.

## Input-logprob alignment

Probe procedure, identical in both arms:
1. Each probe is a fixed user message, tokenized with `/tokenize` (model, messages, `chat_template_kwargs`). The probes are natural prose and mixed code/JSON/Unicode. The exact IDs are stored in `logprob_probes.json`.
2. `POST /generate` with `input_ids`, `return_logprob=true`, `top_logprobs_num=0`, `logprob_start_len=max(0, len(ids)-64)` and `max_new_tokens=1`, `temperature=0`.
3. Each probe is run twice. This gives an intra-arm noise floor for the later comparison.

SGLang versions differ on how `meta_info.input_token_logprobs` lines up with the input. The client therefore detects the alignment from the echoed token IDs and does not assume it. Entries are `[logprob, token_id, text]`. With `s = logprob_start_len` and `n = len(ids)`, the client accepts exactly these layouts:
- **Offset `s`**: entry *k* is for `ids[s+k]`, possibly with one trailing extra entry.
- **Offset `s+1`**: entry *k* is for `ids[s+1+k]`, possibly with one trailing extra entry.

The echoed token IDs must equal `ids[offset:]` exactly, and the entry count may exceed `n - offset` by at most 1. Anything else is a failed check. The trailing extra entry, if present, is kept in the raw data and excluded from `positions`, `logprobs` and `token_ids`.

A probe repeat passes only if all of these hold:
- the alignment is detected;
- at least 63 positions are aligned;
- every logprob is a finite number and not above 1e-3, which tolerates rounding noise;
- all token IDs are ints;
- the server's `prompt_tokens` equals `len(ids)`;
- `completion_tokens == 1`.

The pinned server's `SchedulerLogprobResultProcessor._process_input_token_logprobs` inserts a `None` boundary sentinel even when `s > 0`. The client verifies its echoed ID at exactly position `s`, then excludes that single unscored position. The remaining 63 positions must all have finite logprobs; an interior null, wrong ID, shifted null or NaN fails. Seven alignment regression checks cover these cases. Saved per repeat: the raw entries and aligned scored `positions`/`logprobs`/`token_ids`. The initial client incorrectly rejected this sentinel. The failed data was retained; both full arms completed with the corrected alignment checks, including a complete portable rerun.

Cross-arm comparison with `--compare DIR_A DIR_B` works on the same IDs and by absolute token position, never by list index:
- It first requires the ID lists of the two arms to be identical.
- For each probe it uses the first passing repeat in each arm. It prints common-position count, max and mean |Δ logprob|, token-ID mismatches, and each arm's repeat-to-repeat delta.
- Exit status is 1 only if IDs differ, a probe has no passing repeat, fewer than 63 positions are common, or token IDs mismatch at a common position.
- The size of the logprob deltas is reported and not judged. The coordinator decides whether it is acceptable, using the intra-arm repeat deltas as a reference.
