# Contributing

Reports, questions and patches are all welcome.
This page says what to send and what happens to it.

## A report from your machine

`MODEL=<checkpoint dir> tools/qualify.sh report.md` checks a machine end to end in about 15 minutes.
Open an issue with the "Qualification report" template and paste the file.
A failed check is as useful as a pass: attach the server log that the script names.
A report from a checkout with local changes is filed when the changes do not reach loading, attention or the MoE path. Say what they are.
Accepted reports go under [reports/](reports/README.md) and into the machine table of the README with the next tag.

Numbers measured another way are welcome too.
Say the GPU, the CPU, the memory, the tag and how the number was taken.

## A bug

Open an issue with the "Bug" template: the launch line, the tag and the part of the server log around the failure.
`serve.sh` prints the log's path when it starts.
A suspected cause helps but is not needed.

## A patch

The files under `patches/` are the commits of branch `dsv41-1017-main` of
[05yuki/sglang-upstream](https://github.com/05yuki/sglang-upstream/tree/dsv41-1017-main), one commit a file ([docs/patches.md](docs/patches.md)).
[docs/tier1.md](docs/tier1.md) says what each tier 1 patch does, which files it touches and which switch turns it off.

Either form of pull request works:

- Edit the patch file under `patches/` in this repository. Pull requests #11 and #16 were made this way.
- Or open the pull request against the commit in the SGLang branch.

We fold an accepted change into its source commit and regenerate the patch file, so the merged file can differ from yours in its hunk headers.
Your name stays on the change as `Co-authored-by`.

A new feature goes in its own patch, off by default behind a switch, with a line in `docs/tier1.md`.
A fix goes into the patch that has the bug.

## Before you open the pull request

These run without a GPU, and CI runs them on every pull request:

```
python3 tools/test_qualify_cache.py
python3 tools/lend-transition-fences-test.py
```

The CI job `apply` also applies every patch to the pinned SGLang commit.
The full list of tests, and what each needs, is in [docs/qualification.md](docs/qualification.md#tests).

A change you could not run on a GPU is still welcome. Say so, and we run it on ours.

## The smallest check for your change

You do not need the whole of `tools/qualify.sh` for every change.
Pick the row that matches what you touched.
The rows from the NLL down run against a server started with `./serve.sh`.

| what you changed | the check | what it tells |
|---|---|---|
| a tool, a document, a test | the two CPU tests above | nothing else is needed |
| a patch file | the CPU tests, and CI's `apply` | the whole set still applies to the pinned SGLang commit |
| launching, loading, a switch's default | `MODEL=<dir> tools/check.sh` | the server comes up and answers one request; it starts and stops the server itself |
| what the model computes | `python tools/nll.py score <dir> served.json`, then `python tools/nll.py compare tools/nll-ref-1017.json served.json` | how far the served model is from the reference implementation |
| decode speed | `python tools/bench.py` | the median of four 1,024-token requests |
| prefill speed | `python tools/bench.py --prefill-only --lengths 35000` | one 35,000-token prompt of random ids |
| the prefix cache | the same prompt twice, or [tools/qualify-cache.py](docs/qualification.md#multi-session-prefix-cache-qualification) | the second read comes from the cache |
| tool calls, the chat template | `python tools/tool-call-check.py` | calls, streamed calls and results round-trip |

Two things make a number mean something:

- Compare on one machine, before and after. Say which machine.
- Launches of one build differ by about 0.002 to 0.005 in NLL, because the experts held on the GPU differ. A difference inside that is no difference.

A probe of the prefix cache has to generate a few tokens past the prompt.
A request that ends inside its prefill is stored differently from one that decodes.

To see one tier 1 feature's effect, turn it off with its switch in [docs/tier1.md](docs/tier1.md), where it has one, and run the same row again.

## Labels

| label | meaning |
|---|---|
| `qualification` | a `tools/qualify.sh` report |
| `data point` | a measurement taken another way |
| `bug` | something fails or computes wrongly |
| `fixed in main` | the fix is on `main` and waits for the next tag; the issue closes with the tag's name |
| `question` | a question about a machine or a configuration |
| `enhancement` | a new feature or kernel |
| `hardware we lack` | we cannot run this here; a result from a machine that can is what moves it |

## What happens next

We test every pull request on two RTX 5070 Ti with two EPYC 7452 before merging, with `tools/qualify.sh` on one and two GPUs.
The reply says what we ran and what we measured.
A change that does not fit the recipe's defaults can still land as an opt-in switch.

## People who have contributed

| who | what |
|---|---|
| [howl9003](https://github.com/howl9003) | the SM89 sparse attention fallback (#3), CPU handoff packing and shared-expert overlap (#5), SM89 live top-k and fused merge kernels (#7), measurements in #12 |
| [f-wunderlich](https://github.com/f-wunderlich) | the seeded multi-session prefix-cache qualification (#11), the reader lifetime report (#4), qualification data at 1M context (#9), #13, #16 |
| [tattocau](https://github.com/tattocau) | the first RTX 4090 and AVX-512 Xeon report (#14), the DSPARK draft worker diagnosis (#15) |
| [AloeKing](https://github.com/AloeKing) | one RTX 5060 Ti 16 GB with a Threadripper PRO 3975WX (#10) |
| [andreasgru](https://github.com/andreasgru) | one RTX PRO 6000 96 GB with two EPYC 7302 (#1) |
| [0xSero](https://github.com/0xSero) | the engram storage adapter this recipe's adapter derives from |
