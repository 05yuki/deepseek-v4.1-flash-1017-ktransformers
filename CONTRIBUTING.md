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

Open an issue with the launch line, the tag and the part of the server log around the failure.
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

If your change touches what the model computes, say how its output was compared: `tools/nll.py` against `tools/nll-ref-1017.json` is the measure this repository uses.
If it changes speed, give the number before and after on the same machine, and say which machine.
A change you could not run on a GPU is still welcome. Say so, and we run it on ours.

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
