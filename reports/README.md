# Qualification reports

Each tag's reports live under `reports/<tag>/`, one Markdown file per machine and GPU set.
A report is the unedited output of `tools/qualify.sh` run on a clean checkout of that tag.
A checkout with local changes writes `-dirty` after the tag, and such a report is not filed here.

| tag | machine | answer | tool calls | NLL minus reference | 35K prompt from the cache | decode median | first 35K prefill | report |
|---|---|---|---|---|---|---|---|---|
| 2026-11-04-1912 | 2 × RTX 5070 Ti, 2 × EPYC 7452, Ubuntu 26.04 | pass | 4 of 4 | +0.0028 (SE 0.0045) | 1.9 s again | 53.8 tok/s | 21.2 s | [report](2026-11-04-1912/rtx5070ti-x2-epyc7452-u2604.md) |
| 2026-11-04-1912 | 1 × RTX 5070 Ti, 2 × EPYC 7452, Ubuntu 26.04 | pass | 4 of 4 | -0.0039 (SE 0.0048) | 2.0 s again | 37.9 tok/s | 29.0 s | [report](2026-11-04-1912/rtx5070ti-x1-epyc7452-u2604.md) |
| 2026-11-04-1106 | 2 × RTX 5070 Ti, 2 × EPYC 7452, Ubuntu 26.04 | pass | 4 of 4 | +0.0014 (SE 0.0045) | 1.9 s again | 53.9 tok/s | 22.6 s | [report](2026-11-04-1106/rtx5070ti-x2-epyc7452-u2604.md) |
| 2026-11-04-1106 | 1 × RTX 5070 Ti, 2 × EPYC 7452, Ubuntu 26.04 | pass | 4 of 4 | +0.0029 (SE 0.0039) | 2.0 s again | 38.2 tok/s | 29.3 s | [report](2026-11-04-1106/rtx5070ti-x1-epyc7452-u2604.md) |
| 2026-11-03-1111 | 2 × RTX 5070 Ti, 2 × EPYC 7452, Ubuntu 26.04 | pass | 4 of 4 | +0.0010 (SE 0.0036) | 1.9 s again | 51.6 tok/s | 22.0 s | [report](2026-11-03-1111/rtx5070ti-x2-epyc7452-u2604.md) |
| 2026-11-03-1111 | 1 × RTX 5070 Ti, 2 × EPYC 7452, Ubuntu 26.04 | pass | 4 of 4 | +0.0027 (SE 0.0032) | 2.0 s again | 33.3 tok/s | 36.6 s | [report](2026-11-03-1111/rtx5070ti-x1-epyc7452-u2604.md) |

## Adding a report

Run the tag you have checked out:

```
MODEL=<checkpoint dir> tools/qualify.sh report.md
```

Then open an issue with the "Qualification report" template and paste the file.
The report names the hardware and software, and nothing that identifies the machine.
A report from a GPU or CPU not listed above is the most useful one.
A failed check is filed too, with the server log that `tools/qualify.sh` names.

File names follow `<gpu>-x<count>-<cpu>-<os>.md`, for example `rtx5070ti-x2-epyc7452-u2604.md`.
