# Shipped chunk hints

One file a GPU model and count: the prefill-lend rank files a calibrated setup of `serve.sh`'s
defaults wrote (`lib/kt-lend-auto.sh`, `docs/tier1.md` §13), with the GPU's name and memory. On a
first launch at the defaults with no hint of its own, `serve.sh` seeds the matching file as the
setup's hint, with 512 MiB less free memory than measured, and skips the calibration launch. The
server rewrites the hint from its own measurement at the first long prompt. `KT_LEND_SEED=0`
calibrates instead.

To contribute one from a machine whose GPU is not here: calibrate (the first launch does), serve a
long prompt (`tools/qualify.sh` does), then

```
KT_LEND_PRINT_KEY=1 ./serve.sh            # prints the setup's hint key
tools/lend-hint-export.py <key> --recipe <tag> --note "<CPU, anything to know>"
```

and send the file with the qualification report. The file carries no path, host or user name.

| file | GPU | measured |
| --- | --- | --- |
| `nvidia-geforce-rtx-5070-ti-16g-x1.json` | RTX 5070 Ti 16 GB, one | 2026-11-02, recipe 2026-11-02-1317 |
| `nvidia-geforce-rtx-5070-ti-16g-x2.json` | RTX 5070 Ti 16 GB, two | 2026-11-01, recipe 2026-11-01-1946 |
