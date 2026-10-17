# DeepSeek-V4.1-Flash-1017 on SGLang + KTransformers (CPU/GPU hybrid inference)

This repository adds support for
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
to an existing DeepSeek-V4.1-Flash setup on [SGLang](https://github.com/sgl-project/sglang)
with [KTransformers](https://github.com/kvcache-ai/ktransformers). In that setup the
routed experts run on the CPU through kt-kernel, and attention plus the hot experts
run on the GPU.

If your machine already serves the earlier DeepSeek-V4.1-Flash this way, it can serve
1017. The checkpoint has the same size, the same tensor layout, the same config and
the same tokenizer. Only the weights are new. Running it correctly takes one small
patch to SGLang. That patch is independent of the hardware: one or several GPUs, one
or two CPU sockets, any tensor-parallel size.

## The change: o in FP8 before `wo_a`

1017's reference inference code quantizes the attention output `o` to FP8 and back
before the bf16 `wo_a` projection:

```python
act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)  # blocks of 32, power-of-two scales
```

The earlier release did not have this step. The new weights (all 48 shards changed)
were made with it, so a server that feeds `wo_a` an unrounded bf16 `o` runs 1017 off
its own numerics. The step moves `o` by 1.8-2.6% (relative L2) in every layer.

- `patches/dsv4_o_fp8_qdq.py`: an in-place Triton kernel. It is bit-exact against the
  checkpoint's own tilelang `act_quant`: 0 differing bf16 values at 1, 3, 37 and 4,096
  tokens (`tools/dsv4-o-fp8-qdq-test.py`).
  - On sm89 and newer (RTX 40 / 50, Hopper, Blackwell), the e4m3 rounding uses the
    hardware conversion.
  - On Ampere and older (RTX 30, A100), Triton has no e4m3 type, so the kernel does
    the same rounding in fp32 arithmetic. That emulation matches torch's
    `float8_e4m3fn` cast bit for bit, signed zeros included, on 118,343,666 values:
    every bf16 value at every power-of-two scale, plus random fp32
    (`tools/e4m3-emulation-test.py`, runs on the CPU).
- `patches/sglang-dsv4-o-fp8-qdq.py`: installs the kernel in an SGLang tree behind
  `SGLANG_DSV4_O_FP8_QDQ_BLOCK=<n>` (0 or unset = off).
  - It covers the bf16 / GGUF `wo_a` branch of `MQALayer` and the DSpark draft's
    attention.
  - The fp8 `wo_a` GEMM paths quantize `o` themselves and are left as they are. They
    only run for checkpoints with 128 × 128 weight blocks; 1017's are 32 × 32, so
    1017 always takes the bf16 path.
  - It applies to SGLang's DeepSeek-V4.1 branch (sgl-project/sglang `dsv4.1`, at
    `1aa0e962`), the tree KTransformers' kt-kernel runs inside here.
- `tools/o-fp8-qdq-block.sh`: sourced by a launcher after it sets `$model`. It reads
  the block size from the checkpoint's own `inference/model.py`. The step turns on for
  1017 (block 32) and stays off for the earlier release, so one launcher serves both.

The step works at any tensor-parallel size. It touches one activation inside each
attention layer, and its 32-value blocks lie inside a head (512 values), so splitting
heads across GPUs never cuts a block. It runs on any CUDA GPU that Triton supports.
ROCm is untested.

Everything else 1017 changes is in [docs/1017-changes.md](docs/1017-changes.md).
None of it needs a server change:
- The tokenizer and `config.json` are identical. The safetensors headers of the
  engram shards are identical too, so an offset-based engram manifest carries over.
- The reference indexer's `owns_k` fix does not apply to SGLang's port, which maps
  index-key sources statically.
- The encoding changes only affect tool calls.
- The card's sampling is `temperature` 1.0, `top_p` 1.0.

## Using it

1. Have a DeepSeek-V4.1-Flash setup on SGLang + KTransformers: SGLang's `dsv4.1`
   branch with kt-kernel. Hardware needs are the same as for the earlier release, since the
   checkpoint is the same size (476 GB on disk).
2. Download the checkpoint:
   `hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir <dir>`.
3. Patch SGLang and check the kernel on your GPU:
   ```
   python patches/sglang-dsv4-o-fp8-qdq.py <sglang tree>
   python tools/dsv4-o-fp8-qdq-test.py <dir>   # expects PASS; tilelang is optional
   ```
4. Start the server with `SGLANG_DSV4_O_FP8_QDQ_BLOCK=32` in its environment. A
   launcher can instead source `tools/o-fp8-qdq-block.sh` after setting `$model`.
   Then point `--model-path` (and kt-kernel's weight path) at the new directory.
5. Sample at `temperature` 1.0, `top_p` 1.0. The checkpoint has no
   `generation_config.json`, so these are already SGLang's defaults for a request
   that sets neither.

## Results

Measured on one machine, as an example (not a requirement):
- 2 × AMD EPYC 7452
- 512 GB DDR4
- 2 × RTX 5070 Ti 16 GB
- the checkpoint on NVMe

The setup is SGLang `dsv4.1` + kt-kernel with MXFP4 CPU experts, 5 experts per layer
resident on the GPUs, a 1M-token context and one request at a time. Each column is one
launch, on the same day and with the same launcher.

| | DeepSeek-V4.1-Flash (earlier release) | DeepSeek-V4.1-Flash-1017, this patch |
|---|---|---|
| decode, Japanese prose, median of the server's tok/s | 30.01 tok/s | 30.62 tok/s |
| prefill, 35,000 tokens: first / second prompt | 63.8 s / 44.5 s | 67.0 s / 51.1 s ¹ |
| mean NLL, 6,120 tokens of Japanese prose (24 × 256) | 3.0558 | 3.1369 ² |

¹ The 1017 launch came straight after copying the 476 GB checkpoint onto the NVMe,
with the page cache full. Single launches; not yet repeated.

² Different weights, so this is not a quality comparison between the releases. The
comparison that tests the patch, 1017 with and without the FP8 step on the same
text, is running and will be added here.

## Not done

- The expert placement (which experts stay on the GPU) used for the numbers above
  comes from the earlier release's routing. It loads and runs, but it is not 1017's
  hottest set.
- Tool namespaces and the new `drop_thinking` turn boundary are not in SGLang's
  DeepSeek-V4 chat encoding.

## License

The patches and tools are under Apache-2.0 (see `LICENSE`). The model weights are under
DeepSeek's license (see the model card).
