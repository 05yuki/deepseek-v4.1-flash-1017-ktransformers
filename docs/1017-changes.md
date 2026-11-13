# What DeepSeek-V4.1-Flash-1017 changes

Compared against the DeepSeek-V4.1-Flash release this host ran before
(Hub revision `314113f8`; 1017 is `1bae0e10`).

## Weights

All 48 safetensors shards have new contents (every LFS sha256 differs). The shapes
and dtypes are the same, and so are the safetensors headers of the two engram shards
(47 and 48). As a result, an engram manifest built from byte offsets into those shards
works unchanged. `config.json` and `tokenizer.json` are byte-identical.

## Reference inference code (`inference/model.py`)

1. **The attention output is quantized to FP8 and back before `wo_a`.** Both
   `Attention.forward` and `DSparkAttention.forward` gained

   ```python
   act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)   # block 32, "ue8m0", in place
   ```

   This line sits between the inverse RoPE and the bf16 `wo_a` einsum. `convert.py`
   still dequantizes `wo_a` to bf16, so this is an activation-side step only. The
   in-place quantize-dequantize works like this:
   - per 32 values, amax is floored at 1e-4;
   - the scale is `2^ceil(log2(amax / 448))`;
   - `x / scale` is clamped to ±448, rounded to e4m3, and multiplied back.

   Because the new weights were made with this step, a server that feeds `wo_a` a bf16
   `o` runs 1017 off its training numerics.

2. **Indexer: `shared_attn.index_k` is published by every key-owning layer on every
   call.** Before, an owner published it only when one of its compression groups
   completed (`latent is not None`). In the meantime a reading layer could see the
   previous owner's cache. This was a bug in the reference code. The SGLang port used
   here maps each layer to its source layer statically (`source_layer_of` in the
   DeepSeek-V4 memory pool), so it never had this bug and needs no change.

`kernel.py`, `convert.py`, `generate.py` and `engram.py` are unchanged.

## Encoding (`encoding/`)

- **`drop_thinking`.** Without a tools declaration, reasoning is now stripped only
  before the latest *user* request. Tool results no longer count as a new turn, so
  reasoning inside the current turn survives tool calls.
- **Tool namespaces.** A tool can carry a `namespace` (a string, or a name plus an
  optional description). It is encoded as `namespace::name`, and the parser returns
  the name and the namespace separately.

Neither change touches a plain chat without tools.

## Sampling

The 1017 model card recommends `temperature` 1.0 and `top_p` 1.0. The earlier card
recommended `top_p` "0.95 or 1.0", and its evaluations used 0.95.

## How the recipe runs the FP8 step

1017's reference inference code quantizes the attention output `o` to FP8 and back
before the bf16 `wo_a` projection:

```python
act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)  # blocks of 32, power-of-two scales
```

The earlier release did not have this step; 1017's weights were made with it.

- `patches/dsv4_o_fp8_qdq.py`: an in-place Triton kernel, bit-exact against the
  checkpoint's own tilelang `act_quant` (`tools/dsv4-o-fp8-qdq-test.py`). GPUs older
  than sm89 get an fp32 emulation of the e4m3 rounding, also bit-exact
  (`tools/e4m3-emulation-test.py`).
- Tier 0 (`patches/sglang-tier0/01-o-fp8-qdq.patch`) includes it, behind `SGLANG_DSV4_O_FP8_QDQ_BLOCK`.
  `tools/o-fp8-qdq-block.sh` (sourced by `serve.sh`) reads the block size from the
  checkpoint's own `inference/model.py`. The step therefore turns on for 1017 (block
  32) and stays off for the earlier release.
- `patches/sglang-dsv4-o-fp8-qdq.py` installs the same step in another SGLang tree on
  its own. It applies to SGLang main at `8c7ec052` unmodified.

It works the same at any tensor-parallel size. Its 32-value blocks lie inside a head
(512 values), so splitting heads across GPUs never cuts a block.

## What needs no server change

- The tokenizer and `config.json` are identical.
- The reference indexer's `owns_k` fix does not apply to SGLang's port.
- The encoding changes only affect tool calls.
- The card's sampling is `temperature` 1.0, `top_p` 1.0. These are already SGLang's
  defaults for a request that sets neither, since the checkpoint has no
  `generation_config.json`.
