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
