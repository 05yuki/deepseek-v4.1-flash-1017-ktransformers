# Sourced by a launcher after it sets $model: turns on the FP8 quant-dequant of the
# attention output before wo_a (patches/sglang-dsv4-o-fp8-qdq.py) when the checkpoint's
# own inference code does it, with the block size that code uses. A checkpoint without
# the step (the earlier V4.1 release) leaves it off. SGLANG_DSV4_O_FP8_QDQ_BLOCK set by
# hand (0 = off) wins.
if [[ -z "${SGLANG_DSV4_O_FP8_QDQ_BLOCK:-}" ]] && grep -q "act_quant(o, fp8_block_size" "$model/inference/model.py" 2>/dev/null; then
  SGLANG_DSV4_O_FP8_QDQ_BLOCK="$(sed -n 's/^fp8_block_size = \([0-9]*\).*/\1/p' "$model/inference/model.py" | head -1)"
  export SGLANG_DSV4_O_FP8_QDQ_BLOCK="${SGLANG_DSV4_O_FP8_QDQ_BLOCK:-32}"
fi
