"""DeepSeek-V4: SGLANG_DSV4_O_FP8_QDQ_BLOCK=<n> quantizes the attention output o to FP8
and back (blocks of n along each group's row, power-of-two scales) before a bf16 wo_a.
The DeepSeek-V4.1-Flash-1017 reference does that (act_quant(o, 32, "ue8m0", ...,
inplace=True) ahead of the wo_a einsum, in Attention and in the DSpark draft's
attention); its weights expect it. 0 / unset = as before. The fp8 wo_a GEMM paths
quantize o themselves and are left alone. The launcher sets the block from the
checkpoint's own inference code, so a checkpoint without the step keeps it off.

- models/deepseek_common/o_fp8_qdq.py: the kernel (patches/dsv4_o_fp8_qdq.py)
- models/deepseek_v4.py: MQALayer's bf16 / GGUF wo_a branch
- models/deepseek_v4_dspark.py: DSparkAttention's wo_a einsum

Usage: python sglang-dsv4-o-fp8-qdq.py <sglang tree>
"""
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "python/sglang/srt/models"
# every edit is checked first and the files are written only when all of them fit,
# so a tree it does not fit is left as it was
pending = []


def patch(path, edits):
    s = path.read_text()
    if "SGLANG_DSV4_O_FP8_QDQ_BLOCK" in s:
        print(f"already patched {path}")
        return
    for old, new in edits:
        if s.count(old) != 1:
            raise SystemExit(f"{path}: anchor found {s.count(old)} times, nothing changed:\n{old}")
        s = s.replace(old, new)
    pending.append((path, s))


helper = '''

def _o_fp8_qdq_block() -> int:
    """SGLANG_DSV4_O_FP8_QDQ_BLOCK (sglang-dsv4-o-fp8-qdq.py): FP8 quant-dequant of the
    attention output before a bf16 wo_a, in blocks of this many; 0 = off."""
    import os

    return int(os.environ.get("SGLANG_DSV4_O_FP8_QDQ_BLOCK", "0") or 0)


_O_FP8_QDQ_BLOCK = _o_fp8_qdq_block()
'''

patch(root / "deepseek_v4.py", [
    (
        "def _apply_wo_a_bf16_matmul(\n",
        helper.lstrip("\n") + "\n\ndef _apply_wo_a_bf16_matmul(\n",
    ),
    (
        """            else:
                wo_a_weight = getattr(self.wo_a, "weight", None)
""",
        """            else:
                if _O_FP8_QDQ_BLOCK:
                    from sglang.srt.models.deepseek_common.o_fp8_qdq import o_fp8_qdq_

                    o = o_fp8_qdq_(o, _O_FP8_QDQ_BLOCK)
                wo_a_weight = getattr(self.wo_a, "weight", None)
""",
    ),
])

patch(root / "deepseek_v4_dspark.py", [
    (
        """        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        if self._use_fast_kernel:
""",
        """        import os

        # sglang-dsv4-o-fp8-qdq.py: the 1017 reference quantizes o before wo_a here too
        _qdq = int(os.environ.get("SGLANG_DSV4_O_FP8_QDQ_BLOCK", "0") or 0)
        if _qdq:
            from sglang.srt.models.deepseek_common.o_fp8_qdq import o_fp8_qdq_

            o = o_fp8_qdq_(o, _qdq)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        if self._use_fast_kernel:
""",
    ),
])

shutil.copyfile(Path(__file__).with_name("dsv4_o_fp8_qdq.py"), root / "deepseek_common/o_fp8_qdq.py")
print(f"copied {root / 'deepseek_common/o_fp8_qdq.py'}")
for path, s in pending:
    path.write_text(s)
    print(f"patched {path}")
