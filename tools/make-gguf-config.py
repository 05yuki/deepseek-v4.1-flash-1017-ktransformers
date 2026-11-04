#!/usr/bin/env python3
"""The config directory SGLang reads for a GGUF build: deepseek-ai/DeepSeek-V4.1-Flash-1017's config.json,
tokenizer and encoding/inference code, with the checkpoint's FP8 quantization_config replaced by the GGUF one.
No weights: the dense weights come from the GGUF (KT_GGUF_V41_DENSE), the routed experts through kt-kernel.
The GGUF holds no vision tower, so the model is served as text only (language_model_only).
usage: make-gguf-config.py <dir with the base model's config.json, tokenizer and encoding/> <out dir>"""
import json
import shutil
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for name in ("tokenizer.json", "tokenizer_config.json"):
    shutil.copy2(src / name, out / name)
for d in ("encoding", "inference"):
    if (src / d).is_dir():
        shutil.copytree(src / d, out / d, dirs_exist_ok=True)
cfg = json.loads((src / "config.json").read_text())
cfg["quantization_config"] = {"quant_method": "gguf", "modules_to_not_convert": ["compressor", "indexer"]}
# the GGUF has no vision tower: text only
cfg["language_model_only"] = True
cfg["_gguf_note"] = ("config and tokenizer of deepseek-ai/DeepSeek-V4.1-Flash-1017 without its quantization_config: "
                     "the weights come from the GGUF (dense) and kt-kernel LLAMAFILE (routed experts); the dense "
                     "layers stay in their GGUF types except the compressor and the indexer")
(out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
print(f"wrote {out}")
