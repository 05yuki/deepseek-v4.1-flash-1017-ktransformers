"""The token pieces tools/nll.py and tools/ref-nll.py score: 24 pieces of 480 tokens, 12 from the
start of each public-domain text in tools/nll-text/ (Japanese, then English), each opening with the
tokenizer's BOS. Both scorers cut them with the served model's own tokenizer, so they score the same ids."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEXTS = ("nll-text/ja-kokoro.txt", "nll-text/en-holmes.txt")
PIECE = 480
PER_TEXT = 12


def pieces(tok):
    out = []
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    for name in TEXTS:
        ids = tok.encode(open(os.path.join(HERE, name), encoding="utf-8").read(), add_special_tokens=False)
        body = PIECE - len(bos)
        assert len(ids) >= body * PER_TEXT, f"{name}: {len(ids)} tokens, {body * PER_TEXT} needed"
        out += [bos + ids[i * body:(i + 1) * body] for i in range(PER_TEXT)]
    return out


def tokenizer(model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
