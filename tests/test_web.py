"""Tarayıcı sürümü (web/motor.js) Python modeliyle aynı sonucu veriyor mu?

Node.js kuruluysa JS tokenizer'ı ve JS ileri geçişini çalıştırıp PyTorch ile
karşılaştırır. int8 nicemlemenin etkisini ayırmak için Python tarafında da
ağırlıkları aynı şekilde nicemleyip geri çözüyoruz; böylece fark sadece
hesaplama hatası kadar (çok küçük) olmalı.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from turkce_gpt.chat import encode_prompt
from turkce_gpt.checkpoint import save_checkpoint
from turkce_gpt.config import make_config
from turkce_gpt.export_web import export, quantize_rows
from turkce_gpt.model import GPT
from turkce_gpt.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
TEXT = ("İstanbul'da yaşayan çocuklar evlerimizden koştu! 2026 yılında 12.500 kişi geldi. "
        "Gaziantep'in baklavası ve fıstığı meşhurdur; şehir çok güzel.  Yeni   satır\n")

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js kurulu değil")


def test_quantize_rows_roundtrip():
    w = np.random.default_rng(0).normal(size=(5, 16)).astype(np.float32)
    q, scale = quantize_rows(w)
    assert q.dtype == np.int8 and np.abs(q).max() == 127
    assert np.abs(q * scale[:, None] - w).max() <= scale.max() / 2 + 1e-6


@needs_node
def test_js_matches_python(tmp_path):
    tok = BPETokenizer()
    tok.train([TEXT] * 4, vocab_size=400)
    torch.manual_seed(0)
    model = GPT(make_config("deneme", tok.vocab_size, block_size=32)).eval()
    save_checkpoint(tmp_path / "model.pt", model, tok, {})
    export(tmp_path / "model.pt", tmp_path / "web")

    texts = [TEXT, "Merhaba dünya", "a <|asistan|> b", "🙂 emoji ve ĞÜŞİÖÇ ğüşıöç", ""]
    chat = [{"role": "kullanici", "content": "Gaziantep neyle ünlü?"},
            {"role": "asistan", "content": "Baklavasıyla."},
            {"role": "kullanici", "content": "Peki fıstık?"}]
    tokens = tok.encode(TEXT)[:20]
    (tmp_path / "girdi.json").write_text(json.dumps({"texts": texts, "chat": chat, "tokens": tokens}))
    out = subprocess.run(["node", str(ROOT / "tests" / "web_parity.mjs"), str(tmp_path / "web"),
                          str(tmp_path / "girdi.json")], capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)

    # 1) Tokenizer birebir aynı
    assert js["encoded"] == [tok.encode(t) for t in texts]
    assert js["chat"] == encode_prompt(tok, chat)

    # 2) Model: Python'da ağırlıkları aynı şekilde nicemleyip karşılaştır
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2:
                q, scale = quantize_rows(p.numpy())
                p.copy_(torch.from_numpy(q.astype(np.float32) * scale[:, None]))
        logits, _ = model(torch.tensor([tokens]))
    np.testing.assert_allclose(np.array(js["logits"]), logits[0].numpy(), atol=2e-4)
